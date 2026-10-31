"""
Conventional INT4 Packed GEMM Baseline.
Symmetric 4-bit quantization (2 values per uint8 byte) with INT32 accumulation.
"""

from typing import Tuple
import numpy as np
from numba import njit, prange
from ..core.packing import pack_p4_pair, unpack_p4_pair


@njit(parallel=True, fastmath=True)
def pack_int4_matrix(q: np.ndarray) -> np.ndarray:
    """Pack an (M, K) int8 array (-8..7) into (M, K // 2) uint8 bytes."""
    M, K = q.shape
    K_packed = K // 2
    out = np.empty((M, K_packed), dtype=np.uint8)

    for i in prange(M):
        for k_p in range(K_packed):
            k = k_p * 2
            out[i, k_p] = pack_p4_pair(int(q[i, k]), int(q[i, k + 1]))
    return out


@njit(parallel=True, fastmath=True)
def int4_gemm_kernel(
    A_packed: np.ndarray,
    B_packed: np.ndarray,
    scale_a: float,
    scale_b: float,
    M: int,
    K: int,
    N: int
) -> np.ndarray:
    """
    Parallel INT4 packed GEMM kernel.
    A_packed is (M, K // 2), B_packed is (K, N // 2).
    Unpacks 4-bit nibbles and accumulates into INT32.
    """
    K_half = K // 2
    N_half = N // 2
    scale_ab = np.float32(scale_a * scale_b)
    C = np.zeros((M, N), dtype=np.float32)

    for i in prange(M):
        for j_p in range(N_half):
            j = j_p * 2
            acc0 = 0
            acc1 = 0
            for k_p in range(K_half):
                k = k_p * 2
                # Unpack two values from A: A[i, k] and A[i, k+1]
                a0, a1 = unpack_p4_pair(int(A_packed[i, k_p]))

                # Unpack B[k, j], B[k, j+1] and B[k+1, j], B[k+1, j+1]
                b00, b01 = unpack_p4_pair(int(B_packed[k, j_p]))
                b10, b11 = unpack_p4_pair(int(B_packed[k + 1, j_p]))

                acc0 += int(a0) * int(b00) + int(a1) * int(b10)
                acc1 += int(a0) * int(b01) + int(a1) * int(b11)

            C[i, j] = np.float32(acc0) * scale_ab
            C[i, j + 1] = np.float32(acc1) * scale_ab

    return C


class INT4GEMM:
    """Conventional INT4 Packed GEMM Pipeline."""

    @staticmethod
    def quantize_and_pack(matrix: np.ndarray) -> Tuple[np.ndarray, float]:
        arr = np.asarray(matrix, dtype=np.float32)
        max_abs = float(np.max(np.abs(arr)))
        scale = max_abs / 7.0 if max_abs > 1e-8 else 1.0
        q = np.clip(np.round(arr / scale), -8, 7).astype(np.int8)
        packed = pack_int4_matrix(q)
        return packed, scale

    @staticmethod
    def run_kernel(
        A_packed: np.ndarray,
        B_packed: np.ndarray,
        scale_a: float,
        scale_b: float,
        M: int,
        K: int,
        N: int
    ) -> np.ndarray:
        return int4_gemm_kernel(A_packed, B_packed, scale_a, scale_b, M, K, N)

    @classmethod
    def run_end_to_end(cls, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        M, K = A.shape
        _, N = B.shape
        A_packed, scale_a = cls.quantize_and_pack(A)
        B_packed, scale_b = cls.quantize_and_pack(B)
        return cls.run_kernel(A_packed, B_packed, scale_a, scale_b, M, K, N)
