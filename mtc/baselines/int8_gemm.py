"""
Conventional INT8 Quantized GEMM Baseline.
Symmetric quantization with INT32 accumulation and FP32 output dequantization.
"""

from typing import Tuple
import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def int8_gemm_kernel(
    A_q: np.ndarray,
    B_q: np.ndarray,
    scale_a: float,
    scale_b: float
) -> np.ndarray:
    """
    Parallel INT8 GEMM kernel.
    Multiplies int8 matrices A_q (M, K) and B_q (K, N), accumulates in int32,
    and scales back to float32.
    """
    M, K = A_q.shape
    K2, N = B_q.shape
    scale_ab = np.float32(scale_a * scale_b)
    C = np.zeros((M, N), dtype=np.float32)

    for i in prange(M):
        for j in range(N):
            acc = 0
            for k in range(K):
                acc += int(A_q[i, k]) * int(B_q[k, j])
            C[i, j] = np.float32(acc) * scale_ab
    return C


class INT8GEMM:
    """Conventional INT8 Quantized GEMM Pipeline."""

    @staticmethod
    def quantize(matrix: np.ndarray) -> Tuple[np.ndarray, float]:
        arr = np.asarray(matrix, dtype=np.float32)
        max_abs = float(np.max(np.abs(arr)))
        scale = max_abs / 127.0 if max_abs > 1e-8 else 1.0
        q = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)
        return q, scale

    @staticmethod
    def run_kernel(A_q: np.ndarray, B_q: np.ndarray, scale_a: float, scale_b: float) -> np.ndarray:
        return int8_gemm_kernel(A_q, B_q, scale_a, scale_b)

    @classmethod
    def run_end_to_end(cls, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A_q, scale_a = cls.quantize(A)
        B_q, scale_b = cls.quantize(B)
        return cls.run_kernel(A_q, B_q, scale_a, scale_b)
