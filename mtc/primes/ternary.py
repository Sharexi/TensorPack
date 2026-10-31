"""
Balanced Ternary Computing (F_3) and Multiplier-Free Tensor Operations.
"""

from typing import Tuple
import numpy as np
from numba import njit, prange


@njit(fastmath=True, inline="always")
def pack_5_trits(t0: int, t1: int, t2: int, t3: int, t4: int) -> int:
    """
    Pack 5 balanced trits {-1, 0, 1} into a single uint8 byte (0..242).
    Uses base-3 encoding with offset +1.
    """
    u0 = t0 + 1  # 0, 1, 2
    u1 = t1 + 1
    u2 = t2 + 1
    u3 = t3 + 1
    u4 = t4 + 1
    return u0 + u1 * 3 + u2 * 9 + u3 * 27 + u4 * 81


@njit(fastmath=True, inline="always")
def unpack_5_trits(byte_val: int):
    """
    Unpack a uint8 byte (0..242) into 5 balanced trits {-1, 0, 1}.
    """
    val = byte_val
    u0 = val % 3
    val = val // 3
    u1 = val % 3
    val = val // 3
    u2 = val % 3
    val = val // 3
    u3 = val % 3
    val = val // 3
    u4 = val % 3

    return u0 - 1, u1 - 1, u2 - 1, u3 - 1, u4 - 1


@njit(parallel=True, fastmath=True)
def pack_ternary_vector(trits: np.ndarray) -> np.ndarray:
    """
    Pack a 1D array of balanced trits {-1, 0, 1} into uint8 bytes (5 trits per byte).
    """
    n = len(trits)
    num_bytes = (n + 4) // 5
    out = np.zeros(num_bytes, dtype=np.uint8)

    for i in prange(num_bytes):
        idx = i * 5
        t0 = int(trits[idx]) if idx < n else 0
        t1 = int(trits[idx + 1]) if idx + 1 < n else 0
        t2 = int(trits[idx + 2]) if idx + 2 < n else 0
        t3 = int(trits[idx + 3]) if idx + 3 < n else 0
        t4 = int(trits[idx + 4]) if idx + 4 < n else 0
        out[i] = pack_5_trits(t0, t1, t2, t3, t4)

    return out


@njit(parallel=True, fastmath=True)
def unpack_ternary_vector(packed: np.ndarray, length: int) -> np.ndarray:
    """Unpack uint8 bytes back into a 1D array of balanced trits {-1, 0, 1}."""
    out = np.empty(length, dtype=np.int8)
    num_bytes = len(packed)

    for i in prange(num_bytes):
        t0, t1, t2, t3, t4 = unpack_5_trits(int(packed[i]))
        idx = i * 5
        if idx < length:
            out[idx] = t0
        if idx + 1 < length:
            out[idx + 1] = t1
        if idx + 2 < length:
            out[idx + 2] = t2
        if idx + 3 < length:
            out[idx + 3] = t3
        if idx + 4 < length:
            out[idx + 4] = t4

    return out


@njit(parallel=True, fastmath=True)
def ternary_multiplier_free_gemm(
    W_trits: np.ndarray,
    X: np.ndarray,
    out: np.ndarray
):
    """
    Multiplier-Free GEMM: Y = W @ X
    W_trits has values strictly in {-1, 0, 1}.
    X has arbitrary floating-point values.
    Computes Y using ONLY additions and subtractions (zero multipliers!).
    """
    M, K = W_trits.shape
    K2, N = X.shape
    assert K == K2

    for i in prange(M):
        for j in range(N):
            acc = 0.0
            for k in range(K):
                w = W_trits[i, k]
                if w == 1:
                    acc += X[k, j]
                elif w == -1:
                    acc -= X[k, j]
                # w == 0 requires zero operations
            out[i, j] = acc
