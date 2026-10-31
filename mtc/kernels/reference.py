"""
Reference Implementations and Numerical Validation Metrics for Stage 0.
"""

from typing import Tuple
import numpy as np
from numba import njit, prange


def fp32_gemm_numpy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Standard FP32 BLAS matrix multiplication reference."""
    return np.matmul(A.astype(np.float32), B.astype(np.float32))


@njit(parallel=True, fastmath=True)
def fp32_gemm_numba(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Numba parallel tiled FP32 GEMM for compiler-parity baseline."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = np.zeros((M, N), dtype=np.float32)

    for i in prange(M):
        for k in range(K):
            a_ik = A[i, k]
            for j in range(N):
                C[i, j] += a_ik * B[k, j]
    return C


def relative_frobenius_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Relative Frobenius norm error: ||y_true - y_pred||_F / ||y_true||_F."""
    diff = y_true.astype(np.float64) - y_pred.astype(np.float64)
    norm_diff = np.linalg.norm(diff)
    norm_true = np.linalg.norm(y_true.astype(np.float64))
    if norm_true < 1e-12:
        return float(norm_diff)
    return float(norm_diff / norm_true)


def max_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Maximum absolute difference max(|y_true - y_pred|)."""
    return float(np.max(np.abs(y_true.astype(np.float64) - y_pred.astype(np.float64))))


def signal_to_noise_ratio_db(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Signal-to-Noise Ratio (SNR) in decibels (dB)."""
    signal_power = np.mean(y_true.astype(np.float64) ** 2)
    noise_power = np.mean((y_true.astype(np.float64) - y_pred.astype(np.float64)) ** 2)
    if noise_power < 1e-15:
        return 120.0  # Cap at +120 dB
    if signal_power < 1e-15:
        return 0.0
    return float(10.0 * np.log10(signal_power / noise_power))
