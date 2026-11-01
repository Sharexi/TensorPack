"""
Fair quantization baselines for accuracy-per-bit comparisons.

Every function takes a matrix W of shape (K, N) whose reduction dimension K
runs along the rows (to quantize the left operand A of A @ B, pass A.T and
transpose back), and returns (W_dequant, bits_per_element). bits_per_element
counts ALL stored bits: payload + scales + any side data.

These are simulated (quantize -> dequantize) implementations used for accuracy
studies, as is standard practice; they follow the published format specs:

- Per-tensor INT8 / INT4: one fp32 scale for the whole matrix (the old repo baseline).
- Block INTb (Bs x Bs): the same 16x16 tiling MTC uses, one fp32 scale per block.
- Group INTb (g along K): llama.cpp Q4_0 / GPTQ-RTN style groups, fp16 scale per group.
- MXFP4 / MXINT8: OCP Microscaling v1.0, 32-element blocks along K, E8M0 shared scale.
- NVFP4: 16-element blocks, E2M1 elements, FP8 (E4M3) block scale + fp32 tensor scale.
- Group INT4 + outliers: SpQR / LLM.int8()-style, top fraction of |w| kept in fp16
  with a 32-bit index, the rest group-quantized.
"""

from typing import Tuple
import numpy as np

FP4_E2M1_GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def _sym_int(x: np.ndarray, scale: np.ndarray, bits: int) -> np.ndarray:
    qmax = 2 ** (bits - 1) - 1
    safe = np.where(scale == 0, 1.0, scale)
    return np.clip(np.round(x / safe), -qmax - 1, qmax) * scale


def per_tensor_int(W: np.ndarray, bits: int) -> Tuple[np.ndarray, float]:
    qmax = 2 ** (bits - 1) - 1
    s = np.float32(np.abs(W).max() / qmax)
    return _sym_int(W, s, bits).astype(np.float32), bits + 32.0 / W.size


def block_int(W: np.ndarray, bits: int, Bs: int = 16) -> Tuple[np.ndarray, float]:
    K, N = W.shape
    blk = W.reshape(K // Bs, Bs, N // Bs, Bs)
    s = np.abs(blk).max(axis=(1, 3), keepdims=True) / (2 ** (bits - 1) - 1)
    return _sym_int(blk, s, bits).reshape(K, N).astype(np.float32), bits + 32.0 / (Bs * Bs)


def group_int(W: np.ndarray, bits: int, g: int = 32) -> Tuple[np.ndarray, float]:
    K, N = W.shape
    grp = W.reshape(K // g, g, N)
    s = (np.abs(grp).max(axis=1, keepdims=True) / (2 ** (bits - 1) - 1)).astype(np.float16).astype(np.float32)
    return _sym_int(grp, s, bits).reshape(K, N).astype(np.float32), bits + 16.0 / g


def _round_to_grid(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Round |x| to nearest value of a non-negative sorted grid (saturating), keep sign."""
    a = np.minimum(np.abs(x), grid[-1])
    idx = np.clip(np.searchsorted(grid, a), 1, len(grid) - 1)
    lo, hi = grid[idx - 1], grid[idx]
    return np.sign(x) * np.where(a - lo <= hi - a, lo, hi)


def mxfp4(W: np.ndarray, k: int = 32) -> Tuple[np.ndarray, float]:
    """OCP MXFP4: shared exponent = floor(log2(amax)) - emax_elem (emax=2 for E2M1)."""
    K, N = W.shape
    grp = W.reshape(K // k, k, N)
    amax = np.abs(grp).max(axis=1, keepdims=True)
    e = np.floor(np.log2(np.where(amax > 0, amax, 1.0))) - 2
    X = np.exp2(np.clip(e, -127, 127)).astype(np.float32)
    return (_round_to_grid(grp / X, FP4_E2M1_GRID) * X).reshape(K, N).astype(np.float32), 4 + 8.0 / k


def mxint8(W: np.ndarray, k: int = 32) -> Tuple[np.ndarray, float]:
    """OCP MXINT8: 8-bit two's complement with implicit scale 2^-6, E8M0 shared scale."""
    K, N = W.shape
    grp = W.reshape(K // k, k, N)
    amax = np.abs(grp).max(axis=1, keepdims=True)
    X = np.exp2(np.floor(np.log2(np.where(amax > 0, amax, 1.0)))).astype(np.float32)
    q = np.clip(np.round(grp / X * 64.0), -128, 127) / 64.0
    return (q * X).reshape(K, N).astype(np.float32), 8 + 8.0 / k


def _to_e4m3(x: np.ndarray) -> np.ndarray:
    """Round positive values to nearest FP8 E4M3 (max 448, min subnormal 2^-9)."""
    x = np.clip(x, 0, 448.0)
    e = np.floor(np.log2(np.maximum(x, 2.0 ** -9)))
    e = np.maximum(e, -6)  # subnormal range shares exponent -6
    step = np.exp2(e - 3)
    return np.minimum(np.round(x / step) * step, 448.0)


def nvfp4(W: np.ndarray, k: int = 16) -> Tuple[np.ndarray, float]:
    """NVFP4: E2M1 elements, E4M3 scale per 16 elements, fp32 per-tensor scale."""
    K, N = W.shape
    grp = W.reshape(K // k, k, N)
    amax_t = float(np.abs(W).max()) or 1.0
    g_scale = amax_t / (6.0 * 448.0)
    amax = np.abs(grp).max(axis=1, keepdims=True)
    s = (_to_e4m3(amax / 6.0 / g_scale) * g_scale).astype(np.float32)
    safe = np.where(s == 0, 1.0, s)
    q = _round_to_grid(grp / safe, FP4_E2M1_GRID) * s
    return q.reshape(K, N).astype(np.float32), 4 + 8.0 / k + 32.0 / W.size


def group_int_with_outliers(
    W: np.ndarray, bits: int = 4, g: int = 128, outlier_frac: float = 0.005
) -> Tuple[np.ndarray, float]:
    """Keep the top `outlier_frac` |w| exactly-ish (fp16 + 32-bit index), group-quantize the rest."""
    flat = np.abs(W).ravel()
    n_out = int(round(outlier_frac * flat.size))
    mask = np.zeros(W.shape, dtype=bool)
    if n_out > 0:
        idx = np.argpartition(flat, -n_out)[-n_out:]
        mask.ravel()[idx] = True
    inliers = np.where(mask, 0.0, W).astype(np.float32)
    deq, b = group_int(inliers, bits, g)
    deq = np.where(mask, W.astype(np.float16).astype(np.float32), deq)
    return deq.astype(np.float32), b + n_out * 48.0 / W.size


def awq_group_int(
    W: np.ndarray, X_cal: np.ndarray, bits: int = 4, g: int = 128, n_grid: int = 11
) -> Tuple[np.ndarray, float]:
    """
    AWQ-style activation-aware scaling (Lin et al., 2024): scale input channel k
    by s_k = mean|X[:, k]|^alpha before group quantization and fold 1/s_k back,
    grid-searching alpha on calibration activations. Per-channel scales are
    folded into the previous op in AWQ, so they cost no extra bits.
    """
    act = np.abs(X_cal).mean(axis=0).astype(np.float64)
    act = np.maximum(act, 1e-8)
    Y = X_cal @ W
    best, best_err = None, np.inf
    for alpha in np.linspace(0.0, 1.0, n_grid):
        s = (act ** alpha)
        s = (s / np.sqrt(s.max() * s.min())).astype(np.float32)
        deq, b = group_int(W * s[:, None], bits, g)
        deq = deq / s[:, None]
        err = float(np.linalg.norm(Y - X_cal @ deq))
        if err < best_err:
            best, best_err = (deq.astype(np.float32), b), err
    return best
