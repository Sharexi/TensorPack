"""
Adaptive Precision and Joint Precision Negotiation Engine for MTC-8.
"""

from typing import Tuple, Dict
import numpy as np

from .types import PrecisionMode


def quantize_and_dequantize_block(
    block: np.ndarray,
    precision: PrecisionMode
) -> Tuple[np.ndarray, float]:
    """
    Simulate quantization and dequantization of a 2D block to calculate exact error.
    Returns (reconstructed_block, scale).
    """
    arr = np.asarray(block, dtype=np.float32)
    max_abs = float(np.max(np.abs(arr)))

    if max_abs < 1e-8:
        return np.zeros_like(arr), 1.0

    if precision == PrecisionMode.P2:
        scale = max_abs / 1.0
        q = np.clip(np.round(arr / scale), -2, 1).astype(np.int8)
    elif precision == PrecisionMode.P4:
        scale = max_abs / 7.0
        q = np.clip(np.round(arr / scale), -8, 7).astype(np.int8)
    elif precision == PrecisionMode.P8:
        scale = max_abs / 127.0
        q = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)
    elif precision == PrecisionMode.P16:
        scale = max_abs / 32767.0
        q = np.clip(np.round(arr / scale), -32768, 32767).astype(np.int16)
    else:
        raise ValueError(f"Unknown precision: {precision}")

    recon = q.astype(np.float32) * scale
    return recon, scale


def compute_frobenius_error(orig: np.ndarray, recon: np.ndarray) -> float:
    """Compute Frobenius norm of difference ||orig - recon||_F."""
    diff = orig.astype(np.float32) - recon.astype(np.float32)
    return float(np.linalg.norm(diff))


def compute_relative_error(orig: np.ndarray, recon: np.ndarray) -> float:
    """Compute relative Frobenius error ||orig - recon||_F / ||orig||_F."""
    orig_norm = float(np.linalg.norm(orig))
    if orig_norm < 1e-8:
        return 0.0
    return compute_frobenius_error(orig, recon) / orig_norm


def select_independent_precision(
    block: np.ndarray,
    epsilon: float = 0.05
) -> Tuple[PrecisionMode, float]:
    """
    Standard independent precision selection based on block reconstruction error:
    Select lowest precision where E(block, p) <= epsilon.
    Returns (selected_precision, scale).
    """
    for mode in [PrecisionMode.P2, PrecisionMode.P4, PrecisionMode.P8]:
        recon, scale = quantize_and_dequantize_block(block, mode)
        rel_err = compute_relative_error(block, recon)
        if rel_err <= epsilon:
            return mode, scale

    # Fallback to escape mode P16
    return PrecisionMode.P16, 1.0


def negotiate_joint_precision(
    block_a: np.ndarray,
    block_b: np.ndarray,
    target_epsilon: float = 0.05,
    reduction_tiles: int = 1
) -> Tuple[PrecisionMode, PrecisionMode, float, float]:
    """
    Joint Precision Negotiation between interacting blocks A_ik and B_kj.
    Accounts for output product error:
        Delta(AB) <= ||Delta A|| * ||B|| + ||A|| * ||Delta B|| + ||Delta A|| * ||Delta B||
    Allocates error accounting for accumulation across reduction dimension K:
        tile_tolerance = target_epsilon / sqrt(reduction_tiles)
    Selects (pA, pB) minimizing total bit-budget while meeting tolerance.

    Returns:
        (prec_a, prec_b, scale_a, scale_b)
    """
    norm_a = float(np.linalg.norm(block_a))
    norm_b = float(np.linalg.norm(block_b))
    norm_ab = norm_a * norm_b

    # If either block is practically zero, minimal precision suffices
    if norm_ab < 1e-7:
        return PrecisionMode.P2, PrecisionMode.P2, 1.0, 1.0

    # Target absolute error tolerance for this block product
    tile_allowed_abs_error = (target_epsilon / np.sqrt(max(1, reduction_tiles))) * norm_ab

    modes = [PrecisionMode.P2, PrecisionMode.P4, PrecisionMode.P8, PrecisionMode.P16]

    # Precompute reconstruction errors and scales for all modes
    errors_a: Dict[PrecisionMode, float] = {}
    scales_a: Dict[PrecisionMode, float] = {}
    errors_b: Dict[PrecisionMode, float] = {}
    scales_b: Dict[PrecisionMode, float] = {}

    for m in modes:
        rec_a, sc_a = quantize_and_dequantize_block(block_a, m)
        errors_a[m] = compute_frobenius_error(block_a, rec_a)
        scales_a[m] = sc_a

        rec_b, sc_b = quantize_and_dequantize_block(block_b, m)
        errors_b[m] = compute_frobenius_error(block_b, rec_b)
        scales_b[m] = sc_b

    # Search combinations ordered by total bits: pA + pB ascending
    candidate_pairs = []
    for pa in modes:
        for pb in modes:
            candidate_pairs.append((pa, pb))

    # Sort by sum of bit depths, then max bit depth
    candidate_pairs.sort(key=lambda pair: (pair[0].value + pair[1].value, max(pair[0].value, pair[1].value)))

    for pa, pb in candidate_pairs:
        ea = errors_a[pa]
        eb = errors_b[pb]
        # Frobenius bound: ea * norm_b + norm_a * eb + ea * eb
        est_err = ea * norm_b + norm_a * eb + ea * eb
        if est_err <= tile_allowed_abs_error:
            return pa, pb, scales_a[pa], scales_b[pb]

    # Default to P16 if error cannot be met
    return PrecisionMode.P16, PrecisionMode.P16, scales_a[PrecisionMode.P16], scales_b[PrecisionMode.P16]
