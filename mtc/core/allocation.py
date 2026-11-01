"""
MTC-RD: Rate-Distortion Optimal Block Precision Allocation.

This is the principled version of "joint precision negotiation".

Problem
-------
Quantize a matrix W (K x N, reduction dimension K along rows) into Bs x Bs
blocks, each with its own precision mode p_b in {P0, P2, P4, P8, P16} and its
own scale s_b, so that the *product* error is minimal under a global bit budget:

    minimize   E || X (W - W_hat) ||_F^2
    subject to sum_b bits(p_b) + metadata <= budget

where X is the other operand of the GEMM (activations for a weight matrix).

Under the standard diagonal approximation of the Hessian H = X^T X
(the same approximation used by AWQ-style and HAWQ-style methods),

    E || X dW ||^2  ~=  sum_k h_k || dW[k, :] ||^2,     h_k = sum_t X[t, k]^2

so every block's distortion D_b(p) = sum_{rows r in b} h_r ||dW_r||^2 is
independent of every other block, and the budgeted problem is solved exactly
(up to the discreteness of the modes) by Lagrangian relaxation:

    p_b(lambda) = argmin_p  D_b(p) + lambda * R(p)

with a bisection on lambda to meet the budget, followed by a greedy fill of
the remaining bits by best distortion-reduction-per-bit.

The "joint" part is that precision of one operand is driven by the energy of
the other operand (h). For weights, h comes from calibration activations, so
allocation is fully offline. For a weight x weight product, h comes from the
other matrix directly.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Tuple
import math
import numpy as np
from numba import njit, prange

# Integer code range per mode. P0 = block dropped (all zeros, no payload).
QRANGE: Dict[int, Tuple[int, int]] = {
    2: (-2, 1),
    3: (-4, 3),
    4: (-8, 7),
    5: (-16, 15),
    6: (-32, 31),
    8: (-128, 127),
    16: (-32768, 32767),
}

# Metadata actually stored per block: fp32 scale + 1 byte precision mode.
# (Arena offsets are a prefix sum of the per-block payload sizes, so they are
# recomputable from the modes and are not counted.)
META_BITS_PER_BLOCK = 32 + 8

# P3/P5/P6 are bitstream-packed (Bs*Bs*b bits is always a whole number of bytes
# for Bs = 16). A dense mode lattice matters: with only {2, 4, 8}, a 6-bit budget
# must be met by mixing 4- and 8-bit blocks, which is strictly worse than 6-bit
# codes for homogeneous data (distortion falls exponentially in bits).
DEFAULT_MODES = (0, 2, 3, 4, 5, 6, 8, 16)

# Clipping-ratio search grids (fraction of the block abs-max that maps to
# the largest code magnitude). Clipping matters a lot at 2-4 bits.
_ALPHAS = {
    2: np.linspace(0.20, 1.00, 33),
    3: np.linspace(0.30, 1.00, 29),
    4: np.linspace(0.40, 1.00, 25),
    5: np.linspace(0.50, 1.00, 21),
    6: np.linspace(0.60, 1.00, 17),
    8: np.linspace(0.80, 1.00, 9),
    16: np.array([1.0]),
}


def to_blocks(W: np.ndarray, Bs: int) -> np.ndarray:
    """(K, N) -> (K/Bs, N/Bs, Bs, Bs) view-copy, block (i, j) = W[i*Bs:(i+1)*Bs, j*Bs:(j+1)*Bs]."""
    K, N = W.shape
    return W.reshape(K // Bs, Bs, N // Bs, Bs).transpose(0, 2, 1, 3)


def from_blocks(Bl: np.ndarray) -> np.ndarray:
    nbk, nbn, Bs, _ = Bl.shape
    return Bl.transpose(0, 2, 1, 3).reshape(nbk * Bs, nbn * Bs)


def quantize_blocks(blocks: np.ndarray, mode: int, scales: np.ndarray) -> np.ndarray:
    """Integer codes for every block with given per-block scales (shape (nbk, nbn))."""
    lo, hi = QRANGE[mode]
    s = scales[..., None, None]
    safe = np.where(s == 0.0, 1.0, s)
    return np.clip(np.round(blocks / safe), lo, hi)


@njit(inline="always")
def _round_bf16(x):
    """Round a float to the nearest bfloat16 value (8 significant bits, fp32 exponent range)."""
    if x == 0.0:
        return x
    m, e = math.frexp(x)
    return math.ldexp(round(m * 256.0) / 256.0, e)


@njit(parallel=True, fastmath=False, cache=True)
def _search_kernel(blocks2, h2, amax, lo, hi, qmag, alphas, signs, round_bf16, best_d, best_s):
    """blocks2: (nb, Bs, Bs); h2: (nb, Bs) row importance. Exhaustive scale search per block."""
    nb, Bs, _ = blocks2.shape
    for b in prange(nb):
        bd = np.inf
        bs = np.float32(0.0)
        for ai in range(alphas.shape[0]):
            for si in range(signs.shape[0]):
                s = np.float32(signs[si] * alphas[ai] * amax[b] / qmag)
                if round_bf16:
                    s = np.float32(_round_bf16(float(s)))
                inv = np.float32(1.0) / s
                d = 0.0
                for r in range(Bs):
                    acc = np.float32(0.0)
                    for c in range(Bs):
                        x = blocks2[b, r, c]
                        q = np.float32(np.round(x * inv))
                        if q < lo:
                            q = np.float32(lo)
                        elif q > hi:
                            q = np.float32(hi)
                        e = x - q * s
                        acc += e * e
                    d += h2[b, r] * acc
                if d < bd:
                    bd = d
                    bs = s
        best_d[b] = bd
        best_s[b] = bs


def _search_mode(blocks: np.ndarray, h_rows: np.ndarray, mode: int, round_bf16: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    For one precision mode, find the per-block scale minimizing the
    importance-weighted squared error. Searches clipping ratio and scale sign
    (the P2/P4/P8 grids are asymmetric, e.g. {-2,-1,0,1}, so the sign of the
    scale decides which tail gets the extra level).

    Returns (best_distortion, best_scale), each (nbk, nbn).
    """
    lo, hi = QRANGE[mode]
    qmag = float(max(-lo, hi))
    nbk, nbn, Bs, _ = blocks.shape
    blocks2 = np.ascontiguousarray(blocks, dtype=np.float32).reshape(nbk * nbn, Bs, Bs)
    h2 = np.ascontiguousarray(h_rows, dtype=np.float64).reshape(nbk * nbn, Bs)
    amax = np.abs(blocks2).max(axis=(1, 2)).astype(np.float32)
    amax = np.where(amax < 1e-30, np.float32(1.0), amax).astype(np.float32)
    signs = np.array([1.0] if mode == 16 else [1.0, -1.0], dtype=np.float32)
    best_d = np.empty(nbk * nbn, dtype=np.float64)
    best_s = np.empty(nbk * nbn, dtype=np.float32)
    _search_kernel(blocks2, h2, amax, float(lo), float(hi), qmag,
                   _ALPHAS[mode].astype(np.float32), signs, round_bf16, best_d, best_s)
    return best_d.reshape(nbk, nbn), best_s.reshape(nbk, nbn)


@dataclass
class RDAllocation:
    """Result of MTC-RD quantization of one matrix (blocks indexed [row_block, col_block])."""
    modes: np.ndarray        # (nbk, nbn) uint8 in {0, 2, 4, 8, 16}
    scales: np.ndarray       # (nbk, nbn) float32
    dequant: np.ndarray      # (K, N) float32 reconstruction
    block_size: int
    payload_bits: int
    meta_bits: int

    @property
    def bits_per_element(self) -> float:
        return (self.payload_bits + self.meta_bits) / self.dequant.size

    def mode_histogram(self) -> Dict[int, float]:
        n = self.modes.size
        return {int(m): float((self.modes == m).sum()) / n for m in DEFAULT_MODES}


@dataclass
class RDCandidates:
    """Per-block distortion/scale table for every mode. Budget-independent, so
    it can be computed once and reused for any number of bit budgets."""
    blocks: np.ndarray       # (nbk, nbn, Bs, Bs)
    D: np.ndarray            # (n_modes, nbk, nbn) importance-weighted distortion
    S: np.ndarray            # (n_modes, nbk, nbn) best scale per mode
    R: np.ndarray            # (n_modes,) payload bits per block
    modes: Tuple[int, ...]
    block_size: int
    shape: Tuple[int, int]
    meta_bits_per_block: int = META_BITS_PER_BLOCK


# Metadata per block for each scale storage format (scale + 1 mode byte).
SCALE_FORMAT_META_BITS = {"fp32": 32 + 8, "bf16": 16 + 8}


def rd_candidates(
    W: np.ndarray,
    importance: Optional[np.ndarray] = None,
    block_size: int = 16,
    modes: Sequence[int] = DEFAULT_MODES,
    scale_format: str = "fp32",
) -> RDCandidates:
    """
    Run the per-block, per-mode scale search (the expensive part of MTC-RD).
    scale_format: "fp32" (40 metadata bits/block) or "bf16" (24 bits/block; the
    search only considers bf16-representable scales, so accounting is exact).
    """
    assert scale_format in SCALE_FORMAT_META_BITS
    W = np.asarray(W, dtype=np.float32)
    K, N = W.shape
    Bs = block_size
    assert K % Bs == 0 and N % Bs == 0, f"shape {W.shape} not divisible by block size {Bs}"

    h = np.ones(K, dtype=np.float64) if importance is None else np.asarray(importance, dtype=np.float64)
    assert h.shape == (K,)
    h = h / max(h.mean(), 1e-30)

    blocks = to_blocks(W, Bs)
    nbk, nbn = blocks.shape[:2]
    h_rows = np.broadcast_to(h.reshape(nbk, 1, Bs), (nbk, nbn, Bs))

    modes = tuple(sorted(modes))
    D = np.empty((len(modes), nbk, nbn), dtype=np.float64)
    S = np.zeros((len(modes), nbk, nbn), dtype=np.float32)
    for i, m in enumerate(modes):
        if m == 0:
            D[i] = np.einsum("abrc,abr->ab", blocks.astype(np.float64) ** 2, h_rows)
        else:
            D[i], S[i] = _search_mode(blocks, h_rows, m, scale_format == "bf16")
    R = np.array([m * Bs * Bs for m in modes], dtype=np.float64)
    return RDCandidates(blocks, D, S, R, modes, Bs, (K, N), SCALE_FORMAT_META_BITS[scale_format])


def rd_allocate(
    W: np.ndarray,
    bits_per_element: float,
    importance: Optional[np.ndarray] = None,
    block_size: int = 16,
    modes: Sequence[int] = DEFAULT_MODES,
    scale_format: str = "fp32",
) -> RDAllocation:
    """
    Quantize W (K x N, reduction along rows) with MTC-RD under a total budget of
    `bits_per_element` (payload + metadata) bits per element.

    importance: length-K vector h_k (e.g. mean squared activation of input
        channel k). None -> uniform (plain MSE allocation).
    """
    return rd_select(rd_candidates(W, importance, block_size, modes, scale_format), bits_per_element)


def rd_select(c: RDCandidates, bits_per_element: float) -> RDAllocation:
    """Lagrangian mode selection + greedy fill for one budget, from a candidate table."""
    blocks, D, S, R, modes, Bs = c.blocks, c.D, c.S, c.R, c.modes, c.block_size
    nbk, nbn = blocks.shape[:2]
    n_blocks = nbk * nbn
    size = c.shape[0] * c.shape[1]

    meta_bits = c.meta_bits_per_block * n_blocks
    payload_budget = bits_per_element * size - meta_bits
    if payload_budget < R.min() * n_blocks:
        raise ValueError(
            f"Budget {bits_per_element} b/elem is below the minimum "
            f"{(R.min() * n_blocks + meta_bits) / size:.3f} for this mode set."
        )

    Df = D.reshape(len(modes), -1)

    def choose(lam: float) -> np.ndarray:
        return np.argmin(Df + lam * R[:, None], axis=0)

    # Bisection on log(lambda): larger lambda -> fewer bits.
    scale_ref = max(Df.max(), 1e-30) / R[R > 0].min()
    lo_l, hi_l = np.log(scale_ref * 1e-16), np.log(scale_ref * 1e4)
    best = choose(np.exp(hi_l))
    for _ in range(64):
        mid = 0.5 * (lo_l + hi_l)
        idx = choose(np.exp(mid))
        if R[idx].sum() <= payload_budget:
            best, hi_l = idx, mid
        else:
            lo_l = mid

    # Greedy fill of leftover budget: single-step upgrades by best dD/dR.
    remaining = payload_budget - R[best].sum()
    while remaining > 0:
        cur_r, cur_d = R[best], Df[best, np.arange(n_blocks)]
        gain = np.full(n_blocks, -np.inf)
        nxt = best.copy()
        for j in range(len(modes)):
            dr = R[j] - cur_r
            ok = (dr > 0) & (dr <= remaining)
            g = np.where(ok, (cur_d - Df[j]) / np.where(dr > 0, dr, 1.0), -np.inf)
            upd = g > gain
            gain, nxt = np.where(upd, g, gain), np.where(upd, j, nxt)
        b = int(np.argmax(gain))
        if not np.isfinite(gain[b]) or gain[b] <= 0:
            break
        remaining -= R[nxt[b]] - R[best[b]]
        best[b] = nxt[b]

    mode_arr = np.array(modes, dtype=np.uint8)[best].reshape(nbk, nbn)
    scale_arr = S.reshape(len(modes), -1)[best, np.arange(n_blocks)].reshape(nbk, nbn)

    deq = np.zeros_like(blocks)
    for m in modes:
        if m == 0:
            continue
        sel = mode_arr == m
        if sel.any():
            q = quantize_blocks(blocks[sel][None], m, scale_arr[sel][None])[0]
            deq[sel] = q * scale_arr[sel][:, None, None]
    scale_arr = np.where(mode_arr == 0, 0.0, scale_arr).astype(np.float32)

    return RDAllocation(
        modes=mode_arr,
        scales=scale_arr,
        dequant=from_blocks(deq).astype(np.float32),
        block_size=Bs,
        payload_bits=int(R[best].sum()),
        meta_bits=meta_bits,
    )


def rd_allocate_pair(
    A: np.ndarray,
    B: np.ndarray,
    bits_per_element: float,
    block_size: int = 16,
    modes: Sequence[int] = DEFAULT_MODES,
    use_importance: bool = True,
) -> Tuple[RDAllocation, RDAllocation]:
    """
    Joint allocation for C = A @ B with both operands quantized.
    A's blocks are weighted by the energy of the B rows they multiply, and
    vice versa. Returns allocations with A's in A's own (M, K) orientation.
    """
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    h_for_B = (A.astype(np.float64) ** 2).sum(axis=0) if use_importance else None  # per k
    h_for_A = (B.astype(np.float64) ** 2).sum(axis=1) if use_importance else None  # per k
    alloc_b = rd_allocate(B, bits_per_element, h_for_B, block_size, modes)
    alloc_at = rd_allocate(A.T, bits_per_element, h_for_A, block_size, modes)
    alloc_a = RDAllocation(
        modes=alloc_at.modes.T.copy(),
        scales=alloc_at.scales.T.copy(),
        dequant=alloc_at.dequant.T.copy(),
        block_size=block_size,
        payload_bits=alloc_at.payload_bits,
        meta_bits=alloc_at.meta_bits,
    )
    return alloc_a, alloc_b


def rd_dequant(W: np.ndarray, modes: np.ndarray, scales: np.ndarray, block_size: int = 16) -> np.ndarray:
    """Rebuild the dequantized matrix from W and a stored (modes, scales) allocation."""
    blocks = to_blocks(np.asarray(W, dtype=np.float32), block_size)
    deq = np.zeros_like(blocks)
    for m in np.unique(modes):
        if m == 0:
            continue
        sel = modes == m
        q = quantize_blocks(blocks[sel][None], int(m), scales[sel][None])[0]
        deq[sel] = q * scales[sel][:, None, None]
    return from_blocks(deq).astype(np.float32)
