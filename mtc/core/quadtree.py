"""
MTC-QT: Quadtree (variable block size) rate-distortion allocation.

Fixed 16x16 tiles are a compromise: too coarse where energy changes quickly
(outliers, channel structure), too fine where the matrix is homogeneous (the
per-block metadata is wasted). MTC-QT lets the geometry of the error landscape
pick the tiling:

    each root tile (e.g. 32x32) is either a leaf with its own (mode, scale),
    or splits into four children, recursively down to the finest size (e.g. 8x8).

Cost model (all bits counted):
    leaf     : payload (mode bits x area) + scale (bf16, 16 b) + mode byte (8 b)
    decision : 1 split-flag bit for every node above the finest level

Because the diagonal-Hessian distortion is additive over disjoint leaves, the
Lagrangian problem  min D + lambda * R  is solved EXACTLY for each lambda by a
bottom-up pass:

    J(node) = min( J_leaf(node),  sum_{children} J(child) ) + lambda * flag_bits

and a bisection on lambda meets the budget. This is the same tree-pruning used
for coding-tree decisions in HEVC/VVC video coding, applied to tensor storage.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .allocation import (
    DEFAULT_MODES,
    rd_candidates,
    RDCandidates,
    quantize_blocks,
    to_blocks,
    from_blocks,
)

SPLIT_FLAG_BITS = 1


def _sum4(x: np.ndarray) -> np.ndarray:
    """Sum each 2x2 group of children into its parent: (2a, 2b) -> (a, b)."""
    a, b = x.shape[0] // 2, x.shape[1] // 2
    return x.reshape(a, 2, b, 2).sum(axis=(1, 3))


def _up4(x: np.ndarray) -> np.ndarray:
    """Broadcast a parent grid (a, b) to its children grid (2a, 2b)."""
    return np.repeat(np.repeat(x, 2, axis=0), 2, axis=1)


@dataclass
class QTAllocation:
    sizes: Tuple[int, ...]                 # level block sizes, root first
    leaf: Dict[int, np.ndarray]            # size -> bool grid: node is a leaf
    modes: Dict[int, np.ndarray]           # size -> mode grid (valid where leaf)
    scales: Dict[int, np.ndarray]          # size -> scale grid (valid where leaf)
    dequant: np.ndarray
    total_bits: int

    @property
    def bits_per_element(self) -> float:
        return self.total_bits / self.dequant.size

    def leaf_area_fraction(self) -> Dict[int, float]:
        n = self.dequant.size
        return {s: float(self.leaf[s].sum() * s * s) / n for s in self.sizes}


class QuadtreeRD:
    """
    Precomputes per-level candidate tables once; `select(budget)` is then cheap,
    so many budgets can be evaluated from a single scale search.
    """

    def __init__(
        self,
        W: np.ndarray,
        importance: Optional[np.ndarray] = None,
        sizes: Sequence[int] = (32, 16, 8),
        modes: Sequence[int] = DEFAULT_MODES,
        scale_format: str = "bf16",
    ):
        self.W = np.asarray(W, dtype=np.float32)
        self.sizes = tuple(sorted(sizes, reverse=True))
        for a, b in zip(self.sizes, self.sizes[1:]):
            assert a == 2 * b, "quadtree levels must halve: e.g. (32, 16, 8)"
        K, N = self.W.shape
        assert K % self.sizes[0] == 0 and N % self.sizes[0] == 0, \
            f"shape {self.W.shape} not divisible by root size {self.sizes[0]}"
        self.cands: Dict[int, RDCandidates] = {
            s: rd_candidates(self.W, importance, s, modes, scale_format) for s in self.sizes
        }

    def _solve(self, lam: float):
        """Bottom-up pruning for one lambda. Returns (rate grid at root, decisions)."""
        best_mode_idx, leaf_J, leaf_R = {}, {}, {}
        for s, c in self.cands.items():
            rates = c.R[:, None, None] + c.meta_bits_per_block
            J = c.D + lam * rates
            idx = np.argmin(J, axis=0)
            best_mode_idx[s] = idx
            leaf_J[s] = np.take_along_axis(J, idx[None], 0)[0]
            leaf_R[s] = np.take_along_axis(np.broadcast_to(rates, J.shape), idx[None], 0)[0]

        finest = self.sizes[-1]
        J, R = leaf_J[finest], leaf_R[finest]
        is_leaf = {finest: np.ones_like(J, dtype=bool)}
        for s in reversed(self.sizes[:-1]):
            split_J, split_R = _sum4(J), _sum4(R)
            keep = leaf_J[s] <= split_J
            is_leaf[s] = keep
            J = np.where(keep, leaf_J[s], split_J) + lam * SPLIT_FLAG_BITS
            R = np.where(keep, leaf_R[s], split_R) + SPLIT_FLAG_BITS
        return R, is_leaf, best_mode_idx

    def select(self, bits_per_element: float) -> QTAllocation:
        budget = bits_per_element * self.W.size
        Dmax = max(c.D.max() for c in self.cands.values())
        ref = max(Dmax, 1e-30) / (2 * self.sizes[-1] ** 2)
        lo_l, hi_l = np.log(ref * 1e-16), np.log(ref * 1e6)
        best = self._solve(np.exp(hi_l))
        if best[0].sum() > budget:
            raise ValueError(f"Budget {bits_per_element} b/elem below the minimum {best[0].sum() / self.W.size:.3f}")
        for _ in range(60):
            mid = 0.5 * (lo_l + hi_l)
            sol = self._solve(np.exp(mid))
            if sol[0].sum() <= budget:
                best, hi_l = sol, mid
            else:
                lo_l = mid
        R, is_leaf, mode_idx = best

        # Resolve which nodes are *actual* leaves (a leaf below a leaf ancestor is not used).
        active = {self.sizes[0]: np.ones_like(is_leaf[self.sizes[0]], dtype=bool)}
        final_leaf = {}
        for s, s_next in zip(self.sizes, self.sizes[1:] + (None,)):
            final_leaf[s] = active[s] & is_leaf[s]
            if s_next is not None:
                active[s_next] = _up4(active[s] & ~is_leaf[s])

        modes, scales = {}, {}
        for s in self.sizes:
            c = self.cands[s]
            idx = mode_idx[s]
            mode_grid = np.array(c.modes, dtype=np.uint8)[idx]
            scale_grid = np.take_along_axis(c.S, idx[None], 0)[0]
            modes[s] = np.where(final_leaf[s], mode_grid, 0).astype(np.uint8)
            scales[s] = np.where(final_leaf[s], scale_grid, 0).astype(np.float32)

        deq = qt_dequant(self.W, self.sizes, final_leaf, modes, scales)
        return QTAllocation(self.sizes, final_leaf, modes, scales, deq, int(R.sum()))


def qt_dequant(W: np.ndarray, sizes, leaf, modes, scales) -> np.ndarray:
    """Rebuild the dequantized matrix from W and a stored quadtree allocation (exact)."""
    W = np.asarray(W, dtype=np.float32)
    deq = np.zeros_like(W)
    for s in sizes:
        blocks = to_blocks(W, s)
        lvl = np.zeros_like(blocks)
        for m in np.unique(modes[s][leaf[s]]):
            if m == 0:
                continue
            sel = leaf[s] & (modes[s] == m)
            q = quantize_blocks(blocks[sel][None], int(m), scales[s][sel][None])[0]
            lvl[sel] = q * scales[s][sel][:, None, None]
        deq += from_blocks(lvl)
    return deq


def qt_allocate(
    W: np.ndarray,
    bits_per_element: float,
    importance: Optional[np.ndarray] = None,
    sizes: Sequence[int] = (32, 16, 8),
    modes: Sequence[int] = DEFAULT_MODES,
) -> QTAllocation:
    return QuadtreeRD(W, importance, sizes, modes).select(bits_per_element)
