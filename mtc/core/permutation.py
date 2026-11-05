"""
MTC-QT-P: channel permutation before quadtree rate-distortion allocation.

The rows and columns of a weight matrix have no intrinsic order, so the spatial
neighbourhoods a quadtree exploits are an accident of how the layer was laid
out. Sorting rows (input channels) and columns (output channels) so that
channels with similar RD weight become adjacent turns scattered structure into
large homogeneous rectangles: big, cheap tiles where the matrix is uniform, and
the important channels packed into a few small high-precision tiles.

Because the diagonal-Hessian distortion  sum_k h_k ||dW_k||^2  is invariant
under any joint permutation of (W, h), the quadtree solves the permuted problem
exactly as before; the dequantized matrix is then mapped back.

Inference: for Y = X W with W stored permuted as W[pr][:, pc],
    Y[:, pc] = X[:, pr] @ W_perm
i.e. one gather on the activations and one scatter on the outputs, O(K + N) per
token versus O(K N) for the GEMM. The index arrays are stored data and are
COUNTED in the bit budget: ceil(log2 K) bits per row index, ceil(log2 N) per
column index. (In a transformer most of these permutations can instead be folded
into neighbouring layers at zero cost; that is not assumed here.)
"""

from dataclasses import dataclass
from math import ceil, log2
from typing import Optional, Sequence, Tuple

import numpy as np

from .allocation import DEFAULT_MODES
from .quadtree import QTAllocation, QuadtreeRD, qt_dequant


def channel_orders(W: np.ndarray, importance: Optional[np.ndarray],
                   rows: bool = True, cols: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort keys = each channel's share of the (importance-weighted) distortion scale:
        row k : h_k * ||W_k,:||^2
        col j : sum_k h_k W_kj^2
    """
    K, N = W.shape
    h = np.ones(K) if importance is None else np.asarray(importance, dtype=np.float64)
    hW2 = h[:, None] * (np.asarray(W, dtype=np.float64) ** 2)
    pr = np.argsort(hW2.sum(axis=1), kind="stable") if rows else np.arange(K)
    pc = np.argsort(hW2.sum(axis=0), kind="stable") if cols else np.arange(N)
    return pr, pc


def index_bits(K: int, N: int, rows: bool, cols: bool) -> int:
    """Bits to store the permutations (identity permutations are not stored)."""
    return (K * ceil(log2(K)) if rows else 0) + (N * ceil(log2(N)) if cols else 0)


@dataclass
class PermutedQTAllocation:
    inner: QTAllocation          # allocation of the permuted matrix
    row_perm: np.ndarray
    col_perm: np.ndarray
    dequant: np.ndarray          # in the ORIGINAL channel order
    total_bits: int              # inner bits + permutation index bits

    @property
    def bits_per_element(self) -> float:
        return self.total_bits / self.dequant.size

    def leaf_area_fraction(self):
        return self.inner.leaf_area_fraction()


def unpermute(Wp: np.ndarray, pr: np.ndarray, pc: np.ndarray) -> np.ndarray:
    out = np.empty_like(Wp)
    out[np.ix_(pr, pc)] = Wp
    return out


def permuted_qt_dequant(W: np.ndarray, pr, pc, sizes, leaf, modes, scales) -> np.ndarray:
    """Rebuild the dequantized matrix (original order) from a stored permuted allocation."""
    Wp = np.asarray(W, dtype=np.float32)[pr][:, pc]
    return unpermute(qt_dequant(Wp, sizes, leaf, modes, scales), pr, pc)


class PermutedQuadtreeRD:
    """Drop-in for QuadtreeRD: same budgets, the permutation index bits included."""

    def __init__(
        self,
        W: np.ndarray,
        importance: Optional[np.ndarray] = None,
        sizes: Sequence[int] = (32, 16, 8),
        modes: Sequence[int] = DEFAULT_MODES,
        scale_format: str = "bf16",
        rows: bool = True,
        cols: bool = True,
    ):
        self.W = np.asarray(W, dtype=np.float32)
        h = None if importance is None else np.asarray(importance)
        self.pr, self.pc = channel_orders(self.W, h, rows, cols)
        self.perm_bits = index_bits(*self.W.shape, rows, cols)
        self.qt = QuadtreeRD(self.W[self.pr][:, self.pc], None if h is None else h[self.pr],
                             sizes, modes, scale_format)

    def select(self, bits_per_element: float) -> PermutedQTAllocation:
        inner = self.qt.select(bits_per_element - self.perm_bits / self.W.size)
        return PermutedQTAllocation(inner, self.pr, self.pc, unpermute(inner.dequant, self.pr, self.pc),
                                    inner.total_bits + self.perm_bits)
