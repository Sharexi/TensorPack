"""Tests for channel-permuted quadtree RD allocation (MTC-QT-P)."""

import numpy as np
import pytest

from mtc.core.permutation import (
    PermutedQuadtreeRD,
    channel_orders,
    index_bits,
    permuted_qt_dequant,
)
from mtc.core.quadtree import QuadtreeRD


def _row_scaled(K=256, N=256, seed=0):
    """Rows with widely varying scale in random order: structure a quadtree cannot see unsorted."""
    rng = np.random.RandomState(seed)
    W = (rng.randn(K, N) * np.exp(rng.randn(K, 1) * 1.0)).astype(np.float32)
    h = np.exp(rng.randn(K) * 0.5)
    return W, h


def test_orders_are_permutations():
    W, h = _row_scaled()
    pr, pc = channel_orders(W, h)
    assert np.array_equal(np.sort(pr), np.arange(W.shape[0]))
    assert np.array_equal(np.sort(pc), np.arange(W.shape[1]))


@pytest.mark.parametrize("budget", [2.5, 4.25])
def test_budget_includes_index_bits(budget):
    W, h = _row_scaled()
    a = PermutedQuadtreeRD(W, h, (32, 16, 8)).select(budget)
    assert a.bits_per_element <= budget + 1e-9
    assert a.total_bits == a.inner.total_bits + index_bits(*W.shape, True, True)


def test_rebuild_and_gather_scatter_inference_exact():
    W, h = _row_scaled()
    a = PermutedQuadtreeRD(W, h, (32, 16, 8)).select(3.0)
    inner = a.inner
    rebuilt = permuted_qt_dequant(W, a.row_perm, a.col_perm, inner.sizes, inner.leaf, inner.modes, inner.scales)
    np.testing.assert_array_equal(rebuilt, a.dequant)

    # Inference on the stored (permuted) matrix: gather X, GEMM, scatter Y.
    X = np.random.RandomState(1).randn(8, W.shape[0]).astype(np.float32)
    Y = np.empty((8, W.shape[1]), np.float32)
    Y[:, a.col_perm] = X[:, a.row_perm] @ inner.dequant
    ref = X @ a.dequant  # same products, different summation order: compare at FP32 rounding scale
    assert np.abs(Y - ref).max() <= 1e-6 * np.abs(ref).max()


def test_permutation_helps_on_shuffled_row_structure():
    W, h = _row_scaled()
    base = QuadtreeRD(W, h, (32, 16, 8, 4)).select(3.0)
    perm = PermutedQuadtreeRD(W, h, (32, 16, 8, 4)).select(3.0)

    def werr(Wq):
        return float((h[:, None] * (W - Wq) ** 2).sum())

    assert werr(perm.dequant) < 0.9 * werr(base.dequant)
