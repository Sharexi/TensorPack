"""
Tests for MTC-QT variable-block-size (quadtree) rate-distortion allocation.
"""

import numpy as np
import pytest

from mtc.core.allocation import rd_allocate
from mtc.core.quadtree import QuadtreeRD, qt_allocate, qt_dequant


def _rel(ref, pred):
    return float(np.linalg.norm(ref - pred) / np.linalg.norm(ref))


def _outlier_matrix(seed=0, n=128):
    rng = np.random.RandomState(seed)
    W = (rng.randn(n, n) * 0.05).astype(np.float32)
    mask = rng.rand(n, n) < 0.01
    W[mask] = rng.uniform(-25, 25, size=mask.sum())
    return W


@pytest.mark.parametrize("budget", [2.0, 3.0, 4.5, 7.0])
def test_budget_respected_and_leaves_tile_exactly(budget):
    W = _outlier_matrix()
    a = qt_allocate(W, budget, sizes=(32, 16, 8, 4))
    assert a.bits_per_element <= budget + 1e-9
    # Every element is covered by exactly one leaf
    assert sum(a.leaf_area_fraction().values()) == pytest.approx(1.0)
    cover = np.zeros(W.shape, dtype=int)
    for s in a.sizes:
        cover += np.kron(a.leaf[s].astype(int), np.ones((s, s), dtype=int))
    assert (cover == 1).all()


def test_rebuild_is_exact():
    W = _outlier_matrix(1)
    a = qt_allocate(W, 3.5)
    np.testing.assert_array_equal(qt_dequant(W, a.sizes, a.leaf, a.modes, a.scales), a.dequant)


def test_more_bits_less_error():
    W = _outlier_matrix(2)
    qt = QuadtreeRD(W, None, (32, 16, 8, 4))
    errs = [_rel(W, qt.select(b).dequant) for b in (2.0, 3.0, 4.0, 6.0, 9.0)]
    assert all(e1 > e2 for e1, e2 in zip(errs, errs[1:]))


def test_quadtree_beats_fixed_tiles_on_localized_structure():
    W = _outlier_matrix(3)
    fixed = rd_allocate(W, 3.0, None, 16, scale_format="bf16")
    tree = qt_allocate(W, 3.0, sizes=(32, 16, 8, 4))
    assert _rel(W, tree.dequant) < 0.7 * _rel(W, fixed.dequant)
    # Small leaves are actually used where the outliers are
    assert tree.leaf_area_fraction()[4] > 0.0


def test_quadtree_not_worse_on_homogeneous_data():
    W = np.random.RandomState(4).randn(128, 128).astype(np.float32)
    fixed = rd_allocate(W, 4.2, None, 16, scale_format="bf16")
    tree = qt_allocate(W, 4.2, sizes=(32, 16, 8))
    assert _rel(W, tree.dequant) <= 1.02 * _rel(W, fixed.dequant)
