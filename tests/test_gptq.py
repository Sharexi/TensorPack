"""Tests for GPTQ error compensation on group-INT and fixed (MTC-QT-P) grids."""

import numpy as np
import pytest
import torch

from mtc.baselines import formats as F
from mtc.baselines.gptq import gptq, qt_element_grid
from mtc.core.permutation import PermutedQuadtreeRD
from mtc.core.quadtree import QuadtreeRD, qt_dequant


def _layer(K=256, N=128, T=1024, seed=0):
    rng = np.random.RandomState(seed)
    W = (rng.randn(K, N) * 0.05).astype(np.float32)
    mix = rng.randn(K, K) / np.sqrt(K) + np.eye(K)          # correlated input channels
    X = (rng.randn(T, K) @ mix * np.exp(rng.randn(K) * 0.5)).astype(np.float32)
    return W, X


def test_identity_hessian_group_grid_equals_rtn():
    W, _ = _layer()
    Q = gptq(torch.from_numpy(W), torch.eye(W.shape[0]), grid="group", bits=4, group=128, damp=0.0)
    ref, _ = F.group_int(W, 4, 128)
    np.testing.assert_allclose(Q.numpy(), ref, rtol=0, atol=1e-7)


def test_qt_element_grid_rebuilds_qt_dequant_and_identity_gptq_is_rtn():
    W, X = _layer()
    h = (X.astype(np.float64) ** 2).mean(axis=0)
    a = QuadtreeRD(W, h, (32, 16, 8)).select(3.0)
    S, LO, HI = qt_element_grid(a.sizes, a.leaf, a.modes, a.scales, W.shape)
    safe = np.where(S == 0, 1.0, S)
    rtn = np.clip(np.round(W / safe), LO, HI) * S
    np.testing.assert_array_equal(rtn, qt_dequant(W, a.sizes, a.leaf, a.modes, a.scales))

    Q = gptq(torch.from_numpy(W), torch.eye(W.shape[0]), "fixed", torch.from_numpy(S),
             torch.from_numpy(LO), torch.from_numpy(HI), damp=0.0)
    np.testing.assert_allclose(Q.numpy(), rtn, rtol=0, atol=1e-7)


@pytest.mark.parametrize("act_order", [False, True])
def test_gptq_lowers_output_error(act_order):
    W, X = _layer()
    H = torch.from_numpy((X.T @ X).astype(np.float32))
    Q = gptq(torch.from_numpy(W), H, grid="group", bits=3, group=128, act_order=act_order).numpy()
    rtn, _ = F.group_int(W, 3, 128)
    err = lambda Wq: np.linalg.norm(X @ (W - Wq))
    assert err(Q) < 0.9 * err(rtn)


def test_gptq_on_permuted_qt_grid_stays_in_format_and_helps():
    W, X = _layer()
    h = (X.astype(np.float64) ** 2).mean(axis=0)
    a = PermutedQuadtreeRD(W, h, (32, 16, 8)).select(3.0)
    inner = a.inner
    Sp, LOp, HIp = qt_element_grid(inner.sizes, inner.leaf, inner.modes, inner.scales, W.shape)
    # Map the per-element grid back to the original channel order.
    S = np.empty_like(Sp); LO = np.empty_like(LOp); HI = np.empty_like(HIp)
    ix = np.ix_(a.row_perm, a.col_perm)
    S[ix], LO[ix], HI[ix] = Sp, LOp, HIp

    H = torch.from_numpy((X.T @ X).astype(np.float32))
    Q = gptq(torch.from_numpy(W), H, "fixed", torch.from_numpy(S), torch.from_numpy(LO),
             torch.from_numpy(HI), act_order=True).numpy()
    codes = np.where(S == 0, 0.0, Q / np.where(S == 0, 1.0, S))
    np.testing.assert_allclose(codes, np.round(codes), atol=1e-4)      # integer codes
    assert np.all(codes >= LO - 1e-4) and np.all(codes <= HI + 1e-4)  # inside each tile's range
    assert np.all(Q[S == 0] == 0)                                     # P0 tiles stay empty

    err = lambda Wq: np.linalg.norm(X @ (W - Wq))
    assert err(Q) < 0.9 * err(a.dequant)
