"""
Tests for MTC-RD rate-distortion block precision allocation and its packed format.
"""

import numpy as np
import pytest

from mtc.core.allocation import rd_allocate, rd_allocate_pair, META_BITS_PER_BLOCK
from mtc.kernels.gemm import MTCMatrix, mtc_gemm
from mtc.baselines import formats as F


def _rel(ref, pred):
    return float(np.linalg.norm(ref - pred) / np.linalg.norm(ref))


@pytest.mark.parametrize("budget", [2.3, 3.0, 4.25, 6.0, 8.25])
def test_budget_is_respected(budget):
    W = np.random.RandomState(0).randn(128, 128).astype(np.float32)
    alloc = rd_allocate(W, budget)
    assert alloc.bits_per_element <= budget + 1e-9
    # Greedy fill should leave less than one block-upgrade of slack
    assert alloc.bits_per_element > budget - 16 * 256 / W.size


def test_error_decreases_with_budget():
    W = np.random.RandomState(1).randn(128, 128).astype(np.float32)
    errs = [_rel(W, rd_allocate(W, b).dequant) for b in (2.5, 3.5, 4.5, 6.5, 8.5)]
    assert all(e1 > e2 for e1, e2 in zip(errs, errs[1:]))


def test_homogeneous_data_matches_fixed_width():
    """On iid data the optimal allocation is (near) uniform: it should match a single fixed width."""
    W = np.random.RandomState(2).randn(256, 256).astype(np.float32)
    budget = 4 + META_BITS_PER_BLOCK / 256.0
    adaptive = rd_allocate(W, budget)
    fixed = rd_allocate(W, budget, modes=(4,))
    assert _rel(W, adaptive.dequant) <= _rel(W, fixed.dequant) * 1.001


def test_multiscale_allocation_uses_low_modes_and_beats_fixed():
    rng = np.random.RandomState(3)
    W = rng.randn(256, 256).astype(np.float32)
    W[:128] *= 1e-3  # half the matrix is near-silent
    deq, bits = F.block_int(W, 4, 16)
    alloc = rd_allocate(W, bits)  # same total budget as block INT4
    hist = alloc.mode_histogram()
    assert hist[0] + hist[2] > 0.3, hist  # silent half goes to P0/P2
    assert alloc.bits_per_element <= bits + 1e-9
    assert _rel(W, alloc.dequant) < 0.5 * _rel(W, deq)


def test_importance_weighting_reduces_output_error():
    """Activation-aware allocation must beat plain-MSE allocation on output error."""
    rng = np.random.RandomState(4)
    K, N, T = 256, 256, 128
    W = rng.randn(K, N).astype(np.float32)
    scale = np.ones(K)
    scale[rng.choice(K, 8, replace=False)] = 30.0
    X_cal = (rng.randn(T, K) * scale).astype(np.float32)
    X_test = (rng.randn(T, K) * scale).astype(np.float32)
    h = (X_cal ** 2).mean(axis=0)
    with_h = rd_allocate(W, 3.0, h).dequant
    no_h = rd_allocate(W, 3.0, None).dequant
    Y = X_test @ W
    assert _rel(Y, X_test @ with_h) < 0.8 * _rel(Y, X_test @ no_h)


@pytest.mark.parametrize("budget", [2.6, 3.3, 5.2, 6.3, 9.0])
def test_packed_kernel_matches_simulation(budget):
    """The real arena format + Numba kernel must reproduce the simulated quantization (all modes incl. P0/P3/P5/P6)."""
    rng = np.random.RandomState(5)
    A = rng.randn(128, 96).astype(np.float32)
    B = rng.randn(96, 64).astype(np.float32)
    A[:, :32] *= 1e-4  # forces P0 / low modes
    B[32:64] *= 20.0
    a, b = rd_allocate_pair(A, B, budget)
    Am = MTCMatrix.from_allocation(A, a.modes, a.scales, 16)
    Bm = MTCMatrix.from_allocation(B, b.modes, b.scales, 16)
    np.testing.assert_allclose(Am.to_array(), a.dequant, rtol=0, atol=0)
    np.testing.assert_allclose(Bm.to_array(), b.dequant, rtol=0, atol=0)
    C_sim = a.dequant @ b.dequant
    assert _rel(C_sim, mtc_gemm(Am, Bm)) < 1e-5
    # Arena payload size is exactly what the allocator accounted for
    assert Am.total_bytes_allocated * 8 == a.payload_bits
    assert Bm.total_bytes_allocated * 8 == b.payload_bits


def test_baseline_formats_bit_accounting():
    W = np.random.RandomState(6).randn(256, 256).astype(np.float32)
    assert F.mxfp4(W)[1] == pytest.approx(4.25)
    assert F.mxint8(W)[1] == pytest.approx(8.25)
    assert F.nvfp4(W)[1] == pytest.approx(4.5, abs=1e-3)
    assert F.group_int(W, 4, 32)[1] == pytest.approx(4.5)
    assert F.block_int(W, 4, 16)[1] == pytest.approx(4.125)
    for fn in (F.mxfp4, F.nvfp4, F.mxint8):
        deq, _ = fn(W)
        assert _rel(W, deq) < 0.2
