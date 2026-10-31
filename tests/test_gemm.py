"""
Unit tests for Stage 5: Full GEMM Pipeline and Accuracy Validation.
"""

import numpy as np
import pytest
from mtc.core.types import PrecisionMode
from mtc.kernels.gemm import MTCMatrix, mtc_gemm
from mtc.kernels.reference import relative_frobenius_error, fp32_gemm_numpy


def test_fixed_p8_gemm():
    np.random.seed(42)
    N = 64
    Bs = 16
    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)
    C_ref = fp32_gemm_numpy(A, B)

    mtc_a = MTCMatrix.from_numpy(A, block_size=Bs, precision_mode=PrecisionMode.P8)
    mtc_b = MTCMatrix.from_numpy(B, block_size=Bs, precision_mode=PrecisionMode.P8)
    C_mtc = mtc_gemm(mtc_a, mtc_b)

    err = relative_frobenius_error(C_ref, C_mtc)
    assert err < 0.02, f"P8 GEMM error too high: {err}"


def test_fixed_p4_gemm():
    np.random.seed(42)
    N = 64
    Bs = 16
    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)
    C_ref = fp32_gemm_numpy(A, B)

    mtc_a = MTCMatrix.from_numpy(A, block_size=Bs, precision_mode=PrecisionMode.P4)
    mtc_b = MTCMatrix.from_numpy(B, block_size=Bs, precision_mode=PrecisionMode.P4)
    C_mtc = mtc_gemm(mtc_a, mtc_b)

    err = relative_frobenius_error(C_ref, C_mtc)
    # P4 has higher error than P8 but bounded
    assert err < 0.25, f"P4 GEMM error too high: {err}"


def test_adaptive_independent_gemm():
    np.random.seed(42)
    N = 64
    Bs = 16
    # Matrix with mixed scales: some blocks small magnitude, some normal
    A = np.random.randn(N, N).astype(np.float32)
    A[:32, :] *= 0.01  # Low magnitude block
    B = np.random.randn(N, N).astype(np.float32)
    C_ref = fp32_gemm_numpy(A, B)

    mtc_a = MTCMatrix.from_numpy(A, block_size=Bs, epsilon=0.05)
    mtc_b = MTCMatrix.from_numpy(B, block_size=Bs, epsilon=0.05)
    C_mtc = mtc_gemm(mtc_a, mtc_b)

    err = relative_frobenius_error(C_ref, C_mtc)
    assert err < 0.15, f"Adaptive independent GEMM error too high: {err}"


def test_joint_negotiated_gemm():
    np.random.seed(42)
    N = 64
    Bs = 16
    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)
    C_ref = fp32_gemm_numpy(A, B)

    mtc_a, mtc_b = MTCMatrix.from_numpy_joint(A, B, block_size=Bs, target_epsilon=0.05)
    C_mtc = mtc_gemm(mtc_a, mtc_b)

    err = relative_frobenius_error(C_ref, C_mtc)
    assert err < 0.15, f"Joint negotiated GEMM error too high: {err}"


def test_pointer_hierarchy_flag():
    np.random.seed(42)
    N = 64
    Bs = 16
    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)

    # With pointer hierarchy
    mtc_a_ptr = MTCMatrix.from_numpy(A, block_size=Bs, precision_mode=PrecisionMode.P8, use_pointer_hierarchy=True)
    assert mtc_a_ptr.pointer_tensor is not None

    # Without pointer hierarchy
    mtc_a_noptr = MTCMatrix.from_numpy(A, block_size=Bs, precision_mode=PrecisionMode.P8, use_pointer_hierarchy=False)
    assert mtc_a_noptr.pointer_tensor is None

    # Results match
    mtc_b = MTCMatrix.from_numpy(B, block_size=Bs, precision_mode=PrecisionMode.P8)
    C_ptr = mtc_gemm(mtc_a_ptr, mtc_b)
    C_noptr = mtc_gemm(mtc_a_noptr, mtc_b)
    np.testing.assert_allclose(C_ptr, C_noptr, atol=1e-5)


if __name__ == "__main__":
    test_fixed_p8_gemm()
    test_fixed_p4_gemm()
    test_adaptive_independent_gemm()
    test_joint_negotiated_gemm()
    test_pointer_hierarchy_flag()
    print("All Stage 5 GEMM tests passed!")
