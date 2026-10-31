"""
Unit tests for Stage 2: Elementary 2x2 MTMUL kernel and fused accumulation.
"""

import numpy as np
import pytest
from mtc.core.types import PrecisionMode
from mtc.core.arena import MemoryArena
from mtc.core.microtensor import MicroTensor
from mtc.kernels.mtmul import (
    mtmul_2x2_int,
    mtmul_p2_p2_int32,
    mtmul_p4_p4_int32,
    mtmul_p8_p8_int32,
    mtmul_heterogeneous
)


def test_mtmul_2x2_exact_int():
    A = np.array([[1, -2], [0, 1]], dtype=np.int8)
    B = np.array([[-1, 1], [2, 0]], dtype=np.int8)

    expected = A @ B  # [[-5, 1], [2, 0]]
    accum = np.zeros((2, 2), dtype=np.int32)
    mtmul_2x2_int(
        int(A[0, 0]), int(A[0, 1]), int(A[1, 0]), int(A[1, 1]),
        int(B[0, 0]), int(B[0, 1]), int(B[1, 0]), int(B[1, 1]),
        accum
    )
    np.testing.assert_array_equal(accum, expected)


def test_mtmul_p2_p2_int32():
    arena = MemoryArena(1024)
    A = np.array([[1, -2], [0, 1]], dtype=np.float32)
    B = np.array([[-1, 1], [1, 0]], dtype=np.float32)

    mt_a = MicroTensor.from_array(A, PrecisionMode.P2, arena, scale=1.0)
    mt_b = MicroTensor.from_array(B, PrecisionMode.P2, arena, scale=1.0)

    accum = np.zeros((2, 2), dtype=np.int32)
    mtmul_p2_p2_int32(int(mt_a.raw_bytes[0]), int(mt_b.raw_bytes[0]), accum)

    expected = (A.astype(np.int32) @ B.astype(np.int32))
    np.testing.assert_array_equal(accum, expected)


def test_mtmul_p4_p4_int32():
    arena = MemoryArena(1024)
    A = np.array([[-7, 3], [4, -5]], dtype=np.float32)
    B = np.array([[2, -6], [-3, 1]], dtype=np.float32)

    mt_a = MicroTensor.from_array(A, PrecisionMode.P4, arena, scale=1.0)
    mt_b = MicroTensor.from_array(B, PrecisionMode.P4, arena, scale=1.0)

    accum = np.zeros((2, 2), dtype=np.int32)
    mtmul_p4_p4_int32(mt_a.raw_bytes, 0, mt_b.raw_bytes, 0, accum)

    expected = (A.astype(np.int32) @ B.astype(np.int32))
    np.testing.assert_array_equal(accum, expected)


def test_mtmul_heterogeneous():
    arena = MemoryArena(1024)
    A = np.array([[-1.0, 1.0], [0.0, -1.0]], dtype=np.float32)
    B = np.array([[2.5, -5.0], [1.25, 3.75]], dtype=np.float32)

    mt_a = MicroTensor.from_array(A, PrecisionMode.P2, arena)
    mt_b = MicroTensor.from_array(B, PrecisionMode.P8, arena)

    accum = np.zeros((2, 2), dtype=np.float32)
    mtmul_heterogeneous(
        2, mt_a.raw_bytes, 0, mt_a.scale,
        8, mt_b.raw_bytes, 0, mt_b.scale,
        accum
    )

    expected = mt_a.unpack_to_array() @ mt_b.unpack_to_array()
    np.testing.assert_allclose(accum, expected, atol=1e-5)


if __name__ == "__main__":
    test_mtmul_2x2_exact_int()
    test_mtmul_p2_p2_int32()
    test_mtmul_p4_p4_int32()
    test_mtmul_heterogeneous()
    print("All Stage 2 MTMUL tests passed!")
