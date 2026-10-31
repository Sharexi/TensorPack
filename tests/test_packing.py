"""
Unit tests for Stage 1: Packed micro-tensors lossless packing/unpacking.
"""

import numpy as np
import pytest
from mtc.core.packing import (
    pack_tile_2x2_p2, unpack_tile_2x2_p2,
    pack_tile_2x2_p4, unpack_tile_2x2_p4,
    pack_tile_2x2_p8, unpack_tile_2x2_p8,
    pack_matrix_p2, unpack_matrix_p2,
    pack_matrix_p4, unpack_matrix_p4,
)


def test_tile_2x2_p2_lossless():
    # P2 range: -2, -1, 0, 1
    # Test all 4^4 = 256 possible combinations of 2x2 with values in {-2, -1, 0, 1}
    for q0 in [-2, -1, 0, 1]:
        for q1 in [-2, -1, 0, 1]:
            for q2 in [-2, -1, 0, 1]:
                for q3 in [-2, -1, 0, 1]:
                    orig = np.array([[q0, q1], [q2, q3]], dtype=np.int8)
                    byte_val = pack_tile_2x2_p2(orig)
                    assert 0 <= byte_val <= 255
                    unpacked = unpack_tile_2x2_p2(byte_val)
                    np.testing.assert_array_equal(orig, unpacked)


def test_tile_2x2_p4_lossless():
    # P4 range: -8 to 7
    np.random.seed(42)
    for _ in range(100):
        orig = np.random.randint(-8, 8, size=(2, 2), dtype=np.int8)
        buf = np.zeros(2, dtype=np.uint8)
        pack_tile_2x2_p4(orig, buf, 0)
        unpacked = unpack_tile_2x2_p4(buf, 0)
        np.testing.assert_array_equal(orig, unpacked)


def test_tile_2x2_p8_lossless():
    # P8 range: -128 to 127
    np.random.seed(42)
    for _ in range(100):
        orig = np.random.randint(-128, 128, size=(2, 2), dtype=np.int8)
        buf = np.zeros(4, dtype=np.uint8)
        pack_tile_2x2_p8(orig, buf, 0)
        unpacked = unpack_tile_2x2_p8(buf, 0)
        np.testing.assert_array_equal(orig, unpacked)


def test_matrix_p2_lossless():
    H, W = 64, 64
    np.random.seed(42)
    orig = np.random.choice(np.array([-2, -1, 0, 1], dtype=np.int8), size=(H, W))
    packed = pack_matrix_p2(orig)
    assert packed.nbytes == (H * W) // 4  # 4 values per byte
    unpacked = unpack_matrix_p2(packed, H, W)
    np.testing.assert_array_equal(orig, unpacked)


def test_matrix_p4_lossless():
    H, W = 64, 64
    np.random.seed(42)
    orig = np.random.randint(-8, 8, size=(H, W), dtype=np.int8)
    packed = pack_matrix_p4(orig)
    assert packed.nbytes == (H * W) // 2  # 2 values per byte
    unpacked = unpack_matrix_p4(packed, H, W)
    np.testing.assert_array_equal(orig, unpacked)


if __name__ == "__main__":
    test_tile_2x2_p2_lossless()
    test_tile_2x2_p4_lossless()
    test_tile_2x2_p8_lossless()
    test_matrix_p2_lossless()
    test_matrix_p4_lossless()
    print("All Stage 1 packing tests passed!")
