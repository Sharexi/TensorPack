"""
Unit tests for 3D MicroTensor Computing: MicroCubes, Octree, and 3D Operations.
"""

import numpy as np
import pytest
from mtc.core.arena import MemoryArena
from mtc.spatial3d.types3d import PrecisionMode3D
from mtc.spatial3d.packing3d import (
    pack_cube_2x2x2_p1_spin, unpack_cube_2x2x2_p1_spin,
    pack_cube_2x2x2_p2, unpack_cube_2x2x2_p2,
    pack_cube_2x2x2_p4, unpack_cube_2x2x2_p4,
    pack_cube_2x2x2_p8, unpack_cube_2x2x2_p8
)
from mtc.spatial3d.microcube import MicroCube
from mtc.spatial3d.octree_tensor import OctreeTensor
from mtc.spatial3d.conv3d import conv3d_2x2x2_kernel, trilinear_contraction_3d, conv3d_with_microcube


def test_cube_p1_spin_exact():
    # Test all 2^8 = 256 possible configurations of 1-bit spin cube {-1, 1}
    for byte_val in range(256):
        out = np.empty((2, 2, 2), dtype=np.int8)
        unpack_cube_2x2x2_p1_spin(byte_val, out)
        repacked = pack_cube_2x2x2_p1_spin(out)
        assert repacked == byte_val


def test_cube_p2_p4_p8_roundtrip():
    arena = MemoryArena(1024)

    # P2
    orig_p2 = np.array([
        [[-2, 1], [0, -1]],
        [[1, 0], [-1, -2]]
    ], dtype=np.float32)
    mc_p2 = MicroCube.from_array(orig_p2, PrecisionMode3D.P2, arena, scale=1.0)
    assert mc_p2.descriptor.byte_size == 2
    np.testing.assert_array_equal(mc_p2.unpack_to_array(), orig_p2)

    # P4
    orig_p4 = np.array([
        [[-7, 5], [3, -8]],
        [[2, -1], [6, -4]]
    ], dtype=np.float32)
    mc_p4 = MicroCube.from_array(orig_p4, PrecisionMode3D.P4, arena, scale=1.0)
    assert mc_p4.descriptor.byte_size == 4
    np.testing.assert_array_equal(mc_p4.unpack_to_array(), orig_p4)

    # P8
    orig_p8 = np.random.randint(-128, 128, size=(2, 2, 2)).astype(np.float32)
    mc_p8 = MicroCube.from_array(orig_p8, PrecisionMode3D.P8, arena, scale=1.0)
    assert mc_p8.descriptor.byte_size == 8
    np.testing.assert_array_equal(mc_p8.unpack_to_array(), orig_p8)


def test_octree_vacuum_pruning():
    arena = MemoryArena(1024 * 1024)
    # 8x8x8 volume with sparse active region (only one 2x2x2 octant active)
    vol = np.zeros((8, 8, 8), dtype=np.float32)
    vol[0:2, 0:2, 0:2] = np.array([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]]
    ])

    octree = OctreeTensor.from_array_recursive(
        vol, PrecisionMode3D.P8, arena, vacuum_threshold=1e-3
    )

    stats = octree.count_nodes()
    # 8x8x8 has 64 elementary 2x2x2 cubes. 63 of them are pure vacuum!
    assert stats["vacuum_nodes"] >= 7
    # Bytes allocated should only be 8 bytes for the single active micro-cube!
    assert stats["bytes_allocated"] == 8

    recon = octree.to_array()
    np.testing.assert_allclose(recon, vol, atol=0.1)


def test_conv3d_and_contraction():
    arena = MemoryArena(1024)
    input_vol = np.random.randn(6, 6, 6).astype(np.float32)
    kernel = np.ones((2, 2, 2), dtype=np.float32)

    mc_filter = MicroCube.from_array(kernel, PrecisionMode3D.P8, arena, scale=1.0)
    conv_out = conv3d_with_microcube(input_vol, mc_filter)

    assert conv_out.shape == (5, 5, 5)
    # Check [0, 0, 0]: sum of 2x2x2 patch
    expected_first = float(np.sum(input_vol[0:2, 0:2, 0:2]))
    np.testing.assert_allclose(conv_out[0, 0, 0], expected_first, atol=1e-4)

    # Test Trilinear Contraction
    A = np.random.randn(4, 3, 5).astype(np.float32)
    B = np.random.randn(5, 3, 4).astype(np.float32)
    out = np.zeros((4, 3, 4), dtype=np.float32)
    trilinear_contraction_3d(A, B, out)

    for d in range(4):
        for h in range(3):
            for w in range(4):
                expected = float(np.dot(A[d, h, :], B[:, h, w]))
                np.testing.assert_allclose(out[d, h, w], expected, atol=1e-4)


if __name__ == "__main__":
    test_cube_p1_spin_exact()
    test_cube_p2_p4_p8_roundtrip()
    test_octree_vacuum_pruning()
    test_conv3d_and_contraction()
    print("All 3D Spatial tests passed!")
