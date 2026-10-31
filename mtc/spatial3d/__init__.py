"""
Spatial 3D MicroTensor Computing: MicroCubes, Octree Hierarchy, and 3D Contractions.
"""

from .types3d import PrecisionMode3D, OctantIndex
from .packing3d import (
    pack_cube_2x2x2_p1_spin, unpack_cube_2x2x2_p1_spin,
    pack_cube_2x2x2_p2, unpack_cube_2x2x2_p2,
    pack_cube_2x2x2_p4, unpack_cube_2x2x2_p4,
    pack_cube_2x2x2_p8, unpack_cube_2x2x2_p8
)
from .microcube import MicroCube, CubeDescriptor
from .octree_tensor import OctreeTensor
from .conv3d import conv3d_2x2x2_kernel, trilinear_contraction_3d, conv3d_with_microcube

__all__ = [
    "PrecisionMode3D",
    "OctantIndex",
    "MicroCube",
    "CubeDescriptor",
    "OctreeTensor",
    "pack_cube_2x2x2_p1_spin",
    "unpack_cube_2x2x2_p1_spin",
    "conv3d_2x2x2_kernel",
    "trilinear_contraction_3d",
    "conv3d_with_microcube",
]
