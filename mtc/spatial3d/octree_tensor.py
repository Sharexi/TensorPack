"""
OctreeTensor: Hierarchical 3D Spatial Octree Composition with Vacuum Pruning.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from mtc.core.arena import MemoryArena
from .types3d import PrecisionMode3D, OctantIndex
from .microcube import MicroCube


class OctreeTensor:
    """
    3D hierarchical spatial tensor represented as an Octree.
    Enables sparse 3D spatial representations where vacuum/empty space
    requires zero arena payload memory.
    """

    def __init__(
        self,
        shape: Tuple[int, int, int],
        microcube: Optional[MicroCube] = None,
        children: Optional[List["OctreeTensor"]] = None,
        is_vacuum: bool = False,
        scale: float = 1.0
    ):
        self.shape = shape
        self.microcube = microcube
        self.children = children  # List of 8 OctreeTensor nodes in Z-order
        self.is_vacuum = is_vacuum
        self.scale = scale

    @property
    def is_leaf(self) -> bool:
        return self.microcube is not None

    @property
    def is_composite(self) -> bool:
        return self.children is not None

    @classmethod
    def from_microcube(cls, mc: MicroCube) -> "OctreeTensor":
        """Wrap an elementary 2x2x2 MicroCube as a leaf node."""
        return cls(
            shape=mc.shape,
            microcube=mc,
            is_vacuum=mc.is_vacuum,
            scale=1.0
        )

    @classmethod
    def vacuum_node(cls, shape: Tuple[int, int, int]) -> "OctreeTensor":
        """Construct a pruned vacuum node representing empty space."""
        return cls(
            shape=shape,
            is_vacuum=True,
            scale=0.0
        )

    @classmethod
    def from_octants(cls, octants: List["OctreeTensor"]) -> "OctreeTensor":
        """Compose 8 sub-octree nodes into a macro-octree node."""
        assert len(octants) == 8, f"Expected 8 octants, got {len(octants)}"
        sub_d, sub_h, sub_w = octants[0].shape

        # Check if all 8 children are vacuum
        if all(child.is_vacuum for child in octants):
            return cls.vacuum_node((sub_d * 2, sub_h * 2, sub_w * 2))

        return cls(
            shape=(sub_d * 2, sub_h * 2, sub_w * 2),
            children=octants
        )

    @classmethod
    def from_array_recursive(
        cls,
        volume: np.ndarray,
        precision: PrecisionMode3D,
        arena: MemoryArena,
        vacuum_threshold: float = 1e-4
    ) -> "OctreeTensor":
        """
        Recursively construct a 3D Octree from a (D, H, W) volume.
        Automatically prunes empty / vacuum sub-volumes.
        D, H, W must be powers of 2 (or multiples of 2).
        """
        arr = np.asarray(volume, dtype=np.float32)
        D, H, W = arr.shape
        assert D == H == W and (D & (D - 1) == 0) and D >= 2, f"Shape must be cubic power-of-2 (>=2), got {arr.shape}"

        # Check whole volume for vacuum pruning
        if float(np.max(np.abs(arr))) < vacuum_threshold:
            return cls.vacuum_node((D, H, W))

        # Base case: 2x2x2 elementary leaf
        if (D, H, W) == (2, 2, 2):
            mc = MicroCube.from_array(arr, precision, arena, vacuum_threshold=vacuum_threshold)
            return cls.from_microcube(mc)

        # Recursive subdivision into 8 octants
        half = D // 2
        octant_slices = [
            (slice(0, half),    slice(0, half),    slice(0, half)),    # 000
            (slice(0, half),    slice(0, half),    slice(half, D)),    # 001
            (slice(0, half),    slice(half, D),    slice(0, half)),    # 010
            (slice(0, half),    slice(half, D),    slice(half, D)),    # 011
            (slice(half, D),    slice(0, half),    slice(0, half)),    # 100
            (slice(half, D),    slice(0, half),    slice(half, D)),    # 101
            (slice(half, D),    slice(half, D),    slice(0, half)),    # 110
            (slice(half, D),    slice(half, D),    slice(half, D)),    # 111
        ]

        children = [
            cls.from_array_recursive(arr[sl], precision, arena, vacuum_threshold=vacuum_threshold)
            for sl in octant_slices
        ]

        return cls.from_octants(children)

    def to_array(self) -> np.ndarray:
        """Reconstruct full 3D float32 NumPy array."""
        D, H, W = self.shape
        if self.is_vacuum:
            return np.zeros((D, H, W), dtype=np.float32)

        if self.is_leaf:
            return self.microcube.unpack_to_array() * self.scale

        if self.is_composite:
            half_d, half_h, half_w = D // 2, H // 2, W // 2
            out = np.empty((D, H, W), dtype=np.float32)

            # Reconstruct 8 octants in Z-order
            c = self.children
            out[0:half_d, 0:half_h, 0:half_w] = c[0].to_array()
            out[0:half_d, 0:half_h, half_w:W] = c[1].to_array()
            out[0:half_d, half_h:H, 0:half_w] = c[2].to_array()
            out[0:half_d, half_h:H, half_w:W] = c[3].to_array()

            out[half_d:D, 0:half_h, 0:half_w] = c[4].to_array()
            out[half_d:D, 0:half_h, half_w:W] = c[5].to_array()
            out[half_d:D, half_h:H, 0:half_w] = c[6].to_array()
            out[half_d:D, half_h:H, half_w:W] = c[7].to_array()

            return out * self.scale

        raise RuntimeError("Invalid OctreeTensor state")

    def count_nodes(self) -> Dict[str, Any]:
        """Collect structural statistics on the Octree."""
        if self.is_vacuum:
            return {"total_nodes": 1, "leaf_nodes": 1, "vacuum_nodes": 1, "bytes_allocated": 0}

        if self.is_leaf:
            b = self.microcube.descriptor.byte_size
            return {"total_nodes": 1, "leaf_nodes": 1, "vacuum_nodes": 0, "bytes_allocated": b}

        stats = {"total_nodes": 1, "leaf_nodes": 0, "vacuum_nodes": 0, "bytes_allocated": 0}
        for ch in self.children:
            sub = ch.count_nodes()
            stats["total_nodes"] += sub["total_nodes"]
            stats["leaf_nodes"] += sub["leaf_nodes"]
            stats["vacuum_nodes"] += sub["vacuum_nodes"]
            stats["bytes_allocated"] += sub["bytes_allocated"]

        return stats
