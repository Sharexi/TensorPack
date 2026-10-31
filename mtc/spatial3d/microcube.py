"""
MicroCube: Elementary 3D Computational Unit (2x2x2) for MTC Spatial Geometry.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

from mtc.core.arena import MemoryArena
from .types3d import PrecisionMode3D
from .packing3d import (
    pack_cube_2x2x2_p1_spin, unpack_cube_2x2x2_p1_spin,
    pack_cube_2x2x2_p2, unpack_cube_2x2x2_p2,
    pack_cube_2x2x2_p4, unpack_cube_2x2x2_p4,
    pack_cube_2x2x2_p8, unpack_cube_2x2x2_p8
)


@dataclass(frozen=True)
class CubeDescriptor:
    """Metadata descriptor for a 2x2x2 MicroCube."""
    arena_offset: int
    precision: PrecisionMode3D
    scale: float
    byte_size: int
    is_vacuum: bool = False


class MicroCube:
    """
    Elementary 3D computational volume element (2x2x2 = 8 voxels).
    """

    def __init__(self, descriptor: CubeDescriptor, arena: MemoryArena):
        self.descriptor = descriptor
        self.arena = arena

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (2, 2, 2)

    @property
    def precision(self) -> PrecisionMode3D:
        return self.descriptor.precision

    @property
    def scale(self) -> float:
        return self.descriptor.scale

    @property
    def is_vacuum(self) -> bool:
        return self.descriptor.is_vacuum

    @property
    def raw_bytes(self) -> np.ndarray:
        if self.descriptor.is_vacuum or self.descriptor.byte_size == 0:
            return np.empty(0, dtype=np.uint8)
        return self.arena.view(self.descriptor.arena_offset, self.descriptor.byte_size)

    @classmethod
    def vacuum(cls, arena: MemoryArena) -> "MicroCube":
        """Construct a zero-energy / vacuum micro-cube (0 allocated bytes)."""
        desc = CubeDescriptor(
            arena_offset=0,
            precision=PrecisionMode3D.P1,
            scale=0.0,
            byte_size=0,
            is_vacuum=True
        )
        return cls(desc, arena)

    @classmethod
    def from_array(
        cls,
        cube_3d: np.ndarray,
        precision: PrecisionMode3D,
        arena: MemoryArena,
        scale: Optional[float] = None,
        vacuum_threshold: float = 1e-8
    ) -> "MicroCube":
        """
        Quantize and pack a (2, 2, 2) volume element into arena memory.
        If all elements are below vacuum_threshold, returns a zero-byte vacuum cube.
        """
        arr = np.asarray(cube_3d, dtype=np.float32)
        assert arr.shape == (2, 2, 2), f"Expected shape (2,2,2), got {arr.shape}"

        max_abs = float(np.max(np.abs(arr)))
        if max_abs < vacuum_threshold:
            return cls.vacuum(arena)

        if scale is None:
            if precision == PrecisionMode3D.P1:
                scale = float(np.mean(np.abs(arr)))
            elif precision == PrecisionMode3D.P2:
                scale = max_abs / 1.0
            elif precision == PrecisionMode3D.P4:
                scale = max_abs / 7.0
            elif precision == PrecisionMode3D.P8:
                scale = max_abs / 127.0
            elif precision == PrecisionMode3D.P16:
                scale = max_abs / 32767.0

        if precision == PrecisionMode3D.P1:
            # 1-bit spin {-1, +1}: 8 values = exactly 1 uint8 byte!
            byte_val = np.uint8(pack_cube_2x2x2_p1_spin(arr))
            offset = arena.write_bytes(np.array([byte_val], dtype=np.uint8))
            desc = CubeDescriptor(offset, precision, scale, byte_size=1)
            return cls(desc, arena)

        inv_scale = 1.0 / scale if scale > 1e-9 else 1.0
        q_floats = np.round(arr * inv_scale)

        if precision == PrecisionMode3D.P2:
            q = np.clip(q_floats, -2, 1).astype(np.int8)
            buf = np.empty(2, dtype=np.uint8)
            pack_cube_2x2x2_p2(q, buf, 0)
            offset = arena.write_bytes(buf)
            desc = CubeDescriptor(offset, precision, scale, byte_size=2)

        elif precision == PrecisionMode3D.P4:
            q = np.clip(q_floats, -8, 7).astype(np.int8)
            buf = np.empty(4, dtype=np.uint8)
            pack_cube_2x2x2_p4(q, buf, 0)
            offset = arena.write_bytes(buf)
            desc = CubeDescriptor(offset, precision, scale, byte_size=4)

        elif precision == PrecisionMode3D.P8:
            q = np.clip(q_floats, -128, 127).astype(np.int8)
            buf = np.empty(8, dtype=np.uint8)
            pack_cube_2x2x2_p8(q, buf, 0)
            offset = arena.write_bytes(buf)
            desc = CubeDescriptor(offset, precision, scale, byte_size=8)

        elif precision == PrecisionMode3D.P16:
            q = np.clip(q_floats, -32768, 32767).astype(np.int16)
            raw = q.view(np.uint8).ravel()
            offset = arena.write_bytes(raw, alignment=2)
            desc = CubeDescriptor(offset, precision, scale, byte_size=16)

        else:
            raise ValueError(f"Unknown precision {precision}")

        return cls(desc, arena)

    def unpack_to_array(self) -> np.ndarray:
        """Unpack raw arena bytes and dequantize back to float32 (2, 2, 2) cube."""
        if self.descriptor.is_vacuum:
            return np.zeros((2, 2, 2), dtype=np.float32)

        raw = self.raw_bytes
        precision = self.descriptor.precision
        scale = self.descriptor.scale
        out = np.empty((2, 2, 2), dtype=np.int8)

        if precision == PrecisionMode3D.P1:
            unpack_cube_2x2x2_p1_spin(int(raw[0]), out)
            return out.astype(np.float32) * scale

        elif precision == PrecisionMode3D.P2:
            unpack_cube_2x2x2_p2(raw, 0, out)
            return out.astype(np.float32) * scale

        elif precision == PrecisionMode3D.P4:
            unpack_cube_2x2x2_p4(raw, 0, out)
            return out.astype(np.float32) * scale

        elif precision == PrecisionMode3D.P8:
            unpack_cube_2x2x2_p8(raw, 0, out)
            return out.astype(np.float32) * scale

        elif precision == PrecisionMode3D.P16:
            q16 = np.frombuffer(raw, dtype=np.int16).reshape((2, 2, 2))
            return q16.astype(np.float32) * scale

        raise ValueError(f"Unknown precision {precision}")
