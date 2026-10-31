"""
MicroTensor Definition and Block Descriptors for MTC-8
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

from .types import PrecisionMode
from .arena import MemoryArena
from .packing import (
    pack_tile_2x2_p2, unpack_tile_2x2_p2,
    pack_tile_2x2_p4, unpack_tile_2x2_p4,
    pack_tile_2x2_p8, unpack_tile_2x2_p8,
    pack_p4_pair, unpack_p4_pair
)


@dataclass(frozen=True)
class BlockDescriptor:
    """
    Metadata descriptor for a MicroTensor block.
    Decoupled from the 8-bit payload storage bits.
    """
    arena_offset: int
    precision: PrecisionMode
    shape: Tuple[int, int]
    scale: float
    zero_point: float = 0.0
    byte_size: int = 1


class MicroTensor:
    """
    Elementary computational unit in MTC-8.
    Encapsulates raw payload stored in an arena along with its metadata descriptor.
    """

    def __init__(self, descriptor: BlockDescriptor, arena: MemoryArena):
        self.descriptor = descriptor
        self.arena = arena

    @property
    def precision(self) -> PrecisionMode:
        return self.descriptor.precision

    @property
    def shape(self) -> Tuple[int, int]:
        return self.descriptor.shape

    @property
    def scale(self) -> float:
        return self.descriptor.scale

    @property
    def byte_offset(self) -> int:
        return self.descriptor.arena_offset

    @property
    def raw_bytes(self) -> np.ndarray:
        """Contiguous uint8 slice directly referencing arena memory."""
        return self.arena.view(self.descriptor.arena_offset, self.descriptor.byte_size)

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        precision: PrecisionMode,
        arena: MemoryArena,
        scale: Optional[float] = None,
        zero_point: float = 0.0
    ) -> "MicroTensor":
        """
        Quantize and pack a 2D NumPy array into the memory arena as a MicroTensor.
        """
        arr = np.asarray(array, dtype=np.float32)
        shape = arr.shape
        num_elements = arr.size

        # Compute symmetric scale if not provided
        if scale is None:
            max_abs = float(np.max(np.abs(arr)))
            if precision == PrecisionMode.P2:
                # Quantize into {-2, -1, 0, 1}; symmetric limit 1.0 or 1.5
                scale = max_abs / 1.0 if max_abs > 1e-7 else 1.0
            elif precision == PrecisionMode.P4:
                scale = max_abs / 7.0 if max_abs > 1e-7 else 1.0
            elif precision == PrecisionMode.P8:
                scale = max_abs / 127.0 if max_abs > 1e-7 else 1.0
            elif precision == PrecisionMode.P16:
                scale = max_abs / 32767.0 if max_abs > 1e-7 else 1.0

        if precision == PrecisionMode.P16:
            inv_scale = 1.0 / scale if scale != 0.0 else 1.0
            q = np.clip(np.round(arr * inv_scale), -32768, 32767).astype(np.int16)
            raw_bytes = q.view(np.uint8).ravel()
            offset = arena.write_bytes(raw_bytes, alignment=2)
            desc = BlockDescriptor(
                arena_offset=offset,
                precision=precision,
                shape=shape,
                scale=scale,
                zero_point=0.0,
                byte_size=raw_bytes.nbytes
            )
            return cls(desc, arena)

        # Symmetric integer quantization
        inv_scale = 1.0 / scale if scale != 0.0 else 1.0
        q_floats = np.round(arr * inv_scale)

        if precision == PrecisionMode.P2:
            q = np.clip(q_floats, -2, 1).astype(np.int8)
            if shape == (2, 2):
                byte_val = np.uint8(pack_tile_2x2_p2(q))
                offset = arena.write_bytes(np.array([byte_val], dtype=np.uint8))
                byte_size = 1
            else:
                # Generic P2 packing: 4 elements per byte
                flat = q.ravel()
                pad_len = (4 - (len(flat) % 4)) % 4
                if pad_len > 0:
                    flat = np.pad(flat, (0, pad_len))
                n_bytes = len(flat) // 4
                packed = np.empty(n_bytes, dtype=np.uint8)
                for i in range(n_bytes):
                    idx = i * 4
                    packed[i] = ((flat[idx] & 0x03) |
                                 ((flat[idx + 1] & 0x03) << 2) |
                                 ((flat[idx + 2] & 0x03) << 4) |
                                 ((flat[idx + 3] & 0x03) << 6))
                offset = arena.write_bytes(packed)
                byte_size = n_bytes

        elif precision == PrecisionMode.P4:
            q = np.clip(q_floats, -8, 7).astype(np.int8)
            if shape == (2, 2):
                packed = np.empty(2, dtype=np.uint8)
                pack_tile_2x2_p4(q, packed, 0)
                offset = arena.write_bytes(packed)
                byte_size = 2
            elif shape in ((2, 1), (1, 2)):
                flat = q.ravel()
                byte_val = np.uint8(pack_p4_pair(int(flat[0]), int(flat[1])))
                offset = arena.write_bytes(np.array([byte_val], dtype=np.uint8))
                byte_size = 1
            else:
                flat = q.ravel()
                pad_len = (2 - (len(flat) % 2)) % 2
                if pad_len > 0:
                    flat = np.pad(flat, (0, pad_len))
                n_bytes = len(flat) // 2
                packed = np.empty(n_bytes, dtype=np.uint8)
                for i in range(n_bytes):
                    packed[i] = pack_p4_pair(int(flat[i * 2]), int(flat[i * 2 + 1]))
                offset = arena.write_bytes(packed)
                byte_size = n_bytes

        elif precision == PrecisionMode.P8:
            q = np.clip(q_floats, -128, 127).astype(np.int8)
            packed = q.view(np.uint8).ravel()
            offset = arena.write_bytes(packed)
            byte_size = packed.nbytes

        else:
            raise ValueError(f"Unsupported precision: {precision}")

        desc = BlockDescriptor(
            arena_offset=offset,
            precision=precision,
            shape=shape,
            scale=scale,
            zero_point=zero_point,
            byte_size=byte_size
        )
        return cls(desc, arena)

    def unpack_to_array(self) -> np.ndarray:
        """
        Unpack raw arena payload and dequantize back to float32 NumPy array.
        """
        raw = self.raw_bytes
        shape = self.descriptor.shape
        precision = self.descriptor.precision
        scale = self.descriptor.scale

        if precision == PrecisionMode.P16:
            q = np.frombuffer(raw, dtype=np.int16).reshape(shape)
            return q.astype(np.float32) * scale

        if precision == PrecisionMode.P2:
            if shape == (2, 2):
                q = unpack_tile_2x2_p2(int(raw[0]))
            else:
                # General P2
                flat = np.empty(shape[0] * shape[1], dtype=np.int8)
                idx = 0
                for b in raw:
                    for s in (0, 2, 4, 6):
                        if idx < len(flat):
                            flat[idx] = np.int8(((int(b) >> s) & 0x03) << 6) >> 6
                            idx += 1
                q = flat.reshape(shape)

        elif precision == PrecisionMode.P4:
            if shape == (2, 2):
                q = unpack_tile_2x2_p4(raw, 0)
            elif shape in ((2, 1), (1, 2)):
                q0, q1 = unpack_p4_pair(int(raw[0]))
                q = np.array([q0, q1], dtype=np.int8).reshape(shape)
            else:
                flat = np.empty(shape[0] * shape[1], dtype=np.int8)
                idx = 0
                for b in raw:
                    q0, q1 = unpack_p4_pair(int(b))
                    if idx < len(flat):
                        flat[idx] = q0
                        idx += 1
                    if idx < len(flat):
                        flat[idx] = q1
                        idx += 1
                q = flat.reshape(shape)

        elif precision == PrecisionMode.P8:
            q = np.frombuffer(raw, dtype=np.int8).reshape(shape)

        else:
            raise ValueError(f"Unknown precision: {precision}")

        return q.astype(np.float32) * scale
