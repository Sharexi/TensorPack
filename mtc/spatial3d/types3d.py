"""
Type definitions and precision modes for 3D MicroTensor Computing.
"""

from enum import IntEnum


class PrecisionMode3D(IntEnum):
    """
    Precision modes for 3D MicroCubes (2x2x2 = 8 values).
    """
    P1 = 1    # 1-bit binary/spin {-1, +1} or {0, 1}: 8 values = 1 byte!
    P2 = 2    # 2-bit signed integer {-2..1}: 8 values = 2 bytes (uint16)
    P4 = 4    # 4-bit signed integer {-8..7}: 8 values = 4 bytes (uint32)
    P8 = 8    # 8-bit signed integer {-128..127}: 8 values = 8 bytes (uint64)
    P16 = 16  # 16-bit signed integer {-32768..32767}: 8 values = 16 bytes

    @property
    def cube_bytes(self) -> int:
        """Bytes required to store one 2x2x2 (8-element) micro-cube."""
        return (8 * self.value) // 8


class OctantIndex(IntEnum):
    """
    Standard Z-order (Morton) indexing for 8 octants in a 2x2x2 subdivision:
    Bit 2: Z, Bit 1: Y, Bit 0: X
    """
    O000 = 0  # (0, 0, 0)
    O001 = 1  # (0, 0, 1)
    O010 = 2  # (0, 1, 0)
    O011 = 3  # (0, 1, 1)
    O100 = 4  # (1, 0, 0)
    O101 = 5  # (1, 0, 1)
    O110 = 6  # (1, 1, 0)
    O111 = 7  # (1, 1, 1)
