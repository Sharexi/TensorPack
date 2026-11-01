"""
Core Type Definitions for MicroTensor Computing (MTC-8)
"""

from enum import IntEnum
import numpy as np


class PrecisionMode(IntEnum):
    """Numerical precision mode for MicroTensors."""
    P0 = 0    # Dropped block: all zeros, no payload (selected by MTC-RD allocation)
    P2 = 2    # 2-bit signed integer (-2 to 1)
    P3 = 3    # 3-bit signed integer (-4 to 3), bitstream-packed
    P4 = 4    # 4-bit signed integer (-8 to 7)
    P5 = 5    # 5-bit signed integer (-16 to 15), bitstream-packed
    P6 = 6    # 6-bit signed integer (-32 to 31), bitstream-packed
    P8 = 8    # 8-bit signed integer (-128 to 127)
    P16 = 16  # 16-bit signed integer (-32768 to 32767 escape mode)

    @property
    def values_per_byte(self) -> float:
        """Number of scalar values stored per byte."""
        return float("inf") if self.value == 0 else 8.0 / self.value

    @property
    def bytes_per_value(self) -> float:
        """Bytes occupied per scalar value."""
        return self.value / 8.0


class OpCode(IntEnum):
    """Operation codes for PointerTensor computational nodes."""
    DATA = 0       # Leaf node: points directly to memory arena
    MATMUL = 1     # Matrix multiplication of child nodes
    ADD = 2        # Elementwise addition of child nodes
    SCALE = 3      # Scalar scaling
    TRANSPOSE = 4  # Transpose of child node
