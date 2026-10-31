"""
Core Type Definitions for MicroTensor Computing (MTC-8)
"""

from enum import IntEnum
import numpy as np


class PrecisionMode(IntEnum):
    """Numerical precision mode for MicroTensors."""
    P2 = 2    # 2-bit signed integer (-2 to 1)
    P4 = 4    # 4-bit signed integer (-8 to 7)
    P8 = 8    # 8-bit signed integer (-128 to 127)
    P16 = 16  # 16-bit signed integer (-32768 to 32767 escape mode)

    @property
    def values_per_byte(self) -> float:
        """Number of scalar values stored per byte."""
        return 8.0 / self.value

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
