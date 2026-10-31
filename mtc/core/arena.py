"""
Contiguous Memory Arena for MTC-8 Payload Storage
"""

from typing import Tuple
import numpy as np


class MemoryArena:
    """
    Contiguous uint8 byte arena providing linear, cache-friendly storage for
    all MicroTensor payloads. Offsets are stored as uint32 indices.
    """

    def __init__(self, capacity_bytes: int = 64 * 1024 * 1024):
        self._capacity = int(capacity_bytes)
        self._buffer = np.zeros(self._capacity, dtype=np.uint8)
        self._offset = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def bytes_allocated(self) -> int:
        return self._offset

    @property
    def utilization_pct(self) -> float:
        return (self._offset / self._capacity) * 100.0 if self._capacity > 0 else 0.0

    @property
    def raw_buffer(self) -> np.ndarray:
        return self._buffer

    def allocate(self, size_bytes: int, alignment: int = 1) -> int:
        """
        Allocate a contiguous slice of bytes with specified alignment.
        Returns the byte offset into the arena.
        """
        if alignment > 1:
            remainder = self._offset % alignment
            if remainder != 0:
                self._offset += (alignment - remainder)

        current_offset = self._offset
        new_offset = current_offset + size_bytes
        if new_offset > self._capacity:
            # Grow capacity by 2x or required size
            new_capacity = max(self._capacity * 2, new_offset + 1024 * 1024)
            new_buf = np.zeros(new_capacity, dtype=np.uint8)
            new_buf[:self._offset] = self._buffer[:self._offset]
            self._buffer = new_buf
            self._capacity = new_capacity

        self._offset = new_offset
        return current_offset

    def write_bytes(self, data: np.ndarray, alignment: int = 1) -> int:
        """Write raw uint8 byte buffer into arena and return start offset."""
        if not isinstance(data, np.ndarray) or data.dtype != np.uint8:
            data = np.asarray(data, dtype=np.uint8)
        data_flat = data.ravel()
        size_bytes = data_flat.nbytes
        offset = self.allocate(size_bytes, alignment=alignment)
        self._buffer[offset : offset + size_bytes] = data_flat
        return offset

    def view(self, offset: int, length: int) -> np.ndarray:
        """Return a slice view of the arena without copying."""
        return self._buffer[offset : offset + length]

    def reset(self):
        """Reset allocation offset to reuse the arena."""
        self._offset = 0
