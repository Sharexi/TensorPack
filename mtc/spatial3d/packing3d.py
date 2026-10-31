"""
Bit-Packing and Unpacking Routines for 3D MicroCubes (2x2x2).
Compiled with Numba.
"""

import numpy as np
from numba import njit
from mtc.core.packing import (
    pack_p2_quad, unpack_p2_quad,
    pack_p4_pair, unpack_p4_pair
)


@njit(fastmath=True, inline="always")
def pack_cube_2x2x2_p1_spin(q_cube: np.ndarray) -> int:
    """
    Pack 2x2x2 cube with values in {-1, +1} into a single uint8 byte.
    Bit = 1 if val >= 0 else 0.
    """
    b0 = 1 if q_cube[0, 0, 0] >= 0 else 0
    b1 = 1 if q_cube[0, 0, 1] >= 0 else 0
    b2 = 1 if q_cube[0, 1, 0] >= 0 else 0
    b3 = 1 if q_cube[0, 1, 1] >= 0 else 0
    b4 = 1 if q_cube[1, 0, 0] >= 0 else 0
    b5 = 1 if q_cube[1, 0, 1] >= 0 else 0
    b6 = 1 if q_cube[1, 1, 0] >= 0 else 0
    b7 = 1 if q_cube[1, 1, 1] >= 0 else 0
    return b0 | (b1 << 1) | (b2 << 2) | (b3 << 3) | (b4 << 4) | (b5 << 5) | (b6 << 6) | (b7 << 7)


@njit(fastmath=True, inline="always")
def unpack_cube_2x2x2_p1_spin(byte_val: int, out_buf: np.ndarray):
    """
    Unpack a single uint8 byte into a 2x2x2 int8 cube with values in {-1, +1}.
    """
    out_buf[0, 0, 0] = np.int8(1 if (byte_val & 0x01) != 0 else -1)
    out_buf[0, 0, 1] = np.int8(1 if (byte_val & 0x02) != 0 else -1)
    out_buf[0, 1, 0] = np.int8(1 if (byte_val & 0x04) != 0 else -1)
    out_buf[0, 1, 1] = np.int8(1 if (byte_val & 0x08) != 0 else -1)
    out_buf[1, 0, 0] = np.int8(1 if (byte_val & 0x10) != 0 else -1)
    out_buf[1, 0, 1] = np.int8(1 if (byte_val & 0x20) != 0 else -1)
    out_buf[1, 1, 0] = np.int8(1 if (byte_val & 0x40) != 0 else -1)
    out_buf[1, 1, 1] = np.int8(1 if (byte_val & 0x80) != 0 else -1)


@njit(fastmath=True)
def pack_cube_2x2x2_p2(q_cube: np.ndarray, out_bytes: np.ndarray, out_idx: int = 0):
    """
    Pack 2x2x2 int8 cube into 2 bytes:
    Byte 0: z=0 slice (4 elements)
    Byte 1: z=1 slice (4 elements)
    """
    out_bytes[out_idx] = pack_p2_quad(
        int(q_cube[0, 0, 0]), int(q_cube[0, 0, 1]),
        int(q_cube[0, 1, 0]), int(q_cube[0, 1, 1])
    )
    out_bytes[out_idx + 1] = pack_p2_quad(
        int(q_cube[1, 0, 0]), int(q_cube[1, 0, 1]),
        int(q_cube[1, 1, 0]), int(q_cube[1, 1, 1])
    )


@njit(fastmath=True)
def unpack_cube_2x2x2_p2(raw_bytes: np.ndarray, start_idx: int, out_buf: np.ndarray):
    """Unpack 2 bytes into a 2x2x2 int8 cube."""
    q0, q1, q2, q3 = unpack_p2_quad(int(raw_bytes[start_idx]))
    q4, q5, q6, q7 = unpack_p2_quad(int(raw_bytes[start_idx + 1]))

    out_buf[0, 0, 0] = q0
    out_buf[0, 0, 1] = q1
    out_buf[0, 1, 0] = q2
    out_buf[0, 1, 1] = q3
    out_buf[1, 0, 0] = q4
    out_buf[1, 0, 1] = q5
    out_buf[1, 1, 0] = q6
    out_buf[1, 1, 1] = q7


@njit(fastmath=True)
def pack_cube_2x2x2_p4(q_cube: np.ndarray, out_bytes: np.ndarray, out_idx: int = 0):
    """Pack 2x2x2 int8 cube into 4 bytes (2 values per byte)."""
    out_bytes[out_idx]     = pack_p4_pair(int(q_cube[0, 0, 0]), int(q_cube[0, 0, 1]))
    out_bytes[out_idx + 1] = pack_p4_pair(int(q_cube[0, 1, 0]), int(q_cube[0, 1, 1]))
    out_bytes[out_idx + 2] = pack_p4_pair(int(q_cube[1, 0, 0]), int(q_cube[1, 0, 1]))
    out_bytes[out_idx + 3] = pack_p4_pair(int(q_cube[1, 1, 0]), int(q_cube[1, 1, 1]))


@njit(fastmath=True)
def unpack_cube_2x2x2_p4(raw_bytes: np.ndarray, start_idx: int, out_buf: np.ndarray):
    """Unpack 4 bytes into a 2x2x2 int8 cube."""
    q0, q1 = unpack_p4_pair(int(raw_bytes[start_idx]))
    q2, q3 = unpack_p4_pair(int(raw_bytes[start_idx + 1]))
    q4, q5 = unpack_p4_pair(int(raw_bytes[start_idx + 2]))
    q6, q7 = unpack_p4_pair(int(raw_bytes[start_idx + 3]))

    out_buf[0, 0, 0] = q0
    out_buf[0, 0, 1] = q1
    out_buf[0, 1, 0] = q2
    out_buf[0, 1, 1] = q3
    out_buf[1, 0, 0] = q4
    out_buf[1, 0, 1] = q5
    out_buf[1, 1, 0] = q6
    out_buf[1, 1, 1] = q7


@njit(fastmath=True)
def pack_cube_2x2x2_p8(q_cube: np.ndarray, out_bytes: np.ndarray, out_idx: int = 0):
    """Pack 2x2x2 int8 cube into 8 raw uint8 bytes."""
    idx = out_idx
    for z in range(2):
        for y in range(2):
            for x in range(2):
                out_bytes[idx] = np.uint8(q_cube[z, y, x])
                idx += 1


@njit(fastmath=True)
def unpack_cube_2x2x2_p8(raw_bytes: np.ndarray, start_idx: int, out_buf: np.ndarray):
    """Unpack 8 bytes into a 2x2x2 int8 cube."""
    idx = start_idx
    for z in range(2):
        for y in range(2):
            for x in range(2):
                out_buf[z, y, x] = np.int8(raw_bytes[idx])
                idx += 1
