"""
Bit-Packing and Unpacking Routines for MTC-8 Precision Modes (P2, P4, P8, P16)
Accelerated with Numba.
"""

import numpy as np
from numba import njit, prange
from .types import PrecisionMode


@njit(fastmath=True, inline="always")
def pack_p2_quad(q0: int, q1: int, q2: int, q3: int) -> int:
    """Pack 4 signed 2-bit values (-2..1) into a single uint8 byte."""
    u0 = q0 & 0x03
    u1 = q1 & 0x03
    u2 = q2 & 0x03
    u3 = q3 & 0x03
    return u0 | (u1 << 2) | (u2 << 4) | (u3 << 6)


@njit(fastmath=True, inline="always")
def unpack_p2_quad(byte_val: int):
    """Unpack a single uint8 byte into 4 signed 2-bit values (-2..1)."""
    # Arithmetic right shift via int8 sign extension
    q0 = np.int8((byte_val & 0x03) << 6) >> 6
    q1 = np.int8(((byte_val >> 2) & 0x03) << 6) >> 6
    q2 = np.int8(((byte_val >> 4) & 0x03) << 6) >> 6
    q3 = np.int8(((byte_val >> 6) & 0x03) << 6) >> 6
    return q0, q1, q2, q3


@njit(fastmath=True, inline="always")
def pack_p4_pair(q0: int, q1: int) -> int:
    """Pack 2 signed 4-bit values (-8..7) into a single uint8 byte."""
    u0 = q0 & 0x0F
    u1 = q1 & 0x0F
    return u0 | (u1 << 4)


@njit(fastmath=True, inline="always")
def unpack_p4_pair(byte_val: int):
    """Unpack a single uint8 byte into 2 signed 4-bit values (-8..7)."""
    q0 = np.int8((byte_val & 0x0F) << 4) >> 4
    q1 = np.int8(((byte_val >> 4) & 0x0F) << 4) >> 4
    return q0, q1


# =========================================================================
# Tile-Level (2x2) Pack / Unpack Routines
# =========================================================================

@njit(fastmath=True)
def pack_tile_2x2_p2(q: np.ndarray) -> int:
    """
    Pack a 2x2 int8 array into a single uint8 byte in row-major order:
    q[0,0], q[0,1], q[1,0], q[1,1].
    """
    return pack_p2_quad(int(q[0, 0]), int(q[0, 1]), int(q[1, 0]), int(q[1, 1]))


@njit(fastmath=True)
def unpack_tile_2x2_p2(byte_val: int) -> np.ndarray:
    """Unpack a single uint8 byte into a 2x2 int8 array."""
    q0, q1, q2, q3 = unpack_p2_quad(byte_val)
    out = np.empty((2, 2), dtype=np.int8)
    out[0, 0] = q0
    out[0, 1] = q1
    out[1, 0] = q2
    out[1, 1] = q3
    return out


@njit(fastmath=True)
def pack_tile_2x2_p4(q: np.ndarray, out_bytes: np.ndarray, out_idx: int = 0):
    """Pack a 2x2 int8 array into 2 bytes: row 0 in byte 0, row 1 in byte 1."""
    out_bytes[out_idx] = pack_p4_pair(int(q[0, 0]), int(q[0, 1]))
    out_bytes[out_idx + 1] = pack_p4_pair(int(q[1, 0]), int(q[1, 1]))


@njit(fastmath=True)
def unpack_tile_2x2_p4(bytes_val: np.ndarray, start_idx: int = 0) -> np.ndarray:
    """Unpack 2 uint8 bytes into a 2x2 int8 array."""
    q0, q1 = unpack_p4_pair(int(bytes_val[start_idx]))
    q2, q3 = unpack_p4_pair(int(bytes_val[start_idx + 1]))
    out = np.empty((2, 2), dtype=np.int8)
    out[0, 0] = q0
    out[0, 1] = q1
    out[1, 0] = q2
    out[1, 1] = q3
    return out


@njit(fastmath=True)
def pack_tile_2x2_p8(q: np.ndarray, out_bytes: np.ndarray, out_idx: int = 0):
    """Pack a 2x2 int8 array into 4 raw uint8 bytes."""
    out_bytes[out_idx] = np.uint8(q[0, 0])
    out_bytes[out_idx + 1] = np.uint8(q[0, 1])
    out_bytes[out_idx + 2] = np.uint8(q[1, 0])
    out_bytes[out_idx + 3] = np.uint8(q[1, 1])


@njit(fastmath=True)
def unpack_tile_2x2_p8(bytes_val: np.ndarray, start_idx: int = 0) -> np.ndarray:
    """Unpack 4 uint8 bytes into a 2x2 int8 array."""
    out = np.empty((2, 2), dtype=np.int8)
    out[0, 0] = np.int8(bytes_val[start_idx])
    out[0, 1] = np.int8(bytes_val[start_idx + 1])
    out[1, 0] = np.int8(bytes_val[start_idx + 2])
    out[1, 1] = np.int8(bytes_val[start_idx + 3])
    return out


# =========================================================================
# Bulk Matrix Packing / Unpacking (Tiled 2x2 Layout)
# =========================================================================

@njit(parallel=True, fastmath=True)
def pack_matrix_p2(q_matrix: np.ndarray) -> np.ndarray:
    """
    Pack an (H, W) int8 matrix (H and W must be multiples of 2) into
    a contiguous uint8 byte array of size (H//2) * (W//2).
    """
    H, W = q_matrix.shape
    num_tiles_h = H // 2
    num_tiles_w = W // 2
    out = np.empty((num_tiles_h, num_tiles_w), dtype=np.uint8)

    for ti in prange(num_tiles_h):
        r = ti * 2
        for tj in range(num_tiles_w):
            c = tj * 2
            out[ti, tj] = pack_p2_quad(
                int(q_matrix[r, c]),
                int(q_matrix[r, c + 1]),
                int(q_matrix[r + 1, c]),
                int(q_matrix[r + 1, c + 1])
            )
    return out.ravel()


@njit(parallel=True, fastmath=True)
def unpack_matrix_p2(packed_bytes: np.ndarray, H: int, W: int) -> np.ndarray:
    """Unpack packed P2 bytes back into an (H, W) int8 matrix."""
    num_tiles_h = H // 2
    num_tiles_w = W // 2
    out = np.empty((H, W), dtype=np.int8)
    bytes_2d = packed_bytes.reshape((num_tiles_h, num_tiles_w))

    for ti in prange(num_tiles_h):
        r = ti * 2
        for tj in range(num_tiles_w):
            c = tj * 2
            b = int(bytes_2d[ti, tj])
            q0, q1, q2, q3 = unpack_p2_quad(b)
            out[r, c] = q0
            out[r, c + 1] = q1
            out[r + 1, c] = q2
            out[r + 1, c + 1] = q3
    return out


@njit(parallel=True, fastmath=True)
def pack_matrix_p4(q_matrix: np.ndarray) -> np.ndarray:
    """
    Pack an (H, W) int8 matrix into (H//2, W//2, 2) uint8 bytes.
    """
    H, W = q_matrix.shape
    num_tiles_h = H // 2
    num_tiles_w = W // 2
    out = np.empty((num_tiles_h, num_tiles_w, 2), dtype=np.uint8)

    for ti in prange(num_tiles_h):
        r = ti * 2
        for tj in range(num_tiles_w):
            c = tj * 2
            out[ti, tj, 0] = pack_p4_pair(int(q_matrix[r, c]), int(q_matrix[r, c + 1]))
            out[ti, tj, 1] = pack_p4_pair(int(q_matrix[r + 1, c]), int(q_matrix[r + 1, c + 1]))
    return out.ravel()


@njit(parallel=True, fastmath=True)
def unpack_matrix_p4(packed_bytes: np.ndarray, H: int, W: int) -> np.ndarray:
    """Unpack packed P4 bytes back into an (H, W) int8 matrix."""
    num_tiles_h = H // 2
    num_tiles_w = W // 2
    out = np.empty((H, W), dtype=np.int8)
    bytes_3d = packed_bytes.reshape((num_tiles_h, num_tiles_w, 2))

    for ti in prange(num_tiles_h):
        r = ti * 2
        for tj in range(num_tiles_w):
            c = tj * 2
            q0, q1 = unpack_p4_pair(int(bytes_3d[ti, tj, 0]))
            q2, q3 = unpack_p4_pair(int(bytes_3d[ti, tj, 1]))
            out[r, c] = q0
            out[r, c + 1] = q1
            out[r + 1, c] = q2
            out[r + 1, c + 1] = q3
    return out
