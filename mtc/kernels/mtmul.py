"""
MTMUL: Native Elementary 2x2 MicroTensor Multiplication Primitive with INT32/FP32 Accumulation.
Compiled with Numba.
"""

import numpy as np
from numba import njit
from ..core.packing import unpack_p2_quad, unpack_p4_pair


@njit(fastmath=True, inline="always")
def mtmul_2x2_int(
    a00: int, a01: int, a10: int, a11: int,
    b00: int, b01: int, b10: int, b11: int,
    c: np.ndarray, c_r: int = 0, c_c: int = 0
):
    """
    Elementary 2x2 integer matrix multiply-accumulate:
    c[0,0] += a00*b00 + a01*b10
    c[0,1] += a00*b01 + a01*b11
    c[1,0] += a10*b00 + a11*b10
    c[1,1] += a10*b01 + a11*b11
    Accumulates directly into int32 buffer c.
    """
    c[c_r,     c_c]     += a00 * b00 + a01 * b10
    c[c_r,     c_c + 1] += a00 * b01 + a01 * b11
    c[c_r + 1, c_c]     += a10 * b00 + a11 * b10
    c[c_r + 1, c_c + 1] += a10 * b01 + a11 * b11


@njit(fastmath=True, inline="always")
def mtmul_2x2_fp32(
    a00: float, a01: float, a10: float, a11: float,
    b00: float, b01: float, b10: float, b11: float,
    scale_ab: float,
    c: np.ndarray, c_r: int = 0, c_c: int = 0
):
    """
    Elementary 2x2 floating-point multiply-accumulate:
    Accumulates directly into float32 buffer c.
    """
    c[c_r,     c_c]     += scale_ab * (a00 * b00 + a01 * b10)
    c[c_r,     c_c + 1] += scale_ab * (a00 * b01 + a01 * b11)
    c[c_r + 1, c_c]     += scale_ab * (a10 * b00 + a11 * b10)
    c[c_r + 1, c_c + 1] += scale_ab * (a10 * b01 + a11 * b11)


@njit(fastmath=True)
def mtmul_p2_p2_int32(byte_a: int, byte_b: int, accum_int32: np.ndarray, c_r: int = 0, c_c: int = 0):
    """
    Execute MTMUL on two packed P2 8-bit micro-tensors (1 byte each).
    Unpacks registers and accumulates into int32 array.
    """
    a00, a01, a10, a11 = unpack_p2_quad(byte_a)
    b00, b01, b10, b11 = unpack_p2_quad(byte_b)
    mtmul_2x2_int(a00, a01, a10, a11, b00, b01, b10, b11, accum_int32, c_r, c_c)


@njit(fastmath=True)
def mtmul_p4_p4_int32(bytes_a: np.ndarray, a_idx: int, bytes_b: np.ndarray, b_idx: int, accum_int32: np.ndarray, c_r: int = 0, c_c: int = 0):
    """
    Execute MTMUL on two packed P4 micro-tensors (2 bytes each).
    Accumulates into int32 array.
    """
    a00, a01 = unpack_p4_pair(int(bytes_a[a_idx]))
    a10, a11 = unpack_p4_pair(int(bytes_a[a_idx + 1]))

    b00, b01 = unpack_p4_pair(int(bytes_b[b_idx]))
    b10, b11 = unpack_p4_pair(int(bytes_b[b_idx + 1]))

    mtmul_2x2_int(a00, a01, a10, a11, b00, b01, b10, b11, accum_int32, c_r, c_c)


@njit(fastmath=True)
def mtmul_p8_p8_int32(bytes_a: np.ndarray, a_idx: int, bytes_b: np.ndarray, b_idx: int, accum_int32: np.ndarray, c_r: int = 0, c_c: int = 0):
    """
    Execute MTMUL on two P8 micro-tensors (4 bytes each).
    Accumulates into int32 array.
    """
    a00 = np.int8(bytes_a[a_idx])
    a01 = np.int8(bytes_a[a_idx + 1])
    a10 = np.int8(bytes_a[a_idx + 2])
    a11 = np.int8(bytes_a[a_idx + 3])

    b00 = np.int8(bytes_b[b_idx])
    b01 = np.int8(bytes_b[b_idx + 1])
    b10 = np.int8(bytes_b[b_idx + 2])
    b11 = np.int8(bytes_b[b_idx + 3])

    mtmul_2x2_int(a00, a01, a10, a11, b00, b01, b10, b11, accum_int32, c_r, c_c)


@njit(fastmath=True)
def mtmul_heterogeneous(
    a_prec: int, bytes_a: np.ndarray, a_idx: int, scale_a: float,
    b_prec: int, bytes_b: np.ndarray, b_idx: int, scale_b: float,
    accum_fp32: np.ndarray, c_r: int = 0, c_c: int = 0
):
    """
    General MTMUL handling mixed precision operands (e.g. P2 x P4, P4 x P8, etc.)
    with fused scale accumulation into FP32.
    """
    # Unpack A into floats
    if a_prec == 2:
        qa0, qa1, qa2, qa3 = unpack_p2_quad(int(bytes_a[a_idx]))
        a00 = float(qa0)
        a01 = float(qa1)
        a10 = float(qa2)
        a11 = float(qa3)
    elif a_prec == 4:
        qa0, qa1 = unpack_p4_pair(int(bytes_a[a_idx]))
        qa2, qa3 = unpack_p4_pair(int(bytes_a[a_idx + 1]))
        a00 = float(qa0)
        a01 = float(qa1)
        a10 = float(qa2)
        a11 = float(qa3)
    elif a_prec == 8:
        a00 = float(np.int8(bytes_a[a_idx]))
        a01 = float(np.int8(bytes_a[a_idx + 1]))
        a10 = float(np.int8(bytes_a[a_idx + 2]))
        a11 = float(np.int8(bytes_a[a_idx + 3]))
    else:
        # P16 or general
        a00 = 0.0
        a01 = 0.0
        a10 = 0.0
        a11 = 0.0

    # Unpack B into floats
    if b_prec == 2:
        qb0, qb1, qb2, qb3 = unpack_p2_quad(int(bytes_b[b_idx]))
        b00 = float(qb0)
        b01 = float(qb1)
        b10 = float(qb2)
        b11 = float(qb3)
    elif b_prec == 4:
        qb0, qb1 = unpack_p4_pair(int(bytes_b[b_idx]))
        qb2, qb3 = unpack_p4_pair(int(bytes_b[b_idx + 1]))
        b00 = float(qb0)
        b01 = float(qb1)
        b10 = float(qb2)
        b11 = float(qb3)
    elif b_prec == 8:
        b00 = float(np.int8(bytes_b[b_idx]))
        b01 = float(np.int8(bytes_b[b_idx + 1]))
        b10 = float(np.int8(bytes_b[b_idx + 2]))
        b11 = float(np.int8(bytes_b[b_idx + 3]))
    else:
        b00 = 0.0
        b01 = 0.0
        b10 = 0.0
        b11 = 0.0

    scale_ab = scale_a * scale_b
    mtmul_2x2_fp32(a00, a01, a10, a11, b00, b01, b10, b11, scale_ab, accum_fp32, c_r, c_c)
