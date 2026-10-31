"""
3D Convolution and Trilinear Contraction Primitives for MicroCubes.
Compiled with Numba.
"""

from typing import Tuple
import numpy as np
from numba import njit, prange

from .microcube import MicroCube
from .types3d import PrecisionMode3D
from .packing3d import (
    unpack_cube_2x2x2_p1_spin,
    unpack_cube_2x2x2_p2,
    unpack_cube_2x2x2_p4,
    unpack_cube_2x2x2_p8
)


@njit(parallel=True, fastmath=True)
def conv3d_2x2x2_kernel(
    input_vol: np.ndarray,
    kernel_cube: np.ndarray,
    out: np.ndarray
):
    """
    3D spatial convolution of a (D, H, W) volume with a (2, 2, 2) kernel.
    Output shape is (D - 1, H - 1, W - 1).
    """
    D, H, W = input_vol.shape
    out_d = D - 1
    out_h = H - 1
    out_w = W - 1

    for z in prange(out_d):
        for y in range(out_h):
            for x in range(out_w):
                acc = 0.0
                for dz in range(2):
                    for dy in range(2):
                        for dx in range(2):
                            acc += input_vol[z + dz, y + dy, x + dx] * kernel_cube[dz, dy, dx]
                out[z, y, x] = acc


@njit(parallel=True, fastmath=True)
def trilinear_contraction_3d(
    A: np.ndarray,
    B: np.ndarray,
    out: np.ndarray
):
    """
    3D tensor contraction contracting along shared dimension k:
    C[d, h, w] = sum_k A[d, h, k] * B[k, h, w]
    """
    D, H, K = A.shape
    K2, H2, W = B.shape
    assert K == K2 and H == H2

    for d in prange(D):
        for h in range(H):
            for w in range(W):
                acc = 0.0
                for k in range(K):
                    acc += A[d, h, k] * B[k, h, w]
                out[d, h, w] = acc


def conv3d_with_microcube(
    input_volume: np.ndarray,
    microcube_filter: MicroCube
) -> np.ndarray:
    """
    Execute 3D convolution with an unpacked MicroCube filter.
    """
    kernel = microcube_filter.unpack_to_array()
    D, H, W = input_volume.shape
    out = np.empty((D - 1, H - 1, W - 1), dtype=np.float32)
    conv3d_2x2x2_kernel(input_volume.astype(np.float32), kernel, out)
    return out
