"""
MTC-8 High-Performance GEMM Engine with Adaptive Precision and Pointer Tensors.
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np
from numba import njit, prange

from mtc.core.types import PrecisionMode, OpCode
from mtc.core.arena import MemoryArena
from mtc.core.microtensor import MicroTensor, BlockDescriptor
from mtc.core.pointer_tensor import PointerTensor
from mtc.core.packing import (
    pack_p2_quad, unpack_p2_quad,
    pack_p4_pair, unpack_p4_pair
)
from mtc.core.precision import (
    select_independent_precision,
    negotiate_joint_precision,
    quantize_and_dequantize_block
)


# =========================================================================
# Fast Block-Level Unpackers
# =========================================================================

@njit(fastmath=True, inline="always")
def unpack_block_p2(raw_arena: np.ndarray, offset: int, Bs: int, out_buf: np.ndarray):
    """Unpack Bs x Bs elements from P2 packed bytes into out_buf (float32)."""
    num_elements = Bs * Bs
    num_bytes = num_elements // 4
    for b_idx in range(num_bytes):
        byte_val = int(raw_arena[offset + b_idx])
        q0, q1, q2, q3 = unpack_p2_quad(byte_val)
        elem_idx = b_idx * 4

        r0 = elem_idx // Bs
        c0 = elem_idx % Bs
        out_buf[r0, c0] = float(q0)

        elem_idx += 1
        r1 = elem_idx // Bs
        c1 = elem_idx % Bs
        out_buf[r1, c1] = float(q1)

        elem_idx += 1
        r2 = elem_idx // Bs
        c2 = elem_idx % Bs
        out_buf[r2, c2] = float(q2)

        elem_idx += 1
        r3 = elem_idx // Bs
        c3 = elem_idx % Bs
        out_buf[r3, c3] = float(q3)


@njit(fastmath=True, inline="always")
def unpack_block_p4(raw_arena: np.ndarray, offset: int, Bs: int, out_buf: np.ndarray):
    """Unpack Bs x Bs elements from P4 packed bytes into out_buf (float32)."""
    num_elements = Bs * Bs
    num_bytes = num_elements // 2
    for b_idx in range(num_bytes):
        byte_val = int(raw_arena[offset + b_idx])
        q0, q1 = unpack_p4_pair(byte_val)
        elem_idx = b_idx * 2

        r0 = elem_idx // Bs
        c0 = elem_idx % Bs
        out_buf[r0, c0] = float(q0)

        elem_idx += 1
        r1 = elem_idx // Bs
        c1 = elem_idx % Bs
        out_buf[r1, c1] = float(q1)


@njit(fastmath=True, inline="always")
def unpack_block_p8(raw_arena: np.ndarray, offset: int, Bs: int, out_buf: np.ndarray):
    """Unpack Bs x Bs elements from P8 bytes into out_buf (float32)."""
    for r in range(Bs):
        for c in range(Bs):
            out_buf[r, c] = float(np.int8(raw_arena[offset + r * Bs + c]))


@njit(fastmath=True, inline="always")
def unpack_block_p16(raw_arena: np.ndarray, offset: int, Bs: int, out_buf: np.ndarray):
    """Unpack Bs x Bs elements from P16 (int16, 2 bytes each) into out_buf (float32)."""
    for r in range(Bs):
        for c in range(Bs):
            idx = offset + (r * Bs + c) * 2
            lo = np.uint16(raw_arena[idx])
            hi = np.uint16(raw_arena[idx + 1])
            u = np.uint16((hi << 8) | lo)
            out_buf[r, c] = float(np.int16(u))


@njit(fastmath=True, inline="always")
def unpack_block_bits(raw_arena: np.ndarray, offset: int, Bs: int, bits: int, out_buf: np.ndarray):
    """
    Unpack Bs x Bs signed `bits`-wide codes (3, 5, 6) from a little-endian
    bitstream: element i occupies bits [i*bits, (i+1)*bits) of the block.
    """
    mask = (1 << bits) - 1
    sign = 1 << (bits - 1)
    for i in range(Bs * Bs):
        bitpos = i * bits
        byte = offset + (bitpos >> 3)
        shift = bitpos & 7
        word = int(raw_arena[byte])
        if shift + bits > 8:
            word |= int(raw_arena[byte + 1]) << 8
        v = (word >> shift) & mask
        if v >= sign:
            v -= 1 << bits
        out_buf[i // Bs, i % Bs] = float(v)


# =========================================================================
# Macrotiled Core GEMM Kernel
# =========================================================================

@njit(parallel=True, fastmath=True)
def mtc_gemm_core_kernel(
    arena_a: np.ndarray,
    offsets_a: np.ndarray,
    precisions_a: np.ndarray,
    scales_a: np.ndarray,
    arena_b: np.ndarray,
    offsets_b: np.ndarray,
    precisions_b: np.ndarray,
    scales_b: np.ndarray,
    num_blocks_m: int,
    num_blocks_k: int,
    num_blocks_n: int,
    block_size: int,
    out_c: np.ndarray
):
    """
    Parallel macrotiled GEMM kernel executing over MTC arena representations.
    Iterates over (bi, bj) macro-blocks in parallel, accumulating products
    over reduction dimension bk in INT32 / FP32.
    """
    Bs = block_size

    for bi in prange(num_blocks_m):
        # Thread-local block buffers
        a_buf = np.empty((Bs, Bs), dtype=np.float32)
        b_buf = np.empty((Bs, Bs), dtype=np.float32)
        c_acc = np.zeros((Bs, Bs), dtype=np.float32)

        row_start = bi * Bs
        for bj in range(num_blocks_n):
            col_start = bj * Bs

            # Clear block accumulator
            for r in range(Bs):
                for c in range(Bs):
                    c_acc[r, c] = 0.0

            for bk in range(num_blocks_k):
                p_a = precisions_a[bi, bk]
                off_a = offsets_a[bi, bk]
                sc_a = scales_a[bi, bk]

                p_b = precisions_b[bk, bj]
                off_b = offsets_b[bk, bj]
                sc_b = scales_b[bk, bj]

                # P0 blocks are exactly zero: skip the whole block product
                if p_a == 0 or p_b == 0:
                    continue

                # Unpack Block A
                if p_a == 2:
                    unpack_block_p2(arena_a, off_a, Bs, a_buf)
                elif p_a == 4:
                    unpack_block_p4(arena_a, off_a, Bs, a_buf)
                elif p_a == 8:
                    unpack_block_p8(arena_a, off_a, Bs, a_buf)
                elif p_a == 16:
                    unpack_block_p16(arena_a, off_a, Bs, a_buf)
                else:
                    unpack_block_bits(arena_a, off_a, Bs, p_a, a_buf)

                # Unpack Block B
                if p_b == 2:
                    unpack_block_p2(arena_b, off_b, Bs, b_buf)
                elif p_b == 4:
                    unpack_block_p4(arena_b, off_b, Bs, b_buf)
                elif p_b == 8:
                    unpack_block_p8(arena_b, off_b, Bs, b_buf)
                elif p_b == 16:
                    unpack_block_p16(arena_b, off_b, Bs, b_buf)
                else:
                    unpack_block_bits(arena_b, off_b, Bs, p_b, b_buf)

                # Multiply-accumulate unpacked blocks into c_acc
                scale_prod = sc_a * sc_b
                for r in range(Bs):
                    for k in range(Bs):
                        val_a = a_buf[r, k] * scale_prod
                        for c in range(Bs):
                            c_acc[r, c] += val_a * b_buf[k, c]

            # Commit c_acc into out_c
            for r in range(Bs):
                for c in range(Bs):
                    out_c[row_start + r, col_start + c] = c_acc[r, c]


# =========================================================================
# MTC Matrix Structure
# =========================================================================

class MTCMatrix:
    """
    Representation of an (H, W) matrix composed of MTC-8 blocks stored in an arena.
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        block_size: int,
        arena: MemoryArena,
        offsets: np.ndarray,
        precisions: np.ndarray,
        scales: np.ndarray,
        pointer_tensor: Optional[PointerTensor] = None
    ):
        self.shape = shape
        self.block_size = block_size
        self.arena = arena
        self.offsets = offsets
        self.precisions = precisions
        self.scales = scales
        self.pointer_tensor = pointer_tensor

    @property
    def num_blocks(self) -> Tuple[int, int]:
        return self.offsets.shape

    @property
    def total_bytes_allocated(self) -> int:
        return self.arena.bytes_allocated

    @property
    def average_bits_per_element(self) -> float:
        total_elements = self.shape[0] * self.shape[1]
        if total_elements == 0:
            return 0.0
        return (self.total_bytes_allocated * 8.0) / total_elements

    @classmethod
    def from_numpy(
        cls,
        matrix: np.ndarray,
        block_size: int = 16,
        precision_mode: Optional[PrecisionMode] = None,
        epsilon: float = 0.05,
        arena: Optional[MemoryArena] = None,
        use_pointer_hierarchy: bool = True
    ) -> "MTCMatrix":
        """
        Construct an MTCMatrix from a 2D float NumPy array with adaptive or fixed precision.
        """
        arr = np.asarray(matrix, dtype=np.float32)
        H, W = arr.shape
        Bs = block_size
        assert H % Bs == 0 and W % Bs == 0, f"Dimensions ({H}, {W}) must be divisible by block_size {Bs}"

        num_bh = H // Bs
        num_bw = W // Bs

        if arena is None:
            arena = MemoryArena(capacity_bytes=max(1024 * 1024, H * W * 2))

        offsets = np.zeros((num_bh, num_bw), dtype=np.uint32)
        precisions = np.zeros((num_bh, num_bw), dtype=np.uint8)
        scales = np.zeros((num_bh, num_bw), dtype=np.float32)

        pointer_grid = [] if use_pointer_hierarchy else None

        for bi in range(num_bh):
            row_grid = [] if use_pointer_hierarchy else None
            for bj in range(num_bw):
                block = arr[bi * Bs : (bi + 1) * Bs, bj * Bs : (bj + 1) * Bs]

                # Precision selection
                if precision_mode is not None:
                    p = precision_mode
                    max_abs = float(np.max(np.abs(block)))
                    if p == PrecisionMode.P2:
                        sc = max_abs / 1.0 if max_abs > 1e-7 else 1.0
                    elif p == PrecisionMode.P4:
                        sc = max_abs / 7.0 if max_abs > 1e-7 else 1.0
                    elif p == PrecisionMode.P8:
                        sc = max_abs / 127.0 if max_abs > 1e-7 else 1.0
                    elif p == PrecisionMode.P16:
                        sc = max_abs / 32767.0 if max_abs > 1e-7 else 1.0
                    else:
                        sc = 1.0
                else:
                    p, sc = select_independent_precision(block, epsilon)

                mt = MicroTensor.from_array(block, p, arena, scale=sc)
                offsets[bi, bj] = mt.descriptor.arena_offset
                precisions[bi, bj] = p.value
                scales[bi, bj] = sc

                if use_pointer_hierarchy:
                    row_grid.append(PointerTensor.from_microtensor(mt))

            if use_pointer_hierarchy:
                pointer_grid.append(row_grid)

        pt = PointerTensor.from_grid(pointer_grid) if use_pointer_hierarchy else None

        return cls(
            shape=(H, W),
            block_size=Bs,
            arena=arena,
            offsets=offsets,
            precisions=precisions,
            scales=scales,
            pointer_tensor=pt
        )

    @classmethod
    def from_numpy_joint(
        cls,
        matrix_a: np.ndarray,
        matrix_b: np.ndarray,
        block_size: int = 16,
        target_epsilon: float = 0.05,
        use_pointer_hierarchy: bool = True
    ) -> Tuple["MTCMatrix", "MTCMatrix"]:
        """
        Construct both A and B jointly by negotiating precision between interacting blocks!
        """
        A = np.asarray(matrix_a, dtype=np.float32)
        B = np.asarray(matrix_b, dtype=np.float32)
        M, K = A.shape
        K2, N = B.shape
        assert K == K2
        Bs = block_size

        num_bm = M // Bs
        num_bk = K // Bs
        num_bn = N // Bs

        arena_a = MemoryArena(capacity_bytes=max(1024 * 1024, M * K * 2))
        arena_b = MemoryArena(capacity_bytes=max(1024 * 1024, K * N * 2))

        # Precision maps initialized to P2
        prec_a = np.full((num_bm, num_bk), PrecisionMode.P2.value, dtype=np.uint8)
        scale_a = np.zeros((num_bm, num_bk), dtype=np.float32)
        offsets_a = np.zeros((num_bm, num_bk), dtype=np.uint32)

        prec_b = np.full((num_bk, num_bn), PrecisionMode.P2.value, dtype=np.uint8)
        scale_b = np.zeros((num_bk, num_bn), dtype=np.float32)
        offsets_b = np.zeros((num_bk, num_bn), dtype=np.uint32)

        # Negotiate precision across block interactions
        for bk in range(num_bk):
            for bi in range(num_bm):
                block_a = A[bi * Bs : (bi + 1) * Bs, bk * Bs : (bk + 1) * Bs]
                for bj in range(num_bn):
                    block_b = B[bk * Bs : (bk + 1) * Bs, bj * Bs : (bj + 1) * Bs]
                    pa, pb, _, _ = negotiate_joint_precision(
                        block_a, block_b, target_epsilon=target_epsilon, reduction_tiles=num_bk
                    )
                    if pa.value > prec_a[bi, bk]:
                        prec_a[bi, bk] = pa.value
                    if pb.value > prec_b[bk, bj]:
                        prec_b[bk, bj] = pb.value

        # Pack A according to negotiated precisions with optimal symmetric block scales
        pointer_grid_a = [] if use_pointer_hierarchy else None
        for bi in range(num_bm):
            row_grid = [] if use_pointer_hierarchy else None
            for bk in range(num_bk):
                block = A[bi * Bs : (bi + 1) * Bs, bk * Bs : (bk + 1) * Bs]
                p = PrecisionMode(prec_a[bi, bk])
                mt = MicroTensor.from_array(block, p, arena_a)
                offsets_a[bi, bk] = mt.descriptor.arena_offset
                scale_a[bi, bk] = mt.descriptor.scale
                if use_pointer_hierarchy:
                    row_grid.append(PointerTensor.from_microtensor(mt))
            if use_pointer_hierarchy:
                pointer_grid_a.append(row_grid)

        # Pack B according to negotiated precisions with optimal symmetric block scales
        pointer_grid_b = [] if use_pointer_hierarchy else None
        for bk in range(num_bk):
            row_grid = [] if use_pointer_hierarchy else None
            for bj in range(num_bn):
                block = B[bk * Bs : (bk + 1) * Bs, bj * Bs : (bj + 1) * Bs]
                p = PrecisionMode(prec_b[bk, bj])
                mt = MicroTensor.from_array(block, p, arena_b)
                offsets_b[bk, bj] = mt.descriptor.arena_offset
                scale_b[bk, bj] = mt.descriptor.scale
                if use_pointer_hierarchy:
                    row_grid.append(PointerTensor.from_microtensor(mt))
            if use_pointer_hierarchy:
                pointer_grid_b.append(row_grid)

        pt_a = PointerTensor.from_grid(pointer_grid_a) if use_pointer_hierarchy else None
        pt_b = PointerTensor.from_grid(pointer_grid_b) if use_pointer_hierarchy else None

        mtc_a = cls(
            shape=(M, K),
            block_size=Bs,
            arena=arena_a,
            offsets=offsets_a,
            precisions=prec_a,
            scales=scale_a,
            pointer_tensor=pt_a
        )
        mtc_b = cls(
            shape=(K, N),
            block_size=Bs,
            arena=arena_b,
            offsets=offsets_b,
            precisions=prec_b,
            scales=scale_b,
            pointer_tensor=pt_b
        )
        return mtc_a, mtc_b

    @classmethod
    def from_allocation(cls, matrix: np.ndarray, modes: np.ndarray, scales: np.ndarray, block_size: int) -> "MTCMatrix":
        """
        Pack a matrix into the arena given per-block modes and scales
        (e.g. from mtc.core.allocation.rd_allocate). Vectorized; blocks are laid
        out in row-major block order, P0 blocks occupy zero bytes.
        """
        from mtc.core.allocation import to_blocks, quantize_blocks

        arr = np.asarray(matrix, dtype=np.float32)
        H, W = arr.shape
        Bs = block_size
        blocks = to_blocks(arr, Bs)
        nbh, nbw = blocks.shape[:2]
        modes = np.asarray(modes, dtype=np.uint8).reshape(nbh, nbw)
        scales = np.asarray(scales, dtype=np.float32).reshape(nbh, nbw)

        sizes = (modes.astype(np.int64) * Bs * Bs) // 8
        flat_offsets = np.concatenate([[0], np.cumsum(sizes.ravel())[:-1]])
        total = int(sizes.sum())
        arena = MemoryArena(capacity_bytes=max(total, 1))
        base = arena.allocate(total, alignment=2)
        offsets = (flat_offsets + base).astype(np.uint32).reshape(nbh, nbw)
        buf = arena.raw_buffer

        for m in (2, 3, 4, 5, 6, 8, 16):
            sel = modes == m
            if not sel.any():
                continue
            q = quantize_blocks(blocks[sel][None], m, scales[sel][None])[0]
            q = q.reshape(q.shape[0], Bs * Bs).astype(np.int32)
            if m == 2:
                u = (q & 0x03).reshape(q.shape[0], -1, 4)
                packed = (u[..., 0] | (u[..., 1] << 2) | (u[..., 2] << 4) | (u[..., 3] << 6)).astype(np.uint8)
            elif m == 4:
                u = (q & 0x0F).reshape(q.shape[0], -1, 2)
                packed = (u[..., 0] | (u[..., 1] << 4)).astype(np.uint8)
            elif m == 8:
                packed = q.astype(np.int8).view(np.uint8)
            elif m == 16:
                packed = q.astype("<i2").view(np.uint8)
            else:
                u = (q & ((1 << m) - 1)).astype(np.uint32)
                bitplanes = ((u[..., None] >> np.arange(m, dtype=np.uint32)) & 1).astype(np.uint8)
                packed = np.packbits(bitplanes.reshape(q.shape[0], -1), axis=1, bitorder="little")
            dest = offsets[sel].astype(np.int64)[:, None] + np.arange(packed.shape[1])
            buf[dest] = packed

        return cls(
            shape=(H, W),
            block_size=Bs,
            arena=arena,
            offsets=offsets,
            precisions=modes.copy(),
            scales=np.where(modes == 0, 0.0, scales).astype(np.float32),
            pointer_tensor=None,
        )

    @classmethod
    def from_numpy_rd_pair(
        cls,
        matrix_a: np.ndarray,
        matrix_b: np.ndarray,
        bits_per_element: float,
        block_size: int = 16,
    ) -> Tuple["MTCMatrix", "MTCMatrix"]:
        """
        Joint rate-distortion allocation (MTC-RD) for C = A @ B at a fixed
        bit budget (payload + metadata), packed into real arena storage.
        """
        from mtc.core.allocation import rd_allocate_pair

        alloc_a, alloc_b = rd_allocate_pair(matrix_a, matrix_b, bits_per_element, block_size)
        return (
            cls.from_allocation(matrix_a, alloc_a.modes, alloc_a.scales, block_size),
            cls.from_allocation(matrix_b, alloc_b.modes, alloc_b.scales, block_size),
        )

    def to_array(self) -> np.ndarray:
        """Reconstruct full float32 matrix from blocks in arena."""
        H, W = self.shape
        Bs = self.block_size
        num_bh, num_bw = self.offsets.shape
        out = np.empty((H, W), dtype=np.float32)

        for bi in range(num_bh):
            for bj in range(num_bw):
                p = self.precisions[bi, bj]
                off = self.offsets[bi, bj]
                sc = self.scales[bi, bj]
                temp = np.zeros((Bs, Bs), dtype=np.float32)

                if p == 0:
                    pass
                elif p == 2:
                    unpack_block_p2(self.arena.raw_buffer, off, Bs, temp)
                elif p == 4:
                    unpack_block_p4(self.arena.raw_buffer, off, Bs, temp)
                elif p == 8:
                    unpack_block_p8(self.arena.raw_buffer, off, Bs, temp)
                elif p == 16:
                    unpack_block_p16(self.arena.raw_buffer, off, Bs, temp)
                else:
                    unpack_block_bits(self.arena.raw_buffer, off, Bs, int(p), temp)

                out[bi * Bs : (bi + 1) * Bs, bj * Bs : (bj + 1) * Bs] = temp * sc
        return out


def mtc_gemm(
    A: MTCMatrix,
    B: MTCMatrix
) -> np.ndarray:
    """
    Execute high-performance GEMM between two MTCMatrix instances.
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Shape mismatch: {A.shape} vs {B.shape}"
    assert A.block_size == B.block_size, "Block size mismatch"
    Bs = A.block_size

    num_bm, num_bk = A.num_blocks
    _, num_bn = B.num_blocks

    out_c = np.empty((M, N), dtype=np.float32)

    mtc_gemm_core_kernel(
        A.arena.raw_buffer,
        A.offsets,
        A.precisions,
        A.scales,
        B.arena.raw_buffer,
        B.offsets,
        B.precisions,
        B.scales,
        num_bm,
        num_bk,
        num_bn,
        Bs,
        out_c
    )
    return out_c
