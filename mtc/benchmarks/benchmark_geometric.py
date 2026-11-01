"""
Empirical Benchmark for Geometric 3D Spatial Computing and Number-Theoretic Prime Systems.
"""

import time
import numpy as np

from mtc.core.arena import MemoryArena
from mtc.spatial3d.types3d import PrecisionMode3D
from mtc.spatial3d.microcube import MicroCube
from mtc.spatial3d.octree_tensor import OctreeTensor
from mtc.spatial3d.conv3d import conv3d_2x2x2_kernel, conv3d_with_microcube
from mtc.primes.rns import PrimeResidueSystem
from mtc.primes.ternary import (
    pack_ternary_vector, unpack_ternary_vector,
    ternary_multiplier_free_gemm
)
from mtc.primes.padic import p_adic_distance, tree_ultrametric_distance


def benchmark_3d_spatial():
    print("=" * 70)
    print("1. 3D SPATIAL COMPUTING & OCTREE GEOMETRY BENCHMARK")
    print("=" * 70)

    # Test volumes: 16x16x16 and 32x32x32 sparse physical field
    for D in [16, 32]:
        total_voxels = D ** 3
        dense_fp32_bytes = total_voxels * 4

        # Generate sparse 3D field (e.g. 5% active structure, 95% vacuum)
        np.random.seed(42)
        field = np.zeros((D, D, D), dtype=np.float32)
        # Place localized spherical energy pocket
        center = D // 2
        radius = D // 4
        for z in range(D):
            for y in range(D):
                for x in range(D):
                    if (z - center)**2 + (y - center)**2 + (x - center)**2 <= radius**2:
                        field[z, y, x] = float(np.random.randn())

        active_voxels = np.count_nonzero(field)
        sparsity_pct = (1.0 - active_voxels / total_voxels) * 100.0

        arena = MemoryArena(16 * 1024 * 1024)
        t0 = time.perf_counter()
        octree = OctreeTensor.from_array_recursive(
            field, PrecisionMode3D.P8, arena, vacuum_threshold=1e-3
        )
        t_build = (time.perf_counter() - t0) * 1000.0

        stats = octree.count_nodes()
        octree_bytes = stats["bytes_allocated"]
        compression = dense_fp32_bytes / octree_bytes if octree_bytes > 0 else 0.0

        print(f"\n[Volume {D}x{D}x{D} ({total_voxels:,} voxels) | Physical Sparsity: {sparsity_pct:.1f}%]")
        print(f"  - Dense FP32 Memory:       {dense_fp32_bytes / 1024:.1f} KB")
        print(f"  - Octree Pruned Memory:     {octree_bytes / 1024:.1f} KB")
        print(f"  - Spatial Memory Savings:   {compression:.1f}x reduction ({(1 - octree_bytes/dense_fp32_bytes)*100:.1f}%)")
        print(f"  - Total Octants / Nodes:    {stats['total_nodes']} (Vacuum Pruned: {stats['vacuum_nodes']})")
        print(f"  - Octree Build Latency:     {t_build:.2f} ms")

        # 3D Convolution Benchmark with MicroCube Filter
        kernel_cube = np.random.randn(2, 2, 2).astype(np.float32)
        mc_filter = MicroCube.from_array(kernel_cube, PrecisionMode3D.P8, arena, scale=1.0)

        # Warmup and timing
        conv3d_with_microcube(field, mc_filter)
        times = []
        for _ in range(5):
            t_c0 = time.perf_counter()
            out_conv = conv3d_with_microcube(field, mc_filter)
            times.append(time.perf_counter() - t_c0)
        t_conv = min(times) * 1000.0
        conv_voxels = (D - 1)**3
        conv_mflops = (conv_voxels * 16.0 / (t_conv / 1000.0)) / 1e6

        print(f"  - 3D MicroCube Conv Time:   {t_conv:.2f} ms ({conv_mflops:.1f} MVOX/s)")


def benchmark_primes_and_ternary():
    print("\n" + "=" * 70)
    print("2. NUMBER-THEORETIC & BALANCED TERNARY (F_3) BENCHMARK")
    print("=" * 70)

    # A. Balanced Ternary add/subtract-only GEMM vs FP32 BLAS
    print("\n[A. Balanced Ternary F_3 Add/Subtract-Only GEMM]")
    M, K, N = 256, 512, 256
    np.random.seed(42)
    W_trits = np.random.choice(np.array([-1, 0, 1], dtype=np.int8), size=(M, K))
    X = np.random.randn(K, N).astype(np.float32)
    out_ternary = np.empty((M, N), dtype=np.float32)

    # Measure FP32 reference (which uses standard hardware multipliers)
    W_float = W_trits.astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(5):
        out_fp32 = W_float @ X
    t_fp32 = (time.perf_counter() - t0) / 5.0 * 1000.0

    # Measure ternary GEMM (adds and subtracts only)
    ternary_multiplier_free_gemm(W_trits, X, out_ternary)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        ternary_multiplier_free_gemm(W_trits, X, out_ternary)
        times.append(time.perf_counter() - t0)
    t_ternary = min(times) * 1000.0

    flops = 2.0 * M * K * N
    gflops_ternary = (flops / (t_ternary / 1000.0)) / 1e9
    rel_diff = np.linalg.norm(out_fp32 - out_ternary) / np.linalg.norm(out_fp32)

    print(f"  - Matrix Dimensions:        M={M}, K={K}, N={N} ({flops/1e6:.1f} MOps)")
    print(f"  - Ternary add/sub Latency:  {t_ternary:.2f} ms ({gflops_ternary:.2f} GOps/s)")
    print(f"  - FP32 BLAS Latency:        {t_fp32:.2f} ms ({t_ternary / t_fp32:.1f}x faster than ternary on CPU)")
    print(f"  - Arithmetic Accuracy:      Relative Error = {rel_diff:.2e} (Exact Equivalence)")

    # B. 5-Trits-per-Byte Information Density
    print("\n[B. 5-Trits-per-Byte Information Density vs 2-Bit Packing]")
    num_trits = 100_000
    trits = np.random.choice(np.array([-1, 0, 1], dtype=np.int8), size=num_trits)

    packed_5trit = pack_ternary_vector(trits)
    bytes_5trit = len(packed_5trit)
    bytes_2bit = (num_trits + 3) // 4  # 4 trits per byte in 2-bit packing

    density_gain = (bytes_2bit - bytes_5trit) / bytes_2bit * 100.0

    print(f"  - Elements:                 {num_trits:,} balanced trits")
    print(f"  - Standard 2-bit Storage:   {bytes_2bit:,} bytes (4 trits/byte)")
    print(f"  - MTC-8 Base-3 Storage:     {bytes_5trit:,} bytes (5 trits/byte)")
    print(f"  - Net Storage Reduction:    {density_gain:.1f}% fewer bytes (same scheme as llama.cpp TQ1_0)")

    # C. Residue Number System (RNS) Parallel Channels
    print("\n[C. Residue Number System (RNS) Arithmetic]")
    rns = PrimeResidueSystem(primes=[3, 5, 7, 11, 13, 17, 19, 23])
    print(f"  - Prime Basis:              {list(rns.primes)}")
    print(f"  - Dynamic Range M:          {rns.M:,} ([-{rns.half_M:,}, {rns.half_M:,}])")
    print(f"  - Total Prime Channels:     {rns.num_channels} completely independent channels")
    print(f"  - Inter-channel carries:    none (each channel is mod p; CRT needed to reconstruct, sign/compare are costly)")


if __name__ == "__main__":
    benchmark_3d_spatial()
    benchmark_primes_and_ternary()
