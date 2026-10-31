"""
Unified Benchmark Suite for MTC-8 vs FP32, INT8, and INT4 Baselines.
"""

from typing import Dict, Any, List, Tuple
import time
import gc
import numpy as np

from mtc.core.types import PrecisionMode
from mtc.kernels.reference import (
    fp32_gemm_numpy,
    relative_frobenius_error,
    signal_to_noise_ratio_db
)
from mtc.baselines.int8_gemm import INT8GEMM
from mtc.baselines.int4_gemm import INT4GEMM
from mtc.kernels.gemm import MTCMatrix, mtc_gemm


def generate_matrix(dist_type: str, N: int, seed: int = 42) -> np.ndarray:
    """Generate square (N, N) test matrix according to distribution type."""
    rng = np.random.RandomState(seed)
    if dist_type == "gaussian":
        return rng.randn(N, N).astype(np.float32)

    elif dist_type == "uniform":
        return rng.uniform(-1.0, 1.0, size=(N, N)).astype(np.float32)

    elif dist_type == "outliers":
        # 98% small values, 2% high-magnitude outliers
        base = rng.randn(N, N).astype(np.float32) * 0.05
        mask = rng.rand(N, N) < 0.02
        outliers = rng.uniform(-25.0, 25.0, size=(N, N)).astype(np.float32)
        base[mask] = outliers[mask]
        return base

    elif dist_type == "multi_scale":
        # Blockwise varying scales across 16x16 blocks
        base = rng.randn(N, N).astype(np.float32)
        Bs = 16
        num_b = N // Bs
        for bi in range(num_b):
            for bj in range(num_b):
                # Scale between 0.001 and 100.0
                scale_exp = rng.uniform(-3.0, 2.0)
                base[bi * Bs : (bi + 1) * Bs, bj * Bs : (bj + 1) * Bs] *= (10.0 ** scale_exp)
        return base

    else:
        raise ValueError(f"Unknown distribution: {dist_type}")


def time_callable(fn, warmup: int = 1, repeats: int = 3) -> float:
    """Warmup and measure median execution time in seconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(np.median(times))


def benchmark_fp32(A: np.ndarray, B: np.ndarray) -> Dict[str, Any]:
    N = A.shape[0]
    flops = 2.0 * (N ** 3)
    t_kernel = time_callable(lambda: fp32_gemm_numpy(A, B))
    gflops = (flops / t_kernel) / 1e9

    return {
        "name": "FP32 (BLAS)",
        "kernel_ms": t_kernel * 1000.0,
        "e2e_ms": t_kernel * 1000.0,
        "packing_ms": 0.0,
        "gflops": gflops,
        "e2e_gflops": gflops,
        "bytes": A.nbytes + B.nbytes,
        "compression": 1.0,
        "avg_bits": 32.0,
        "rel_err": 0.0,
        "snr_db": 120.0
    }


def benchmark_int8(A: np.ndarray, B: np.ndarray, C_ref: np.ndarray) -> Dict[str, Any]:
    N = A.shape[0]
    flops = 2.0 * (N ** 3)

    # Measure packing
    t_pack = time_callable(lambda: (INT8GEMM.quantize(A), INT8GEMM.quantize(B)))
    A_q, sc_a = INT8GEMM.quantize(A)
    B_q, sc_b = INT8GEMM.quantize(B)

    # Warmup kernel
    INT8GEMM.run_kernel(A_q, B_q, sc_a, sc_b)
    t_kernel = time_callable(lambda: INT8GEMM.run_kernel(A_q, B_q, sc_a, sc_b))

    t_e2e = t_pack + t_kernel
    C_pred = INT8GEMM.run_kernel(A_q, B_q, sc_a, sc_b)

    rel_err = relative_frobenius_error(C_ref, C_pred)
    snr = signal_to_noise_ratio_db(C_ref, C_pred)

    total_bytes = A_q.nbytes + B_q.nbytes
    fp32_bytes = A.nbytes + B.nbytes
    comp = fp32_bytes / total_bytes

    return {
        "name": "INT8 GEMM",
        "kernel_ms": t_kernel * 1000.0,
        "e2e_ms": t_e2e * 1000.0,
        "packing_ms": t_pack * 1000.0,
        "gflops": (flops / t_kernel) / 1e9,
        "e2e_gflops": (flops / t_e2e) / 1e9,
        "bytes": total_bytes,
        "compression": comp,
        "avg_bits": 8.0,
        "rel_err": rel_err,
        "snr_db": snr
    }


def benchmark_int4(A: np.ndarray, B: np.ndarray, C_ref: np.ndarray) -> Dict[str, Any]:
    N = A.shape[0]
    flops = 2.0 * (N ** 3)

    t_pack = time_callable(lambda: (INT4GEMM.quantize_and_pack(A), INT4GEMM.quantize_and_pack(B)))
    A_p, sc_a = INT4GEMM.quantize_and_pack(A)
    B_p, sc_b = INT4GEMM.quantize_and_pack(B)

    # Warmup kernel
    INT4GEMM.run_kernel(A_p, B_p, sc_a, sc_b, N, N, N)
    t_kernel = time_callable(lambda: INT4GEMM.run_kernel(A_p, B_p, sc_a, sc_b, N, N, N))

    t_e2e = t_pack + t_kernel
    C_pred = INT4GEMM.run_kernel(A_p, B_p, sc_a, sc_b, N, N, N)

    rel_err = relative_frobenius_error(C_ref, C_pred)
    snr = signal_to_noise_ratio_db(C_ref, C_pred)

    total_bytes = A_p.nbytes + B_p.nbytes
    fp32_bytes = A.nbytes + B.nbytes
    comp = fp32_bytes / total_bytes

    return {
        "name": "INT4 GEMM",
        "kernel_ms": t_kernel * 1000.0,
        "e2e_ms": t_e2e * 1000.0,
        "packing_ms": t_pack * 1000.0,
        "gflops": (flops / t_kernel) / 1e9,
        "e2e_gflops": (flops / t_e2e) / 1e9,
        "bytes": total_bytes,
        "compression": comp,
        "avg_bits": 4.0,
        "rel_err": rel_err,
        "snr_db": snr
    }


def benchmark_mtc(
    A: np.ndarray,
    B: np.ndarray,
    C_ref: np.ndarray,
    target_epsilon: float = 0.05,
    mode: str = "joint",
    use_pointers: bool = True
) -> Dict[str, Any]:
    N = A.shape[0]
    flops = 2.0 * (N ** 3)
    Bs = 16

    if mode == "joint":
        t_pack = time_callable(lambda: MTCMatrix.from_numpy_joint(
            A, B, block_size=Bs, target_epsilon=target_epsilon, use_pointer_hierarchy=use_pointers
        ))
        mtc_a, mtc_b = MTCMatrix.from_numpy_joint(
            A, B, block_size=Bs, target_epsilon=target_epsilon, use_pointer_hierarchy=use_pointers
        )
        tag = f"MTC Joint (eps={target_epsilon})"
    elif mode == "independent":
        t_pack = time_callable(lambda: (
            MTCMatrix.from_numpy(A, block_size=Bs, epsilon=target_epsilon, use_pointer_hierarchy=use_pointers),
            MTCMatrix.from_numpy(B, block_size=Bs, epsilon=target_epsilon, use_pointer_hierarchy=use_pointers)
        ))
        mtc_a = MTCMatrix.from_numpy(A, block_size=Bs, epsilon=target_epsilon, use_pointer_hierarchy=use_pointers)
        mtc_b = MTCMatrix.from_numpy(B, block_size=Bs, epsilon=target_epsilon, use_pointer_hierarchy=use_pointers)
        tag = f"MTC Indep (eps={target_epsilon})"
    else:
        raise ValueError(f"Unknown mode {mode}")

    if not use_pointers:
        tag += " [No Pointers]"

    # Warmup kernel
    mtc_gemm(mtc_a, mtc_b)
    t_kernel = time_callable(lambda: mtc_gemm(mtc_a, mtc_b))

    t_e2e = t_pack + t_kernel
    C_pred = mtc_gemm(mtc_a, mtc_b)

    rel_err = relative_frobenius_error(C_ref, C_pred)
    snr = signal_to_noise_ratio_db(C_ref, C_pred)

    total_bytes = mtc_a.total_bytes_allocated + mtc_b.total_bytes_allocated
    fp32_bytes = A.nbytes + B.nbytes
    comp = fp32_bytes / total_bytes if total_bytes > 0 else 1.0
    avg_bits = (total_bytes * 8.0) / (2 * N * N)

    return {
        "name": tag,
        "kernel_ms": t_kernel * 1000.0,
        "e2e_ms": t_e2e * 1000.0,
        "packing_ms": t_pack * 1000.0,
        "gflops": (flops / t_kernel) / 1e9,
        "e2e_gflops": (flops / t_e2e) / 1e9,
        "bytes": total_bytes,
        "compression": comp,
        "avg_bits": avg_bits,
        "rel_err": rel_err,
        "snr_db": snr
    }
