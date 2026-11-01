"""
End-to-End Neural Network Inference Benchmark: FP32 vs INT8 vs MTC-Adaptive vs Balanced Ternary.
"""

import time
import numpy as np

from mtc.nn.model import DeepMLP_FP32, DeepMLP_INT8, DeepMLP_MTCAdaptive, DeepMLP_Ternary


def benchmark_model_inference(batch_sizes=[1, 16, 64], repeats=10):
    print("=" * 75)
    print("  END-TO-END NEURAL NETWORK INFERENCE BENCHMARK")
    print("  Architecture: 4-Layer DeepMLP (256 -> 512 -> 512 -> 256 -> 16)")
    print("  Total Parameters: ~528,000 weights")
    print("=" * 75)

    # Initialize models
    m_fp32 = DeepMLP_FP32(256, 512, 16, seed=42)
    m_int8 = DeepMLP_INT8(256, 512, 16, seed=42)
    m_mtc = DeepMLP_MTCAdaptive(256, 512, 16, seed=42)
    m_ternary = DeepMLP_Ternary(256, 512, 16, seed=42)

    fp32_kb = m_fp32.total_bytes / 1024.0
    int8_kb = m_int8.total_bytes / 1024.0
    mtc_kb = m_mtc.total_bytes / 1024.0
    ternary_kb = m_ternary.total_bytes / 1024.0

    print(f"\nModel Parameter Storage Footprint:")
    print(f"  - FP32 Model:          {fp32_kb:.1f} KB (1.0x baseline, 32.0 bits/weight)")
    print(f"  - INT8 Model:          {int8_kb:.1f} KB ({fp32_kb / int8_kb:.1f}x compression, 8.0 bits/weight)")
    print(f"  - MTC-8 Adaptive:      {mtc_kb:.1f} KB ({fp32_kb / mtc_kb:.1f}x compression, adaptive bits)")
    print(f"  - Balanced Ternary:    {ternary_kb:.1f} KB ({fp32_kb / ternary_kb:.1f}x compression, 1.58 bits/weight, 5 trits/byte)")

    for B in batch_sizes:
        print(f"\n---> Evaluating Batch Size = {B}")
        np.random.seed(123)
        X = np.random.randn(B, 256).astype(np.float32)

        # Warmup and time FP32
        m_fp32.forward(X)
        times_fp32 = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            out_fp32, _ = m_fp32.forward(X)
            times_fp32.append(time.perf_counter() - t0)
        t_fp32 = min(times_fp32) * 1000.0

        # Warmup and time INT8
        m_int8.forward(X)
        times_int8 = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            out_int8 = m_int8.forward(X)
            times_int8.append(time.perf_counter() - t0)
        t_int8 = min(times_int8) * 1000.0

        # Warmup and time MTC Adaptive
        m_mtc.forward(X)
        times_mtc = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            out_mtc = m_mtc.forward(X)
            times_mtc.append(time.perf_counter() - t0)
        t_mtc = min(times_mtc) * 1000.0

        # Warmup and time Balanced Ternary (add/subtract-only kernel)
        m_ternary.forward(X)
        times_ternary = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            out_ternary, _ = m_ternary.forward(X)
            times_ternary.append(time.perf_counter() - t0)
        t_ternary = min(times_ternary) * 1000.0

        fps_fp32 = B / (t_fp32 / 1000.0)
        fps_mtc = B / (t_mtc / 1000.0)
        fps_ternary = B / (t_ternary / 1000.0)

        err_int8 = np.linalg.norm(out_fp32 - out_int8) / np.linalg.norm(out_fp32)
        err_mtc = np.linalg.norm(out_fp32 - out_mtc) / np.linalg.norm(out_fp32)

        print(f"  [Batch {B:2d}] Latency & Throughput:")
        print(f"    - FP32 Reference:     {t_fp32:.3f} ms | {fps_fp32:8.1f} samples/sec")
        print(f"    - INT8 Quantized:     {t_int8:.3f} ms | Err: {err_int8:.4f}")
        print(f"    - MTC-8 Adaptive:     {t_mtc:.3f} ms | {fps_mtc:8.1f} samples/sec | Err: {err_mtc:.4f} (BLAS on cached dequantized weights)")
        print(f"    - Balanced Ternary:   {t_ternary:.3f} ms | {fps_ternary:8.1f} samples/sec | add/sub only")


if __name__ == "__main__":
    benchmark_model_inference()
