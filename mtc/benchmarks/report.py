"""
Comprehensive Benchmark Execution and Markdown Report Generator.
"""

from typing import List, Dict, Any
import sys
import numpy as np
from mtc.benchmarks.benchmark_runner import (
    generate_matrix,
    benchmark_fp32,
    benchmark_int8,
    benchmark_int4,
    benchmark_mtc
)
from mtc.kernels.reference import fp32_gemm_numpy


def run_benchmark_suite(
    sizes: List[int] = [128, 256],
    distributions: List[str] = ["gaussian", "uniform", "outliers", "multi_scale"]
) -> List[Dict[str, Any]]:
    all_results = []

    print("=" * 70)
    print("Starting MTC-8 Empirical Benchmark Suite")
    print(f"Matrix Sizes: {sizes}")
    print(f"Distributions: {distributions}")
    print("=" * 70)

    for dist in distributions:
        print(f"\n---> Running Distribution: {dist.upper()}")
        for N in sizes:
            print(f"  [Size N={N}x{N}] Generating matrices...")
            A = generate_matrix(dist, N, seed=100)
            B = generate_matrix(dist, N, seed=200)

            print("    Running FP32 Reference...")
            C_ref = fp32_gemm_numpy(A, B)
            res_fp32 = benchmark_fp32(A, B)
            res_fp32.update({"dist": dist, "N": N})
            all_results.append(res_fp32)

            print("    Running INT8 Baseline...")
            res_int8 = benchmark_int8(A, B, C_ref)
            res_int8.update({"dist": dist, "N": N})
            all_results.append(res_int8)

            print("    Running INT4 Baseline...")
            res_int4 = benchmark_int4(A, B, C_ref)
            res_int4.update({"dist": dist, "N": N})
            all_results.append(res_int4)

            print("    Running MTC Joint Negotiation (eps=0.05)...")
            res_mtc_joint = benchmark_mtc(A, B, C_ref, target_epsilon=0.05, mode="joint", use_pointers=True)
            res_mtc_joint.update({"dist": dist, "N": N})
            all_results.append(res_mtc_joint)

            print("    Running MTC Independent (eps=0.05)...")
            res_mtc_indep = benchmark_mtc(A, B, C_ref, target_epsilon=0.05, mode="independent", use_pointers=True)
            res_mtc_indep.update({"dist": dist, "N": N})
            all_results.append(res_mtc_indep)

            print("    Running MTC without Pointer Hierarchy...")
            res_mtc_noptr = benchmark_mtc(A, B, C_ref, target_epsilon=0.05, mode="joint", use_pointers=False)
            res_mtc_noptr.update({"dist": dist, "N": N})
            all_results.append(res_mtc_noptr)

    return all_results


def format_results_markdown(results: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("# MTC-8 Empirical Performance and Accuracy Evaluation\n")
    lines.append("| Dist | N | Architecture | Compression | Avg Bits | Rel Error | SNR (dB) | Kernel (ms) | Kernel GFLOPS | E2E (ms) | Pack (ms) |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for r in results:
        lines.append(
            f"| {r['dist']} | {r['N']} | **{r['name']}** | {r['compression']:.1f}x | "
            f"{r['avg_bits']:.1f}b | {r['rel_err']:.4f} | {r['snr_db']:.1f} dB | "
            f"{r['kernel_ms']:.2f} | {r['gflops']:.2f} | {r['e2e_ms']:.2f} | {r['packing_ms']:.2f} |"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    # By default run N=128, 256 for rapid thorough validation
    sizes = [128, 256]
    if len(sys.argv) > 1:
        sizes = [int(x) for x in sys.argv[1].split(",")]

    results = run_benchmark_suite(sizes=sizes)
    md_table = format_results_markdown(results)
    print("\n" + md_table)
