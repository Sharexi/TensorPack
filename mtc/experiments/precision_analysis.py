"""
Experiment: Precision Distribution Analysis Across Tolerances and Data Regimes.
"""

from typing import Dict
import numpy as np

from mtc.core.types import PrecisionMode
from mtc.benchmarks.benchmark_runner import generate_matrix
from mtc.kernels.gemm import MTCMatrix


def analyze_precision_allocation(dist: str, N: int = 128, epsilons = [0.01, 0.05, 0.10, 0.20]) -> Dict:
    print(f"\n=======================================================")
    print(f"Precision Distribution Analysis: {dist.upper()} (N={N})")
    print(f"=======================================================")
    A = generate_matrix(dist, N, seed=42)
    B = generate_matrix(dist, N, seed=84)

    results = {}
    for eps in epsilons:
        mtc_a, mtc_b = MTCMatrix.from_numpy_joint(A, B, block_size=16, target_epsilon=eps)
        prec_counts = {2: 0, 4: 0, 8: 0, 16: 0}
        total_blocks = mtc_a.precisions.size + mtc_b.precisions.size

        for p in mtc_a.precisions.ravel():
            prec_counts[int(p)] += 1
        for p in mtc_b.precisions.ravel():
            prec_counts[int(p)] += 1

        p2_pct = (prec_counts[2] / total_blocks) * 100.0
        p4_pct = (prec_counts[4] / total_blocks) * 100.0
        p8_pct = (prec_counts[8] / total_blocks) * 100.0
        p16_pct = (prec_counts[16] / total_blocks) * 100.0
        avg_bits = (mtc_a.total_bytes_allocated + mtc_b.total_bytes_allocated) * 8.0 / (2 * N * N)

        results[eps] = {
            "p2_pct": p2_pct,
            "p4_pct": p4_pct,
            "p8_pct": p8_pct,
            "p16_pct": p16_pct,
            "avg_bits": avg_bits
        }

        print(f"Tolerance eps={eps:.2f} -> Avg Bits: {avg_bits:.2f}b | P2: {p2_pct:.1f}% | P4: {p4_pct:.1f}% | P8: {p8_pct:.1f}% | P16: {p16_pct:.1f}%")

    return results


if __name__ == "__main__":
    for d in ["gaussian", "uniform", "outliers", "multi_scale"]:
        analyze_precision_allocation(d, N=128)
