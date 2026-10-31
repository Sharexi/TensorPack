"""
Experiment: Error Accumulation Analysis Across Reduction Dimension K.
"""

import numpy as np

from mtc.benchmarks.benchmark_runner import generate_matrix
from mtc.kernels.reference import fp32_gemm_numpy, relative_frobenius_error
from mtc.baselines.int8_gemm import INT8GEMM
from mtc.baselines.int4_gemm import INT4GEMM
from mtc.kernels.gemm import MTCMatrix, mtc_gemm


def run_error_vs_reduction_k(K_list = [64, 128, 256, 512], M: int = 128, N: int = 128):
    print("=" * 70)
    print("Error Accumulation Across Reduction Dimension K")
    print(f"Fixed Output Shape: M={M}, N={N}")
    print("=" * 70)

    for K in K_list:
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)

        C_ref = fp32_gemm_numpy(A, B)

        C_int8 = INT8GEMM.run_end_to_end(A, B)
        err_int8 = relative_frobenius_error(C_ref, C_int8)

        C_int4 = INT4GEMM.run_end_to_end(A, B)
        err_int4 = relative_frobenius_error(C_ref, C_int4)

        mtc_a, mtc_b = MTCMatrix.from_numpy_joint(A, B, block_size=16, target_epsilon=0.05)
        C_mtc = mtc_gemm(mtc_a, mtc_b)
        err_mtc = relative_frobenius_error(C_ref, C_mtc)

        print(f"K={K:4d} | INT8 Err: {err_int8:.4f} | INT4 Err: {err_int4:.4f} | MTC-8 Joint Err: {err_mtc:.4f} (Target eps=0.05)")


if __name__ == "__main__":
    run_error_vs_reduction_k()
