"""
Single-command reproduction script for TensorPack / MTC-8.

Usage:
    python reproduce.py            # tests + accuracy-per-bit Pareto benchmark + legacy demos
    python reproduce.py --llm      # additionally run the Qwen2.5-0.5B perplexity benchmark (~90 min CPU)
"""

import sys

import pytest

RUN_LLM = "--llm" in sys.argv


def banner(text):
    print("\n" + "=" * 75 + f"\n  {text}\n" + "=" * 75)


banner("TENSORPACK / MICRO-TENSOR COMPUTING (MTC-8) REPRODUCTION SUITE")

print("\n[Step 1] Unit tests")
ret = pytest.main(["-q", "tests/"])
print("[PASS] All tests passed." if ret == 0 else "[FAIL] Some tests failed (see above).")

print("\n[Step 2] Accuracy per bit: MTC-RD / MTC-QT vs fair baselines (synthetic) -> results/pareto.*")
from mtc.benchmarks import benchmark_pareto
benchmark_pareto.main()

if RUN_LLM:
    print("\n[Step 3] Qwen2.5-0.5B WikiText-2 perplexity -> results/llm_ppl.*")
    from mtc.benchmarks import benchmark_llm
    sys.argv = [sys.argv[0]]
    benchmark_llm.main()
else:
    print("\n[Step 3] Skipped real-LLM perplexity benchmark (pass --llm to run it).")

print("\n[Step 4] Legacy exploratory demos (3D octrees, primes, small MLP).")
print("         These are illustrative demonstrations; no results are claimed for them.")
from mtc.benchmarks.benchmark_geometric import benchmark_3d_spatial, benchmark_primes_and_ternary
from mtc.benchmarks.benchmark_inference import benchmark_model_inference
from mtc.benchmarks.benchmark_training import benchmark_training_convergence

benchmark_3d_spatial()
benchmark_primes_and_ternary()
benchmark_model_inference(batch_sizes=[16], repeats=5)
benchmark_training_convergence(epochs=5, batch_size=32)

banner("REPRODUCTION COMPLETE")
