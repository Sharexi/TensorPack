# Handoff: open work on MTC-QT-P

Status as of 2026-11-06, branch `rd-allocation-quadtree` (PR #2 into `main`).

## Where things stand

- **MTC-QT-P** (`mtc/core/permutation.py`): rows and columns are sorted by RD weight before the quadtree RD allocation; the index bits are counted in the budget. The Qwen2.5-0.5B results are in `results/llm_ppl.md` and the README.
- **GPTQ** (`mtc/baselines/gptq.py`, `mtc/benchmarks/benchmark_gptq.py`): code and tests are done. **The full Qwen run has never completed** (on the local 16 GB laptop the OOM killer stopped it). `results/llm_gptq.*` does not exist yet, and the README says no GPTQ results are reported.
- Tests: `python -m pytest -q` should report 67 passed.

Qwen2.5-0.5B, WikiText-2 perplexity (FP32 = 13.99), MTC-QT → MTC-QT-P:

| Bits | 2.5 | 3.125 | 4.125 | 4.25 | 4.5 |
|---|---|---|---|---|---|
| MTC-QT | 266.5 | 29.8 | 15.70 | 15.33 | 14.91 |
| MTC-QT-P | 52.5 | 19.8 | 15.19 | 14.89 | 14.74 |

## Setup

```bash
pip install -r requirements.txt   # torch, transformers, datasets, numba, ...
python -m pytest -q
```

You need network access to Hugging Face (`Qwen/Qwen2.5-0.5B`, `wikitext-2-raw-v1`) and about 8 GB of free RAM or more. Without a GPU, calibration runs on the CPU (slower, but it works).

## Task 1: GPTQ comparison (highest priority)

```bash
python -m mtc.benchmarks.benchmark_gptq                      # defaults: --methods gptq,qtp_gptq --budgets 2.5,3.125,4.125,4.5
```

- It writes `results/llm_gptq.json` / `.md` after every method, and a rerun skips the methods already done.
- Optional extra methods: `qt_gptq` (unpermuted quadtree + GPTQ) and `qtp_rtn_seq` (MTC-QT-P with sequential calibration but no GPTQ). The second one separates the gain of GPTQ from the gain of sequential calibration.
- **Comparability check:** perplexity depends slightly on the torch/transformers versions. Rerun FP32 and one RTN baseline in the same environment (`python -m mtc.benchmarks.benchmark_llm --methods fp32,baselines`, written to `results/llm_ppl.json`, so back up the committed file first). Compare GPTQ numbers only with baselines from the same environment.
- Then update the README:
  - Add a GPTQ column or rows to the Qwen table.
  - Remove the bullet "GPTQ-style error compensation has not been compared yet".
  - Remove the GPTQ limitation bullet.
  - State plainly whether MTC-QT-P + GPTQ beats GPTQ g128 at equal bits. Do not overclaim.
- Memory: the script keeps a bf16 copy of all quantized weights (`originals`, about 1 GB). If memory is tight, reload them from disk in `restore()` instead.

## Task 2: Per-matrix permute-or-not

Reordering is 20× worse on the synthetic multi-scale matrix (`benchmark_pareto.py`): a 1D sort scatters 2D blocks. The fix:

- Solve every matrix both with and without the permutation, and keep the one with the lower RD cost at the same λ (or the lower distortion at the same bits).
- Store 1 flag bit per matrix, and skip the index bits when the matrix is not permuted.
- Add it as a `PermutedQuadtreeRD(..., auto=True)` option or as a separate solver.
- Add a test showing that the auto mode is never worse than either option on the synthetic suite.
- Rerun `benchmark_pareto` and the `qt_perm` group of `benchmark_llm`.

## Task 3: Rows-only ablation

Compare `PermutedQuadtreeRD(rows=True, cols=False)`, the reverse, and both, on Qwen at 3.125 and 4.25 b. Rows-only permutations can be folded into the previous layer (no runtime gather), so a small loss compared with rows-and-columns would make the deployment story much simpler.

## Conventions

- `Wt` is `(K, N) = (in_features, out_features)`. The importance is `h_k = E[x_k^2]`.
- The bit counts include **everything stored**: payload, scales, mode bytes, split flags and permutation indices.
- The evaluation protocol is fixed: 24×1024 test tokens, 32×512 calibration tokens (seed 0), FP32 evaluation, embeddings unquantized. Do not change it, or the numbers stop being comparable.
- The README reports only measured results; known regressions go in "Where it loses".
