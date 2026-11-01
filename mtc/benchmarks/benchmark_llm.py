"""
Real-model evaluation: WikiText-2 perplexity with weight-only quantization.

All nn.Linear layers inside the transformer blocks are quantized (embeddings and
the tied LM head stay FP32, the standard protocol in GPTQ/AWQ papers). Every
method is evaluated on the SAME token slice, and reported bits are total stored
bits per quantized weight (payload + scales + side data).

Activation-aware methods (AWQ, MTC-RD) use calibration statistics from the
WikiText-2 *train* split; evaluation uses the *test* split.

Usage:
    python -m mtc.benchmarks.benchmark_llm                    # Qwen2.5-0.5B, 24 x 1024 test tokens
    python -m mtc.benchmarks.benchmark_llm --n-eval 0         # full test set (slow on CPU)
"""

import argparse
import json
import math
import os
import time
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

from mtc.baselines import formats as F
from mtc.core.allocation import rd_candidates, rd_select, rd_dequant


# ---------------------------------------------------------------------------
# Data / model
# ---------------------------------------------------------------------------

def load(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    test_ids = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids
    train_ids = tok("\n\n".join(train["text"][:20000]), return_tensors="pt").input_ids
    return model, test_ids, train_ids


def quantizable_linears(model) -> Dict[str, torch.nn.Linear]:
    return {n: m for n, m in model.model.layers.named_modules() if isinstance(m, torch.nn.Linear)}


@torch.no_grad()
def perplexity(model, ids: torch.Tensor, seqlen: int, n_seqs: int, batch: int = 4) -> float:
    total = ids.shape[1] // seqlen
    n = total if n_seqs <= 0 else min(n_seqs, total)
    x = ids[:, : n * seqlen].reshape(n, seqlen)
    nll, count = 0.0, 0
    for i in range(0, n, batch):
        xb = x[i : i + batch]
        loss = model(xb, labels=xb).loss.item()
        tokens = xb.shape[0] * (seqlen - 1)
        nll += loss * tokens
        count += tokens
    return math.exp(nll / count)


@torch.no_grad()
def calibrate(model, ids: torch.Tensor, n_seqs: int, seqlen: int, rows_kept: int, seed: int = 0):
    """Per-linear input statistics: mean x^2 per input channel, plus a row subsample for AWQ."""
    rng = np.random.RandomState(seed)
    starts = rng.randint(0, ids.shape[1] - seqlen - 1, size=n_seqs)
    linears = quantizable_linears(model)
    sumsq = {n: torch.zeros(m.in_features, dtype=torch.float64) for n, m in linears.items()}
    rows: Dict[str, List[torch.Tensor]] = {n: [] for n in linears}
    per_seq = max(1, rows_kept // n_seqs)
    state = {"count": 0}

    def hook(name):
        def fn(mod, inp, out):
            x = inp[0].reshape(-1, inp[0].shape[-1]).double()
            sumsq[name] += (x ** 2).sum(0)
            sel = torch.from_numpy(rng.choice(x.shape[0], per_seq, replace=False))
            rows[name].append(x[sel].float())
        return fn

    handles = [m.register_forward_hook(hook(n)) for n, m in linears.items()]
    for s in starts:
        model(ids[:, s : s + seqlen])
        state["count"] += seqlen
    for h in handles:
        h.remove()
    h_stats = {n: (sumsq[n] / state["count"]).numpy() for n in linears}
    x_cal = {n: torch.cat(rows[n]).numpy() for n in linears}  # (xcal_rows, in) per linear
    return h_stats, x_cal


# ---------------------------------------------------------------------------
# Methods. Each acts on Wt = W.T (in_features x out_features: reduction along rows).
# ---------------------------------------------------------------------------

def fixed_methods() -> Dict[str, Callable]:
    return {
        "RTN g128 INT3": lambda Wt, h, X: F.group_int(Wt, 3, 128),
        "RTN g128 INT4": lambda Wt, h, X: F.group_int(Wt, 4, 128),
        "RTN g32 INT4 (Q4_0-style)": lambda Wt, h, X: F.group_int(Wt, 4, 32),
        "MXFP4": lambda Wt, h, X: F.mxfp4(Wt),
        "NVFP4": lambda Wt, h, X: F.nvfp4(Wt),
        "AWQ g128 INT3": lambda Wt, h, X: F.awq_group_int(Wt, X, 3, 128),
        "AWQ g128 INT4": lambda Wt, h, X: F.awq_group_int(Wt, X, 4, 128),
    }


RD_BUDGETS = [2.5, 3.125, 4.125, 4.25, 4.5]
RD_ABLATION_BUDGETS = [3.125, 4.125]
QT_BUDGETS = [2.5, 3.125, 4.125, 4.25, 4.5]
METHOD_GROUPS = ("fp32", "baselines", "rd", "rd_ablation", "rd_bf16", "qt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--n-eval", type=int, default=24, help="test sequences (0 = full test set)")
    ap.add_argument("--n-calib", type=int, default=32)
    ap.add_argument("--calib-len", type=int, default=512)
    ap.add_argument("--xcal-rows", type=int, default=256)
    ap.add_argument("--out", default="results")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--methods", default=",".join(METHOD_GROUPS),
                    help=f"comma-separated subset of {METHOD_GROUPS}")
    ap.add_argument("--tag", default="", help="suffix for output files")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    groups = set(args.methods.split(","))
    assert groups <= set(METHOD_GROUPS), groups
    out_json = os.path.join(args.out, f"llm_ppl{args.tag}.json")

    t0 = time.perf_counter()
    model, test_ids, train_ids = load(args.model)
    linears = quantizable_linears(model)

    # Memory: the checkpoint is bf16, so keeping originals as bf16 is lossless
    # (verified below) and halves the copy. Quantized weights are written into
    # the model layer by layer, never materialized as a second full model.
    originals = {n: m.weight.detach().to(torch.bfloat16).clone() for n, m in linears.items()}
    for n, m in linears.items():
        assert torch.equal(originals[n].float(), m.weight.detach()), f"{n} is not bf16-exact"
    n_params = sum(w.numel() for w in originals.values())
    print(f"Loaded {args.model}: {len(linears)} linears, {n_params / 1e6:.1f}M quantizable weights "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)

    def orig_t(n: str) -> np.ndarray:
        """Original weight as (in, out) fp32, reduction dim along rows."""
        return originals[n].float().numpy().T.copy()

    def set_weight(n: str, Wt: np.ndarray):
        linears[n].weight.data = torch.from_numpy(np.ascontiguousarray(Wt.T, dtype=np.float32))

    def restore():
        for n in linears:
            linears[n].weight.data = originals[n].float()

    h_stats, x_cal = calibrate(model, train_ids, args.n_calib, args.calib_len, args.xcal_rows)
    print(f"Calibrated on {args.n_calib} x {args.calib_len} train tokens", flush=True)

    # Resume: methods already in the output JSON (same config) are skipped.
    results = {"config": vars(args), "eval_tokens": None, "methods": []}
    if os.path.exists(out_json):
        with open(out_json) as f:
            prev = json.load(f)
        same = all(prev.get("config", {}).get(k) == getattr(args, k)
                   for k in ("model", "seqlen", "n_eval", "n_calib", "calib_len", "xcal_rows"))
        if same:
            results["methods"] = prev["methods"]
            print(f"Resuming: {len(prev['methods'])} methods already done", flush=True)
    done = {m["name"] for m in results["methods"]}

    def record(name: str, family: str, bits: float, ppl: float, secs: float):
        results["methods"].append({"name": name, "family": family, "bits": round(bits, 4), "ppl": ppl,
                                   "quant_seconds": round(secs, 1)})
        with open(out_json, "w") as f:
            json.dump(results, f, indent=1)
        print(f"  {name:38s} bits={bits:6.3f}  ppl={ppl:8.3f}  (quant {secs:.0f}s)", flush=True)

    def ppl_now() -> float:
        return perplexity(model, test_ids, args.seqlen, args.n_eval)

    results["eval_tokens"] = (args.n_eval if args.n_eval > 0 else test_ids.shape[1] // args.seqlen) * args.seqlen
    if "fp32" in groups and "FP32 (unquantized)" not in done:
        record("FP32 (unquantized)", "reference", 32.0, ppl_now(), 0.0)

    # Fixed-format baselines: quantize layer by layer straight into the model.
    for name, q in (fixed_methods().items() if "baselines" in groups else []):
        if name in done:
            continue
        t = time.perf_counter()
        bits = 0.0
        for n in linears:
            Wt = orig_t(n)
            deq, b = q(Wt, h_stats[n], x_cal[n])
            set_weight(n, deq)
            bits += b * Wt.size
        secs = time.perf_counter() - t
        record(name, name.split()[0], bits / n_params, ppl_now(), secs)
        restore()

    # MTC-RD: one candidate search per layer, then every budget from the compact cache.
    rd_runs = [
        ("rd", "MTC-RD", True, RD_BUDGETS, "fp32"),
        ("rd_ablation", "MTC-RD, no importance (ablation)", False, RD_ABLATION_BUDGETS, "fp32"),
        ("rd_bf16", "MTC-RD 16x16, bf16 scales (ablation)", True, RD_ABLATION_BUDGETS, "bf16"),
    ]
    for key, label, use_h, budgets, fmt in rd_runs:
        budgets = [b for b in budgets if f"{label} @ {b:g} b" not in done]
        if key not in groups or not budgets:
            continue
        t = time.perf_counter()
        allocs: Dict[float, Dict[str, Tuple]] = {b: {} for b in budgets}
        bits = {b: 0.0 for b in budgets}
        for n in linears:
            Wt = orig_t(n)
            cand = rd_candidates(Wt, h_stats[n] if use_h else None, 16, scale_format=fmt)
            for b in budgets:
                a = rd_select(cand, b)
                allocs[b][n] = (a.modes, a.scales)
                bits[b] += a.bits_per_element * Wt.size
            del cand
        secs = (time.perf_counter() - t) / len(budgets)
        for b in budgets:
            for n in linears:
                set_weight(n, rd_dequant(orig_t(n), *allocs[b][n], 16))
            record(f"{label} @ {b:g} b", label, bits[b] / n_params, ppl_now(), secs)
            restore()

    if "qt" in groups:
        from mtc.core.quadtree import QuadtreeRD, qt_dequant
        label = "MTC-QT quadtree 32-4"
        budgets = [b for b in QT_BUDGETS if f"{label} @ {b:g} b" not in done]
        if budgets:
            t = time.perf_counter()
            # Keep only the compact allocation (leaf masks, modes, scales); rebuild exactly at eval time.
            qt_allocs: Dict[float, Dict[str, Tuple]] = {b: {} for b in budgets}
            bits = {b: 0.0 for b in budgets}
            for n in linears:
                Wt = orig_t(n)
                qt = QuadtreeRD(Wt, h_stats[n], (32, 16, 8, 4))
                for b in budgets:
                    a = qt.select(b)
                    qt_allocs[b][n] = (a.sizes, a.leaf, a.modes, a.scales)
                    bits[b] += a.bits_per_element * Wt.size
                del qt
            secs = (time.perf_counter() - t) / len(budgets)
            for b in budgets:
                for n in linears:
                    set_weight(n, qt_dequant(orig_t(n), *qt_allocs[b][n]))
                record(f"{label} @ {b:g} b", "MTC-QT", bits[b] / n_params, ppl_now(), secs)
                restore()

    write_markdown(results, os.path.join(args.out, f"llm_ppl{args.tag}.md"))
    print(f"Done in {(time.perf_counter() - t0) / 60:.1f} min -> {out_json}")


def write_markdown(results: Dict, path: str):
    cfg = results["config"]
    lines = [
        f"# WikiText-2 perplexity, {cfg['model']} (weight-only quantization)", "",
        f"Evaluated on {results['eval_tokens']:,} test tokens (seqlen {cfg['seqlen']}), identical for all methods. "
        f"Calibration: {cfg['n_calib']} x {cfg['calib_len']} train tokens. Bits = total stored bits per quantized "
        "weight (payload + scales + side data). Embeddings / tied LM head kept in FP32 for all methods.", "",
        "| Method | Bits / weight | Perplexity |", "|---|---|---|",
    ]
    for m in sorted(results["methods"], key=lambda r: (-r["bits"] if r["family"] == "reference" else r["bits"])):
        lines.append(f"| {m['name']} | {m['bits']:.3f} | {m['ppl']:.3f} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
