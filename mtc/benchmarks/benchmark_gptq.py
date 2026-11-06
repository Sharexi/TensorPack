"""
GPTQ comparison on a real LLM: GPTQ baselines and MTC-QT-P + GPTQ.

Protocol (identical to benchmark_llm.py, so numbers are comparable to results/llm_ppl.md):
same model, same 24 x 1024 WikiText-2 test tokens, same 32 x 512 calibration
tokens (train split, seed 0), FP32 evaluation on the CPU, embeddings / tied LM
head unquantized, bits = total stored bits per quantized weight.

Calibration is sequential, as in GPTQ: decoder layer l is calibrated on the
outputs of the already-quantized layers 0..l-1. One decoder layer at a time is
moved to the GPU (if available), so a 2 GB card is enough for Qwen2.5-0.5B.

For MTC-QT-P the quadtree RD allocation (tiling, modes, scales, permutation) is
computed from the original weights and the diagonal of the sequential Hessian;
GPTQ then chooses the integer codes inside that fixed grid. The result is a valid
MTC-QT-P encoding with exactly the bits of the allocation.

Usage:
    python -m mtc.benchmarks.benchmark_gptq
    python -m mtc.benchmarks.benchmark_gptq --methods gptq,qtp_gptq --budgets 3.125,4.125
"""

import argparse
import json
import os
import time
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

from mtc.baselines import formats as F
from mtc.baselines.gptq import gptq, group_int_bits, qt_element_grid
from mtc.benchmarks.benchmark_llm import load, perplexity, quantizable_linears

METHOD_GROUPS = ("gptq", "qtp_gptq", "qt_gptq", "qtp_rtn_seq")


class _Catcher(torch.nn.Module):
    """Stands in for decoder layer 0 to record its inputs, then aborts the forward pass."""

    def __init__(self, layer, store):
        super().__init__()
        self.layer, self.store = layer, store
        if hasattr(layer, "attention_type"):
            self.attention_type = layer.attention_type

    def forward(self, x, *args, **kwargs):
        self.store["x"].append(x.detach())
        self.store["kwargs"] = kwargs
        raise StopIteration


def _to(obj, dev):
    if torch.is_tensor(obj):
        return obj.to(dev)
    if isinstance(obj, tuple):
        return tuple(_to(o, dev) for o in obj)
    if isinstance(obj, list):
        return [_to(o, dev) for o in obj]
    if isinstance(obj, dict):
        return {k: _to(v, dev) for k, v in obj.items()}
    return obj


@torch.no_grad()
def capture_layer0_inputs(model, ids, n_seqs, seqlen, seed=0):
    """Same calibration windows as benchmark_llm.calibrate (same RNG, same draws)."""
    rng = np.random.RandomState(seed)
    starts = rng.randint(0, ids.shape[1] - seqlen - 1, size=n_seqs)
    layers = model.model.layers
    store = {"x": [], "kwargs": None}
    layers[0] = _Catcher(layers[0], store)
    for s in starts:
        try:
            model(ids[:, s: s + seqlen], use_cache=False)
        except StopIteration:
            pass
    layers[0] = layers[0].layer
    kwargs = {k: v for k, v in store["kwargs"].items() if k not in ("past_key_values", "past_key_value", "use_cache")}
    kwargs["use_cache"] = False
    return store["x"], kwargs


def _layer_out(layer, x, kwargs):
    out = layer(x, **kwargs)
    return out[0] if isinstance(out, (tuple, list)) else out


@torch.no_grad()
def quantize_sequential(model, inps: List[torch.Tensor], kwargs: Dict, quantize: Callable, dev) -> float:
    """
    quantize(name, Wt (K,N) np.float32, H (K,K) torch on dev, n_tokens) -> (deq (K,N) np.float32, bits/weight).
    Writes the quantized weights into the model and returns total bits per quantized weight.
    """
    kw = _to(kwargs, dev)
    xs = [x.to(dev) for x in inps]
    total_bits, total_w = 0.0, 0
    for li, layer in enumerate(model.model.layers):
        layer.to(dev)
        lins = {n: m for n, m in layer.named_modules() if isinstance(m, torch.nn.Linear)}
        H = {n: torch.zeros(m.in_features, m.in_features, device=dev) for n, m in lins.items()}
        cnt = {n: 0 for n in lins}

        def hook(n):
            def fn(mod, inp, out):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                H[n].addmm_(x.T, x)
                cnt[n] += x.shape[0]
            return fn

        hs = [m.register_forward_hook(hook(n)) for n, m in lins.items()]
        for x in xs:
            _layer_out(layer, x, kw)
        for h in hs:
            h.remove()

        for n, m in lins.items():
            Wt = m.weight.detach().float().cpu().numpy().T.copy()
            deq, bits = quantize(f"{li}.{n}", Wt, H[n] / cnt[n], cnt[n])
            m.weight.data = torch.from_numpy(np.ascontiguousarray(deq.T, dtype=np.float32)).to(dev)
            total_bits += bits * Wt.size
            total_w += Wt.size
        del H
        xs = [_layer_out(layer, x, kw) for x in xs]
        layer.to("cpu")
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        print(f"    layer {li:2d} done", flush=True)
    return total_bits / total_w


def gptq_group_quantizer(bits: int, act_order: bool, dev):
    def q(name, Wt, H, n):
        Q = gptq(torch.from_numpy(Wt).to(dev), H, grid="group", bits=bits, group=128, act_order=act_order)
        return Q.cpu().numpy(), group_int_bits(bits, 128)
    return q


def qt_quantizer(budget: float, permute: bool, use_gptq: bool, dev):
    from mtc.core.permutation import PermutedQuadtreeRD
    from mtc.core.quadtree import QuadtreeRD

    def q(name, Wt, H, n):
        h = torch.diag(H).double().cpu().numpy()  # = mean x_k^2 over the sequential calibration inputs
        if permute:
            a = PermutedQuadtreeRD(Wt, h, (32, 16, 8, 4)).select(budget)
            inner, pr, pc = a.inner, a.row_perm, a.col_perm
        else:
            a = QuadtreeRD(Wt, h, (32, 16, 8, 4)).select(budget)
            inner, pr, pc = a, np.arange(Wt.shape[0]), np.arange(Wt.shape[1])
        if not use_gptq:
            return a.dequant, a.bits_per_element
        Sp, LOp, HIp = qt_element_grid(inner.sizes, inner.leaf, inner.modes, inner.scales, Wt.shape)
        S, LO, HI = (np.empty_like(Sp) for _ in range(3))
        ix = np.ix_(pr, pc)
        S[ix], LO[ix], HI[ix] = Sp, LOp, HIp
        t = lambda x: torch.from_numpy(x).to(dev)
        Q = gptq(torch.from_numpy(Wt).to(dev), H, "fixed", t(S), t(LO), t(HI), act_order=True)
        return Q.cpu().numpy(), a.bits_per_element
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--n-eval", type=int, default=24)
    ap.add_argument("--n-calib", type=int, default=32)
    ap.add_argument("--calib-len", type=int, default=512)
    ap.add_argument("--budgets", default="2.5,3.125,4.125,4.5")
    ap.add_argument("--methods", default="gptq,qtp_gptq", help=f"comma-separated subset of {METHOD_GROUPS}")
    ap.add_argument("--out", default="results")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    groups = args.methods.split(",")
    assert set(groups) <= set(METHOD_GROUPS), groups
    budgets = [float(b) for b in args.budgets.split(",")]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_json = os.path.join(args.out, "llm_gptq.json")

    t0 = time.perf_counter()
    model, test_ids, train_ids = load(args.model)
    linears = quantizable_linears(model)
    originals = {n: m.weight.detach().to(torch.bfloat16).clone() for n, m in linears.items()}
    for n, m in linears.items():
        assert torch.equal(originals[n].float(), m.weight.detach()), f"{n} is not bf16-exact"
    print(f"Loaded {args.model} ({time.perf_counter() - t0:.0f}s), device for calibration: "
          f"{torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'}", flush=True)

    inps, kwargs = capture_layer0_inputs(model, train_ids, args.n_calib, args.calib_len)

    results = {"config": vars(args), "methods": []}
    if os.path.exists(out_json):
        with open(out_json) as f:
            prev = json.load(f)
        keys = ("model", "seqlen", "n_eval", "n_calib", "calib_len")
        if all(prev.get("config", {}).get(k) == getattr(args, k) for k in keys):
            results["methods"] = prev["methods"]
    done = {m["name"] for m in results["methods"]}

    def restore():
        for n in linears:
            linears[n].weight.data = originals[n].float()

    runs: List[Tuple[str, str, Callable]] = []
    if "gptq" in groups:
        for b in (3, 4):
            runs.append((f"GPTQ g128 INT{b}", "GPTQ", gptq_group_quantizer(b, False, dev)))
            runs.append((f"GPTQ g128 INT{b}, act-order", "GPTQ", gptq_group_quantizer(b, True, dev)))
    for key, label, permute, use_gptq in [
        ("qtp_gptq", "MTC-QT-P + GPTQ", True, True),
        ("qt_gptq", "MTC-QT + GPTQ", False, True),
        ("qtp_rtn_seq", "MTC-QT-P, sequential calib. (ablation)", True, False),
    ]:
        if key in groups:
            for b in budgets:
                runs.append((f"{label} @ {b:g} b", label, qt_quantizer(b, permute, use_gptq, dev)))

    for name, family, qfn in runs:
        if name in done:
            print(f"  skip {name} (done)")
            continue
        print(f"  {name}: quantizing", flush=True)
        t = time.perf_counter()
        bits = quantize_sequential(model, inps, kwargs, qfn, dev)
        model.to("cpu")
        secs = time.perf_counter() - t
        ppl = perplexity(model, test_ids, args.seqlen, args.n_eval)
        restore()
        results["methods"].append({"name": name, "family": family, "bits": round(bits, 4), "ppl": ppl,
                                   "quant_seconds": round(secs, 1)})
        with open(out_json, "w") as f:
            json.dump(results, f, indent=1)
        print(f"  {name:44s} bits={bits:6.3f}  ppl={ppl:8.3f}  (quant {secs:.0f}s)", flush=True)

    write_markdown(results, os.path.join(args.out, "llm_gptq.md"))
    print(f"Done in {(time.perf_counter() - t0) / 60:.1f} min -> {out_json}")


def write_markdown(results: Dict, path: str):
    cfg = results["config"]
    lines = [
        f"# GPTQ comparison, {cfg['model']}, WikiText-2 perplexity (weight-only)", "",
        "Same protocol as `llm_ppl.md` (same test tokens, calibration windows and bit accounting). "
        "Calibration is sequential (each layer sees the outputs of the already-quantized layers). "
        "GPTQ baselines use the same symmetric g128 grid with an fp16 scale as the RTN g128 baseline.", "",
        "| Method | Bits / weight | Perplexity |", "|---|---|---|",
    ]
    for m in sorted(results["methods"], key=lambda r: (r["bits"], r["name"])):
        lines.append(f"| {m['name']} | {m['bits']:.3f} | {m['ppl']:.3f} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
