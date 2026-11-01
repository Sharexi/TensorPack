"""
Accuracy-per-bit (Pareto) benchmark: MTC-RD vs fair, state-of-the-art baselines.

Every method is scored by relative Frobenius error of the GEMM output against
FP32, as a function of TOTAL stored bits per element (payload + metadata).

Scenarios
---------
- gaussian / uniform / outliers / multi_scale: the repository's original
  synthetic distributions, C = A @ B with BOTH operands quantized.
- llm_like (weight-only): W (K x N) with heavy-tailed entries, activations with
  ~1% outlier channels (as observed in LLMs). Only W is quantized; activation-
  aware methods see a calibration batch X_cal, error is measured on a disjoint
  test batch X_test.

Usage:
    python -m mtc.benchmarks.benchmark_pareto            # writes results/pareto.{png,md,json}
"""

import json
import os
import time
from typing import Callable, Dict, List, Tuple

import numpy as np

from mtc.benchmarks.benchmark_runner import generate_matrix
from mtc.baselines import formats as F
from mtc.core.allocation import rd_allocate

Quantizer = Callable[[np.ndarray], Tuple[np.ndarray, float]]

RD_BUDGETS = [2.2, 2.5, 3.0, 3.5, 4.25, 5.0, 6.0, 8.25]


def rel_err(ref: np.ndarray, pred: np.ndarray) -> float:
    return float(np.linalg.norm(ref - pred) / np.linalg.norm(ref))


# ---------------------------------------------------------------------------
# Method registry. Each entry: name -> list of (label, quantizer factory).
# A quantizer acts on a (K, N) matrix with K = reduction dim along rows and
# receives `h` (per-row importance) and `X_cal` (calibration activations or None).
# ---------------------------------------------------------------------------

def baseline_methods() -> Dict[str, List[Tuple[str, Callable]]]:
    return {
        "Per-tensor INT (old baseline)": [
            (f"INT{b}", lambda W, h, X, b=b: F.per_tensor_int(W, b)) for b in (4, 8)
        ],
        "Block 16x16 INT (RTN)": [
            (f"INT{b}", lambda W, h, X, b=b: F.block_int(W, b, 16)) for b in (2, 3, 4, 5, 6, 8)
        ],
        "Group INT g32 (Q4_0-style)": [
            (f"INT{b}", lambda W, h, X, b=b: F.group_int(W, b, 32)) for b in (2, 3, 4, 5, 6, 8)
        ],
        "Group INT g128 (GPTQ-RTN-style)": [
            (f"INT{b}", lambda W, h, X, b=b: F.group_int(W, b, 128)) for b in (2, 3, 4, 5, 6, 8)
        ],
        "MX / NVFP4 (OCP, NVIDIA)": [
            ("MXFP4", lambda W, h, X: F.mxfp4(W)),
            ("NVFP4", lambda W, h, X: F.nvfp4(W)),
            ("MXINT8", lambda W, h, X: F.mxint8(W)),
        ],
        # Outlier fraction swept; only the Pareto envelope is reported (best fraction per budget).
        "Group INT g128 + fp16 outliers (best of 0.5-4%)": [
            (f"INT{b}+{f:.1%}", lambda W, h, X, b=b, f=f: F.group_int_with_outliers(W, b, 128, f))
            for b in (2, 3, 4, 5) for f in (0.005, 0.01, 0.02, 0.04)
        ],
    }


def fixed_mode_methods() -> List[Tuple[str, Callable]]:
    """Ablation: MTC's importance-weighted scale search, but every block at the same width (no allocation)."""
    from mtc.core.allocation import META_BITS_PER_BLOCK
    return [
        (f"P{b}", lambda W, h, X, b=b: (lambda a: (a.dequant, a.bits_per_element))(
            rd_allocate(W, b + META_BITS_PER_BLOCK / 256.0, h, 16, modes=(b,))))
        for b in (2, 3, 4, 5, 6, 8)
    ]


def qt_methods(sizes=(32, 16, 8, 4)) -> List[Tuple[str, Callable]]:
    """MTC-QT: one quadtree candidate search per (matrix, importance), reused for every budget."""
    import hashlib
    from mtc.core.quadtree import QuadtreeRD
    cache: Dict[str, QuadtreeRD] = {}

    def get(W, h):
        key = hashlib.blake2b(W.tobytes() + (b"" if h is None else np.asarray(h).tobytes())).hexdigest()
        if key not in cache:
            cache.clear()  # keep memory bounded: callers alternate over at most a few matrices
            cache[key] = QuadtreeRD(W, h, sizes)
        return cache[key]

    def q(b):
        def run(W, h, X):
            a = get(W, h).select(b)
            return a.dequant, a.bits_per_element
        return run

    return [(f"{b:g}b", q(b)) for b in RD_BUDGETS]


def awq_methods() -> List[Tuple[str, Callable]]:
    return [(f"INT{b}", lambda W, h, X, b=b: F.awq_group_int(W, X, b, 128)) for b in (2, 3, 4, 5)]


def rd_methods(use_importance: bool) -> List[Tuple[str, Callable]]:
    return [
        (f"{b:g}b", lambda W, h, X, b=b, u=use_importance: (lambda a: (a.dequant, a.bits_per_element))(
            rd_allocate(W, b, h if u else None, 16)))
        for b in RD_BUDGETS
    ]


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def scenario_both_operands(dist: str, N: int = 512):
    A = generate_matrix(dist, N, seed=42)
    B = generate_matrix(dist, N, seed=43)
    C = A @ B
    hA = (B.astype(np.float64) ** 2).sum(axis=1)  # importance for rows of A.T (= cols of A)
    hB = (A.astype(np.float64) ** 2).sum(axis=0)  # importance for rows of B

    def evaluate(q: Callable) -> Tuple[float, float]:
        At, ba = q(A.T, hA, None)
        Bq, bb = q(B, hB, None)
        return (ba + bb) / 2.0, rel_err(C, At.T @ Bq)

    return evaluate


def make_llm_like(K: int = 1024, N: int = 1024, T: int = 512, seed: int = 0):
    rng = np.random.RandomState(seed)
    W = (rng.standard_t(df=4, size=(K, N)) * 0.02).astype(np.float32)
    # Per-input-channel weight scale variation (LayerNorm-folded weights vary by channel)
    W *= np.exp(rng.randn(K, 1) * 0.5).astype(np.float32)
    chan = np.exp(rng.randn(K) * 0.3)
    outlier_ch = rng.choice(K, size=max(1, K // 100), replace=False)
    chan[outlier_ch] *= 30.0

    def acts(n):
        return (rng.randn(n, K) * chan).astype(np.float32)

    return W, acts(T), acts(T)


def scenario_weight_only(W: np.ndarray, X_cal: np.ndarray, X_test: np.ndarray):
    Y = X_test @ W
    h = (X_cal.astype(np.float64) ** 2).mean(axis=0)

    def evaluate(q: Callable) -> Tuple[float, float]:
        Wq, b = q(W, h, X_cal)
        return b, rel_err(Y, X_test @ Wq)

    return evaluate


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def pareto_front(pts: List[Dict]) -> List[Dict]:
    """Keep only non-dominated (bits, error) points, sorted by bits."""
    front, best = [], np.inf
    for p in sorted(pts, key=lambda p: (p["bits"], p["rel_err"])):
        if p["rel_err"] < best:
            front.append(p)
            best = p["rel_err"]
    return front


def run_scenario(evaluate: Callable, weight_only: bool) -> Dict[str, List[Dict]]:
    methods = baseline_methods()
    if weight_only:
        methods["AWQ g128 (activation-aware)"] = awq_methods()
    methods["MTC scales, fixed width (ablation)"] = fixed_mode_methods()
    methods["MTC-RD, no importance (ablation)"] = rd_methods(False)
    methods["MTC-RD fixed 16x16 (v1)"] = rd_methods(True)
    methods["MTC-QT quadtree 32-4 (ours)"] = qt_methods()

    out: Dict[str, List[Dict]] = {}
    for family, entries in methods.items():
        pts = []
        for label, q in entries:
            bits, err = evaluate(q)
            pts.append({"label": label, "bits": round(bits, 4), "rel_err": err})
        out[family] = pareto_front(pts)
    return out


def run_all(N: int = 512) -> Dict[str, Dict]:
    results = {}
    for dist in ["gaussian", "uniform", "outliers", "multi_scale"]:
        t0 = time.perf_counter()
        results[f"{dist} (A@B, both quantized)"] = run_scenario(scenario_both_operands(dist, N), False)
        print(f"  {dist:12s} done in {time.perf_counter() - t0:.1f}s")
    t0 = time.perf_counter()
    W, X_cal, X_test = make_llm_like()
    results["llm_like (weight-only, calibrated)"] = run_scenario(scenario_weight_only(W, X_cal, X_test), True)
    print(f"  llm_like     done in {time.perf_counter() - t0:.1f}s")
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

# Fixed categorical order (reference palette, light surface). Color follows the
# method family; the ablation shares MTC-RD's hue and is distinguished by dashes.
STYLE = {
    "MTC-QT quadtree 32-4 (ours)":          dict(color="#2a78d6", marker="o", ls="-",  lw=2.6, z=11),
    "MTC-RD fixed 16x16 (v1)":             dict(color="#2a78d6", marker="o", ls=":",  lw=1.4, z=10, mfc="white"),
    "MTC-RD, no importance (ablation)":    dict(color="#2a78d6", marker="o", ls="--", lw=1.0, z=9, mfc="white"),
    "Block 16x16 INT (RTN)":               dict(color="#eb6834", marker="s", ls="-",  lw=1.4, z=5),
    "MTC scales, fixed width (ablation)":  dict(color="#eb6834", marker="s", ls="--", lw=1.4, z=8, mfc="white"),
    "Group INT g32 (Q4_0-style)":          dict(color="#1baf7a", marker="^", ls="-",  lw=1.4, z=5),
    "Group INT g128 (GPTQ-RTN-style)":     dict(color="#1baf7a", marker="v", ls=":",  lw=1.4, z=5),
    "MX / NVFP4 (OCP, NVIDIA)":            dict(color="#6250d6", marker="D", ls="none", lw=0, z=6),
    "Group INT g128 + fp16 outliers (best of 0.5-4%)": dict(color="#e34948", marker="P", ls="-", lw=1.4, z=6),
    "AWQ g128 (activation-aware)":         dict(color="#eda100", marker="X", ls="-",  lw=1.4, z=7),
    "Per-tensor INT (old baseline)":       dict(color="#8a8a85", marker="x", ls="none", lw=0, z=4),
}


def plot(results: Dict[str, Dict], path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    axes = axes.ravel()
    for ax, name in zip(axes, names):
        for family, pts in results[name].items():
            st = STYLE[family]
            xs = [p["bits"] for p in pts]
            ys = [p["rel_err"] for p in pts]
            ax.plot(xs, ys, color=st["color"], ls=st["ls"], lw=st["lw"], marker=st["marker"],
                    ms=6, mfc=st.get("mfc", st["color"]), mec=st["color"], zorder=st["z"], label=family)
            if family.startswith("MX"):
                for p in pts:
                    ax.annotate(p["label"], (p["bits"], p["rel_err"]), textcoords="offset points",
                                xytext=(5, 4), fontsize=7.5, color="#55554f")
        ax.set_yscale("log")
        ax.set_xlim(1.8, 9.0)
        ax.set_title(name, fontsize=11, loc="left", color="#1f1f1d")
        ax.set_xlabel("total bits / element (payload + metadata)", fontsize=9, color="#55554f")
        ax.set_ylabel("relative output error (log)", fontsize=9, color="#55554f")
        ax.grid(True, which="major", color="#e6e5e0", lw=0.8)
        ax.grid(True, which="minor", color="#f2f1ed", lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c3c2b7")
        ax.tick_params(colors="#55554f", labelsize=8)

    leg_ax = axes[len(names)] if len(names) < len(axes) else None
    handles, labels = [], []
    seen = set()
    for ax in axes[: len(names)]:
        for hd, lb in zip(*ax.get_legend_handles_labels()):
            if lb not in seen:
                seen.add(lb)
                handles.append(hd)
                labels.append(lb)
    if leg_ax is not None:
        leg_ax.axis("off")
        leg_ax.legend(handles, labels, loc="center", frameon=False, fontsize=10,
                      title="Lower-left is better", title_fontsize=10)
    fig.suptitle("Accuracy per bit: MTC-QT / MTC-RD vs. fair quantization baselines", fontsize=14,
                 x=0.01, ha="left", color="#1f1f1d")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _interp_err(pts: List[Dict], bits: float) -> float:
    """
    Linear interpolation of a method's error at a given bit budget (nan if out
    of range). Linear-in-error is what is actually realizable between two
    operating points (use width b1 on a fraction of the blocks and b2 on the
    rest); log-linear interpolation would flatter fixed-width curves.
    """
    xs = np.array([p["bits"] for p in pts])
    ys = np.array([p["rel_err"] for p in pts])
    if len(xs) < 2 or bits < xs.min() - 1e-9 or bits > xs.max() + 1e-9:
        return float("nan")
    return float(np.interp(bits, xs, ys))


def write_markdown(results: Dict[str, Dict], path: str):
    lines = ["# Accuracy-per-bit results (relative output error; lower is better)", "",
             "Bits are total stored bits per element including scales and side data. "
             "Curves are compared at matched budgets by linear (realizable) interpolation between measured points; "
             "single-point formats are listed at their native bit width.", ""]
    budgets = [2.5, 3.0, 4.25, 5.0, 6.0]
    for name, fams in results.items():
        lines += [f"## {name}", "", "| Method | " + " | ".join(f"@{b} b" for b in budgets) + " |",
                  "|---|" + "---|" * len(budgets)]
        for fam, pts in fams.items():
            if len(pts) >= 2 and not fam.startswith("MX") and not fam.startswith("Per-tensor"):
                row = [_interp_err(pts, b) for b in budgets]
                lines.append(f"| {fam} | " + " | ".join("—" if np.isnan(v) else f"{v:.4f}" for v in row) + " |")
        lines.append("")
        lines.append("Fixed formats: " + ", ".join(
            f"{p['label']} {p['rel_err']:.4f} @ {p['bits']:.2f} b"
            for fam in ("MX / NVFP4 (OCP, NVIDIA)", "Per-tensor INT (old baseline)") for p in fams[fam]))
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(out_dir: str = "results"):
    os.makedirs(out_dir, exist_ok=True)
    print("Running accuracy-per-bit Pareto benchmark...")
    results = run_all()
    with open(os.path.join(out_dir, "pareto.json"), "w") as f:
        json.dump(results, f, indent=1)
    write_markdown(results, os.path.join(out_dir, "pareto.md"))
    plot(results, os.path.join(out_dir, "pareto.png"))
    print(f"Wrote {out_dir}/pareto.png, pareto.md, pareto.json")


if __name__ == "__main__":
    main()
