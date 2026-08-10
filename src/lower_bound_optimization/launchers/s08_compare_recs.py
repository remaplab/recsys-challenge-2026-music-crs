"""s08 — Paired Empirical-Bernstein comparison of two s07 runs.

Inputs: two directories produced by s07 (`--a`, `--b`) that ran on the SAME
StratifiedEvaluator subset set (use s07 --cache_subsets PATH for both runs
to guarantee alignment).

For each (metric, strategy) pair: compute Δ = scores_A - scores_B per subset,
report paired empirical-Bernstein CI per LBO doc C. Significance gate:
the CI's lower bound strictly above 0 → A is reliably better than B.

Run:
    uv run python -m launchers.s08_compare_recs \\
        --a models/.../eval_run_A \\
        --b models/.../eval_run_B \\
        --out_dir models/.../comparison
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PKG_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np

from lbo.paths import ROOT
from lbo.shift.multi_comp import paired_empirical_bernstein_ci
from lbo.evaluator.plots import (
    plot_ab_cdf_overlay,
    plot_comparison_forest,
    plot_delta_distribution,
)


METRICS = ("ndcg@20", "recall@200")
STRATEGIES = ("calibrated", "density", "stratified")


def _resolve(p: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p).resolve()


def _metric_dir(root: Path, metric: str) -> Path:
    return root / metric.replace("@", "_at_")


def _load_scores(run_dir: Path, metric: str, strat: str) -> np.ndarray:
    path = _metric_dir(run_dir, metric) / f"per_subset_{strat}.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing per-subset scores: {path}")
    return np.load(path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--a", type=Path, required=True,
                   help="s07 output dir for model A.")
    p.add_argument("--b", type=Path, required=True,
                   help="s07 output dir for model B.")
    p.add_argument("--label_a", type=str, default="A")
    p.add_argument("--label_b", type=str, default="B")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--delta", type=float, default=0.05,
                   help="Paired EB two-sided miscoverage level (default 0.05 = 95% CI).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.a = _resolve(args.a)
    args.b = _resolve(args.b)
    args.out_dir = _resolve(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load both summaries for metadata consistency check.
    sum_a = json.loads((args.a / "summary.json").read_text())
    sum_b = json.loads((args.b / "summary.json").read_text())

    # Hard constraint: same subset configuration (otherwise pairing is invalid).
    for key in ("n_subsets", "subset_size", "seed", "strat_cols"):
        if sum_a.get(key) != sum_b.get(key):
            raise ValueError(
                f"runs A and B differ on '{key}' "
                f"(A={sum_a.get(key)}, B={sum_b.get(key)}). "
                "Re-run both with the same --cache_subsets to align subsets."
            )

    print(f"[s08] A: {args.label_a} @ {args.a}")
    print(f"[s08] B: {args.label_b} @ {args.b}")
    print(f"[s08] n_subsets={sum_a['n_subsets']} "
          f"subset_size={sum_a['subset_size']} seed={sum_a['seed']}")
    print(f"[s08] strat_cols={sum_a['strat_cols']}")
    print(f"[s08] paired-EB delta={args.delta} → "
          f"{int(100*(1-args.delta))}% CI")

    report: dict = {
        "label_a": args.label_a, "label_b": args.label_b,
        "a_dir": str(args.a), "b_dir": str(args.b),
        "n_subsets": sum_a["n_subsets"],
        "subset_size": sum_a["subset_size"],
        "seed": sum_a["seed"],
        "strat_cols": sum_a["strat_cols"],
        "delta": args.delta,
        "comparisons": {},
    }

    for metric in METRICS:
        report["comparisons"][metric] = {}
        for strat in STRATEGIES:
            try:
                a = _load_scores(args.a, metric, strat)
                b = _load_scores(args.b, metric, strat)
            except FileNotFoundError as e:
                report["comparisons"][metric][strat] = {"error": str(e)}
                continue
            if a.shape != b.shape:
                report["comparisons"][metric][strat] = {
                    "error": f"shape mismatch a={a.shape}, b={b.shape}",
                }
                continue
            mask = ~(np.isnan(a) | np.isnan(b))
            d = (a - b)[mask]
            if d.size < 2:
                report["comparisons"][metric][strat] = {
                    "n": int(d.size), "mean_delta": float("nan"),
                    "ci_lo": float("nan"), "ci_hi": float("nan"),
                    "a_wins": False, "b_wins": False,
                }
                continue
            mean = float(d.mean())
            lo, hi = paired_empirical_bernstein_ci(d, delta=args.delta)
            entry = {
                "n": int(d.size),
                "mean_delta": mean,
                "ci_lo": float(lo),
                "ci_hi": float(hi),
                "mean_a": float(a[mask].mean()),
                "mean_b": float(b[mask].mean()),
                "a_wins": bool(lo > 0),
                "b_wins": bool(hi < 0),
            }
            report["comparisons"][metric][strat] = entry

            # per (metric, strategy) plots
            out_ms = args.out_dir / metric.replace("@", "_at_") / strat
            out_ms.mkdir(parents=True, exist_ok=True)
            plot_delta_distribution(
                d, lo, hi, mean,
                out_ms / "delta_hist.png",
                label_a=args.label_a, label_b=args.label_b,
                title=f"{metric} — {strat}",
            )
            plot_ab_cdf_overlay(
                a[mask], b[mask],
                out_ms / "cdf_overlay.png",
                label_a=args.label_a, label_b=args.label_b,
                title=f"{metric} — {strat} — CDF",
                metric_label=metric,
            )

    (args.out_dir / "comparison.json").write_text(json.dumps(report, indent=2))

    # forest plot across all comparisons
    plot_comparison_forest(
        report["comparisons"],
        args.out_dir / "forest_all.png",
        label_a=args.label_a,
        label_b=args.label_b,
    )

    # ── readable digest ────────────────────────────────────────────────────
    print()
    print(f"[s08] === comparison {args.label_a} vs {args.label_b} ===")
    for metric, by_strat in report["comparisons"].items():
        print(f"  {metric}:")
        for strat, e in by_strat.items():
            if "error" in e:
                print(f"    {strat:11s}: SKIP — {e['error']}")
                continue
            verdict = (
                f"{args.label_a} wins" if e["a_wins"] else
                f"{args.label_b} wins" if e["b_wins"] else
                "tie (CI straddles 0)"
            )
            print(
                f"    {strat:11s}: Δ(mean)={e['mean_delta']:+.4f} "
                f"CI=[{e['ci_lo']:+.4f}, {e['ci_hi']:+.4f}] "
                f"(A={e['mean_a']:.4f}, B={e['mean_b']:.4f}) → {verdict}"
            )

    print(f"\n[s08] saved → {args.out_dir}")
    print(f"        comparison.json, forest_all.png, "
          f"<metric>/<strat>/{{delta_hist,cdf_overlay}}.png")


if __name__ == "__main__":
    main()
