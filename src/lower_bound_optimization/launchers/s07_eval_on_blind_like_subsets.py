"""s07 — CLI wrapper around `BlindLikeEvaluator`.

Evaluates a model's recommendations on Blind-A-like subsets of any GT split.
All real work lives in `lbo.evaluator.BlindLikeEvaluator` — this launcher only
parses args, loads the recs parquet, calls `score_all()`, writes per-subset
arrays + summary.json.

Run on a single GT file:
    PYTHONPATH=src uv run python -m launchers.s07_eval_on_blind_like_subsets \\
        --gt_path data/splitK/holdout_test.parquet \\
        --recs_path models/.../recs.parquet \\
        --out_dir models/.../eval/holdout_test/<model> \\
        --cache_subsets models/.../eval_cache/holdout_test.npz

Loop across all splitK CV val folds + holdout:
    for f in data/splitK/fold_*_val.parquet data/splitK/holdout_test.parquet; do
        label=$(basename "$f" .parquet)
        uv run python -m launchers.s07_eval_on_blind_like_subsets \\
            --gt_path "$f" --recs_path recs.parquet \\
            --out_dir "exp/eval/${label}/<model>" \\
            --cache_subsets "models/.../eval_cache/${label}.npz"
    done
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PKG_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import polars as pl

from lbo.paths import BLIND_PARQUET, ROOT, SHIFT_V2_OUT, TRACKS_META
from lbo.evaluator import (
    DEFAULT_STRAT_COLS,
    DEFAULT_STRATEGIES,
    BlindLikeEvaluator,
)
from lbo.evaluator.plots import (
    plot_ci_summary,
    plot_cdf,
    plot_distribution,
    plot_distribution_overlay,
    plot_violin_by_strategy,
)


METRICS = ("ndcg@20", "recall@200")


def _resolve(p: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p).resolve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--gt_path", type=Path, required=True,
                   help="splitK-schema parquet (CV-fold val or holdout_test).")
    p.add_argument("--recs_path", type=Path, required=True,
                   help="Model recommendations: cols (session_id, "
                        "{turn|turn_number}, track_ids: list[str], scores: list[float]).")
    p.add_argument("--density_ratio", type=Path,
                   default=SHIFT_V2_OUT / "density_ratio.parquet")
    p.add_argument("--blind_parquet", type=Path, default=BLIND_PARQUET)
    p.add_argument("--tracks_meta", type=Path, default=TRACKS_META)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--label", type=str, default="model")
    p.add_argument("--n_subsets", type=int, default=2000)
    p.add_argument("--subset_size", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cvar_alpha", type=float, default=0.7)
    p.add_argument("--target_turn", type=int, default=None,
                   help="Force every session's scored turn to this value "
                        "(e.g. 1 for turn-1-only/one-shot CGs). Subset SAMPLING "
                        "is independent of the target turn, so the subset cache "
                        "stays shared/comparable across runs. Omit = sample "
                        "max_turn from the blind multiturn PMF (default).")
    p.add_argument("--strat_cols", type=str, nargs="+",
                   default=list(DEFAULT_STRAT_COLS),
                   help=f"Default: {list(DEFAULT_STRAT_COLS)}")
    p.add_argument("--strategies", type=str, nargs="+",
                   default=list(DEFAULT_STRATEGIES),
                   choices=["calibrated", "density", "stratified", "blend"])
    p.add_argument("--cache_subsets", type=Path, default=None,
                   help="Path to .npz cache; shared cache across runs → "
                        "identical subsets → s08 paired-EB comparison valid.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Resolve every path against repo root (so launcher works from any CWD).
    args.gt_path = _resolve(args.gt_path)
    args.recs_path = _resolve(args.recs_path)
    args.density_ratio = _resolve(args.density_ratio)
    args.blind_parquet = _resolve(args.blind_parquet)
    args.tracks_meta = _resolve(args.tracks_meta)
    args.out_dir = _resolve(args.out_dir)
    if args.cache_subsets is not None:
        args.cache_subsets = _resolve(args.cache_subsets)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"[s07] gt_path:        {args.gt_path}")
    print(f"[s07] recs_path:      {args.recs_path}")
    print(f"[s07] density_ratio:  {args.density_ratio}")
    print(f"[s07] strat_cols:     {args.strat_cols}")
    print(f"[s07] strategies:     {args.strategies}")

    # ── instantiate evaluator (loads features + density ratio + prepares
    #    subsets once) ────────────────────────────────────────────────────
    ev = BlindLikeEvaluator(
        gt_path=args.gt_path,
        blind_parquet=args.blind_parquet,
        tracks_meta=args.tracks_meta,
        density_ratio=args.density_ratio,
        n_subsets=args.n_subsets,
        subset_size=args.subset_size,
        seed=args.seed,
        strat_cols=args.strat_cols,
        strategies=args.strategies,
        cache_path=args.cache_subsets,
        cvar_alpha=args.cvar_alpha,
    )

    # ── force the scored turn (one-shot CGs) ────────────────────────────────
    # The wrapper samples a per-session max_turn from the blind multiturn PMF;
    # recs/GT are joined at target_turn = max_turn + 1. For a turn-`t`-only CG
    # we pin every session's target turn to `t` (max_turn = t-1). Subset
    # SAMPLING is independent of max_turn, so this is safe after a cache load
    # and keeps subsets identical (hence comparable) across CGs.
    if args.target_turn is not None:
        ev.evaluator._prep.max_turn_per_row[:] = args.target_turn - 1
        print(f"[s07] target_turn forced to {args.target_turn}")

    # ── load recs ───────────────────────────────────────────────────────────
    recs = pl.read_parquet(args.recs_path)
    print(f"[s07] recs rows: {recs.height}")

    # ── score each metric across all strategies ───────────────────────────
    full_summary: dict = {
        "label": args.label,
        "gt_path": str(args.gt_path),
        "recs_path": str(args.recs_path),
        "density_ratio": str(args.density_ratio),
        "n_subsets": args.n_subsets,
        "subset_size": args.subset_size,
        "seed": args.seed,
        "target_turn": args.target_turn,
        "strat_cols": list(args.strat_cols),
        "strategies": list(args.strategies),
        "cache_subsets": str(args.cache_subsets) if args.cache_subsets else None,
        "eval_sessions": ev.eval_features.height,
        "blind_sessions": ev.blind_features.height,
        "density_coverage": (
            float(ev.density_coverage) if ev.density_coverage is not None else None
        ),
        "metrics": {},
    }

    for metric in METRICS:
        print(f"[s07] evaluating {metric}…")
        out_metric = args.out_dir / metric.replace("@", "_at_")
        out_metric.mkdir(parents=True, exist_ok=True)
        results = ev.score_all(recs, metric=metric)
        metric_summary: dict[str, dict] = {}
        for strat, r in results.items():
            np.save(out_metric / f"per_subset_{strat}.npy", r.per_subset)
            metric_summary[strat] = r.to_dict()
            # per-strategy histogram — CIs as empirical quantiles of per_subset
            clean = r.per_subset[~np.isnan(r.per_subset)]
            if clean.size == 0:
                continue
            s = r.to_dict()
            for ci_key, alpha in (("ci90", 0.90), ("ci95", 0.95), ("ci99", 0.99)):
                tail = (1.0 - alpha) / 2.0
                s[ci_key] = [
                    float(np.quantile(clean, tail)),
                    float(np.quantile(clean, 1.0 - tail)),
                ]
            plot_distribution(
                r.per_subset, s,
                out_metric / f"hist_{strat}.png",
                title=f"{args.label} — {metric} ({strat})",
                bins=50,
            )
        full_summary["metrics"][metric] = metric_summary

        # multi-strategy plots (only when ≥1 strategy scored)
        if results:
            scores_map = {strat: r.per_subset for strat, r in results.items()}
            summary_map = {
                strat: {
                    "mean": r.mean, "std": r.std,
                    "ci95_lo": r.ci95_lo, "ci95_hi": r.ci95_hi,
                    "median": r.median,
                }
                for strat, r in results.items()
            }
            plot_distribution_overlay(
                scores_map, {s: {"mean": r.mean} for s, r in results.items()},
                out_metric / "hist_overlay.png",
                title=f"{args.label} — {metric} — strategy overlay",
            )
            plot_violin_by_strategy(
                scores_map, summary_map,
                out_metric / "violin_by_strategy.png",
                title=f"{args.label} — {metric} — violin by strategy",
                metric_label=metric,
            )
            plot_cdf(
                scores_map,
                out_metric / "cdf_by_strategy.png",
                title=f"{args.label} — {metric} — CDF by strategy",
                metric_label=metric,
            )
            plot_ci_summary(
                summary_map,
                out_metric / "ci_summary.png",
                title=f"{args.label} — {metric} — mean ± CI95",
                metric_label=metric,
            )

    (args.out_dir / "summary.json").write_text(json.dumps(full_summary, indent=2))

    # ── readable digest ────────────────────────────────────────────────────
    print()
    print(f"[s07] === summary ({args.label}) ===")
    for metric, ms in full_summary["metrics"].items():
        print(f"  {metric}:")
        for strat, s in ms.items():
            if s.get("n_valid", 0) == 0:
                print(f"    {strat:11s}: (empty)")
                continue
            kl = s.get("kl_marginal_sum", 0.0)
            print(
                f"    {strat:11s}: mean={s['mean']:.4f} "
                f"CI95=[{s['ci95_lo']:.4f}, {s['ci95_hi']:.4f}] "
                f"CVaR_{args.cvar_alpha:.1f}={s['cvar_score']:.4f} "
                f"KL_sum={kl:.3f}"
            )

    print(f"\n[s07] done in {time.time() - t0:.1f}s. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
