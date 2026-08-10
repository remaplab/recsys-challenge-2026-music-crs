"""s07b — run s07 (blind-like subset eval) for every CG, then aggregate.

Discovers every candidate generator under `--cg_root` that exposes
`datasets/holdout_candidates.parquet`, evaluates each on the holdout GT via
`s07_eval_on_blind_like_subsets` (one shared subset cache → identical subsets →
cross-CG comparable + s08-paired-valid), and writes a combined leaderboard.

`*_oneshot` CGs emit turn-1 candidates only, so they are scored with
`--target_turn 1` (every session pinned to its turn-1 GT). All other CGs are
scored across turns (max_turn sampled from the blind multiturn PMF) as usual.

Outputs under `--out_dir` (default models/LowerBoundOptimization/CG_analysis/):
    <cg>/                 full s07 output (per-subset arrays, plots, summary.json)
    _subset_cache/<gt>.npz   shared subset cache
    leaderboard.csv       long-format: one row per (cg, metric, strategy)
    leaderboard.md        ranked tables per metric for the primary strategy

Run:
    PYTHONPATH=src uv run python -m launchers.s07b_eval_all_cgs
    PYTHONPATH=src uv run python -m launchers.s07b_eval_all_cgs --only bm25_oneshot heuristic_session_dro
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PKG_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import polars as pl

from lbo.paths import ROOT

_S07_MODULE = "launchers.s07_eval_on_blind_like_subsets"
_METRICS = ("ndcg@20", "recall@200")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cg_root", type=Path, default=ROOT / "models/CG_crossvalidation")
    p.add_argument("--out_dir", type=Path,
                   default=ROOT / "models/LowerBoundOptimization/CG_analysis")
    p.add_argument("--gt_path", type=Path, default=ROOT / "data/splitK/holdout_test.parquet")
    p.add_argument("--n_subsets", type=int, default=2000)
    p.add_argument("--subset_size", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cvar_alpha", type=float, default=0.7)
    p.add_argument("--only", type=str, nargs="+", default=None,
                   help="Restrict to these CG names (dir names under cg_root).")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip CGs that already have a summary.json in out_dir.")
    p.add_argument("--primary_metric", type=str, default="ndcg@20", choices=_METRICS)
    p.add_argument("--primary_strategy", type=str, default="calibrated")
    p.add_argument("--aggregate_only", action="store_true",
                   help="Skip eval; just rebuild leaderboard from existing summaries.")
    return p.parse_args()


def discover_cgs(cg_root: Path, only: list[str] | None) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for recs in sorted(cg_root.glob("*/datasets/holdout_candidates.parquet")):
        cg = recs.parents[1].name
        if only and cg not in only:
            continue
        out.append((cg, recs))
    return out


def run_one(cg: str, recs: Path, args: argparse.Namespace, cache: Path) -> bool:
    out_dir = args.out_dir / cg
    if args.skip_existing and (out_dir / "summary.json").exists():
        print(f"[s07b] skip {cg} (summary.json exists)")
        return True
    cmd = [
        sys.executable, "-m", _S07_MODULE,
        "--gt_path", str(args.gt_path),
        "--recs_path", str(recs),
        "--out_dir", str(out_dir),
        "--label", cg,
        "--n_subsets", str(args.n_subsets),
        "--subset_size", str(args.subset_size),
        "--seed", str(args.seed),
        "--cvar_alpha", str(args.cvar_alpha),
        "--cache_subsets", str(cache),
    ]
    if cg.endswith("_oneshot"):
        cmd += ["--target_turn", "1"]
    print(f"[s07b] ▶ {cg}{'  (turn-1)' if cg.endswith('_oneshot') else ''}")
    r = subprocess.run(cmd, cwd=str(_PKG_ROOT))
    if r.returncode != 0:
        print(f"[s07b] ✗ {cg} failed (exit {r.returncode}) — skipping in leaderboard")
        return False
    return True


def aggregate(cgs: list[str], args: argparse.Namespace) -> None:
    rows: list[dict] = []
    for cg in cgs:
        sp = args.out_dir / cg / "summary.json"
        if not sp.exists():
            continue
        s = json.loads(sp.read_text())
        for metric, ms in s.get("metrics", {}).items():
            for strat, d in ms.items():
                if not d or d.get("n_valid", 0) == 0:
                    continue
                rows.append({
                    "cg": cg,
                    "target_turn": s.get("target_turn"),
                    "metric": metric,
                    "strategy": strat,
                    "mean": d.get("mean"),
                    "std": d.get("std"),
                    "median": d.get("median"),
                    "ci95_lo": d.get("ci95_lo"),
                    "ci95_hi": d.get("ci95_hi"),
                    "cvar": d.get("cvar_score"),
                    "kl_sum": d.get("kl_marginal_sum"),
                    "n_valid": d.get("n_valid"),
                    "eval_sessions": s.get("eval_sessions"),
                })
    if not rows:
        print("[s07b] no summaries found — nothing to aggregate.")
        return
    df = pl.DataFrame(rows)
    csv_path = args.out_dir / "leaderboard.csv"
    df.write_csv(csv_path)
    print(f"[s07b] wrote {csv_path} ({df.height} rows)")

    # markdown: one ranked table per metric for the primary strategy
    lines: list[str] = [
        f"# CG blind-like leaderboard ({args.primary_strategy} strategy)",
        "",
        f"GT: `{args.gt_path}` · n_subsets={args.n_subsets} · "
        f"subset_size={args.subset_size} · CVaR α={args.cvar_alpha}",
        "",
        "`*_oneshot` CGs scored at turn 1; all others across turns "
        "(blind multiturn PMF). Ranked by CVaR (worst-tail mean).",
        "",
    ]
    for metric in _METRICS:
        sub = (df.filter((pl.col("metric") == metric)
                         & (pl.col("strategy") == args.primary_strategy))
                 .sort("cvar", descending=True))
        if sub.height == 0:
            continue
        lines += [f"## {metric}", "",
                  "| # | CG | turn | mean | CVaR | CI95 | n |",
                  "|--:|----|:----:|-----:|-----:|------|--:|"]
        for i, r in enumerate(sub.iter_rows(named=True), 1):
            turn = r["target_turn"] if r["target_turn"] is not None else "all"
            ci = f"[{r['ci95_lo']:.4f}, {r['ci95_hi']:.4f}]"
            lines.append(
                f"| {i} | {r['cg']} | {turn} | {r['mean']:.4f} | "
                f"{r['cvar']:.4f} | {ci} | {r['n_valid']} |")
        lines.append("")
    md_path = args.out_dir / "leaderboard.md"
    md_path.write_text("\n".join(lines))
    print(f"[s07b] wrote {md_path}")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Cache is keyed by path only (the evaluator ignores n_subsets on load), so
    # encode the sampling params into the filename — otherwise a stale cache
    # with a different n_subsets is silently reused.
    cache = (args.out_dir / "_subset_cache"
             / f"{args.gt_path.stem}_n{args.n_subsets}_s{args.subset_size}_seed{args.seed}.npz")
    cache.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    found = discover_cgs(args.cg_root, args.only)
    if not found:
        sys.exit(f"[s07b] no CGs with datasets/holdout_candidates.parquet under {args.cg_root}")
    print(f"[s07b] {len(found)} CGs: {[c for c, _ in found]}")

    if not args.aggregate_only:
        for cg, recs in found:
            run_one(cg, recs, args, cache)

    aggregate([c for c, _ in found], args)
    print(f"[s07b] done in {time.time() - t0:.1f}s. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
