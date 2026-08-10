"""Evaluate a CG model turn-by-turn on fold 0 cg_val.

Trains on fold 0 cg_train (80%), evaluates on fold 0 cg_val (16%),
and reports ndcg@20, recall@20, recall@200 broken down by target turn.

Reads best hyperparameters from configs/cv_best_{model}_{urm_mode}_{metric}{K}.yaml
(same lookup logic as retrain_and_export.py).

Output:
  - Formatted table printed to stdout
  - CSV saved to {storage_dir}/{model}_{urm_mode}/turn_metrics_{objective}{K}.csv

Usage:
    cd src/basic_candidate_generators

    uv run python -m launchers_crossvalidation.eval_by_turn \\
        --model session_knn --urm_mode session

    uv run python -m launchers_crossvalidation.eval_by_turn \\
        --model session_knn --urm_mode session \\
        --objective recall --objective_k 200
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _PKG_ROOT / "src"
_CV_ROOT  = _PKG_ROOT / "launchers_crossvalidation"
sys.path.insert(0, str(_SRC_ROOT))
sys.path.insert(0, str(_CV_ROOT))

import polars as pl   # noqa: E402
import yaml           # noqa: E402

from _cv_utils import (    # noqa: E402
    _per_row_metric,
    instantiate_rec,
    load_eval,
    load_fold,
    pkg_path,
    repo_path,
    resolve_param_paths,
    run_inference_dispatch,
)

_TRACK_META_PATH = (
    "data/talkpl-ai/TalkPlayData-Challenge-Track-Metadata/"
    "data/all_tracks-00000-of-00001.parquet"
)
_MODELS_NO_META = {"gf_cf"}

_EVAL_METRICS = [
    ("ndcg",   20),
    ("recall", 20),
    ("recall", 200),
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a CG model turn-by-turn on fold 0 cg_val."
    )
    p.add_argument("--model",       required=True, help="Model key (must match config filename)")
    p.add_argument("--urm_mode",    required=True, choices=["session", "user"])
    p.add_argument("--config",      default=None,
                   help="Explicit path to a cv_best YAML (overrides auto-discovery).")
    p.add_argument("--top_k",       type=int, default=200,
                   help="Candidates per session (must be >= max K in metrics, i.e. 200)")
    p.add_argument("--storage_dir", default="models/CG_crossvalidation",
                   help="Repo-root relative root dir for output CSV")
    p.add_argument("--splitk_dir",  default="data/splitK")
    p.add_argument("--objective",   choices=("ndcg", "recall"), default="ndcg",
                   help="Objective tag of the cv_best YAML to load.")
    p.add_argument("--objective_k", type=int, default=None,
                   help="K for the objective. Defaults: ndcg=20, recall=200.")
    return p.parse_args()


_OBJECTIVE_DEFAULT_K = {"ndcg": 20, "recall": 200}


# ---------------------------------------------------------------------------
# Per-turn metric computation
# ---------------------------------------------------------------------------

def _compute_turn_table(recs: pl.DataFrame) -> list[dict]:
    """Return one dict per gt_turn_number with n, ndcg@20, recall@20, recall@200."""
    buckets: dict[int, dict[str, list[float]]] = {}
    col_keys = [f"{m}@{k}" for m, k in _EVAL_METRICS]

    for row in recs.iter_rows(named=True):
        if row["gt_track_id"] is None:
            continue
        t = int(row.get("gt_turn_number") or 0)
        if t not in buckets:
            buckets[t] = {col: [] for col in col_keys}
        for metric, k in _EVAL_METRICS:
            v = _per_row_metric(row, metric, k)
            if v is not None:
                buckets[t][f"{metric}@{k}"].append(v)

    rows = []
    for t in sorted(buckets.keys()):
        b = buckets[t]
        n = len(b[col_keys[0]])
        rows.append({
            "turn": t,
            "n": n,
            **{col: (sum(b[col]) / len(b[col]) if b[col] else 0.0) for col in col_keys},
        })
    return rows


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _print_table(rows: list[dict]) -> None:
    header = f"{'turn':>6} | {'n':>6} | {'ndcg@20':>10} | {'recall@20':>10} | {'recall@200':>10}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['turn']:>6} | {r['n']:>6} | "
            f"{r['ndcg@20']:>10.4f} | {r['recall@20']:>10.4f} | {r['recall@200']:>10.4f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    objective_k = args.objective_k or _OBJECTIVE_DEFAULT_K[args.objective]
    suffix = f"_{args.objective}{objective_k}"
    folder_key = f"{args.model}_{args.urm_mode}"

    # Resolve config path
    if args.config:
        cfg_path = Path(args.config)
    else:
        tagged = pkg_path(f"configs/cv_best_{folder_key}{suffix}.yaml")
        bare   = pkg_path(f"configs/cv_best_{folder_key}.yaml")
        cfg_path = tagged if tagged.exists() else bare
    if not cfg_path.exists():
        sys.exit(f"[eval_by_turn] Config not found: {cfg_path}\nRun extract_best_params.py first.")

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    class_name     = cfg["class"]
    module_name    = cfg["module"]
    best_params    = resolve_param_paths(
        {**(cfg.get("fixed_params") or {}), **(cfg.get("best_params") or {})}
    )
    inference_mode = cfg.get("inference_mode", "standard")

    splitk_dir = repo_path(args.splitk_dir)
    out_dir    = repo_path(args.storage_dir) / folder_key
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[eval_by_turn] model:     {args.model} ({class_name})")
    print(f"[eval_by_turn] urm_mode:  {args.urm_mode}")
    print(f"[eval_by_turn] inference: {inference_mode}")
    print(f"[eval_by_turn] top_k:     {args.top_k}")
    print(f"[eval_by_turn] fold:      0  (cg_train → cg_val)")

    if args.model in _MODELS_NO_META:
        track_meta = None
    else:
        print("\nLoading track metadata...")
        track_meta = pl.read_parquet(repo_path(_TRACK_META_PATH))

    print("\nLoading fold 0 data...")
    train_df = load_fold(splitk_dir, fold=0, parts=["cg_train"])
    eval_df  = load_eval(splitk_dir, fold=0)
    print(f"  train: {train_df.shape[0]} rows  |  eval: {eval_df.shape[0]} rows")

    print("\nFitting...")
    t0  = time.time()
    rec = instantiate_rec(class_name, module_name, best_params, args.urm_mode)
    rec.fit(train_df, track_metadata=track_meta)
    print(f"  done in {time.time() - t0:.1f}s")

    print("\nRunning inference (all turns)...")
    t0   = time.time()
    recs = run_inference_dispatch(rec, eval_df, args.top_k, inference_mode, track_meta)
    print(f"  {recs.shape[0]} rows in {time.time() - t0:.1f}s")

    rows = _compute_turn_table(recs)

    print("\n")
    _print_table(rows)

    csv_name = f"turn_metrics_{args.objective}{objective_k}.csv"
    csv_path = out_dir / csv_name
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["turn", "n", "ndcg@20", "recall@20", "recall@200"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[eval_by_turn] saved → {csv_path}")


if __name__ == "__main__":
    main()
