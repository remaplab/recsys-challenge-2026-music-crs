"""Backfill ``gt_track_id`` into already-exported one-shot CG datasets.

One-shot components (``export_oneshot_candidates.py``) historically wrote
candidate parquets WITHOUT a ``gt_track_id`` column, unlike the session CGs.
Downstream CG calibration (``cg_calibration._wide_to_long_with_gt``) filters on
``gt_track_id`` and crashes when it is absent. This patches the existing files in
place instead of re-running the (expensive) model refits.

GT for a ``(session_id, turn)`` is the track played at that ``turn_number`` in the
corresponding splitK parquet — the same mapping the exporter / RRF fuser use:

    fold_{k}_oof_cg_val.parquet        ← splitK/fold_{k}_cg_val.parquet
    fold_{k}_oof_reranker_val.parquet  ← splitK/fold_{k}_reranker_val.parquet
    holdout_candidates.parquet         ← splitK/holdout_test.parquet
    blind_candidates.parquet           ← (no GT — set null)

Only ``*_oneshot`` dataset dirs are touched; session CGs already carry GT.

Run:
    cd src/basic_candidate_generators
    uv run python -m launchers_dro_oneshot.patch_oneshot_gt
    uv run python -m launchers_dro_oneshot.patch_oneshot_gt --only dense_text_8b tower_a
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PKG_ROOT.parent.parent

import polars as pl    # noqa: E402


def _gt_map(gt_parquet: Path) -> pl.DataFrame:
    """(session_id, turn, gt_track_id) from a splitK long parquet (all turns)."""
    return (
        pl.read_parquet(gt_parquet, columns=["session_id", "turn_number", "track_id"])
        .select(
            "session_id",
            pl.col("turn_number").cast(pl.Int64).alias("turn"),
            pl.col("track_id").alias("gt_track_id"),
        )
        .unique(subset=["session_id", "turn"], keep="first")
    )


def _patch_file(ds_path: Path, gt_parquet: Path | None) -> str:
    """Join GT (or null) into one dataset parquet, overwriting it. Returns a
    short status string for logging."""
    df = pl.read_parquet(ds_path)
    if "gt_track_id" in df.columns:
        df = df.drop("gt_track_id")
    if gt_parquet is None:
        df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("gt_track_id"))
        n_gt = 0
    else:
        df = df.with_columns(pl.col("turn").cast(pl.Int64)).join(
            _gt_map(gt_parquet), on=["session_id", "turn"], how="left",
        )
        n_gt = df.filter(pl.col("gt_track_id").is_not_null()).height
    df.write_parquet(ds_path)
    return f"rows={df.height} gt={n_gt}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--storage_dir", type=Path,
                    default=_REPO_ROOT / "models" / "CG_crossvalidation")
    ap.add_argument("--splitk_dir", type=Path,
                    default=_REPO_ROOT / "data" / "splitK")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--only", nargs="*",
                    help="restrict to these model names (without the _oneshot suffix).")
    args = ap.parse_args()

    # filename → splitK GT parquet (None = blind, null GT).
    file_to_gt: dict[str, Path | None] = {}
    for k in range(args.n_folds):
        file_to_gt[f"fold_{k}_oof_cg_val.parquet"] = args.splitk_dir / f"fold_{k}_cg_val.parquet"
        file_to_gt[f"fold_{k}_oof_reranker_val.parquet"] = args.splitk_dir / f"fold_{k}_reranker_val.parquet"
    file_to_gt["holdout_candidates.parquet"] = args.splitk_dir / "holdout_test.parquet"
    file_to_gt["blind_candidates.parquet"] = None

    only = set(args.only) if args.only else None
    n_dirs = 0
    for model_dir in sorted(args.storage_dir.iterdir()):
        if not model_dir.is_dir() or not model_dir.name.endswith("_oneshot"):
            continue
        if only is not None and model_dir.name[: -len("_oneshot")] not in only:
            continue
        ds_dir = model_dir / "datasets"
        if not ds_dir.is_dir():
            continue
        touched = []
        for fname, gt_parquet in file_to_gt.items():
            ds_path = ds_dir / fname
            if not ds_path.exists():
                continue
            if gt_parquet is not None and not gt_parquet.exists():
                print(f"  [warn] {model_dir.name}/{fname}: missing GT {gt_parquet} — skip")
                continue
            status = _patch_file(ds_path, gt_parquet)
            touched.append(f"{fname}({status})")
        if touched:
            n_dirs += 1
            print(f"{model_dir.name:34s} {len(touched)} files: " + ", ".join(touched))

    print(f"[patch_oneshot_gt] patched {n_dirs} oneshot dataset dir(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
