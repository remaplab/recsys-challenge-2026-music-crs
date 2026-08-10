"""Repair CG dataset parquets whose `turn` column is constant (always the
session's max turn) and whose real per-row turn lives in `gt_turn_number`.

Symptom (observed on `split_hidim_xattn_hardneg_query_session`):
    turn                  i64  → always 8
    gt_turn_number        i64  → correct (1..8)
    fallback_used         i64  → extra column not in the canonical schema

Canonical schema (matches `heuristic_session`, `item_knn_session`, ...):
    session_id (str), user_id (str), turn (i64),
    track_ids (list[str]), scores (list[f64]), gt_track_id (str)

Fix per parquet:
    * If `gt_turn_number` exists → overwrite `turn` with it and drop the column.
    * Optionally drop `fallback_used` (with `--drop_fallback_used`).
    * If `turn` is already non-constant AND there is no `gt_turn_number` the
      file is considered healthy and skipped.

Usage (paths are always relative to repository root):
    cd src/basic_candidate_generators
    uv run python -u -m launchers_crossvalidation.fix_dataset_columns \
        --path models/CG_crossvalidation/split_hidim_xattn_hardneg_query_session/datasets

    # Dry-run is the default. Add --apply to write changes.
    uv run python -u -m launchers_crossvalidation.fix_dataset_columns \
        --path models/CG_crossvalidation/split_hidim_xattn_hardneg_query_session/datasets \
        --apply --drop_fallback_used
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import polars as pl


# Repo root = parents[3] from src/basic_candidate_generators/launchers_crossvalidation/...
_REPO_ROOT = Path(__file__).resolve().parents[3]

CANONICAL_COLS = (
    "session_id", "user_id", "turn", "track_ids", "scores", "gt_track_id",
)


def _resolve(p: str) -> Path:
    """Resolve `p` against repo root unless it's already absolute."""
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return _REPO_ROOT / pp


def _diagnose(df: pl.DataFrame) -> dict:
    """Return a small dict describing what (if anything) is wrong."""
    has_gt_turn = "gt_turn_number" in df.columns
    has_turn = "turn" in df.columns
    has_fallback = "fallback_used" in df.columns
    turn_distinct = (
        df["turn"].n_unique() if has_turn else 0
    )
    turn_min = int(df["turn"].min()) if has_turn and df.height else None
    turn_max = int(df["turn"].max()) if has_turn and df.height else None
    return {
        "n_rows": df.height,
        "columns": df.columns,
        "has_gt_turn_number": has_gt_turn,
        "has_fallback_used": has_fallback,
        "turn_n_distinct": int(turn_distinct),
        "turn_min": turn_min,
        "turn_max": turn_max,
        "broken_constant_turn": has_gt_turn and turn_distinct == 1,
        "broken_schema_only": has_gt_turn and turn_distinct > 1,
    }


def _apply_fix(
    df: pl.DataFrame, *, drop_fallback_used: bool,
) -> pl.DataFrame:
    out = df
    if "gt_turn_number" in out.columns:
        # Replace `turn` with the row-correct value and drop the helper col.
        out = out.with_columns(
            pl.col("gt_turn_number").alias("turn")
        ).drop("gt_turn_number")
    if drop_fallback_used and "fallback_used" in out.columns:
        out = out.drop("fallback_used")
    # Reorder canonical columns first, then any remaining (informational).
    ordered = [c for c in CANONICAL_COLS if c in out.columns]
    extras = [c for c in out.columns if c not in ordered]
    return out.select(ordered + extras)


def _short_cols(cols: list[str]) -> str:
    s = ", ".join(cols)
    return s if len(s) < 110 else s[:107] + "..."


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--path", required=True,
        help="Folder of parquets to inspect (path relative to repository root, "
             "e.g. models/CG_crossvalidation/<cg>/datasets).",
    )
    p.add_argument(
        "--apply", action="store_true",
        help="Write fixed parquets in place (default: dry-run / report only).",
    )
    p.add_argument(
        "--no_backup", action="store_true",
        help="Skip the .bak backup before overwriting (default: backup is kept).",
    )
    p.add_argument(
        "--drop_fallback_used", action="store_true",
        help="Also drop the `fallback_used` column to match the canonical schema.",
    )
    p.add_argument(
        "--glob", default="*.parquet",
        help="Glob pattern under --path (default *.parquet).",
    )
    args = p.parse_args()

    target = _resolve(args.path)
    if not target.exists():
        print(f"[fix] path not found: {target}", file=sys.stderr)
        return 2
    if target.is_file():
        files = [target]
    else:
        files = sorted(target.glob(args.glob))
    if not files:
        print(f"[fix] no parquets found under {target} (glob={args.glob!r})")
        return 0

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[fix] mode={mode}  root={_REPO_ROOT}")
    print(f"[fix] target={target}  files={len(files)}")
    print()

    n_broken = 0
    n_healthy = 0
    n_unknown = 0
    for fp in files:
        try:
            df = pl.read_parquet(fp)
        except Exception as e:                                       # noqa: BLE001
            print(f"  [skip] {fp.name}: failed to read ({e})")
            n_unknown += 1
            continue
        diag = _diagnose(df)
        tag = (
            "BROKEN(constant_turn)" if diag["broken_constant_turn"]
            else "BROKEN(schema_only)" if diag["broken_schema_only"]
            else "ok"
        )
        print(f"- {fp.relative_to(target.parent) if target.is_dir() else fp.name}")
        print(f"    rows={diag['n_rows']:>8}  turn_n_distinct={diag['turn_n_distinct']:>2} "
              f"(min={diag['turn_min']}, max={diag['turn_max']})  "
              f"has_gt_turn_number={diag['has_gt_turn_number']}  "
              f"has_fallback_used={diag['has_fallback_used']}")
        print(f"    schema: {_short_cols(diag['columns'])}")

        if not (diag["broken_constant_turn"] or diag["broken_schema_only"]):
            print(f"    -> {tag}")
            n_healthy += 1
            continue

        fixed = _apply_fix(df, drop_fallback_used=args.drop_fallback_used)
        diag_after = _diagnose(fixed)
        print(f"    after fix: turn_n_distinct={diag_after['turn_n_distinct']}  "
              f"(min={diag_after['turn_min']}, max={diag_after['turn_max']})")
        print(f"    after fix schema: {_short_cols(diag_after['columns'])}")
        print(f"    -> {tag}")
        n_broken += 1

        if args.apply:
            if not args.no_backup:
                bak = fp.with_suffix(fp.suffix + ".bak")
                if not bak.exists():
                    shutil.copy2(fp, bak)
                    print(f"    backup -> {bak.name}")
                else:
                    print(f"    backup already exists ({bak.name}), not overwritten")
            fixed.write_parquet(fp)
            print(f"    [written] {fp.name}")
        print()

    print()
    print(f"[fix] SUMMARY  broken={n_broken}  healthy={n_healthy}  unreadable={n_unknown}")
    if not args.apply and n_broken:
        print("[fix] re-run with --apply to write changes "
              "(originals saved as <name>.parquet.bak unless --no_backup).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
