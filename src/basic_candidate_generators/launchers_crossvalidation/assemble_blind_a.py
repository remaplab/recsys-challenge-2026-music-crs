"""assemble_blind_a — gather ALL-turn candidates for the 80 Blind-A sessions.

TEMPORARY fast path. Blind-A is mixed into splitK, so each Blind-A session's
*history* turns already appear (with a real ``gt_track_id``) inside every CG's
OOF fold candidate parquets. The *submission* turn (the withheld final turn,
``gt_track_id`` null) lives in that CG's ``blind_candidates.parquet``.

For each CG under ``models/CG_crossvalidation`` this script unions:

    history turns  = Blind-A rows from fold_{k}_oof_{cg_val,reranker_val}.parquet
    submission turn= blind_candidates.parquet

→ ``<cg>/datasets/blind_a_all_turns_candidates.parquet`` (same schema:
``session_id, user_id, turn, track_ids, scores, gt_track_id``). History rows win
on a (session, turn) collision because they carry the label.

The proper general path (new blinds, blind-B) is each CG folder's
``export_blind.py``; this script only exists because Blind-A is already in the
candidate store.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]
CG_STORE = ROOT / "models" / "CG_crossvalidation"
BLIND_A = ROOT / "data/talkpl-ai/TalkPlayData-Challenge-Blind-A/data/test-00000-of-00001.parquet"
OUT_NAME = "blind_a_all_turns_candidates.parquet"
N_FOLDS = 5
KEY = ["session_id", "turn"]
CANON = ["session_id", "user_id", "turn", "track_ids", "scores", "gt_track_id"]


# Null dtype to inject when a canonical column is absent from a CG parquet.
_CANON_NULL_DTYPE = {
    "session_id": pl.Utf8, "user_id": pl.Utf8, "turn": pl.Int64,
    "track_ids": pl.List(pl.Utf8), "scores": pl.List(pl.Float64), "gt_track_id": pl.Utf8,
}


def _canon(df: pl.DataFrame) -> pl.DataFrame:
    """Coerce to the canonical schema: keep CANON cols in order, backfilling a
    typed null for any column a given CG parquet omits (some lack user_id /
    gt_track_id)."""
    missing = [c for c in CANON if c not in df.columns]
    if missing:
        df = df.with_columns([pl.lit(None, dtype=_CANON_NULL_DTYPE[c]).alias(c) for c in missing])
    return df.select(CANON)


def _skip(name: str) -> bool:
    """Individual one-shot CGs are consumed only through their aggregator
    ``rrf_oneshot`` (which exports all turns), so they never feed the reranker
    roster — skip them. Everything else (incl. rrf_oneshot) is kept."""
    return name.endswith("_oneshot") and name != "rrf_oneshot"


def _cand_dir(model_dir: Path) -> Path | None:
    """Resolve where a CG keeps its candidate parquets (``datasets/`` or root)."""
    for cd in (model_dir / "datasets", model_dir):
        if (cd / "blind_candidates.parquet").exists():
            return cd
    return None


def _history(cand_dir: Path, aids: list[str], n_folds: int) -> pl.DataFrame:
    parts = []
    for k in range(n_folds):
        for s in ("cg_val", "reranker_val"):
            p = cand_dir / f"fold_{k}_oof_{s}.parquet"
            if not p.exists():
                continue
            d = pl.read_parquet(p).filter(pl.col("session_id").is_in(aids))
            if d.height:
                parts.append(d)
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts, how="vertical_relaxed").unique(subset=KEY, keep="first")


def _gt_map(cg_store: Path, aids: list[str], n_folds: int) -> pl.DataFrame:
    """Master ``(session_id, turn) → gt_track_id`` map. GT is a property of the
    blind session/turn, not the CG, so any CG that has it can supply the label
    for CGs (one-shot) whose candidate parquets carry null gt."""
    want = {*KEY, "gt_track_id"}
    parts = []
    for model_dir in sorted(cg_store.iterdir()):
        if not model_dir.is_dir():
            continue
        cand_dir = _cand_dir(model_dir)
        if cand_dir is None:
            continue
        for k in range(n_folds):
            for s in ("cg_val", "reranker_val"):
                p = cand_dir / f"fold_{k}_oof_{s}.parquet"
                if not p.exists():
                    continue
                # Column projection: only the 3 GT columns are read from the
                # parquet — the big track_ids/scores list columns are skipped, so
                # this is IO-cheap even across all CGs (vs reading full files).
                lf = pl.scan_parquet(p)
                if not want.issubset(set(lf.collect_schema().names())):
                    continue
                d = (lf.select(*KEY, "gt_track_id")
                       .filter(pl.col("session_id").is_in(aids)
                               & pl.col("gt_track_id").is_not_null())
                       .collect())
                if d.height:
                    parts.append(d)
    if not parts:
        return pl.DataFrame(schema={"session_id": pl.Utf8, "turn": pl.Int64, "gt_track_id": pl.Utf8})
    return pl.concat(parts, how="vertical_relaxed").unique(subset=KEY, keep="first")


def _assemble_one(cand_dir: Path, aids: list[str], n_folds: int, gt_map: pl.DataFrame) -> pl.DataFrame:
    hist = _history(cand_dir, aids, n_folds)
    sub = _canon(pl.read_parquet(cand_dir / "blind_candidates.parquet")
                 .filter(pl.col("session_id").is_in(aids)))
    if hist.height:
        # History first so unique() keeps the labelled row on a (session,turn) tie.
        combined = pl.concat([_canon(hist), sub], how="vertical_relaxed")
    else:
        combined = sub
    combined = combined.unique(subset=KEY, keep="first").sort(["session_id", "turn"])
    # Backfill gt from the master map for CGs (one-shot) whose rows lack it.
    combined = (combined.join(gt_map, on=KEY, how="left", suffix="_map")
                .with_columns(pl.coalesce("gt_track_id", "gt_track_id_map").alias("gt_track_id"))
                .drop("gt_track_id_map"))
    return combined.select(CANON).sort(["session_id", "turn"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cg-store", type=Path, default=CG_STORE)
    ap.add_argument("--blind", type=Path, default=BLIND_A)
    ap.add_argument("--out-name", default=OUT_NAME)
    ap.add_argument("--n-folds", type=int, default=N_FOLDS)
    ap.add_argument("--only", nargs="*", help="restrict to these CG folder names")
    args = ap.parse_args()

    aids = pl.read_parquet(args.blind, columns=["session_id"])["session_id"].to_list()
    print(f"[assemble_blind_a] Blind-A sessions: {len(aids)}")

    gt_map = _gt_map(args.cg_store, aids, args.n_folds)
    print(f"[assemble_blind_a] GT map: {gt_map.height} labelled (session,turn) keys")

    n_ok = 0
    for model_dir in sorted(args.cg_store.iterdir()):
        if not model_dir.is_dir():
            continue
        if args.only and model_dir.name not in args.only:
            continue
        """
        if not args.only and _skip(model_dir.name):
            continue
        """
        cand_dir = _cand_dir(model_dir)
        if cand_dir is None:
            continue
        df = _assemble_one(cand_dir, aids, args.n_folds, gt_map)
        out = cand_dir / args.out_name
        df.write_parquet(out)
        n_gt = df.filter(pl.col("gt_track_id").is_not_null()).height
        print(f"  {model_dir.name:34s} rows={df.height:>4} sess={df['session_id'].n_unique():>3} "
              f"turns={df['turn'].min()}-{df['turn'].max()} gt={n_gt} → {out.relative_to(args.cg_store)}")
        n_ok += 1

    print(f"[assemble_blind_a] wrote {n_ok} CG candidate file(s)")


if __name__ == "__main__":
    main()
