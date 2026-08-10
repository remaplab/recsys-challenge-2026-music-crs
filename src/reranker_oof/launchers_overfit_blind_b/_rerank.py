"""Shared reranker helpers for the Blind-B overfit pipeline (s05 tune, s06 submit).

- Resolve the assembled dataset chunk paths (train pool + the three eval sets).
- Re-shard chunks to N groups/file for the XGB DataIter (non-dropping).
- Build inference DMatrices (sharing dtrain's bins) for per-trial scoring.
- ndcg+recall@K of a scored frame, and the submission-JSON records.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

_PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG_ROOT))
sys.path.insert(0, str(_PKG_ROOT / "src"))

from src.eval import evaluate                                           # noqa: E402
from src.features.pipeline import fold_chunk_paths                      # noqa: E402
from src.paths import active_dataset_dir, active_subsamples_dir         # noqa: E402
from src.rerankers.base import feature_columns                          # noqa: E402
from src.rerankers.xgb_ranker import ParquetRankerIter                  # noqa: E402

from launchers_overfit_blind_b._common import (                         # noqa: E402
    blind_session_ids, last_turn_gt, load_gt,
)

# Eval-set identifiers used across the tuner / submitter.
EVAL_SETS = ("holdout", "blind_b_all", "blind_b_last")


# ---------------------------------------------------------------------------
# Dataset chunk resolution
# ---------------------------------------------------------------------------

def _all_fold_chunks(split: str, max_folds: int = 16) -> list[Path]:
    paths: list[Path] = []
    for k in range(max_folds):
        paths += fold_chunk_paths(active_dataset_dir(), split, fold_idx=k)
    return paths


def holdout_chunks() -> list[Path]:
    return fold_chunk_paths(active_dataset_dir(), "holdout", fold_idx=0)


def blind_chunks() -> list[Path]:
    return fold_chunk_paths(active_dataset_dir(), "blind_b", fold_idx=0)


def blind_a_chunks() -> list[Path]:
    return fold_chunk_paths(active_dataset_dir(), "blind_a", fold_idx=0)


def train_pool_paths(target: str) -> list[Path]:
    """Training chunks for a given val target. The target's own source is
    excluded from TRAIN (holdout target → train = OOF folds only; blind targets
    → train = OOF folds + holdout)."""
    base = _all_fold_chunks("train") + _all_fold_chunks("val")
    if target != "holdout":
        base += holdout_chunks()
    return base


def eval_gt(which: str) -> pl.DataFrame:
    """(session_id, turn_number, track_id) GT for an eval set. ``which`` ∈
    {holdout, blind_b_all, blind_b_last, blind_a_all, blind_a_last}."""
    if which == "holdout":
        return load_gt("holdout")
    base, _, suffix = which.rpartition("_")          # blind_b/blind_a + all/last
    gt = load_gt(base)
    return last_turn_gt(gt) if suffix == "last" else gt


def eval_keys(which: str) -> pl.DataFrame:
    return eval_gt(which).select("session_id", "turn_number").unique()


# ---------------------------------------------------------------------------
# Feature-column resolution (blind_b-schema aware)
# ---------------------------------------------------------------------------

def resolve_feats(sample: pl.DataFrame, keep, blind_sample: pl.DataFrame) -> list[str]:
    """Feature columns for a run that must score blind_b.

    The blind_b raw test parquet carries no goal text, so its assembled chunks
    lack the ~17 ``goal_*`` features that every other split has. A model that
    scores blind_b can only use columns present in the blind_b schema — any
    other feature is unusable at serve time (and would crash the DataIter when
    a blind_b chunk is ingested). So:

    - ``keep`` null → ``feature_columns(sample)`` ∩ blind_b columns (drops the
      goal_* set, with a notice).
    - ``keep`` given → validate every kept column against BOTH the train sample
      and the blind_b schema; error out otherwise.
    """
    base = feature_columns(sample)
    blind_cols = set(blind_sample.columns)
    if not keep:
        feats = [c for c in base if c in blind_cols]
        dropped = [c for c in base if c not in blind_cols]
        if dropped:
            print(f"[feats] dropping {len(dropped)} cols absent from blind_b "
                  f"schema (e.g. {dropped[:6]})")
        return feats
    missing_train = [c for c in keep if c not in set(base)]
    if missing_train:
        raise SystemExit(f"feat_cols_keep missing from train schema: {missing_train[:5]}")
    missing_blind = [c for c in keep if c not in blind_cols]
    if missing_blind:
        raise SystemExit(f"feat_cols_keep not in blind_b schema (unusable at "
                         f"serve time): {missing_blind[:5]}")
    return list(keep)


# ---------------------------------------------------------------------------
# Re-shard to N groups/file (non-dropping), per source chunk
# ---------------------------------------------------------------------------

def subsample_train(paths: list[Path], cfg: dict) -> list[Path]:
    """Cap each TRAIN chunk to ``train_subsample.max_groups_per_file`` non-blind
    groups (drops the rest, deterministic hash-by-session) for fast fitting —
    the DRO-style speed lever. Blind-A ∪ Blind-B session groups are ALWAYS kept
    (the overfit signal). NEVER applied to eval/val. No-op when the cap is absent.
    Idempotent by mtime."""
    ts = cfg.get("train_subsample") or {}
    max_g = ts.get("max_groups_per_file")
    if not max_g:
        return list(paths)
    max_g = int(max_g)
    seed = int(ts.get("seed", 42))
    protect = blind_session_ids()                          # A ∪ B
    root = active_subsamples_dir() / f"trainsub{max_g}"
    out: list[Path] = []
    for p in paths:                          # per-parent subdir avoids name clash
        d = root / p.parent.name
        d.mkdir(parents=True, exist_ok=True)
        op = d / p.name
        if op.exists() and op.stat().st_mtime >= p.stat().st_mtime:
            out.append(op); continue
        df = pl.read_parquet(p)
        groups = df.select("session_id", "turn_number").unique()
        is_prot = pl.col("session_id").is_in(protect)
        prot = groups.filter(is_prot)
        rest = groups.filter(~is_prot)
        if rest.height > max_g:               # deterministic by session hash
            rest = (rest.with_columns(pl.col("session_id").hash(seed=seed).alias("_h"))
                    .sort("_h").head(max_g).drop("_h"))
        keep = pl.concat([prot, rest])
        df.join(keep, on=["session_id", "turn_number"], how="semi").write_parquet(op)
        out.append(op)
        print(f"[train_subsample] {p.parent.name}/{p.name}: {groups.height} → "
              f"{keep.height} groups (protected blind {prot.height})", flush=True)
    return out


def reshard(paths: list[Path], out_dir: Path, groups_per_chunk: int,
            restrict: pl.DataFrame | None = None, force: bool = False) -> list[Path]:
    """Split each source chunk into ``groups_per_chunk``-group parquets under
    ``out_dir`` (whole groups, no row drop). ``restrict`` semi-joins to a key
    subset first. Idempotent: reuses existing output unless ``force``."""
    import time
    label = out_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("chunk_*.parquet"))
    if existing and not force:
        print(f"[reshard:{label}] reuse {len(existing)} existing chunks "
              f"({out_dir})", flush=True)
        return existing
    for f in existing:
        f.unlink()
    print(f"[reshard:{label}] {len(paths)} source files → {groups_per_chunk} "
          f"groups/chunk → {out_dir}", flush=True)
    t0 = time.time()
    out: list[Path] = []
    cidx = 0
    g_total = 0
    for i, p in enumerate(paths, 1):
        df = pl.read_parquet(p)
        if restrict is not None:
            df = df.join(restrict, on=["session_id", "turn_number"], how="semi")
        if df.height == 0:
            print(f"  [{i:>3d}/{len(paths)}] {p.name}: empty after restrict — skip",
                  flush=True)
            continue
        df = df.sort("session_id", "turn_number", "track_id")
        keys = (df.select("session_id", "turn_number").unique(maintain_order=True)
                .with_row_index("_g"))
        n_groups = keys.height
        df = df.join(keys, on=["session_id", "turn_number"]).with_columns(
            (pl.col("_g") // groups_per_chunk).alias("_b"))
        parts = sorted(df.partition_by("_b", as_dict=True).items())
        for (b,), sub in parts:
            cp = out_dir / f"chunk_{cidx:04d}.parquet"
            sub.drop("_g", "_b").write_parquet(cp)
            out.append(cp); cidx += 1
        g_total += n_groups
        print(f"  [{i:>3d}/{len(paths)}] {p.name}: {df.height:,} rows, "
              f"{n_groups:,} groups → {len(parts)} sub-chunks "
              f"({time.time() - t0:.1f}s)", flush=True)
    print(f"[reshard:{label}] done: {len(out)} chunks, {g_total:,} groups "
          f"({time.time() - t0:.1f}s)", flush=True)
    return out


# ---------------------------------------------------------------------------
# Inference DMatrix (shares dtrain bins via ref)
# ---------------------------------------------------------------------------

def build_infer_dmatrix(paths: list[Path], feat_cols: list[str], device: str,
                        ref: "xgb.DMatrix",
                        max_bin: int | None = None) -> tuple["xgb.DMatrix", pl.DataFrame]:
    """QuantileDMatrix over ``paths`` (carrying labels for early-stop) plus the
    (session, turn, track) meta in the same row order for ``predict_dval``.

    ``max_bin`` MUST match the value used to build ``ref`` (dtrain) and the
    booster — XGBoost rejects QuantileDMatrices with inconsistent ``max_bin``."""
    it = ParquetRankerIter(paths, feat_cols, tag="eval", device=device)
    kw = {"max_bin": int(max_bin)} if max_bin is not None else {}
    dmat = xgb.QuantileDMatrix(it, ref=ref, **kw)
    dmat.feature_names = feat_cols
    meta = pl.concat([
        pl.read_parquet(p).sort("session_id", "turn_number", "track_id")
        .select("session_id", "turn_number", "track_id")
        for p in paths
    ])
    assert meta.height == dmat.num_row(), "meta/dmatrix row-order drift"
    return dmat, meta


# ---------------------------------------------------------------------------
# Metrics + submission
# ---------------------------------------------------------------------------

def _topk(scored: pl.DataFrame, k: int) -> dict[tuple[str, int], list[str]]:
    ranked = (
        scored.with_columns(
            pl.col("score").rank("ordinal", descending=True)
            .over("session_id", "turn_number").alias("_r")
        ).filter(pl.col("_r") <= k).sort("session_id", "turn_number", "_r")
        .group_by("session_id", "turn_number", maintain_order=True)
        .agg(pl.col("track_id").alias("lst"))
    )
    return {(r[0], r[1]): r[2] for r in ranked.iter_rows()}


def eval_scored(scored: pl.DataFrame, gt: pl.DataFrame,
                ks: list[int]) -> dict[int, dict[str, float]]:
    """Macro-by-turn ndcg@k and recall@k of ``scored`` (session, turn, track,
    score) vs ``gt`` for every k. ``evaluate`` already averages within each
    turn_number then across turns (mean of per_turn values), so this IS the
    double-averaged macro. Only ``gt``'s turns are scored."""
    gtp = gt.rename({"track_id": "ground_truth"}).to_pandas()
    out: dict[int, dict[str, float]] = {}
    for k in ks:
        e = evaluate(_topk(scored, k), gtp, k=k)
        out[k] = {"ndcg": float(e[f"ndcg@{k}"]), "recall": float(e[f"recall@{k}"])}
    return out


def submission_records(scored: pl.DataFrame, blind_raw: Path) -> list[dict]:
    """Competition records for the withheld (gt-null) submission turns: top-20
    by score, user_id from the blind raw parquet."""
    sub = scored.filter(pl.col("gt_track_id").is_null())
    users = pl.read_parquet(blind_raw, columns=["session_id", "user_id"]).unique()
    top20 = (
        sub.with_columns(
            pl.col("score").rank("ordinal", descending=True)
            .over("session_id", "turn_number").alias("_r")
        ).filter(pl.col("_r") <= 20).sort("session_id", "turn_number", "_r")
        .group_by("session_id", "turn_number", maintain_order=True)
        .agg(pl.col("track_id").alias("predicted_track_ids"))
        .join(users, on="session_id", how="left")
    )
    return [
        {
            "session_id": r["session_id"],
            "user_id": r["user_id"],
            "turn_number": int(r["turn_number"]),
            "predicted_track_ids": list(r["predicted_track_ids"][:20]),
            "predicted_response": "",
        }
        for r in top20.iter_rows(named=True)
    ]
