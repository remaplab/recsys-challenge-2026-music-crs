"""Shared helpers for the standalone Blind-B overfit pipeline.

Responsibilities
----------------
- Load the ``dataset.yaml`` config and register its CG paths on the path module.
- Assert every listed CG ships the candidate parquet for the split(s) we use.
- Build the per-(session, turn) RRF rank pool for a split and (optionally) the
  test-tracks candidate filter (applied to Blind-A ∪ Blind-B sessions only).
- Weighted RRF fusion (per-(cg, turn-bucket) weights) → top-K, byte-identical to
  ``src.features.fusion_select`` so tuning and dataset assembly agree.
- Ground-truth loaders + macro-by-turn recall@k via ``src.eval.evaluate``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import yaml

_PKG_ROOT = Path(__file__).resolve().parents[1]            # src/reranker_oof
sys.path.insert(0, str(_PKG_ROOT))
sys.path.insert(0, str(_PKG_ROOT / "src"))

from src.eval import evaluate                                           # noqa: E402
from src.features.fusion_select import turn_bucket_expr                 # noqa: E402
from src.features.resources import cg_candidate_path                    # noqa: E402
from src.paths import (                                                 # noqa: E402
    BLIND_A_RAW, BLIND_B_RAW, REPO_ROOT, SPLITK_DIR,
    apply_feature_builder_config,
)

BLIND_B_GT = REPO_ROOT / "data" / "exploded_blind" / "blind-b.parquet"
BLIND_A_GT = REPO_ROOT / "data" / "exploded_blind" / "blind-a.parquet"
HOLDOUT_GT = SPLITK_DIR / "holdout_test.parquet"
_GT_SOURCES = {"blind_b": BLIND_B_GT, "blind_a": BLIND_A_GT, "holdout": HOLDOUT_GT}
PLOT_KS = [1, 5, 10, 20, 50, 100, 200]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: "str | Path") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not cfg.get("cgs"):
        raise SystemExit(f"no `cgs:` in {path}")
    if not cfg.get("name"):
        raise SystemExit(f"no `name:` in {path}")
    return cfg


def register_cg_paths(cfg: dict) -> None:
    """Push the config's ``cg_store`` / ``cg_folder_map`` / ``filenames`` (incl.
    the ``blind_b`` split) onto the path module so ``cg_candidate_path`` resolves."""
    apply_feature_builder_config({
        "name": cfg["name"],
        "cg_store": cfg.get("cg_store", "models/CG_crossvalidation"),
        "cg_folder_map": cfg.get("cg_folder_map", {}),
        "filenames": cfg["filenames"],
    })


def assert_cgs_have(cfg: dict, splits: list[str]) -> None:
    """Hard-fail if any listed CG is missing a candidate parquet for any split."""
    missing: list[str] = []
    for cg in cfg["cgs"]:
        for split in splits:
            p = cg_candidate_path(cg, split, 0)
            if not p.exists():
                missing.append(f"{cg}:{split} → {p}")
    if missing:
        raise SystemExit(
            "missing CG candidate parquets:\n  " + "\n  ".join(missing)
        )


def turn_buckets(cfg: dict) -> list[tuple[str, list[int]]]:
    return [(b["name"], list(b["turns"])) for b in cfg["turn_buckets"]]


# ---------------------------------------------------------------------------
# test-tracks candidate filter (Blind-A ∪ Blind-B sessions only)
# ---------------------------------------------------------------------------

def blind_session_ids() -> list[str]:
    """Session ids of Blind-A ∪ Blind-B — the sessions whose GT is guaranteed to
    live in the test-tracks catalogue, so restricting their candidates loses no
    positive. Cold splitK sessions are deliberately left untouched."""
    a = pl.read_parquet(BLIND_A_RAW, columns=["session_id"])["session_id"].to_list()
    b = pl.read_parquet(BLIND_B_RAW, columns=["session_id"])["session_id"].to_list()
    return sorted(set(a) | set(b))


def test_tracks_spec(cfg: dict) -> dict | None:
    """Resolve the test-tracks filter from the config. Returns
    ``{"tracks": [...], "blind_ids": [...]}`` or ``None`` when disabled."""
    tt = cfg.get("test_tracks") or {}
    if not tt.get("enabled"):
        return None
    path = Path(tt["path"])
    if not path.is_absolute():
        path = REPO_ROOT / path
    tracks = pl.read_parquet(path, columns=["track_id"])["track_id"].unique().to_list()
    blind = blind_session_ids()
    print(f"[blind_b/test_tracks] filter: {len(blind):,} blind sessions, "
          f"{len(tracks):,} test tracks")
    return {"tracks": tracks, "blind_ids": blind}


def filter_candidates(frame: pl.DataFrame, spec: dict | None) -> pl.DataFrame:
    """Drop candidates outside ``test_tracks`` for Blind sessions only (GT row
    always kept). Non-Blind sessions are left whole. No-op when ``spec`` is None.
    Needs ``session_id`` + ``track_id`` (+ ``gt_track_id`` to keep the GT)."""
    if spec is None or frame.height == 0:
        return frame
    keep = (~pl.col("session_id").is_in(spec["blind_ids"])) | \
        pl.col("track_id").is_in(spec["tracks"])
    if "gt_track_id" in frame.columns:
        keep = keep | (pl.col("track_id") == pl.col("gt_track_id"))
    return frame.filter(keep)


# ---------------------------------------------------------------------------
# RRF rank pool
# ---------------------------------------------------------------------------

def load_rank_pool(cfg: dict, split: str) -> pl.DataFrame:
    """Long-format rank pool for ``split``: one row per (session, turn, cg,
    track) with the CG's 1-based rank. Truncated to ``max_candidates_per_cg``.

    Columns: session_id, turn_number, cg, track_id, rank, gt_track_id.
    """
    k_per_cg = int(cfg.get("max_candidates_per_cg", 200))
    frames = []
    for cg in cfg["cgs"]:
        df = pl.read_parquet(
            cg_candidate_path(cg, split, 0),
            columns=["session_id", "turn", "track_ids", "gt_track_id"],
        )
        df = (
            df.with_columns(
                pl.int_ranges(1, pl.col("track_ids").list.len() + 1).alias("rank")
            )
            .explode(["track_ids", "rank"])
            .rename({"track_ids": "track_id", "turn": "turn_number"})
            .filter(pl.col("rank") <= k_per_cg)
            .with_columns(pl.lit(cg).alias("cg"))
            .select("session_id", "turn_number", "cg", "track_id", "rank", "gt_track_id")
        )
        frames.append(df)
    pool = pl.concat(frames)
    return pool.with_columns(
        turn_bucket_expr(turn_buckets(cfg)).alias("turn_bucket")
    )


# ---------------------------------------------------------------------------
# Weighted RRF fusion (per-(cg, bucket) weights) → top-K
# ---------------------------------------------------------------------------

def fuse_rrf(pool: pl.DataFrame, weights: dict[str, list[float]], k: float,
             *, top_k: int) -> pl.DataFrame:
    """Weighted RRF over the long rank pool. ``weights[cg]`` is the per-bucket
    weight list (bucket -1 → fallback 1.0). Returns one row per (session, turn)
    with ``track_ids`` = the top-``top_k`` fused list.

    contrib(cg, row) = weights[cg][turn_bucket] / (k + rank);
    fused_score(track) = Σ_cg contrib.
    """
    wl = [
        {"cg": cg, "turn_bucket": b, "weight": float(w[b])}
        for cg, w in weights.items()
        for b in range(len(w))
    ]
    wdf = pl.DataFrame(wl, schema={"cg": pl.Utf8, "turn_bucket": pl.Int32,
                                   "weight": pl.Float64})
    scored = (
        pool.join(wdf, on=["cg", "turn_bucket"], how="left")
        .with_columns(pl.col("weight").fill_null(1.0))
        .with_columns((pl.col("weight") / (k + pl.col("rank"))).alias("_c"))
        .group_by("session_id", "turn_number", "track_id")
        .agg(pl.col("_c").sum().alias("_s"))
    )
    return (
        scored.with_columns(
            pl.col("_s").rank("ordinal", descending=True)
            .over("session_id", "turn_number").alias("_r")
        )
        .filter(pl.col("_r") <= top_k)
        .sort("session_id", "turn_number", "_r")
        .group_by("session_id", "turn_number", maintain_order=True)
        .agg(pl.col("track_id").alias("track_ids"))
    )


# ---------------------------------------------------------------------------
# Ground truth + recall
# ---------------------------------------------------------------------------

def load_gt(target: str) -> pl.DataFrame:
    src = _GT_SOURCES[target]
    return pl.read_parquet(src, columns=["session_id", "turn_number", "track_id"])


def last_turn_gt(gt: pl.DataFrame) -> pl.DataFrame:
    """Keep only the last GT turn of each session."""
    last = gt.group_by("session_id").agg(pl.col("turn_number").max())
    return gt.join(last, on=["session_id", "turn_number"], how="inner")


def recall_at(recs: pl.DataFrame, gt: pl.DataFrame, ks: list[int]) -> dict[int, float]:
    """Macro-by-turn recall@k of fused ``recs`` (session, turn, track_ids list)
    against ``gt``, for every k in ``ks``."""
    preds = {
        (r[0], r[1]): r[2]
        for r in recs.select("session_id", "turn_number", "track_ids").iter_rows()
    }
    gtp = gt.rename({"track_id": "ground_truth"}).to_pandas()
    return {k: float(evaluate(preds, gtp, k=k)[f"recall@{k}"]) for k in ks}
