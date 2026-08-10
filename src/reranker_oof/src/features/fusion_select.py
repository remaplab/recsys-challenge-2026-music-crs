"""Pure weighted rank/score fusion + top-K selection.

Shared by the fusion tuner and the FeatureBuilder pipeline
(``features.pipeline``) so the on-disk dataset uses byte-identical
fusion math to what was tuned. No I/O, no global state.

Fusion families (all "higher = better"):
    rank-based  : rrf (k), isr, borda, logrank   — read ``rank_<cg>``
    score-based : combsum, combmnz               — read a per-CG score column

Each CG contribution is multiplied by a per-(cg, turn-bucket) weight; the
bucket is read from a ``turn_bucket`` Int column (-1 → fallback weight 1.0).
"""
from __future__ import annotations

import polars as pl

RANK_METHODS = frozenset({"rrf", "isr", "borda", "logrank"})
SCORE_METHODS = frozenset({"combsum", "combmnz"})
FUSION_METHODS = RANK_METHODS | SCORE_METHODS

# Borda credit width (matches pool.add_fusion_features N_FULL).
_BORDA_N = 200


def turn_bucket_expr(buckets: list[tuple[str, list[int]]]) -> pl.Expr:
    """Map ``turn_number`` → bucket index (Int32); -1 if covered by no bucket."""
    e = pl.lit(-1, dtype=pl.Int32)
    for idx, (_name, turns) in enumerate(buckets):
        e = pl.when(pl.col("turn_number").is_in(list(turns))).then(
            pl.lit(idx, dtype=pl.Int32)
        ).otherwise(e)
    return e


def _weight_expr(weights_cg: list[float]) -> pl.Expr:
    """Per-row weight for one CG given its per-bucket weight list."""
    e = pl.lit(1.0)                                    # fallback (bucket == -1)
    for b in range(len(weights_cg) - 1, -1, -1):
        e = pl.when(pl.col("turn_bucket") == b).then(
            pl.lit(float(weights_cg[b]))
        ).otherwise(e)
    return e


def _cg_contrib(cg, w, method, params, score_col):
    rank = pl.col(f"rank_{cg}")
    has = rank.is_not_null()
    if method == "rrf":
        k = float(params.get("k", 60))
        return pl.when(has).then(w * (1.0 / (k + rank))).otherwise(0.0)
    if method == "isr":
        return pl.when(has).then(w * (1.0 / rank ** 2)).otherwise(0.0)
    if method == "borda":
        return pl.when(has).then(w * (_BORDA_N - rank + 1)).otherwise(0.0)
    if method == "logrank":
        return pl.when(has).then(w * (1.0 / (rank + 1).log(2))).otherwise(0.0)
    if method in ("combsum", "combmnz"):
        return w * pl.col(score_col(cg)).fill_null(0.0)
    raise ValueError(f"unknown fusion method {method!r}")


def fusion_score_expr(
    cgs: list[str],
    weights: dict[str, list[float]],
    method: str,
    params: dict,
    *,
    score_col=lambda cg: f"score_minmax_{cg}",
    out: str = "fusion_tuned_score",
) -> pl.Expr:
    """Build the fused-score expression. Requires a ``turn_bucket`` column and
    ``rank_<cg>`` (+ ``score_col(cg)`` for score-based methods).

    ``weights[cg]`` is the per-bucket weight list. ``score_col`` maps a CG name
    to its score column (default ``score_minmax_<cg>``)."""
    total = sum(
        (_cg_contrib(cg, _weight_expr(weights[cg]), method, params, score_col)
         for cg in cgs)
    )
    if method == "combmnz":
        n_retr = sum(
            (pl.col(f"rank_{cg}").is_not_null().cast(pl.Int32) for cg in cgs)
        )
        total = total * n_retr
    return total.alias(out)


def add_fused_and_truncate(
    pool: pl.DataFrame,
    cgs: list[str],
    weights: dict[str, list[float]],
    method: str,
    params: dict,
    buckets: list[tuple[str, list[int]]],
    *,
    top_k: int,
    score_col=lambda cg: f"score_minmax_{cg}",
    group_keys: tuple[str, str] = ("session_id", "turn_number"),
) -> pl.DataFrame:
    """Add ``fusion_tuned_score`` and keep the top-``top_k`` candidates per
    group by it (always keeping the GT row when ``gt_track_id`` is present, so
    label/recall stays measurable). Adds ``turn_bucket`` if absent.

    Used by the FeatureBuilder pipeline to replace the full union pool with the
    tuned fused selection across every split.
    """
    if "turn_bucket" not in pool.columns:
        pool = pool.with_columns(turn_bucket_expr(buckets).alias("turn_bucket"))
    pool = pool.with_columns(
        fusion_score_expr(cgs, weights, method, params, score_col=score_col)
    )
    rank_in_group = (
        pl.col("fusion_tuned_score").rank("ordinal", descending=True)
        .over(list(group_keys))
    )
    keep = rank_in_group <= top_k
    if "gt_track_id" in pool.columns:
        keep = keep | (pl.col("track_id") == pl.col("gt_track_id"))
    return pool.filter(keep)
