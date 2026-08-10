"""Density-ratio weight post-processing for distribution_shift_v2.

Given an ensemble probability `p̂(blind | x)`, produces:
  - raw importance weight  w = p̂ / (1 - p̂) * n_train / n_blind
  - clipped weight at quantile q  (CVaR-equivalence per LBO doc B)
  - group_id ∈ {0..8}  (turn_bucket × dr_bucket per LBO doc B Group-DRO recipe)
  - per-augmentation table for reranker handoff
"""
from __future__ import annotations

import numpy as np
import polars as pl

TURN_EDGES = np.array([0.0, 2.0, 5.0, np.inf])
DR_EDGES = np.array([0.0, 0.5, 1.5, np.inf])


def clip_at_quantile(w: np.ndarray, q: float) -> np.ndarray:
    """Cap weights at the q-quantile; values below are untouched."""
    cap = float(np.quantile(w, q))
    return np.minimum(w, cap)


def _bucket(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Right-open bucketing: returns index in [0, len(edges)-2]."""
    idx = np.digitize(x, edges[1:-1], right=False)
    return np.clip(idx, 0, len(edges) - 2).astype(np.int64)


def assign_group_id(turn_number: np.ndarray, dr_raw: np.ndarray) -> np.ndarray:
    """group_id = turn_bucket * 3 + dr_bucket, each in {0,1,2}."""
    tb = _bucket(np.asarray(turn_number), TURN_EDGES)
    db = _bucket(np.asarray(dr_raw), DR_EDGES)
    return (tb * 3 + db).astype(np.int64)


def raw_density_ratio(p_ens: np.ndarray, n_train: int, n_blind: int) -> np.ndarray:
    """w_raw = p̂ / (1 - p̂) * n_train / n_blind. Clips p̂ off the boundary."""
    p = np.clip(np.asarray(p_ens, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return (p / (1.0 - p)) * (float(n_train) / float(max(n_blind, 1)))


def build_weight_table(
    df: pl.DataFrame,
    *,
    n_train: int,
    n_blind: int,
    clip_q: float = 0.99,
    turn_col: str = "max_turn",
) -> pl.DataFrame:
    """Per-augmentation weight table.

    Input df must have: session_id, aug_id, <turn_col>, ensemble_prob.
    Default `turn_col` matches the upstream feature_matrix column "max_turn".
    Returns columns:
        session_id, aug_id, max_turn, weight_raw, weight_clipped,
        ensemble_prob, group_id.
    """
    p_ens = df["ensemble_prob"].to_numpy().astype(np.float64)
    turn = df[turn_col].to_numpy().astype(np.int64)

    w_raw = raw_density_ratio(p_ens, n_train, n_blind)
    w_clip = clip_at_quantile(w_raw, q=clip_q)
    gid = assign_group_id(turn, w_raw)

    return df.with_columns(
        pl.Series("max_turn", turn),
        pl.Series("weight_raw", w_raw),
        pl.Series("weight_clipped", w_clip),
        pl.Series("group_id", gid),
    ).select([
        "session_id", "aug_id", "max_turn",
        "weight_raw", "weight_clipped", "ensemble_prob", "group_id",
    ])


def aggregate_per_session(per_aug: pl.DataFrame) -> pl.DataFrame:
    """One row per session_id."""
    return (
        per_aug
        .group_by("session_id")
        .agg(
            pl.col("weight_raw").mean().alias("weight_mean"),
            pl.col("weight_clipped").mean().alias("weight_p99_clipped"),
            pl.col("ensemble_prob").mean().alias("ensemble_prob"),
            pl.col("group_id").mode().first().alias("group_id_mode"),
        )
    )
