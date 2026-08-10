"""Truncate sessions to a `max_turn` sampled from blind n_music_turns PMF.

For each session we emit `aug_id` rows. Each row holds:
    session_id, user_id, aug_id, max_turn, source

For blind: 1 row per session, max_turn = real number of completed music turns
in the session (= n_music_turns in raw blind = count of gt_track_id non-null).
For train: `n_augs` rows per session, max_turn sampled from blind PMF.

Sampling is deterministic given seed + (session_id, aug_id).
"""
from __future__ import annotations

import hashlib

import numpy as np
import polars as pl

from lbo.paths import BLIND_MUSIC_TURNS_PMF


def _seeded_choice(values: np.ndarray, probs: np.ndarray, key: str, seed: int) -> int:
    """Deterministic categorical draw from (values, probs) keyed by `key`."""
    h = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    u = int.from_bytes(h[:8], "big") / 2**64
    cdf = np.cumsum(probs)
    return int(values[np.searchsorted(cdf, u)])


def build_augmented(per_turn_df: pl.DataFrame, n_augs: int, seed: int) -> pl.DataFrame:
    """Return (session_id, user_id, aug_id, max_turn, source) for all augmentations.

    `per_turn_df` is the output of `assemble.assemble()`.
    """
    pmf_items = sorted(BLIND_MUSIC_TURNS_PMF.items())
    values = np.array([k for k, _ in pmf_items], dtype=np.int64)
    counts = np.array([v for _, v in pmf_items], dtype=np.float64)
    probs = counts / counts.sum()

    # blind: max_turn = count of (gt_track_id not null) per session
    blind_meta = (
        per_turn_df.filter(pl.col("source") == "blind")
        .group_by("session_id")
        .agg(
            pl.col("user_id").first(),
            pl.col("gt_track_id").is_not_null().sum().alias("max_turn"),
        )
        .with_columns(pl.lit(0, dtype=pl.Int64).alias("aug_id"), pl.lit("blind").alias("source"))
    )

    train_sessions = (
        per_turn_df.filter(pl.col("source") == "train")
        .group_by("session_id")
        .agg(pl.col("user_id").first())
    )

    rows = []
    for sid, uid in zip(train_sessions["session_id"].to_list(), train_sessions["user_id"].to_list()):
        for aug_id in range(n_augs):
            mt = _seeded_choice(values, probs, key=f"{sid}:{aug_id}", seed=seed)
            rows.append((sid, uid, aug_id, mt, "train"))

    train_aug = pl.DataFrame(
        rows, schema=["session_id", "user_id", "aug_id", "max_turn", "source"], orient="row"
    )

    return pl.concat([blind_meta, train_aug], how="diagonal_relaxed").select(
        "session_id", "user_id", "aug_id", "max_turn", "source"
    )


def truncate_turns(per_turn_df: pl.DataFrame, aug_df: pl.DataFrame) -> pl.DataFrame:
    """Inner-join per-turn rows with (session_id, aug_id, max_turn) and keep
    turns 1..max_turn+1. Output has columns of `per_turn_df` plus `aug_id`,
    `max_turn`.

    `max_turn` = number of completed prior (user, music, assistant) cycles.
    Prediction query is at `turn_number == max_turn + 1`. Music history covers
    `turn_number ∈ [1, max_turn]` (rows where `gt_track_id` is the listened track).
    The row at `turn_number == max_turn + 1` carries only the prediction query;
    its `gt_track_id` is the GT (train) or null (blind).
    """
    joined = per_turn_df.join(
        aug_df.select("session_id", "aug_id", "max_turn"),
        on="session_id",
        how="inner",
    )
    return joined.filter(pl.col("turn_number") <= pl.col("max_turn") + 1)


if __name__ == "__main__":
    from lbo.shift.assemble import assemble

    df = assemble()
    aug = build_augmented(df, n_augs=3, seed=42)
    print("aug rows:", aug.shape[0])
    print(
        "max_turn dist per source:",
        aug.group_by("source", "max_turn").agg(pl.len()).sort("source", "max_turn"),
    )
    trunc = truncate_turns(df, aug)
    print("truncated rows:", trunc.shape[0])
