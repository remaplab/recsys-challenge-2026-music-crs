"""Concat all feature blocks into a final session-level matrix.

Blocks (one row per (session_id, aug_id, max_turn, source)):
    [meta]         session_id, aug_id, max_turn, source, user_id, label
    [profile]      age_group, country_code, gender, preferred_language,
                   preferred_musical_culture, category, specificity
    [structure]    max_turn (also feature), n_queries_kept
    [lexical]      query_len_mean, query_len_std, query_chars_mean,
                   query_words_mean, qmark_rate
    [track_stats]  n_prior, n_unique_artists, n_unique_albums,
                   pop_mean, pop_std, year_mean, year_std, year_missing,
                   dur_mean, dur_std, tag_entropy, tag_diversity, top_tag,
                   prior_empty
    [emb_q_*]      qwen3 query mean (1024 dims)
    [emb_t_*]      qwen3 track mean over prior (1024 dims)

Categorical columns are kept as strings; the classifier layer ordinal-encodes
them per-fold from training data only.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from lbo.shift.assemble import assemble
from lbo.shift.augment import build_augmented, truncate_turns
from lbo.shift.embeddings import EMB_DIM, aggregate_embeddings
from lbo.shift.tracks import aggregate_tracks

CATEGORICAL_COLS = [
    "age_group", "country_code", "gender", "preferred_language",
    "preferred_musical_culture", "category", "specificity", "top_tag",
]

PROFILE_COLS = [
    "age_group", "country_code", "gender", "preferred_language",
    "preferred_musical_culture", "category", "specificity",
]


def _lexical_block(truncated: pl.DataFrame) -> pl.DataFrame:
    """Lexical aggregates over the kept user_query rows per (session, aug_id)."""
    return (
        truncated.with_columns(
            pl.col("user_query").str.len_chars().alias("q_chars"),
            pl.col("user_query").str.split(" ").list.len().alias("q_words"),
            pl.col("user_query").str.count_matches("\\?").alias("q_qmark"),
        )
        .group_by("session_id", "aug_id")
        .agg(
            pl.col("q_chars").mean().alias("query_chars_mean"),
            pl.col("q_chars").std().fill_null(0.0).alias("query_chars_std"),
            pl.col("q_words").mean().alias("query_words_mean"),
            pl.col("q_words").std().fill_null(0.0).alias("query_words_std"),
            pl.col("q_qmark").mean().alias("qmark_rate"),
            pl.len().alias("n_queries_kept"),
        )
    )


def _profile_block(per_turn_df: pl.DataFrame) -> pl.DataFrame:
    """One row per session with profile + goal cols (taken from any turn)."""
    return per_turn_df.group_by("session_id").agg(*[pl.col(c).first() for c in PROFILE_COLS])


def build_feature_matrix(n_augs: int, seed: int) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Return (tabular_df, q_emb, t_emb) row-aligned by (session_id, aug_id)."""
    print("[features] assembling per-turn…")
    per_turn = assemble()
    print("[features] augmenting…")
    aug = build_augmented(per_turn, n_augs=n_augs, seed=seed)
    print("[features] truncating…")
    trunc = truncate_turns(per_turn, aug)

    print("[features] lexical block…")
    lex = _lexical_block(trunc)
    print("[features] profile block…")
    prof = _profile_block(per_turn)
    print("[features] track stats block…")
    tstats = aggregate_tracks(aug)

    print("[features] joining tabular blocks…")
    tab = (
        aug.join(prof, on="session_id", how="left")
        .join(lex, on=["session_id", "aug_id"], how="left")
        .join(tstats, on=["session_id", "aug_id"], how="left")
        .with_columns(pl.when(pl.col("source") == "blind").then(1).otherwise(0).alias("label"))
    )

    print("[features] aggregating embeddings…")
    q_emb, t_emb, sids, augids = aggregate_embeddings(aug, per_turn)
    emb_order = pl.DataFrame({"session_id": sids, "aug_id": augids}).with_row_index("emb_row")
    tab_idx = tab.with_row_index("tab_row").select("tab_row", "session_id", "aug_id")
    join_key = tab_idx.join(emb_order, on=["session_id", "aug_id"], how="left")
    # Reorder emb rows to match tab order
    perm = join_key["emb_row"].to_numpy()
    q_emb = q_emb[perm]
    t_emb = t_emb[perm]

    return tab, q_emb, t_emb


if __name__ == "__main__":
    tab, q, t = build_feature_matrix(n_augs=3, seed=42)
    print("tab shape:", tab.shape)
    print("tab cols:", tab.columns)
    print("label dist:", tab.group_by("label").agg(pl.len()))
    print("q emb shape:", q.shape, "t emb shape:", t.shape)
    print("EMB_DIM:", EMB_DIM)
