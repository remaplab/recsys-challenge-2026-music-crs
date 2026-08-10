"""Assemble train+test+blind into a long per-turn DataFrame.

Output columns:
    session_id, user_id, turn_number, source ∈ {train, blind},
    user_query, gt_track_id (nullable),
    category, specificity,
    age_group, country_code, gender, preferred_language, preferred_musical_culture
"""
from __future__ import annotations

import polars as pl

from lbo.paths import BLIND_PARQUET, TEST_PARQUET, TRAIN_PARQUET


def _explode_turns(df: pl.DataFrame, source: str) -> pl.DataFrame:
    """Return one row per (session, turn) with user_query + (optional) GT track."""
    base = df.with_columns(
        pl.col("user_profile").struct.field("age_group").alias("age_group"),
        pl.col("user_profile").struct.field("country_code").alias("country_code"),
        pl.col("user_profile").struct.field("gender").alias("gender"),
        pl.col("user_profile").struct.field("preferred_language").alias("preferred_language"),
        pl.col("user_profile").struct.field("preferred_musical_culture").alias("preferred_musical_culture"),
        pl.col("conversation_goal").struct.field("category").alias("category"),
        pl.col("conversation_goal").struct.field("specificity").alias("specificity"),
    ).drop("user_profile", "conversation_goal", "session_date", "goal_progress_assessments")

    queries = (
        base.explode("conversations").unnest("conversations")
        .filter(pl.col("role") == "user")
        .rename({"content": "user_query"})
        .drop("role", "thought")
    )
    music = (
        base.select("session_id", "conversations").explode("conversations").unnest("conversations")
        .filter(pl.col("role") == "music")
        .rename({"content": "gt_track_id"})
        .select("session_id", "turn_number", "gt_track_id")
    )
    out = queries.join(music, on=["session_id", "turn_number"], how="left")
    return out.with_columns(pl.lit(source).alias("source"))


def assemble() -> pl.DataFrame:
    train = pl.concat([pl.read_parquet(TRAIN_PARQUET), pl.read_parquet(TEST_PARQUET)])
    blind = pl.read_parquet(BLIND_PARQUET)
    out = pl.concat([
        _explode_turns(train, source="train"),
        _explode_turns(blind, source="blind"),
    ], how="diagonal_relaxed")
    return out


if __name__ == "__main__":
    df = assemble()
    print("rows:", df.shape[0])
    print("by source:", df.group_by("source").agg(pl.col("session_id").n_unique().alias("sessions"), pl.len().alias("rows")))
    print(df.head(3))
