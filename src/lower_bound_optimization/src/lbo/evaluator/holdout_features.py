"""Build per-session feature DataFrames for the evaluator from raw splitK
holdout / blind-A parquets.

Output columns per row (one row per session):
    session_id, user_id,
    preferred_musical_culture, country_code, top_tag,
    pop_mean, year_mean,
    max_turn, n_queries, specificity, category

Identity-laden cols (preferred_musical_culture, country_code, top_tag) are
kept for backward compatibility but should NOT be used as stratification axes
in cold-user split — they don't align with Blind-A users.
Conversation-structure cols (max_turn, n_queries, specificity, category)
ARE controllable across users and recommended for stratification.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

YEAR_RE = re.compile(r"(\d{4})")


def _year(s: str | None) -> float | None:
    if s is None:
        return None
    m = YEAR_RE.match(s)
    return float(m.group(1)) if m else None


def _load_tracks(tracks_path: Path) -> pl.DataFrame:
    t = pl.read_parquet(tracks_path).with_columns(
        pl.col("release_date").map_elements(_year, return_dtype=pl.Float64).alias("year"),
        pl.col("tag_list").alias("tags"),
    ).select("track_id", "popularity", "year", "tags")
    return t


def _aggregate_per_session(
    long_df: pl.DataFrame, tracks: pl.DataFrame,
) -> pl.DataFrame:
    """Compute pop_mean / year_mean / top_tag per session from listened tracks.

    `long_df` is expected to be one-row-per-(session, turn) with `track_id`.
    """
    rows = (
        long_df.select("session_id", "track_id")
        .drop_nulls("track_id")
        .join(tracks, on="track_id", how="left")
    )
    sids = rows["session_id"].to_list()
    pops = rows["popularity"].to_list()
    years = rows["year"].to_list()
    tags = rows["tags"].to_list()

    per: dict[str, dict] = {}
    for sid, pop, yr, tg in zip(sids, pops, years, tags):
        d = per.setdefault(sid, {"pop": [], "year": [], "tags": Counter()})
        if pop is not None:
            d["pop"].append(float(pop))
        if yr is not None:
            d["year"].append(float(yr))
        if tg:
            for tag in tg:
                d["tags"][tag] += 1

    out_rows = []
    for sid, d in per.items():
        pop_mean = float(np.mean(d["pop"])) if d["pop"] else float("nan")
        year_mean = float(np.mean(d["year"])) if d["year"] else float("nan")
        top_tag = d["tags"].most_common(1)[0][0] if d["tags"] else "__missing__"
        out_rows.append({
            "session_id": sid, "pop_mean": pop_mean,
            "year_mean": year_mean, "top_tag": top_tag,
        })
    return pl.DataFrame(out_rows)


def _extract_structural_cols(df: pl.DataFrame) -> pl.DataFrame:
    """Pull conversation-structure cols. Operates on the splitK long-form df.

    Returns one row per session_id with:
        max_turn   = max(turn_number)
        n_queries  = number of distinct user_query strings (proxy for n music turns)
        specificity, category = first non-null conversation_goal fields
    """
    base = df
    if "conversation_goal" in df.columns:
        base = base.with_columns(
            pl.col("conversation_goal").struct.field("specificity").alias("specificity"),
            pl.col("conversation_goal").struct.field("category").alias("category"),
        )
    else:
        base = base.with_columns(
            pl.lit("__missing__").alias("specificity"),
            pl.lit("__missing__").alias("category"),
        )

    agg_cols = [
        pl.col("turn_number").max().alias("max_turn"),
        pl.col("specificity").drop_nulls().first().alias("specificity"),
        pl.col("category").drop_nulls().first().alias("category"),
    ]
    if "user_query" in base.columns:
        agg_cols.append(pl.col("user_query").n_unique().alias("n_queries"))
    else:
        agg_cols.append(pl.col("turn_number").n_unique().alias("n_queries"))

    return (
        base.group_by("session_id")
        .agg(*agg_cols)
        .with_columns(
            pl.col("specificity").fill_null("__missing__"),
            pl.col("category").fill_null("__missing__"),
        )
    )


def build_holdout_features(
    holdout_path: Path, tracks_path: Path,
) -> pl.DataFrame:
    """splitK holdout_test.parquet is already long (one row per turn)."""
    df = pl.read_parquet(holdout_path)
    if "user_profile" in df.columns:
        df = df.with_columns(
            pl.col("user_profile").struct.field("country_code").alias("country_code"),
            pl.col("user_profile").struct.field("preferred_musical_culture")
                .alias("preferred_musical_culture"),
        )
    tracks = _load_tracks(tracks_path)
    track_aggs = _aggregate_per_session(df, tracks)
    structural = _extract_structural_cols(df)
    per_session = (
        df.group_by("session_id")
        .agg(
            pl.col("user_id").first(),
            pl.col("preferred_musical_culture").first(),
            pl.col("country_code").first(),
        )
        .join(track_aggs, on="session_id", how="left")
        .join(structural, on="session_id", how="left")
    )
    return per_session


def build_blind_features(
    blind_raw_path: Path, tracks_path: Path,
) -> pl.DataFrame:
    """Build the same feature schema from raw Blind-A parquet."""
    raw = pl.read_parquet(blind_raw_path)
    raw = raw.with_columns(
        pl.col("user_profile").struct.field("country_code").alias("country_code"),
        pl.col("user_profile").struct.field("preferred_musical_culture")
            .alias("preferred_musical_culture"),
        pl.col("conversation_goal").struct.field("specificity").alias("specificity"),
        pl.col("conversation_goal").struct.field("category").alias("category"),
    )

    # Long form: one row per turn (any role) for structural counts.
    long = raw.explode("conversations").unnest("conversations")

    # Track aggregates from music turns only.
    music = long.filter(pl.col("role") == "music").rename({"content": "track_id"}) \
        .select("session_id", "track_id")
    tracks = _load_tracks(tracks_path)
    track_aggs = _aggregate_per_session(music, tracks)

    structural = (
        long.group_by("session_id")
        .agg(
            pl.col("turn_number").max().alias("max_turn"),
            pl.col("role").filter(pl.col("role") == "user").count().alias("n_queries"),
        )
    )

    per_session = (
        raw.select(
            "session_id", "user_id",
            "preferred_musical_culture", "country_code",
            "specificity", "category",
        )
        .unique(subset="session_id")
        .join(track_aggs, on="session_id", how="left")
        .join(structural, on="session_id", how="left")
        .with_columns(
            pl.col("specificity").fill_null("__missing__"),
            pl.col("category").fill_null("__missing__"),
        )
    )
    return per_session


def build_holdout_gt(holdout_path: Path) -> pl.DataFrame:
    """GT = (session_id, turn_number, gt_track_id) from splitK holdout."""
    df = pl.read_parquet(holdout_path).select("session_id", "turn_number", "track_id")
    return df.rename({"track_id": "gt_track_id"})
