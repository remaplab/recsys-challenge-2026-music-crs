"""Helpers for the Blind-A-matched split produced by
`scripts/01b_rebuild_split.py`.

Schema of that parquet:
    session_id            (str)
    split                 ("train" | "val" | "test")
    predict_turn_number   (int)   <- the turn whose music track is the GT

Semantics
=========
- TRAIN sessions contribute ALL their (session, turn) rows to training; their
  `predict_turn_number` is just bookkeeping (= number of music turns).
- VAL / TEST sessions are pinned to EXACTLY ONE (session_id, predict_turn_number)
  row each — that single turn is the evaluation target. This is what makes the
  fold match Blind A on the K (predict-turn) covariate.

We therefore expose:
    sessions_for_fold(df, "train")   -> set of session_ids (use ALL their turns)
    turn_pairs_for_fold(df, "val")   -> set of (session_id, turn) pairs (one per session)

ASSUMPTION
==========
`predict_turn_number` is matched against the query meta's `turn_number`. This
holds when a session's music turns carry contiguous turn_numbers and each has a
user message at the same turn (the standard TalkPlay layout, and what
`03_encode_queries.py` emits). If your data numbers turns differently, this is
the single place to change the (session_id, turn_number) mapping.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl


REQUIRED_COLUMNS = {"session_id", "split", "predict_turn_number"}


def load_blinda_split(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing columns {sorted(missing)}; expected the output of "
            f"scripts/01b_rebuild_split.py with columns {sorted(REQUIRED_COLUMNS)}."
        )
    return df.with_columns(
        pl.col("session_id").cast(pl.Utf8),
        pl.col("predict_turn_number").cast(pl.Int64),
    )


def sessions_for_fold(df: pl.DataFrame, fold: str) -> set[str]:
    return set(df.filter(pl.col("split") == fold)["session_id"].to_list())


def turn_pairs_for_fold(df: pl.DataFrame, fold: str) -> set[tuple[str, int]]:
    sub = df.filter(pl.col("split") == fold)
    return {
        (str(s), int(k))
        for s, k in zip(sub["session_id"].to_list(), sub["predict_turn_number"].to_list())
    }


def _pair_key(session_id, turn) -> str:
    return f"{session_id}::{int(turn)}"


def select_pair_indices(meta_df: pl.DataFrame, pairs: set[tuple[str, int]]) -> list[int]:
    """Row indices of `meta_df` whose (session_id, turn_number) is in `pairs`.

    Returned ascending, so callers can index a parallel embedding array with the
    same list and keep alignment.
    """
    keyset = {_pair_key(s, k) for (s, k) in pairs}
    sids = meta_df["session_id"].to_list()
    tns = meta_df["turn_number"].to_list()
    return [i for i, (s, t) in enumerate(zip(sids, tns)) if _pair_key(s, t) in keyset]


def filter_meta_to_pairs(meta_df: pl.DataFrame, pairs: set[tuple[str, int]]) -> pl.DataFrame:
    """Subset `meta_df` to the rows in `pairs`, preserving original row order."""
    idx = select_pair_indices(meta_df, pairs)
    return (
        meta_df.with_row_index(name="__row")
        .filter(pl.col("__row").is_in(idx))
        .drop("__row")
    )