"""VADER sentiment scoring helper for the reranker feature builder.

VADER (``vaderSentiment``) is a lexicon + rule-based sentiment analyser — no
model download, pure-Python, deterministic. We only ever use the ``compound``
score (a single float in [-1, 1]) as a feature.

The analyser is instantiated once (the lexicon load is the only non-trivial
cost) and reused. :func:`score_text_column` dedupes the distinct strings in a
column before scoring so each unique utterance is scored exactly once — the
per-turn text tables are tiny (one row per (session, turn)), but a session's
goal text repeats across all its turns, so deduping still pays off.
"""
from __future__ import annotations

from functools import lru_cache

import polars as pl


@lru_cache(maxsize=1)
def _analyzer():
    """Return a process-wide singleton ``SentimentIntensityAnalyzer``."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer()


def score_text_column(
    df: pl.DataFrame, text_col: str, out_col: str,
) -> pl.DataFrame:
    """Add ``out_col`` = VADER compound sentiment of ``text_col``.

    Null / empty strings score 0.0 (neutral). Distinct strings are scored
    once and joined back, so the cost scales with the number of *unique*
    utterances, not the number of rows.
    """
    if text_col not in df.columns:
        return df.with_columns(pl.lit(0.0, dtype=pl.Float64).alias(out_col))

    sia = _analyzer()
    uniq = df.select(pl.col(text_col)).unique().drop_nulls()
    texts = uniq[text_col].to_list()
    scores = [
        sia.polarity_scores(t)["compound"] if t else 0.0 for t in texts
    ]
    lut = uniq.with_columns(pl.Series(out_col, scores, dtype=pl.Float64))
    return df.join(lut, on=text_col, how="left").with_columns(
        pl.col(out_col).fill_null(0.0)
    )
