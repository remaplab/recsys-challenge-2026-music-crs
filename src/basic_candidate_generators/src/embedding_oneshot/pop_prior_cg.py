"""PopPriorCG — profile-conditioned (and global) popularity for turn-1 cold-start.

At turn 1 there is no session history, so the strongest non-semantic signal is
"what do users like this tend to open first". This CG counts turn-1 target
tracks in the training fold, bucketed by a tuple of user_profile attributes
(e.g. country_code + preferred_language + preferred_musical_culture + age_group),
and recommends each bucket's most frequent tracks, backing off to the global
popularity ranking when a bucket is unseen or too small.

`profile_keys = []` degenerates to pure global popularity → the `global_pop`
component. With keys set it is `profile_pop`. Cold-user-safe: eval users are
unseen, but their profile ATTRIBUTES bucket against training counts; unknown
buckets fall back to global.

Standard inference mode: subclasses BaseRecommender and consumes the
`user_profile` struct that build_context_df now passes through on context_df.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import numpy as np
import polars as pl

from BaseRecommender import BaseRecommender

from recommenders.interactions import parse_date

from embedding_based.dense_query_cg import _build_release_dates

_MISSING = "∅"  # placeholder for null profile fields so buckets stay hashable


class PopPriorCG(BaseRecommender):
    RECOMMENDER_NAME = "PopPriorCG"

    def __init__(
        self,
        profile_keys: list[str] | None = None,
        min_bucket_count: int = 20,    # buckets with fewer obs back off to global
        top_n: int = 1000,             # cap stored candidates per bucket / global
        max_future_years: float = 2.0,
        urm_mode: str = "session",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.profile_keys = list(profile_keys or [])
        self.min_bucket_count = int(min_bucket_count)
        self.top_n = int(top_n)
        self.max_future_years = float(max_future_years)
        self.urm_mode = urm_mode

        self.track_ids: np.ndarray | None = None
        self.track_to_idx: dict[str, int] = {}
        self.release_dates: np.ndarray | None = None
        self.global_rank: np.ndarray | None = None        # track idx, desc by count
        self.bucket_rank: dict[tuple, np.ndarray] = {}     # bucket -> track idx array

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, train_df, track_metadata: pl.DataFrame | None = None, **kwargs: Any) -> None:
        if track_metadata is None:
            raise ValueError(f"[{self.RECOMMENDER_NAME}] track_metadata required (catalogue)")
        t0 = time.time()
        self.track_ids = track_metadata["track_id"].to_numpy()
        self.track_to_idx = {t: i for i, t in enumerate(self.track_ids)}
        self.release_dates = _build_release_dates(self.track_ids, track_metadata)

        rows = train_df.filter(pl.col("track_id").is_not_null())
        track_idx = np.fromiter(
            (self.track_to_idx.get(t, -1) for t in rows["track_id"].to_list()),
            dtype=np.int64, count=rows.height)

        # global counts
        gcounts = defaultdict(int)
        for ti in track_idx:
            if ti >= 0:
                gcounts[ti] += 1
        self.global_rank = self._rank_from_counts(gcounts)

        # per-bucket counts
        self.bucket_rank = {}
        if self.profile_keys:
            buckets = self._bucket_keys(rows)
            bcounts: dict[tuple, dict[int, int]] = defaultdict(lambda: defaultdict(int))
            for bk, ti in zip(buckets, track_idx):
                if ti >= 0:
                    bcounts[bk][ti] += 1
            for bk, cnt in bcounts.items():
                if sum(cnt.values()) >= self.min_bucket_count:
                    self.bucket_rank[bk] = self._rank_from_counts(cnt)

        print(f"[{self.RECOMMENDER_NAME}] fit in {time.time()-t0:.1f}s — "
              f"{len(self.track_ids)} tracks, {len(self.bucket_rank)} profile buckets, "
              f"keys={self.profile_keys}")

    def _rank_from_counts(self, counts: dict[int, int]) -> np.ndarray:
        if not counts:
            return np.empty(0, dtype=np.int64)
        idx = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
        cnt = np.fromiter(counts.values(), dtype=np.int64, count=len(counts))
        order = np.argsort(-cnt, kind="stable")[: self.top_n]
        return idx[order]

    def _bucket_keys(self, df: pl.DataFrame) -> list[tuple]:
        """Tuple of profile-attribute values per row (nulls → placeholder)."""
        cols = [
            df["user_profile"].struct.field(k).fill_null(_MISSING).to_list()
            for k in self.profile_keys
        ]
        return list(zip(*cols)) if cols else [() for _ in range(df.height)]

    # ------------------------------------------------------------------
    # recommend
    # ------------------------------------------------------------------

    def recommend(
        self, context_df: pl.DataFrame, top_k: int = 200, remove_seen: bool = True,
        max_future_years: float | None = None, **kwargs: Any,
    ) -> pl.DataFrame:
        if self.global_rank is None:
            raise RuntimeError("Recommender not fitted")
        if "target_turn" not in context_df.columns:
            raise ValueError("context_df missing 'target_turn'. Use build_context_df().")
        if max_future_years is None:
            max_future_years = self.max_future_years

        keep = ["session_id", "user_id", "session_date", "target_turn"]
        if self.profile_keys and "user_profile" in context_df.columns:
            keep.append("user_profile")
        meta = context_df.select(keep).unique(subset=["session_id"])

        # per-session seen tracks (turn 1 → usually none)
        ctx_map: dict[str, list[str]] = {}
        if "track_id" in context_df.columns and context_df.height > 0:
            grp = (context_df.filter(pl.col("track_id").is_not_null())
                   .group_by("session_id").agg(pl.col("track_id")))
            ctx_map = dict(zip(grp["session_id"].to_list(), grp["track_id"].to_list()))

        buckets = (self._bucket_keys(meta)
                   if self.profile_keys and "user_profile" in meta.columns
                   else [() for _ in range(meta.height)])
        rows = meta.to_dicts()

        out_tracks: list[list[str]] = []
        out_scores: list[list[float]] = []
        for r, bk in zip(rows, buckets):
            ranked = self._candidates_for(bk)
            tracks, scores = self._finalize(
                ranked, r, ctx_map, top_k, remove_seen, float(max_future_years))
            out_tracks.append(tracks)
            out_scores.append(scores)

        return pl.DataFrame(
            {"session_id": [r["session_id"] for r in rows],
             "user_id": [r["user_id"] for r in rows],
             "turn": [r["target_turn"] for r in rows],
             "track_ids": out_tracks, "scores": out_scores},
            schema={"session_id": pl.Utf8, "user_id": pl.Utf8, "turn": pl.Int64,
                    "track_ids": pl.List(pl.Utf8), "scores": pl.List(pl.Float64)},
        )

    def _candidates_for(self, bucket: tuple) -> np.ndarray:
        """Bucket ranking then global backoff (deduped, order preserved)."""
        b = self.bucket_rank.get(bucket)
        if b is None or b.size == 0:
            return self.global_rank
        rest = self.global_rank[~np.isin(self.global_rank, b)]
        return np.concatenate([b, rest])

    def _finalize(self, ranked, r, ctx_map, top_k, remove_seen, max_future_years):
        if ranked.size == 0:
            return [], []
        bad = np.zeros(ranked.size, dtype=bool)
        sd = parse_date(r["session_date"])
        if sd is not None:
            cutoff = np.datetime64(sd, "D") + np.timedelta64(int(max_future_years * 365), "D")
            rd = self.release_dates[ranked]
            bad |= (rd > cutoff) & ~np.isnat(rd)
        if remove_seen:
            seen = {self.track_to_idx.get(t, -1) for t in ctx_map.get(r["session_id"], [])}
            if seen:
                bad |= np.isin(ranked, np.fromiter(seen, dtype=np.int64, count=len(seen)))
        kept = ranked[~bad][:top_k]
        n = kept.size
        return ([self.track_ids[i] for i in kept],
                [float(n - j) for j in range(n)])   # descending positional score

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def _get_model_state(self) -> dict:
        return {
            "profile_keys": self.profile_keys, "min_bucket_count": self.min_bucket_count,
            "top_n": self.top_n, "max_future_years": self.max_future_years,
            "urm_mode": self.urm_mode, "track_ids": self.track_ids,
            "track_to_idx": self.track_to_idx, "release_dates": self.release_dates,
            "global_rank": self.global_rank, "bucket_rank": self.bucket_rank,
        }

    def _set_model_state(self, state: dict) -> None:
        for k, v in state.items():
            setattr(self, k, v)
