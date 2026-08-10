"""TopPopular with selectable temporal windows.

Popularity is computed at inference time per (session_date, window):
    popularity[t] = #plays in train where t was played within
                    [session_date - window, session_date)

Future tracks (release_date > session_date) are filtered out automatically.

Window choices (string):
    "1w", "2w"
    "1m", "2m", "3m", "6m"
    "1y", "2y", "3y", "5y"
    "all"   (default — full train history up to session_date)

This recommender does NOT use a session profile. It returns the same
ranking for all sessions sharing a session_date. Therefore it is also a
sensible default fallback (set fallback=None when constructing it to
avoid recursion).
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

from .interactions import explode_music_turns, parse_date
from .user_base import UserRecommender


_WINDOW_DAYS: dict[str, int | None] = {
    "1w": 7, "2w": 14,
    "1m": 30, "2m": 60, "3m": 90, "6m": 180,
    "1y": 365, "2y": 730, "3y": 1095, "5y": 1825,
    "all": None,
}


class TopPopularRecommender(UserRecommender):
    RECOMMENDER_NAME = "TopPopular"

    def __init__(self, window: str = "all", log_scale: bool = True, **kwargs):
        if window not in _WINDOW_DAYS:
            raise ValueError(f"window must be one of {list(_WINDOW_DAYS)}")
        kwargs.setdefault("fallback", None)  # TopPopular handles cold rows itself
        super().__init__(**kwargs)
        self.window = window
        self.log_scale = log_scale
        # Per-interaction track index + session_date (sorted by date)
        self._sorted_track_idx: np.ndarray | None = None
        self._sorted_dates: np.ndarray | None = None  # datetime64[D]

    def _fit_model(self, urm: csr_matrix) -> None:
        # We need (track_id, session_date) for every interaction. Stash from explode.
        # Note: we don't have train_df here; subclass overrides fit to capture it.
        ...

    def fit(self, train_df: pl.DataFrame, track_metadata: pl.DataFrame | None = None, **kwargs: Any) -> None:
        super().fit(train_df, track_metadata=track_metadata, **kwargs)
        long = explode_music_turns(train_df).filter(pl.col("session_date").is_not_null())
        long = long.sort("session_date")
        track_idx = np.array(
            [self.id_map.track_to_idx[t] for t in long["track_id"].to_list()],
            dtype=np.int32,
        )
        dates = np.array(
            [np.datetime64(d, "D") for d in long["session_date"].to_list()],
            dtype="datetime64[D]",
        )
        self._sorted_track_idx = track_idx
        self._sorted_dates = dates
        print(f"[{self.RECOMMENDER_NAME}] indexed {len(track_idx)} dated interactions")

    # Override recommend: popularity doesn't use a user profile
    def recommend(
        self,
        context_df: pl.DataFrame,
        top_k: int = 20,
        remove_seen: bool = True,
        max_future_years: float | None = None,
        **kwargs,
    ) -> pl.DataFrame:
        if self.id_map is None:
            raise RuntimeError("Recommender not fitted")
        if max_future_years is None:
            max_future_years = self.max_future_years

        session_meta = (
            context_df
            .select(["session_id", "user_id", "session_date"])
            .unique(subset=["session_id"])
        )
        ctx_map: dict[str, list[str]] = {}
        for sid, group in context_df.group_by("session_id"):
            sid_str = sid[0] if isinstance(sid, tuple) else sid
            ctx_map[sid_str] = group["track_id"].to_list()

        out_session, out_user, out_tracks, out_scores = [], [], [], []
        for row in session_meta.iter_rows(named=True):
            sess_id = row["session_id"]
            user_id = row["user_id"]
            sd = parse_date(row["session_date"])
            mask = self._filter_candidate_mask(sd, max_future_years)
            pop_scores = self._popularity_for_date(sd)

            context_tracks = [t for t in ctx_map.get(sess_id, []) if t is not None]
            seen_idxs = {
                self.id_map.track_to_idx[t] for t in context_tracks if t in self.id_map.track_to_idx
            }
            recs, scs = self._topk_from_scores(pop_scores, seen_idxs, top_k, mask, remove_seen)
            out_session.append(sess_id)
            out_user.append(user_id)
            out_tracks.append(recs)
            out_scores.append(scs)

        return pl.DataFrame({"session_id": out_session, "user_id": out_user,
                             "track_ids": out_tracks, "scores": out_scores})

    def _popularity_for_date(self, session_date: date | None) -> np.ndarray:
        n = self.id_map.n_tracks
        if self._sorted_dates is None or session_date is None:
            counts = np.bincount(self._sorted_track_idx or [], minlength=n) if self._sorted_track_idx is not None else np.zeros(n)
            return np.log1p(counts) if self.log_scale else counts.astype(np.float64)

        sd64 = np.datetime64(session_date, "D")
        wd = _WINDOW_DAYS[self.window]
        lo64 = np.datetime64(session_date - timedelta(days=wd), "D") if wd is not None else None
        # binary search to slice [lo, sd)
        hi = int(np.searchsorted(self._sorted_dates, sd64, side="left"))
        lo = int(np.searchsorted(self._sorted_dates, lo64, side="left")) if lo64 is not None else 0
        if hi <= lo:
            return np.zeros(n, dtype=np.float64)
        sl = self._sorted_track_idx[lo:hi]
        counts = np.bincount(sl, minlength=n).astype(np.float64)
        return np.log1p(counts) if self.log_scale else counts

    # _score_session_profile is unused (recommend overridden); provide stub
    def _score_session_profile(self, profile):
        raise NotImplementedError("TopPopular doesn't use session profile")

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st.update(
            {
                "window": self.window,
                "log_scale": self.log_scale,
                "_sorted_track_idx": self._sorted_track_idx,
                "_sorted_dates": self._sorted_dates,
            }
        )
        return st

    def _set_model_state(self, state: dict) -> None:
        super()._set_model_state(state)
        self.window = state["window"]
        self.log_scale = state["log_scale"]
        self._sorted_track_idx = state["_sorted_track_idx"]
        self._sorted_dates = state["_sorted_dates"]
