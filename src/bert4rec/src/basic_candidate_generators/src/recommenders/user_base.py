from __future__ import annotations

import sys as _sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

_HERE = Path(__file__).resolve()
_sys.path.insert(0, str(_HERE.parent.parent))

from BaseRecommender import BaseRecommender

from .fallback import AbstractFallback, PopularityFallback

def build_context_df(
    test_df: pl.DataFrame,
    last_turn: int | None = None,
    inject_multi_session: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    max_turns = test_df.group_by("session_id").agg(
        pl.col("turn_number").max().alias("max_turn")
    )
    test_df = test_df.join(max_turns, on="session_id", how="left")

    gt_df = test_df.filter(pl.col("turn_number") == pl.col("max_turn")).select(
        ["user_id", "session_id", "session_date", "track_id", "turn_number"]
    )

    ctx_turns = test_df.filter(pl.col("turn_number") < pl.col("max_turn"))

    sess_dates = test_df.select(["user_id", "session_id", "session_date"]).unique(
        subset=["session_id"]
    )

    earlier_rows: list[pl.DataFrame] = []
    users_with_multiple = (
        sess_dates.group_by("user_id")
        .agg(pl.col("session_id").n_unique().alias("n"))
        .filter(pl.col("n") > 1)["user_id"]
        .to_list()
    )

    if inject_multi_session and users_with_multiple:
        multi_df = test_df.filter(pl.col("user_id").is_in(users_with_multiple))
        for uid, user_sessions in multi_df.group_by("user_id"):
            sess_order = (
                user_sessions.select(["session_id", "session_date"])
                .unique(subset=["session_id"])
                .sort("session_date")
            )
            ordered_sids = sess_order["session_id"].to_list()
            ordered_dates = sess_order["session_date"].to_list()
            for k, (target_sid, target_date) in enumerate(
                zip(ordered_sids, ordered_dates)
            ):
                prior_sids = ordered_sids[:k]
                if not prior_sids:
                    continue
                prior_rows = (
                    user_sessions.filter(pl.col("session_id").is_in(prior_sids))
                    .select(["user_id", "track_id", "turn_number"])
                    .with_columns(
                        pl.lit(target_sid).alias("session_id"),
                        pl.lit(target_date).alias("session_date"),
                    )
                    .select(["user_id", "session_id", "session_date", "track_id", "turn_number"])
                )
                earlier_rows.append(prior_rows)

    ctx_base = ctx_turns.select(["user_id", "session_id", "session_date", "track_id", "turn_number"])

    if earlier_rows:
        extra = pl.concat(earlier_rows)
        context_df = pl.concat([ctx_base, extra]).unique(
            subset=["session_id", "track_id"]
        )
    else:
        context_df = ctx_base

    all_sess_meta = test_df.select(["user_id", "session_id", "session_date"]).unique(
        subset=["session_id"]
    )
    present_sids = (
        set(context_df["session_id"].to_list()) if context_df.height > 0 else set()
    )
    missing = all_sess_meta.filter(~pl.col("session_id").is_in(present_sids))
    if missing.height > 0:
        context_df = pl.concat(
            [
                context_df,
                missing.with_columns(
                    pl.lit(None, dtype=pl.Utf8).alias("track_id"),
                    pl.lit(None, dtype=pl.Int64).alias("turn_number"),
                ),
            ]
        )

    context_df = context_df.join(
        max_turns.rename({"max_turn": "target_turn"}),
        on="session_id",
        how="left",
    )

    if "conversation_goal" in test_df.columns:
        goal_meta = test_df.select(["session_id", "conversation_goal"]).unique(
            subset=["session_id"]
        )
        context_df = context_df.join(goal_meta, on="session_id", how="left")

    return context_df, gt_df

def run_inference(
    recommender,
    test_df: pl.DataFrame,
    top_k: int = 100,
    remove_seen: bool = True,
    max_future_years: float | None = None,
) -> pl.DataFrame:
    inject = getattr(recommender, "urm_mode", "user") == "user"
    context_df, gt_df = build_context_df(test_df, inject_multi_session=inject)

    kwargs = {}
    if max_future_years is not None:
        kwargs["max_future_years"] = max_future_years

    recs = recommender.recommend(
        context_df, top_k=top_k, remove_seen=remove_seen, **kwargs
    )

    recs = recs.join(
        gt_df.rename(
            {"track_id": "gt_track_id", "turn_number": "gt_turn_number"}
        ).select(["session_id", "gt_track_id", "gt_turn_number"]),
        on="session_id",
        how="left",
    )
    return recs

from .interactions import (
    IdMap,
    build_icm,
    build_id_map,
    build_track_release_dates,
    build_urm,
    build_user_seen_items,
    explode_music_turns,
    parse_date,
)

class UserRecommender(BaseRecommender):

    RECOMMENDER_NAME = "UserRecommender"

    def __init__(
        self,
        fallback: AbstractFallback | None = "default",
        max_future_years: float = 2.0,
        urm_mode: str = "user",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if fallback == "default":
            fallback = PopularityFallback()
        if urm_mode not in ("user", "session"):
            raise ValueError(f"urm_mode must be 'user' or 'session', got {urm_mode!r}")
        self.fallback = fallback
        self.max_future_years = max_future_years
        self.urm_mode = urm_mode

        self.id_map: IdMap | None = None
        self.urm: csr_matrix | None = None
        self.icm: csr_matrix | None = None
        self.track_release_dates: np.ndarray | None = None
        self.user_history: dict[str, frozenset[str]] = {}

    def fit(
        self,
        train_df: pl.DataFrame,
        track_metadata: pl.DataFrame | None = None,
        **kwargs: Any,
    ) -> None:
        long = explode_music_turns(train_df)
        extra = (
            track_metadata["track_id"].to_list() if track_metadata is not None else None
        )
        self.id_map = build_id_map(long, extra_track_ids=extra, mode=self.urm_mode)
        self.urm = build_urm(long, self.id_map)
        self.user_history = build_user_seen_items(long)

        if track_metadata is not None:
            self.track_release_dates = build_track_release_dates(
                track_metadata, self.id_map
            )
            self.icm = build_icm(track_metadata, self.id_map, interactions=long)
            n_catalog = track_metadata["track_id"].n_unique()
            icm_items = self.icm.shape[0]
            mismatch = " *** MISMATCH ***" if icm_items != n_catalog else ""
            print(
                f"[{self.RECOMMENDER_NAME}] ICM: {self.icm.shape}, nnz={self.icm.nnz}"
                f"  (catalog={n_catalog}){mismatch}"
            )

        urm_items = self.urm.shape[1]
        mode_tag = f"mode={self.urm_mode}"
        if track_metadata is not None:
            mismatch = " *** MISMATCH ***" if urm_items != n_catalog else ""
            print(
                f"[{self.RECOMMENDER_NAME}] URM: {self.urm.shape}, nnz={self.urm.nnz}"
                f"  ({mode_tag}, catalog={n_catalog}){mismatch}"
            )
        else:
            print(
                f"[{self.RECOMMENDER_NAME}] URM: {self.urm.shape}, nnz={self.urm.nnz}"
                f"  ({mode_tag})"
            )

        self._set_seeds()
        self._fit_model(self.urm)

        if self.fallback is not None:
            self.fallback.fit(train_df, track_metadata=track_metadata)

    def _set_seeds(self, seed: int | None = None) -> None:
        import os
        import random as _random
        import numpy as _np
        if seed is not None:
            s = int(seed)
        elif os.environ.get("RECSYS_SEED") is not None:
            s = int(os.environ["RECSYS_SEED"])
        else:
            s = int(getattr(self, "seed", 42))
        print(f"[seed] training RNGs seeded with {s}", flush=True)
        _random.seed(s)
        _np.random.seed(s)
        try:
            import torch as _torch
            _torch.manual_seed(s)
            if _torch.cuda.is_available():
                _torch.cuda.manual_seed_all(s)
        except ImportError:
            pass

    def _fit_model(self, urm: csr_matrix) -> None:
        raise NotImplementedError

    def _score_session_profile(self, profile: csr_matrix) -> np.ndarray:
        raise NotImplementedError

    def _filter_candidate_mask(
        self, session_date: date | None, max_future_years: float
    ) -> np.ndarray | None:
        if self.track_release_dates is None or session_date is None:
            return None
        sd64 = np.datetime64(session_date, "D")
        future_cutoff = sd64 + np.timedelta64(int(max_future_years * 365), "D")
        rd = self.track_release_dates
        return (rd <= future_cutoff) | np.isnat(rd)

    def _build_profile_vector(self, track_ids: list[str]) -> csr_matrix:
        if self.id_map is None:
            raise RuntimeError("Recommender not fitted")
        idxs = [
            self.id_map.track_to_idx[t]
            for t in track_ids
            if t in self.id_map.track_to_idx
        ]
        if not idxs:
            return csr_matrix((1, self.id_map.n_tracks), dtype=np.float32)
        data = np.ones(len(idxs), dtype=np.float32)
        rows = np.zeros(len(idxs), dtype=np.int32)
        cols = np.array(idxs, dtype=np.int32)
        return csr_matrix((data, (rows, cols)), shape=(1, self.id_map.n_tracks))

    def _topk_from_scores(
        self,
        scores: np.ndarray,
        seen_idxs: set[int],
        top_k: int,
        candidate_mask: np.ndarray | None,
        remove_seen: bool,
    ) -> tuple[list[str], list[float]]:
        s = scores.astype(np.float64, copy=True)
        if remove_seen and seen_idxs:
            s[list(seen_idxs)] = -np.inf
        if candidate_mask is not None:
            s[~candidate_mask] = -np.inf
        finite_count = int(np.isfinite(s).sum())
        if finite_count == 0:
            return [], []
        k = min(top_k, finite_count)
        idx = np.argpartition(-s, k - 1)[:k]
        idx = idx[np.argsort(-s[idx])]
        track_ids = [self.id_map.idx_to_track[i] for i in idx]
        score_vals = [float(s[i]) for i in idx]
        return track_ids, score_vals

    def recommend(
        self,
        context_df: pl.DataFrame,
        top_k: int = 20,
        remove_seen: bool = True,
        max_future_years: float | None = None,
        **kwargs: Any,
    ) -> pl.DataFrame:
        if self.id_map is None:
            raise RuntimeError("Recommender not fitted")
        if "target_turn" not in context_df.columns:
            raise ValueError(
                "context_df is missing 'target_turn' column. "
                "Use build_context_df() from build_inference_context to build context."
            )
        if max_future_years is None:
            max_future_years = self.max_future_years

        if "track_id" not in context_df.columns:
            context_df = explode_music_turns(context_df)

        out_session: list[str] = []
        out_user: list[str] = []
        out_turn: list[int] = []
        out_tracks: list[list[str]] = []
        out_scores: list[list[float]] = []

        session_meta = context_df.select(
            ["session_id", "user_id", "session_date", "target_turn"]
        ).unique(subset=["session_id"])

        if context_df.height > 0:
            ctx_map: dict[str, list[str]] = {}
            for sid, group in context_df.group_by("session_id"):
                sid_str = sid[0] if isinstance(sid, tuple) else sid
                ctx_map[sid_str] = group["track_id"].to_list()
        else:
            ctx_map = {}

        for row in session_meta.iter_rows(named=True):
            sess_id = row["session_id"]
            user_id = row["user_id"]
            target_turn = row["target_turn"]
            sd = parse_date(row["session_date"])
            mask = self._filter_candidate_mask(sd, max_future_years)

            context_tracks = [t for t in ctx_map.get(sess_id, []) if t is not None]

            seen_idxs = {
                self.id_map.track_to_idx[t]
                for t in context_tracks
                if t in self.id_map.track_to_idx
            }

            if self.urm_mode == "user":
                train_history: frozenset[str] = self.user_history.get(
                    user_id, frozenset()
                )
                profile_tracks = list(set(context_tracks) | train_history)
            else:
                sid_idx = self.id_map.user_to_idx.get(sess_id)
                if sid_idx is not None:
                    urm_row = self.urm[sid_idx]
                    session_train_tracks = [
                        self.id_map.idx_to_track[j] for j in urm_row.indices
                    ]
                else:
                    session_train_tracks = []
                profile_tracks = list(set(context_tracks) | set(session_train_tracks))
            if not profile_tracks:

                if self.fallback is not None:
                    recs, scs = self.fallback.recommend_one(sess_id, 0, sd, top_k)
                else:
                    recs, scs = [], []
            else:
                profile = self._build_profile_vector(profile_tracks)
                if profile.nnz == 0:

                    if self.fallback is not None:
                        recs, scs = self.fallback.recommend_one(sess_id, 0, sd, top_k)
                    else:
                        recs, scs = [], []
                else:
                    scores = self._score_session_profile(profile)
                    recs, scs = self._topk_from_scores(
                        scores, seen_idxs, top_k, mask, remove_seen
                    )

            out_session.append(sess_id)
            out_user.append(user_id)
            out_turn.append(target_turn)
            out_tracks.append(recs)
            out_scores.append(scs)

        return pl.DataFrame(
            {
                "session_id": out_session,
                "user_id": out_user,
                "turn": out_turn,
                "track_ids": out_tracks,
                "scores": out_scores,
            }
        )

    def _get_model_state(self) -> dict:
        return {
            "id_map": self.id_map,
            "urm": self.urm,
            "icm": self.icm,
            "track_release_dates": self.track_release_dates,
            "user_history": self.user_history,
            "fallback": self.fallback,
            "max_future_years": self.max_future_years,
            "urm_mode": self.urm_mode,
        }

    def _set_model_state(self, state: dict) -> None:
        self.id_map = state.get("id_map")
        self.urm = state.get("urm")
        self.icm = state.get("icm")
        self.track_release_dates = state.get("track_release_dates")
        self.user_history = state.get("user_history", {})
        self.fallback = state.get("fallback")
        self.max_future_years = state.get("max_future_years", 2.0)
        self.urm_mode = state.get("urm_mode", "user")
