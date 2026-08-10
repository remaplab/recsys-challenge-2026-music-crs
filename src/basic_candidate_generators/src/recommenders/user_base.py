"""UserRecommender — CF base class supporting user-based and session-based URM.

Subclasses implement:
    _fit_model(self, urm: csr_matrix) -> None
    _score_session_profile(self, profile_csr: csr_matrix) -> np.ndarray  # shape (n_tracks,)

urm_mode controls the row granularity of the URM at training time:
    "user"    — one row per user, all their sessions' tracks merged.
                Captures long-term preferences; fewer but richer rows.
    "session" — one row per session, treated as an independent entity.
                Captures within-session patterns; more but sparser rows.

At inference:
    "user"    — profile = full train history (by user_id) + context tracks.
    "session" — profile = context tracks only.
    seen_items keyed by user_id, used only for profile in user mode.
    Seen-item filter (remove_seen) always scoped to current session context only.

Future-track filter:
    Tracks released up to `max_future_years` years after session_date are candidates.
    Set max_future_years=0 to block all future tracks (strict cutoff).
"""

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

from BaseRecommender import BaseRecommender  # noqa: E402

from .fallback import AbstractFallback, PopularityFallback  # noqa: E402

# ---------------------------------------------------------------------------
# Context-building utilities (usable by any recommender)
# ---------------------------------------------------------------------------


def build_context_df(
    test_df: pl.DataFrame,
    last_turn: int | None = None,
    inject_multi_session: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split each test session into context + ground-truth.

    Parameters
    ----------
    test_df:
        Long-format DataFrame with columns (user_id, session_id, session_date,
        turn_number, track_id).  All turns of each test session.
    last_turn:
        If None, the last turn for each session is auto-detected (max turn_number).
        Explicit value useful when all sessions have a fixed number of turns.
    inject_multi_session:
        If True (default), earlier test sessions of the same user are injected
        into the context of later sessions.  Only correct for urm_mode="user"
        (URM row = user); for urm_mode="session" the profile should stay
        session-scoped to match the training distribution.

    Returns
    -------
    context_df:
        (user_id, session_id, session_date, track_id, target_turn) — context turns only.
        target_turn is the GT turn number being predicted.
    gt_df:
        (user_id, session_id, session_date, track_id) — the last (GT) turn only.
    """
    # maintain_order: unordered group_by row order otherwise leaks into GPU
    # batch composition downstream (float32 reduction noise → top-k reshuffle).
    max_turns = test_df.group_by("session_id", maintain_order=True).agg(
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
        sess_dates.group_by("user_id", maintain_order=True)
        .agg(pl.col("session_id").n_unique().alias("n"))
        .filter(pl.col("n") > 1)["user_id"]
        .to_list()
    )

    if inject_multi_session and users_with_multiple:
        multi_df = test_df.filter(pl.col("user_id").is_in(users_with_multiple))
        for uid, user_sessions in multi_df.group_by("user_id", maintain_order=True):
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

    # turn_number is preserved so per-turn-aware recommenders (e.g. the
    # query-injection variants) can recover the original turn index of each
    # prior track at scoring time. Backwards compatible: parent recommend()
    # already conditionally sorts by turn_number if present.
    ctx_base = ctx_turns.select(["user_id", "session_id", "session_date", "track_id", "turn_number"])

    if earlier_rows:
        extra = pl.concat(earlier_rows)
        context_df = pl.concat([ctx_base, extra]).unique(
            subset=["session_id", "track_id"]
        )
    else:
        context_df = ctx_base

    # Ensure every session appears in context_df even with no context turns (L=1).
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

    # Attach target_turn so recommenders emit the correct turn in their output.
    context_df = context_df.join(
        max_turns.rename({"max_turn": "target_turn"}),
        on="session_id",
        how="left",
    )

    # Optional pass-through of session-level conversation_goal struct
    # (used by HeuristicRecommender for cat/spec; ignored by other recommenders).
    if "conversation_goal" in test_df.columns:
        goal_meta = test_df.select(["session_id", "conversation_goal"]).unique(
            subset=["session_id"]
        )
        context_df = context_df.join(goal_meta, on="session_id", how="left")

    # Optional pass-through of session-level user_profile struct
    # (used by PopPriorCG for profile-conditioned popularity; ignored otherwise).
    if "user_profile" in test_df.columns:
        profile_meta = test_df.select(["session_id", "user_profile"]).unique(
            subset=["session_id"]
        )
        context_df = context_df.join(profile_meta, on="session_id", how="left")

    return context_df, gt_df


def run_inference(
    recommender,
    test_df: pl.DataFrame,
    top_k: int = 100,
    remove_seen: bool = True,
    max_future_years: float | None = None,
) -> pl.DataFrame:
    """Build context, run recommender, attach GT if available.

    Returns DataFrame with
    (session_id, user_id, turn, track_ids, scores, gt_track_id, gt_turn_number).
    gt_turn_number is the turn position of the last music turn per session,
    used by downstream macro-by-turn scoring.
    """
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


from .interactions import (  # noqa: E402
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
    """See module docstring. Subclasses fill in _fit_model + _score_session_profile."""

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

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df: pl.DataFrame,
        track_metadata: pl.DataFrame | None = None,
        precomputed_icm: csr_matrix | None = None,
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
            # The ICM is HP-independent (depends only on this fold's train_df +
            # track_metadata, against a deterministic id_map). Reuse a prebuilt one
            # across tuning trials when given; rebuild on shape mismatch.
            if (precomputed_icm is not None
                    and precomputed_icm.shape[0] == self.id_map.n_tracks):
                self.icm = precomputed_icm
            else:
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
        self._fit_model(self.urm)

        if self.fallback is not None:
            self.fallback.fit(train_df, track_metadata=track_metadata)

    def _fit_model(self, urm: csr_matrix) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # recommend
    # ------------------------------------------------------------------

    def _score_session_profile(self, profile: csr_matrix) -> np.ndarray:
        raise NotImplementedError

    def _filter_candidate_mask(
        self, session_date: date | None, max_future_years: float
    ) -> np.ndarray | None:
        """Bool mask over track indices: True = valid candidate.

        Tracks released up to max_future_years after session_date are kept.
        Tracks with missing release_date metadata are always kept.
        """
        if self.track_release_dates is None or session_date is None:
            return None
        sd64 = np.datetime64(session_date, "D")
        future_cutoff = sd64 + np.timedelta64(int(max_future_years * 365), "D")
        rd = self.track_release_dates
        return (rd <= future_cutoff) | np.isnat(rd)

    def _build_profile_vector(self, track_ids: list[str]) -> csr_matrix:
        """Return (1, n_tracks) binary CSR from a list of track IDs.

        Unknown track IDs (outside vocabulary) are silently dropped.
        Returns an all-zero row if no given ID maps to the vocabulary.
        """
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
        """Convert raw score vector to ranked (track_ids, scores) lists.

        Applies seen-item removal and candidate_mask by setting excluded entries
        to -inf before selection.  Uses argpartition (O(n)) then sorts only the
        top-k slice, keeping inference fast even on large catalogues.
        """
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
        """Recommend for each session in context_df.

        context_df must have columns: session_id, user_id, session_date, track_id,
        target_turn.  target_turn is the turn number being predicted (GT turn).
        Use build_inference_context.build_context_df to produce a conforming DataFrame.
        Returns one row per session: session_id, user_id, turn, track_ids, scores.
        """
        if self.id_map is None:
            raise RuntimeError("Recommender not fitted")
        if "target_turn" not in context_df.columns:
            raise ValueError(
                "context_df is missing 'target_turn' column. "
                "Use build_context_df() from build_inference_context to build context."
            )
        if max_future_years is None:
            max_future_years = self.max_future_years

        # Normalize to long format if needed
        if "track_id" not in context_df.columns:
            context_df = explode_music_turns(context_df)

        out_session: list[str] = []
        out_user: list[str] = []
        out_turn: list[int] = []
        out_tracks: list[list[str]] = []
        out_scores: list[list[float]] = []

        # Collect all unique (session_id, user_id, session_date, target_turn) combos first
        # so we can handle sessions with zero context rows
        session_meta = context_df.select(
            ["session_id", "user_id", "session_date", "target_turn"]
        ).unique(subset=["session_id"])

        # Build per-session context track lists
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

            # seen filter: only current session context tracks
            seen_idxs = {
                self.id_map.track_to_idx[t]
                for t in context_tracks
                if t in self.id_map.track_to_idx
            }

            # profile: merge context tracks with known training history
            # user mode:    history = all train tracks for user_id
            # session mode: history = URM row for sess_id (if session seen in training)
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
                # fully cold — no history at all
                if self.fallback is not None:
                    recs, scs = self.fallback.recommend_one(sess_id, 0, sd, top_k)
                else:
                    recs, scs = [], []
            else:
                profile = self._build_profile_vector(profile_tracks)
                if profile.nnz == 0:
                    # all tracks outside vocabulary — cold
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

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

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
