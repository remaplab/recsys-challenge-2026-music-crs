"""SequentialRules: item→item transition matrix from ordered session turns.

Each co-occurring (i→j) pair within max_dist turns contributes weight 1/distance.
Score = profile @ S.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

from .interactions import explode_music_turns
from .user_base import UserRecommender


class SequentialRulesRecommender(UserRecommender):
    """Item→item transition matrix from ordered turns, weight 1/distance."""

    RECOMMENDER_NAME = "SeqRules"

    def __init__(self, max_dist: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.max_dist = max_dist
        self.S: csr_matrix | None = None

    def fit(self, train_df, track_metadata=None, **kw):
        super().fit(train_df, track_metadata=track_metadata, **kw)

        tt = self.id_map.track_to_idx
        idx_df = pl.DataFrame({
            "track_id": list(tt.keys()),
            "idx": np.fromiter(tt.values(), dtype=np.int64),
        })

        long = (
            explode_music_turns(train_df)
            .join(idx_df, on="track_id")
            .drop_nulls("idx")
        )
        a = long.select(["session_id", "turn_number", "idx"])
        b = a.rename({"turn_number": "t2", "idx": "j"})
        pairs = (
            a.join(b, on="session_id")
            .filter(
                (pl.col("t2") > pl.col("turn_number")) &
                (pl.col("t2") - pl.col("turn_number") <= self.max_dist)
            )
            .with_columns((1.0 / (pl.col("t2") - pl.col("turn_number"))).alias("w"))
            .group_by(["idx", "j"]).agg(pl.col("w").sum())
        )

        n = self.id_map.n_tracks
        self.S = csr_matrix(
            (pairs["w"].to_numpy(), (pairs["idx"].to_numpy(), pairs["j"].to_numpy())),
            shape=(n, n),
            dtype=np.float32,
        )

    def _fit_model(self, urm: csr_matrix) -> None:
        pass

    def _score_session_profile(self, profile: csr_matrix) -> np.ndarray:
        return np.asarray((profile @ self.S).todense()).ravel()

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st.update({"max_dist": self.max_dist, "S": self.S})
        return st

    def _set_model_state(self, state: dict) -> None:
        super()._set_model_state(state)
        self.max_dist = state["max_dist"]
        self.S = state["S"]
