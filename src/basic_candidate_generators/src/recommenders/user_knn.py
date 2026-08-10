"""User-KNN: here "user" = train session.

For each test session profile, find the most similar training sessions
(cosine similarity over their track sets) and aggregate their track plays.
Concretely: scores = profile_norm @ URM_norm^T  →  sim_to_train_sessions
            then sim_top_k @ URM  →  track scores.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.sparse import csr_matrix, hstack as sp_hstack
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm

from .user_base import UserRecommender


class UserKNNRecommender(UserRecommender):
    RECOMMENDER_NAME = "UserKNN"

    def __init__(self, k: int = 100, icm_weight: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.icm_weight = icm_weight
        self.urm_normed: csr_matrix | None = None
        self._icm_normed: csr_matrix | None = None  # cached for scoring

    def _fit_model(self, urm: csr_matrix) -> None:
        if self.icm is not None and self.icm_weight > 0:
            # Per-session content profile: sum of normalized ICM rows for listened tracks
            self._icm_normed = normalize(self.icm.astype(np.float32), norm="l2", axis=1).tocsr()
            sess_icm = urm.astype(np.float32) @ self._icm_normed          # (n_sessions, n_icm)
            sess_icm_norm = normalize(sess_icm, norm="l2", axis=1) * self.icm_weight
            urm_norm = normalize(urm.astype(np.float32), norm="l2", axis=1)
            aug = sp_hstack([urm_norm, sess_icm_norm], format="csr")
            self.urm_normed = normalize(aug, norm="l2", axis=1).tocsr()
        else:
            self.urm_normed = normalize(urm.astype(np.float32), norm="l2", axis=1).tocsr()
        print(f"[{self.RECOMMENDER_NAME}] fitted (lazy: neighbours computed at inference)")

    def _score_session_profile(self, profile: csr_matrix) -> np.ndarray:
        if self._icm_normed is not None and self.icm_weight > 0:
            p_icm = profile.astype(np.float32) @ self._icm_normed
            p_icm_norm = normalize(p_icm, norm="l2", axis=1) * self.icm_weight
            p_urm = normalize(profile.astype(np.float32), norm="l2", axis=1)
            p = normalize(sp_hstack([p_urm, p_icm_norm], format="csr"), norm="l2", axis=1)
        else:
            p = normalize(profile.astype(np.float32), norm="l2", axis=1)

        sims = p.dot(self.urm_normed.T).toarray().flatten()
        if self.k < len(sims):
            top = np.argpartition(-sims, self.k)[: self.k]
        else:
            top = np.arange(len(sims))
        top = top[sims[top] > 0]
        if len(top) == 0:
            # scores only over CF items (first urm.shape[1] dims)
            return np.zeros(self.urm.shape[1], dtype=np.float32)
        weights = sims[top]
        sub = self.urm_normed[top]
        scores = (sub.multiply(weights[:, None])).sum(axis=0)
        scores_arr = np.asarray(scores).flatten()
        # urm_normed may have extra ICM cols; return only the first n_items cols
        return scores_arr[:self.urm.shape[1]]

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st.update({"k": self.k, "icm_weight": self.icm_weight,
                   "urm_normed": self.urm_normed, "_icm_normed": self._icm_normed})
        return st

    def _set_model_state(self, state: dict) -> None:
        super()._set_model_state(state)
        self.k = state["k"]
        self.icm_weight = state.get("icm_weight", 0.0)
        self.urm_normed = state["urm_normed"]
        self._icm_normed = state.get("_icm_normed")
