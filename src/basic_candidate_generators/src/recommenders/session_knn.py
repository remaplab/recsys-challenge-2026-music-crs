"""SessionKNN (VS-KNN): find similar training sessions via IDF-weighted cosine,
score their items by weighted vote.

Contrast with ItemKNN: ItemKNN pre-trains an item×item W matrix and scores via
`profile @ W`. SessionKNN keeps the raw URM and at inference time finds the k
nearest training sessions, then aggregates their item vectors weighted by
session similarity — no pre-trained item similarity needed.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from .user_base import UserRecommender


class SessionKNNRecommender(UserRecommender):
    """VS-KNN: find similar training sessions (IDF cosine), score their items."""

    RECOMMENDER_NAME = "SessionKNN"

    def __init__(self, k: int = 200, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self._urm_train: csr_matrix | None = None
        self._urm_T: csr_matrix | None = None
        self._idf: np.ndarray | None = None
        self._sess_norm: np.ndarray | None = None

    def _fit_model(self, urm: csr_matrix) -> None:
        self._urm_train = urm.tocsr().astype(np.float32)
        self._urm_T = self._urm_train.T.tocsr()
        m = self._urm_train.shape[0]
        df = np.asarray((self._urm_train > 0).sum(axis=0)).ravel()
        self._idf = (np.log((1.0 + m) / (1.0 + df)) + 1.0).astype(np.float32)
        sq = self._urm_train.multiply(self._idf[None, :]).power(2)
        self._sess_norm = np.sqrt(
            np.asarray(sq.sum(axis=1)).ravel()
        ).astype(np.float32)
        self._sess_norm = np.maximum(self._sess_norm, 1e-8)

    def _score_session_profile(self, profile: csr_matrix) -> np.ndarray:
        idxs = profile.indices
        pw = self._idf[idxs]
        pnorm = float(np.sqrt((pw * pw).sum())) + 1e-8
        pw_row = csr_matrix(
            (pw, (np.zeros_like(idxs), idxs)),
            shape=(1, self._urm_train.shape[1]),
            dtype=np.float32,
        )
        sims = np.asarray((pw_row @ self._urm_T).todense()).ravel()  # (m,)
        sims /= pnorm * self._sess_norm
        k = min(self.k, sims.shape[0])
        nb = np.argpartition(-sims, k - 1)[:k]
        nb = nb[sims[nb] > 0]
        if nb.size == 0:
            return np.zeros(self._urm_train.shape[1], dtype=np.float32)
        scores = self._urm_train[nb].T @ sims[nb]
        return np.asarray(scores).ravel() * self._idf

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st.update({
            "k": self.k,
            "_urm_train": self._urm_train,
            "_urm_T": self._urm_T,
            "_idf": self._idf,
            "_sess_norm": self._sess_norm,
        })
        return st

    def _set_model_state(self, state: dict) -> None:
        super()._set_model_state(state)
        self.k = state["k"]
        self._urm_train = state["_urm_train"]
        self._urm_T = state["_urm_T"]
        self._idf = state["_idf"]
        self._sess_norm = state["_sess_norm"]
