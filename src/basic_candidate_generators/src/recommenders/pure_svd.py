"""PureSVD: truncated SVD of the URM.

score(session) = profile @ V^T @ V
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from .user_base import UserRecommender


class PureSVDRecommender(UserRecommender):
    """Truncated SVD of URM. score = profile @ V^T @ V."""

    RECOMMENDER_NAME = "PureSVD"

    def __init__(self, n_factors: int = 256, **kwargs):
        super().__init__(**kwargs)
        self.n_factors = n_factors
        self.VT: np.ndarray | None = None  # (f, n_items)

    def _fit_model(self, urm: csr_matrix) -> None:
        f = min(self.n_factors, min(urm.shape) - 1)
        svd = TruncatedSVD(n_components=f, random_state=0)
        svd.fit(urm.astype(np.float32))
        self.VT = svd.components_.astype(np.float32)

    def _score_session_profile(self, profile: csr_matrix) -> np.ndarray:
        proj = profile @ self.VT.T  # (1, f)
        return np.asarray(proj @ self.VT).ravel()  # (n_items,)

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st.update({"n_factors": self.n_factors, "VT": self.VT})
        return st

    def _set_model_state(self, state: dict) -> None:
        super()._set_model_state(state)
        self.n_factors = state["n_factors"]
        self.VT = state["VT"]
