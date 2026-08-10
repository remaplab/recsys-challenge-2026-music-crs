"""Prod2Vec: item embeddings from shifted positive PMI co-occurrence + SVD.

SGNS-equivalent without actual word2vec training:
  1. Compute item co-occurrence C = B^T B (B = binary URM)
  2. Apply SPPMI: PPMI[i,j] = max(0, log(C[i,j]*total/(r_i*r_j)) - log(neg))
  3. Truncated SVD on SPPMI → embeddings E = U * sqrt(sigma)
  4. score = profile @ E @ E^T
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from .user_base import UserRecommender


class Prod2VecRecommender(UserRecommender):
    """Item embeddings via shifted-PPMI co-occurrence + SVD."""

    RECOMMENDER_NAME = "Prod2Vec"

    def __init__(self, n_factors: int = 200, neg: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.n_factors = n_factors
        self.neg = neg
        self.E: np.ndarray | None = None  # (n_items, f)

    def _fit_model(self, urm: csr_matrix) -> None:
        B = (urm > 0).astype(np.float32)
        C = (B.T @ B).tocsr()
        C.setdiag(0)
        C.eliminate_zeros()

        total = C.sum()
        rs = np.asarray(C.sum(axis=1)).ravel() + 1e-8
        c = C.tocoo()
        pmi = np.log((c.data * total) / (rs[c.row] * rs[c.col]) + 1e-12) - np.log(self.neg)
        pmi = np.maximum(pmi, 0.0)  # SPPMI
        mask = pmi > 0
        sppmi = csr_matrix(
            (pmi[mask], (c.row[mask], c.col[mask])),
            shape=C.shape,
            dtype=np.float32,
        )

        f = min(self.n_factors, min(sppmi.shape) - 1)
        svd = TruncatedSVD(n_components=f, random_state=0)
        U = svd.fit_transform(sppmi)
        self.E = (U * np.sqrt(svd.singular_values_)).astype(np.float32)

    def _score_session_profile(self, profile: csr_matrix) -> np.ndarray:
        proj = profile @ self.E  # (1, f)
        return np.asarray(proj @ self.E.T).ravel()  # (n_items,)

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st.update({"n_factors": self.n_factors, "neg": self.neg, "E": self.E})
        return st

    def _set_model_state(self, state: dict) -> None:
        super()._set_model_state(state)
        self.n_factors = state["n_factors"]
        self.neg = state["neg"]
        self.E = state["E"]
