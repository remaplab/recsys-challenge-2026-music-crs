"""EASE — Embarrassingly Shallow Auto-Encoder (Steck 2019).

Closed-form solution:
    G = R^T R + lambda * I
    P = G^-1
    B = P / -diag(P)
    diag(B) = 0
    scores(u) = R[u] @ B
"""

from __future__ import annotations

import time

import numpy as np
from scipy.sparse import csr_matrix

from .interactions import build_item_cbf_similarity
from .user_base import UserRecommender


class EASERecommender(UserRecommender):
    RECOMMENDER_NAME = "EASE"

    def __init__(self, lambda_reg: float = 500.0, icm_weight: float = 0.0, k_icm: int = 150, **kwargs):
        super().__init__(**kwargs)
        self.lambda_reg = lambda_reg
        self.icm_weight = icm_weight
        self.k_icm = k_icm
        self.B: np.ndarray | None = None  # (n_items, n_items)

    def _fit_model(self, urm: csr_matrix) -> None:
        t0 = time.time()
        G = (urm.T @ urm).toarray().astype(np.float32)
        idx = np.diag_indices_from(G)
        G[idx] += self.lambda_reg
        P = np.linalg.inv(G)
        diag_P = np.diag(P)
        B = P / -diag_P
        B[idx[0], idx[1]] = 0.0
        B_cf = B.astype(np.float32)
        print(f"[{self.RECOMMENDER_NAME}] λ={self.lambda_reg} CF fit in {time.time()-t0:.1f}s")

        if self.icm is not None and self.icm_weight > 0:
            t1 = time.time()
            W_cbf = build_item_cbf_similarity(self.icm, self.k_icm)
            print(f"[{self.RECOMMENDER_NAME}] CBF fit in {time.time()-t1:.1f}s, nnz={W_cbf.nnz}")
            self.B = (1.0 - self.icm_weight) * B_cf + self.icm_weight * W_cbf.toarray()
        else:
            self.B = B_cf

    def _score_session_profile(self, profile: csr_matrix) -> np.ndarray:
        return np.asarray(profile @ self.B).flatten()

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st.update({"lambda_reg": self.lambda_reg, "icm_weight": self.icm_weight,
                   "k_icm": self.k_icm, "B": self.B})
        return st

    def _set_model_state(self, state: dict) -> None:
        super()._set_model_state(state)
        self.lambda_reg = state["lambda_reg"]
        self.icm_weight = state.get("icm_weight", 0.0)
        self.k_icm = state.get("k_icm", 150)
        self.B = state["B"]
