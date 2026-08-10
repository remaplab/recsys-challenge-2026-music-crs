"""RP3Beta: random-walk based item-item similarity (Paudel et al. 2017).

Formula:
    Pu_i  = R / row_sum(R)            # user→item transition prob
    Pi_u  = R.T / col_sum(R)          # item→user transition prob
    W_ij  = sum_u  Pi_u[i,u] * Pu_i[u,j] / pop_j^beta,  then raised to alpha

Only the top-k neighbours per item are kept.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.sparse import csr_matrix, diags

from .interactions import build_item_cbf_similarity
from .user_base import UserRecommender


class RP3BetaRecommender(UserRecommender):
    RECOMMENDER_NAME = "RP3Beta"

    def __init__(self, alpha: float = 0.5, beta: float = 0.3, top_k: int = 100,
                 icm_weight: float = 0.0, k_icm: int = 150, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.top_k_neighbours = top_k
        self.icm_weight = icm_weight
        self.k_icm = k_icm
        self.W: csr_matrix | None = None

    def _fit_model(self, urm: csr_matrix) -> None:
        t0 = time.time()
        W_rp3 = self._fit_rp3(urm)
        print(f"[{self.RECOMMENDER_NAME}] fit in {time.time()-t0:.1f}s, nnz={W_rp3.nnz}")

        if self.icm is not None and self.icm_weight > 0:
            t1 = time.time()
            W_cbf = build_item_cbf_similarity(self.icm, self.k_icm)
            print(f"[{self.RECOMMENDER_NAME}] CBF fit in {time.time()-t1:.1f}s, nnz={W_cbf.nnz}")
            self.W = (1.0 - self.icm_weight) * W_rp3 + self.icm_weight * W_cbf
        else:
            self.W = W_rp3

    def _fit_rp3(self, urm: csr_matrix) -> csr_matrix:
        R = urm.astype(np.float32)
        row_sums = np.array(R.sum(axis=1)).flatten()
        col_sums = np.array(R.sum(axis=0)).flatten()

        Du_inv = diags(1.0 / np.maximum(row_sums, 1e-10))
        Di_inv = diags(1.0 / np.maximum(col_sums, 1e-10))
        Pu_i = Du_inv @ R          # (n_users, n_items) — row-stochastic
        Pi_u = Di_inv @ R.T        # (n_items, n_users) — col-stochastic

        W_csr = (Pi_u @ Pu_i).tocsr()   # (n_items, n_items)

        # popularity damping
        pop = np.maximum(col_sums, 1.0)
        W_csr = (W_csr @ diags(np.power(pop, -self.beta, dtype=np.float32))).tocsr()

        # alpha exponent
        W_csr.data = np.power(np.maximum(W_csr.data, 0.0), self.alpha).astype(np.float32)
        W_csr.setdiag(0.0)
        W_csr.eliminate_zeros()

        # keep top-k per item
        K = self.top_k_neighbours
        n = W_csr.shape[0]
        rows, cols, data = [], [], []
        for i in range(n):
            s, e = W_csr.indptr[i], W_csr.indptr[i + 1]
            if s == e:
                continue
            r_data = W_csr.data[s:e]
            r_cols = W_csr.indices[s:e]
            top = np.argpartition(-r_data, min(K, len(r_data)) - 1)[:K] if K < len(r_data) else np.arange(len(r_data))
            m = r_data[top] > 0
            rows.extend([i] * int(m.sum()))
            cols.extend(r_cols[top[m]].tolist())
            data.extend(r_data[top[m]].tolist())
        return csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)

    def _score_session_profile(self, profile: csr_matrix) -> np.ndarray:
        return np.asarray(profile.dot(self.W).todense()).flatten()

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st.update({"alpha": self.alpha, "beta": self.beta,
                   "top_k_neighbours": self.top_k_neighbours,
                   "icm_weight": self.icm_weight, "k_icm": self.k_icm, "W": self.W})
        return st

    def _set_model_state(self, state: dict) -> None:
        super()._set_model_state(state)
        self.alpha = state["alpha"]
        self.beta = state["beta"]
        self.top_k_neighbours = state["top_k_neighbours"]
        self.icm_weight = state.get("icm_weight", 0.0)
        self.k_icm = state.get("k_icm", 150)
        self.W = state["W"]
