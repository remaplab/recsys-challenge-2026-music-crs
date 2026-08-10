"""SLIM ElasticNet — coordinate descent, numba parallel, active-set, sparse output.

For each item j, solves:
    min_w  0.5/n * ||R[:,j] - R w||^2
         + alpha * l1_ratio * ||w||_1
         + 0.5 * alpha * (1 - l1_ratio) * ||w||^2
    s.t.  w[j] = 0,  w >= 0

Active-set reduction: Gram built only over items with at least one interaction
(n_active × n_active float32). No dense W materialization — CD writes sparse COO
directly. Memory: O(n_active^2) vs O(n_items^2 * 2).
"""

from __future__ import annotations

import os
import time
import warnings

import numpy as np
from numba import njit, prange, set_num_threads
from scipy.sparse import csr_matrix

from .interactions import build_item_cbf_similarity_fast
from .user_base import UserRecommender


@njit(cache=True)
def _soft_threshold(x: float, t: float) -> float:
    if x > t:
        return x - t
    if x < -t:
        return x + t
    return 0.0


@njit(cache=True)
def _top_k_thresh(abs_w: np.ndarray, top_k: int) -> float:
    """Value of the top_k-th largest element (0.0 if top_k >= n)."""
    n = len(abs_w)
    if top_k >= n:
        return 0.0
    tmp = abs_w.copy()
    tmp.sort()
    return tmp[n - top_k]


@njit(cache=True)
def _cd_item(
    G: np.ndarray,
    gj: np.ndarray,
    j_local: int,
    l1: float,
    l2: float,
    max_iter: int,
    tol: float,
    top_k: int,
) -> np.ndarray:
    """Coordinate descent for item j in local active-item space (G is float32, n_active×n_active)."""
    n = len(gj)
    w = np.zeros(n)
    r = gj.copy()

    for _ in range(max_iter):
        max_change = 0.0
        for k in range(n):
            if k == j_local:
                continue
            rk = r[k] + G[k, k] * w[k]
            new_wk = _soft_threshold(rk, l1) / (G[k, k] + l2)
            if new_wk < 0.0:
                new_wk = 0.0
            delta = new_wk - w[k]
            if delta == 0.0:
                continue
            d_abs = delta if delta > 0.0 else -delta
            if d_abs > max_change:
                max_change = d_abs
            for l in range(n):
                r[l] -= G[l, k] * delta
            w[k] = new_wk
        w[j_local] = 0.0
        if max_change < tol:
            break

    abs_w = np.abs(w)
    thresh = _top_k_thresh(abs_w, top_k)
    for k in range(n):
        if abs_w[k] < thresh:
            w[k] = 0.0
    w[j_local] = 0.0
    return w


@njit(parallel=True, cache=True)
def _fit_active_sparse(
    G: np.ndarray,
    l1: float,
    l2: float,
    max_iter: int,
    tol: float,
    top_k: int,
):
    """Parallel CD over active-item Gram; returns sparse COO arrays (no full W_dense)."""
    n = G.shape[0]
    real_k = min(top_k, n - 1)
    out_cols = np.full((n, real_k), -1, dtype=np.int32)
    out_vals = np.zeros((n, real_k), dtype=np.float32)
    out_nnz = np.zeros(n, dtype=np.int32)

    for j in prange(n):
        gj = G[:, j].copy()
        w = _cd_item(G, gj, j, l1, l2, max_iter, tol, real_k)
        nz = np.int32(0)
        for k in range(n):
            if w[k] > np.float32(0.0) and nz < real_k:
                out_cols[j, nz] = k
                out_vals[j, nz] = np.float32(w[k])
                nz += np.int32(1)
        out_nnz[j] = nz
    return out_cols, out_vals, out_nnz


class SLIMElasticNetRecommender(UserRecommender):
    RECOMMENDER_NAME = "SLIM_ElasticNet"

    def __init__(
        self,
        top_k: int = 100,
        alpha: float = 1e-4,
        l1_ratio: float = 0.5,
        max_iter: int = 100,
        tol: float = 1e-4,
        workers: int = 15,
        icm_weight: float = 0.0,
        k_icm: int = 150,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.workers = workers
        self.icm_weight = icm_weight
        self.k_icm = k_icm
        self.W: csr_matrix | None = None

    def _fit_model(self, urm: csr_matrix) -> None:
        t0 = time.time()
        n_users, n_items = urm.shape

        # Active-set: only items with at least one interaction
        col_nnz = np.diff(urm.tocsc().indptr)
        active = np.where(col_nnz > 0)[0].astype(np.int32)
        n_active = len(active)

        gram_gb = n_active ** 2 * 4 / 1e9  # float32
        if gram_gb > 8.0:
            warnings.warn(
                f"[{self.RECOMMENDER_NAME}] Gram ~{gram_gb:.1f} GB "
                f"({n_active} active items). Consider a smaller model."
            )
        print(f"[{self.RECOMMENDER_NAME}] Active {n_active}/{n_items} ({n_active/n_items:.1%})  "
              f"Gram {n_active}×{n_active} ({gram_gb:.2f} GB float32)...")

        R_act = urm[:, active].astype(np.float32)
        G = np.asfortranarray(R_act.T.dot(R_act).toarray(), dtype=np.float32) / n_users

        l1 = self.alpha * self.l1_ratio
        l2 = self.alpha * (1.0 - self.l1_ratio)

        set_num_threads(self.workers)
        print(f"[{self.RECOMMENDER_NAME}] CD threads={self.workers} "
              f"max_iter={self.max_iter} top_k={self.top_k}...")

        out_cols, out_vals, out_nnz = _fit_active_sparse(
            G, l1, l2, self.max_iter, self.tol, self.top_k
        )
        del G, R_act

        # Flatten local COO → global indices
        total = int(out_nnz.sum())
        rows_l = np.empty(total, np.int32)
        cols_l = np.empty(total, np.int32)
        vals_f = np.empty(total, np.float32)
        pos = 0
        for i in range(n_active):
            cnt = int(out_nnz[i])
            if cnt:
                rows_l[pos:pos + cnt] = i
                cols_l[pos:pos + cnt] = out_cols[i, :cnt]
                vals_f[pos:pos + cnt] = out_vals[i, :cnt]
                pos += cnt

        W_cd = csr_matrix(
            (vals_f, (active[rows_l], active[cols_l])),
            shape=(n_items, n_items), dtype=np.float32,
        )
        print(f"[{self.RECOMMENDER_NAME}] CD done nnz={W_cd.nnz} in {time.time()-t0:.1f}s")

        # Restore full parallelism for ICM
        import numba
        set_num_threads(min(os.cpu_count(), numba.get_num_threads()))

        if self.icm is not None and self.icm_weight > 0:
            t1 = time.time()
            W_cbf = build_item_cbf_similarity_fast(self.icm, self.k_icm, use_gpu=False)
            print(f"[{self.RECOMMENDER_NAME}] CBF fit in {time.time()-t1:.1f}s, nnz={W_cbf.nnz}")
            self.W = (1.0 - self.icm_weight) * W_cd + self.icm_weight * W_cbf
        else:
            self.W = W_cd

    def _score_session_profile(self, profile: csr_matrix) -> np.ndarray:
        return np.asarray(profile.dot(self.W).todense()).flatten()

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st.update({k: getattr(self, k) for k in (
            "top_k", "alpha", "l1_ratio", "max_iter", "tol", "workers", "icm_weight", "k_icm", "W"
        )})
        return st

    def _set_model_state(self, state: dict) -> None:
        super()._set_model_state(state)
        for k in ("top_k", "alpha", "l1_ratio", "max_iter", "tol", "workers", "W"):
            setattr(self, k, state[k])
        self.icm_weight = state.get("icm_weight", 0.0)
        self.k_icm = state.get("k_icm", 150)
