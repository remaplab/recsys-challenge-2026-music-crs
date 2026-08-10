"""SLIM BPR candidate generation (Numba JIT accelerated).

Training: Hogwild! BPR SGD, all epochs in a single numba call.
Sparsification: parallel top-k per column via numba (no dense transpose copy).
ICM similarity: O(n_touched) tracked inverted-index kernel.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numba
import numpy as np
from numba import njit, prange
from scipy.sparse import csr_matrix

from .interactions import build_item_cbf_similarity_fast
from .user_base import UserRecommender


# ---------------------------------------------------------------------------
# BPR kernels
# ---------------------------------------------------------------------------

@njit(nogil=True, parallel=True)
def bpr_update_epoch(
    indptr: np.ndarray,
    indices: np.ndarray,
    S: np.ndarray,
    n_users: int,
    n_items: int,
    n_samples: int,
    lr: float,
    reg_pos: float,
    reg_neg: float,
):
    """One Hogwild! BPR epoch over n_samples triplets."""
    for _sample_step in prange(n_samples):
        u = np.random.randint(n_users)
        start = indptr[u]
        end = indptr[u + 1]
        if end == start:
            continue

        pos_idx = np.random.randint(start, end)
        i = indices[pos_idx]

        j = -1
        for _retry in range(100):
            j_cand = np.random.randint(n_items)
            is_pos = False
            for idx in range(start, end):
                if indices[idx] == j_cand:
                    is_pos = True
                    break
            if not is_pos:
                j = j_cand
                break
        if j == -1:
            continue

        x_ui = 0.0
        x_uj = 0.0
        for idx in range(start, end):
            k = indices[idx]
            if k != i:
                x_ui += S[k, i]
            if k != j:
                x_uj += S[k, j]

        x_uij = x_ui - x_uj
        if x_uij > 50.0:
            sig = 0.0
        elif x_uij < -50.0:
            sig = 1.0
        else:
            sig = 1.0 / (1.0 + np.exp(x_uij))

        for idx in range(start, end):
            k = indices[idx]
            if k != i:
                S[k, i] += lr * (sig - reg_pos * S[k, i])
            if k != j:
                S[k, j] += lr * (-sig - reg_neg * S[k, j])


@njit(cache=True)
def _bpr_all_epochs(indptr, indices, S, n_users, n_items, n_samples,
                    lr, reg_pos, reg_neg, epochs):
    """All epochs in one numba call — no Python overhead between epochs."""
    for _ in range(epochs):
        bpr_update_epoch(indptr, indices, S, n_users, n_items, n_samples,
                         lr, reg_pos, reg_neg)


# ---------------------------------------------------------------------------
# Sparsification kernels
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _sparsify_kernel(S, k_eff, out_rows, out_vals, out_nnz):
    """Parallel top-k per column of S (Fortran-order for cache efficiency)."""
    n = S.shape[0]
    for target_i in prange(n):
        nz = np.int32(0)
        for j in range(n):
            if S[j, target_i] > np.float32(0.0):
                nz += np.int32(1)
        if nz == 0:
            out_nnz[target_i] = 0
            continue

        nz_idx = np.empty(nz, dtype=np.int32)
        nz_val = np.empty(nz, dtype=np.float32)
        pos = np.int32(0)
        for j in range(n):
            if S[j, target_i] > np.float32(0.0):
                nz_idx[pos] = j
                nz_val[pos] = S[j, target_i]
                pos += np.int32(1)

        actual_k = min(k_eff, nz)
        order = np.argsort(nz_val[:nz])
        for r in range(actual_k):
            idx = order[nz - 1 - r]
            out_rows[target_i, r] = nz_idx[idx]
            out_vals[target_i, r] = nz_val[idx]
        out_nnz[target_i] = actual_k


@njit(cache=True)
def _flatten_coo(out_rows, out_vals, out_nnz, n_items):
    """Prefix-sum scatter: (n_items, k) dense → flat COO arrays."""
    prefix = np.empty(n_items + 1, dtype=np.int64)
    prefix[0] = np.int64(0)
    for i in range(n_items):
        prefix[i + 1] = prefix[i] + np.int64(out_nnz[i])
    total = prefix[n_items]
    rows_o = np.empty(total, dtype=np.int32)
    cols_o = np.empty(total, dtype=np.int32)
    vals_o = np.empty(total, dtype=np.float32)
    for target_i in range(n_items):
        p = prefix[target_i]
        cnt = out_nnz[target_i]
        for r in range(cnt):
            rows_o[p + r] = out_rows[target_i, r]
            cols_o[p + r] = target_i
            vals_o[p + r] = out_vals[target_i, r]
    return rows_o, cols_o, vals_o


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

class SLIMBPRRecommender(UserRecommender):
    """SLIM optimized via BPR (Hogwild! SGD, Numba JIT)."""

    RECOMMENDER_NAME = "SLIM_BPR"

    def __init__(
        self,
        top_k: int = 100,
        epochs: int = 50,
        learning_rate: float = 0.05,
        reg_pos: float = 0.001,
        reg_neg: float = 0.001,
        workers: int = 15,
        icm_weight: float = 0.0,
        k_icm: int = 150,
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.reg_pos = reg_pos
        self.reg_neg = reg_neg
        self.workers = workers
        self.icm_weight = icm_weight
        self.k_icm = k_icm
        self.W: csr_matrix | None = None

    def _fit_model(self, urm: csr_matrix) -> None:
        n_users, n_items = urm.shape
        mem_gb = (n_items ** 2 * 4) / 1024 ** 3
        print(f"[{self.RECOMMENDER_NAME}] Allocating {n_items}×{n_items} S (~{mem_gb:.2f} GB)...")

        # Fortran order: column-contiguous access pattern matches BPR S[k, i/j] writes
        S = np.zeros((n_items, n_items), dtype=np.float32, order="F")

        # BPR uses restricted thread count (tuned hyperparameter)
        numba.set_num_threads(self.workers)
        t_bpr = time.time()
        _bpr_all_epochs(
            urm.indptr, urm.indices, S,
            n_users, n_items, urm.nnz,
            self.learning_rate, self.reg_pos, self.reg_neg, self.epochs,
        )
        print(f"[{self.RECOMMENDER_NAME}] BPR {self.epochs} epochs: {time.time() - t_bpr:.1f}s")

        # Restore full parallelism for embarrassingly parallel sparsify / ICM steps
        numba.set_num_threads(min(os.cpu_count(), numba.get_num_threads()))

        t_sp = time.time()
        k_eff = min(self.top_k, n_items)
        out_rows = np.full((n_items, k_eff), -1, dtype=np.int32)
        out_vals = np.zeros((n_items, k_eff), dtype=np.float32)
        out_nnz = np.zeros(n_items, dtype=np.int32)
        _sparsify_kernel(S, k_eff, out_rows, out_vals, out_nnz)
        rows_o, cols_o, vals_o = _flatten_coo(out_rows, out_vals, out_nnz, n_items)
        W_bpr = csr_matrix((vals_o, (rows_o, cols_o)), shape=(n_items, n_items))
        print(f"[{self.RECOMMENDER_NAME}] Sparsify: {time.time() - t_sp:.1f}s  nnz={W_bpr.nnz}")

        if self.icm is not None and self.icm_weight > 0:
            t1 = time.time()
            W_cbf = build_item_cbf_similarity_fast(self.icm, self.k_icm, use_gpu=False)
            print(f"[{self.RECOMMENDER_NAME}] ICM similarity: {time.time() - t1:.1f}s  nnz={W_cbf.nnz}")
            self.W = (1.0 - self.icm_weight) * W_bpr + self.icm_weight * W_cbf
        else:
            self.W = W_bpr

    def _score_session_profile(self, profile: csr_matrix) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("Model not fitted.")
        return profile.dot(self.W).toarray().ravel()

    def _get_model_state(self) -> dict:
        state = super()._get_model_state()
        state.update({"W": self.W, "icm_weight": self.icm_weight, "k_icm": self.k_icm})
        return state

    def _set_model_state(self, state: dict) -> None:
        super()._set_model_state(state)
        self.W = state.get("W")
        self.icm_weight = state.get("icm_weight", 0.0)
        self.k_icm = state.get("k_icm", 150)
