"""Item-KNN cosine similarity (sessions as users).

Computes item-item cosine similarity over the session × track URM, keeps top-k
neighbours per item, then scores sessions as profile @ W.

W_cf uses a numba parallel inverted-index kernel when numba is available;
falls back to batched scipy otherwise.
W_cbf uses build_item_cbf_similarity_fast (CPU-only).
"""

from __future__ import annotations

import os
import time

import numpy as np
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

from .interactions import build_item_cbf_similarity_fast
from .user_base import UserRecommender

try:
    import numba as _numba

    @_numba.njit(parallel=True, cache=True)
    def _cf_topk_kernel(
        data, indices, indptr,
        data_T, indices_T, indptr_T,
        norms, n, k, shrink,
        out_cols, out_vals, out_nnz,
    ):
        """Parallel inverted-index top-k CF similarity with shrink."""
        shrink_f = np.float32(shrink)
        for i in _numba.prange(n):
            acc = np.zeros(n, dtype=np.float32)
            for ui in range(indptr[i], indptr[i + 1]):
                u = indices[ui]
                v = data[ui]
                for ji in range(indptr_T[u], indptr_T[u + 1]):
                    acc[indices_T[ji]] += v * data_T[ji]
            acc[i] = np.float32(0.0)
            norm_i = norms[i]

            nz = 0
            for j in range(n):
                if acc[j] > np.float32(0.0):
                    nz += 1

            nz_idx = np.empty(nz, dtype=np.int32)
            nz_val = np.empty(nz, dtype=np.float32)
            pos = 0
            for j in range(n):
                if acc[j] > np.float32(0.0):
                    denom = norm_i * norms[j] + shrink_f
                    nz_idx[pos] = j
                    nz_val[pos] = acc[j] / denom
                    pos += 1

            actual_k = min(k, nz)
            if actual_k == 0:
                out_nnz[i] = 0
                continue
            order = np.argsort(nz_val[:nz])
            for r in range(actual_k):
                idx = order[nz - 1 - r]
                out_cols[i, r] = nz_idx[idx]
                out_vals[i, r] = nz_val[idx]
            out_nnz[i] = actual_k

    _NUMBA_CF_OK = True
except ImportError:
    _NUMBA_CF_OK = False


def _slurm_numba_threads() -> None:
    """Cap numba threads to SLURM allocation when running on a cluster."""
    if not _NUMBA_CF_OK:
        return
    n = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    if n > 0:
        _numba.set_num_threads(n)


def build_cf_similarity(
    urm: csr_matrix, k: int, shrink: float, desc: str = "ItemKNN",
) -> csr_matrix:
    """Item-item cosine CF similarity over a session/user x track URM.

    Top-k neighbours per item, shrink-damped cosine. Numba inverted-index kernel
    when available, batched scipy fallback otherwise. Shared by
    ItemKNNRecommender and EmbeddingItemKNN.
    """
    _slurm_numba_threads()
    n     = urm.shape[1]
    urm_T = urm.T.tocsr().astype(np.float32)
    norms = np.sqrt(np.array(urm_T.power(2).sum(axis=1)).flatten()).astype(np.float32)
    norms = np.maximum(norms, np.float32(1e-10))

    if _NUMBA_CF_OK:
        urm_ = urm_T.T.tocsr().astype(np.float32)
        out_cols = np.full((n, k), -1, dtype=np.int32)
        out_vals = np.zeros((n, k), dtype=np.float32)
        out_nnz  = np.zeros(n, dtype=np.int32)
        _cf_topk_kernel(
            urm_T.data, urm_T.indices, urm_T.indptr,
            urm_.data,  urm_.indices,  urm_.indptr,
            norms, n, k, float(shrink),
            out_cols, out_vals, out_nnz,
        )
        total  = int(out_nnz.sum())
        rows_o = np.empty(total, np.int32)
        cols_o = np.empty(total, np.int32)
        vals_o = np.empty(total, np.float32)
        pos = 0
        for i in range(n):
            cnt = int(out_nnz[i])
            if cnt:
                rows_o[pos:pos + cnt] = i
                cols_o[pos:pos + cnt] = out_cols[i, :cnt]
                vals_o[pos:pos + cnt] = out_vals[i, :cnt]
                pos += cnt
        return csr_matrix((vals_o, (rows_o, cols_o)), shape=(n, n), dtype=np.float32)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    batch = 1000
    for start in tqdm(range(0, n, batch), desc=f"  {desc}"):
        end = min(start + batch, n)
        sims = urm_T[start:end].dot(urm_T.T).toarray()
        for li, item_idx in enumerate(range(start, end)):
            s = sims[li] / (norms * norms[item_idx] + shrink)
            s[item_idx] = 0.0
            top = np.argpartition(-s, k)[:k] if k < n else np.arange(n)
            m = s[top] > 0
            rows.extend([item_idx] * int(m.sum()))
            cols.extend(top[m].tolist())
            data.extend(s[top][m].tolist())
    return csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)


class ItemKNNRecommender(UserRecommender):
    RECOMMENDER_NAME = "ItemKNN"

    def __init__(self, k: int = 150, shrink: float = 10.0, icm_weight: float = 0.0, k_icm: int = 150, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.shrink = shrink
        self.icm_weight = icm_weight
        self.k_icm = k_icm
        self.W: csr_matrix | None = None

    def _fit_model(self, urm: csr_matrix) -> None:
        t0 = time.time()
        W_cf = build_cf_similarity(urm, self.k, self.shrink, desc=self.RECOMMENDER_NAME)
        print(f"[{self.RECOMMENDER_NAME}] CF fit in {time.time()-t0:.1f}s, nnz={W_cf.nnz}")

        if self.icm is not None and self.icm_weight > 0:
            t1 = time.time()
            W_cbf = build_item_cbf_similarity_fast(self.icm, self.k_icm, use_gpu=False)
            print(f"[{self.RECOMMENDER_NAME}] CBF fit in {time.time()-t1:.1f}s, nnz={W_cbf.nnz}")
            self.W = (1.0 - self.icm_weight) * W_cf + self.icm_weight * W_cbf
        else:
            self.W = W_cf

    def _score_session_profile(self, profile: csr_matrix) -> np.ndarray:
        return np.asarray(profile.dot(self.W).todense()).flatten()

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st.update({"k": self.k, "shrink": self.shrink, "icm_weight": self.icm_weight, "k_icm": self.k_icm, "W": self.W})
        return st

    def _set_model_state(self, state: dict) -> None:
        super()._set_model_state(state)
        self.k = state["k"]
        self.shrink = state["shrink"]
        self.icm_weight = state.get("icm_weight", 0.0)
        self.k_icm = state.get("k_icm", 150)
        self.W = state["W"]
