"""iALS — Implicit ALS (Hu et al. 2008), Numba-accelerated.

Trains item factors Y via alternating least squares on implicit feedback.
Inference: fold-in session profile (one ALS solve per session, no re-training).
"""

from __future__ import annotations

import os
import time

import numba
import numpy as np
from scipy.sparse import csr_matrix

from .user_base import UserRecommender


def _slurm_numba_threads() -> None:
    """Cap numba threads to SLURM allocation when running on a cluster."""
    n = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    if n > 0:
        numba.set_num_threads(n)


@numba.njit(parallel=True, cache=True)
def _als_update(indptr, indices, F_other, FtF, lam, alpha, n_rows, f):
    """Solve x_r = (FtF + alpha * sum_i f_i f_i^T + lam I)^-1 ((1+alpha) sum_i f_i)."""
    out = np.zeros((n_rows, f), dtype=np.float32)
    for r in numba.prange(n_rows):
        A = FtF.copy()
        for d in range(f):
            A[d, d] += lam
        b = np.zeros(f, dtype=np.float64)
        for p in range(indptr[r], indptr[r + 1]):
            i = indices[p]
            fi = F_other[i]
            for a in range(f):
                bi = fi[a]
                b[a] += (1.0 + alpha) * bi
                for c in range(f):
                    A[a, c] += alpha * bi * fi[c]
        out[r] = np.linalg.solve(A, b).astype(np.float32)
    return out


class IALSRecommender(UserRecommender):
    """Implicit ALS with fold-in inference."""

    RECOMMENDER_NAME = "iALS"

    def __init__(
        self,
        n_factors: int = 64,
        reg: float = 10.0,
        alpha: float = 40.0,
        iters: int = 12,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_factors = n_factors
        self.reg = reg
        self.alpha = alpha
        self.iters = iters
        self.Y: np.ndarray | None = None    # (n_items, f) item factors
        self.YtY: np.ndarray | None = None  # (f, f) precomputed for fold-in

    def _fit_model(self, urm: csr_matrix) -> None:
        _slurm_numba_threads()
        R = urm.tocsr().astype(np.float32)
        Rt = R.T.tocsr()
        m, n = R.shape
        f = self.n_factors
        rng = np.random.default_rng(0)
        X = (rng.standard_normal((m, f)) * 0.01).astype(np.float32)
        self.Y = (rng.standard_normal((n, f)) * 0.01).astype(np.float32)

        for it in range(self.iters):
            t = time.time()
            YtY = (self.Y.T @ self.Y).astype(np.float64)
            X = _als_update(
                R.indptr.astype(np.int64), R.indices.astype(np.int64),
                self.Y, YtY, self.reg, self.alpha, m, f,
            )
            XtX = (X.T @ X).astype(np.float64)
            self.Y = _als_update(
                Rt.indptr.astype(np.int64), Rt.indices.astype(np.int64),
                X, XtX, self.reg, self.alpha, n, f,
            )
            print(f"  [{self.RECOMMENDER_NAME}] iter {it + 1}/{self.iters}: {time.time() - t:.1f}s")

        self.YtY = (self.Y.T @ self.Y).astype(np.float64)

    def _score_session_profile(self, profile: csr_matrix) -> np.ndarray:
        f = self.n_factors
        A = self.YtY + self.reg * np.eye(f)
        b = np.zeros(f, dtype=np.float64)
        for i in profile.indices:
            yi = self.Y[i].astype(np.float64)
            A += self.alpha * np.outer(yi, yi)
            b += (1.0 + self.alpha) * yi
        x = np.linalg.solve(A, b).astype(np.float32)
        return (self.Y @ x).ravel()

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st.update({
            "n_factors": self.n_factors,
            "reg": self.reg,
            "alpha": self.alpha,
            "iters": self.iters,
            "Y": self.Y,
            "YtY": self.YtY,
        })
        return st

    def _set_model_state(self, state: dict) -> None:
        super()._set_model_state(state)
        self.n_factors = state["n_factors"]
        self.reg = state["reg"]
        self.alpha = state["alpha"]
        self.iters = state["iters"]
        self.Y = state["Y"]
        self.YtY = state["YtY"]
