"""GF-CF — Graph-Filter Collaborative Filtering (Shen et al. 2021).

S = R̂^T R̂ + α · D_I^(-a) V_K Σ_K^β V_K^T D_I^(+a)
where R̂ is the symmetrically normalized URM and V_K, Σ_K are the top-K
SVD components of R̂.

scores(profile) = profile @ S

Filter variants: standard / linear / power / svd_only / combined.

GPU path memory budget (per phase):

  Phase 1 — densify + SVD + S_base:
    R_hat_t  (n_users × n_items × 4 B)
    S_base_t (n_items² × 4 B)
    peak ≈ (n_users + n_items) × n_items × 4 B
    → S_base moved to CPU numpy; both GPU tensors freed

  Phase 2 — S_svd GEMM (per fit or refit):
    S_svd_t  (n_items² × 4 B)   ← only live tensor on GPU
    → moved to CPU numpy; S_svd freed

  Persistent GPU (between trials): V_t + sigma_t + item_d ≈ 39 MB

  splitF (29k items): phase1 peak ≈ 4.8 GB, phase2 ≈ 3.4 GB
  splitA (38k items): phase1 peak ≈ 8.2 GB, phase2 ≈ 5.8 GB
  Both fit on 16 GB.  Falls back to CPU automatically on OOM.

Fast refit for Optuna tuning:
  After fit(), call refit_S(alpha, beta, degree_exp) to recompute only
  the S_svd GEMM — V_t/sigma_t stay in VRAM, S_base stays in RAM.
"""

from __future__ import annotations

import gc
import time

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import svds

from .user_base import UserRecommender


class GFCFRecommender(UserRecommender):
    RECOMMENDER_NAME = "GFCF"

    def __init__(
        self,
        K: int = 256,
        alpha: float = 1.0,
        degree_exp: float = 0.5,
        beta: float = 1.0,
        filter_type: str = "power",
        device: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.K = K
        self.alpha = alpha
        self.degree_exp = degree_exp
        self.beta = beta
        self.filter_type = filter_type
        self.device = device  # resolved to "cuda"/"cpu" in _fit_model if None
        self.S: np.ndarray | None = None
        self._cache: dict | None = None   # pre-computed components for refit_S()

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def _fit_model(self, urm: csr_matrix) -> None:
        import torch

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        t0 = time.time()

        R = urm.astype(np.float32)
        user_deg = np.maximum(np.array(R.sum(axis=1)).flatten(), 1e-10)
        item_deg = np.maximum(np.array(R.sum(axis=0)).flatten(), 1e-10)
        R_hat = (
            diags(1.0 / np.sqrt(user_deg)) @ R @ diags(1.0 / np.sqrt(item_deg))
        ).astype(np.float32)

        K_actual = min(self.K, min(R_hat.shape) - 1)

        if self.device != "cpu":
            try:
                self.S = self._fit_gpu(R_hat, item_deg, K_actual)
            except torch.cuda.OutOfMemoryError:
                print(f"[{self.RECOMMENDER_NAME}] GPU OOM — falling back to CPU")
                torch.cuda.empty_cache()
                gc.collect()
                self.S = self._fit_cpu(R_hat, item_deg, K_actual)
        else:
            self.S = self._fit_cpu(R_hat, item_deg, K_actual)

        np.fill_diagonal(self.S, 0.0)
        gc.collect()
        print(
            f"[{self.RECOMMENDER_NAME}] {self.filter_type} K={self.K} "
            f"fit in {time.time()-t0:.1f}s on {self.device}"
        )

    # ------------------------------------------------------------------
    # CPU path  (scipy sparse SVD + numpy BLAS)
    # ------------------------------------------------------------------

    def _fit_cpu(self, R_hat: csr_matrix, item_deg: np.ndarray, K: int) -> np.ndarray:
        _, sigma_np, Vt_np = svds(R_hat, k=K)          # ascending order
        order = np.argsort(-sigma_np)
        sigma = sigma_np[order].astype(np.float32)      # (K,)
        V     = Vt_np[order].astype(np.float32)         # (K, n_items)
        del sigma_np, Vt_np

        needs_base = self.filter_type in ("standard", "linear", "power", "combined")
        S_base = (R_hat.T @ R_hat).toarray().astype(np.float32) if needs_base else None
        del R_hat
        gc.collect()

        self._cache = {
            "backend":  "cpu",
            "V":        V,
            "sigma":    sigma,
            "item_deg": item_deg.astype(np.float32),
            "S_base":   S_base,
        }

        S_svd = self._S_svd_np(V, sigma, item_deg)
        S_svd *= self.alpha
        return (S_base + S_svd) if S_base is not None else S_svd

    def _S_svd_np(self, V: np.ndarray, sigma: np.ndarray, item_deg: np.ndarray) -> np.ndarray:
        if self.filter_type in ("standard", "svd_only"):
            D_neg = np.power(item_deg, -self.degree_exp, dtype=np.float32)
            D_pos = np.power(item_deg, +self.degree_exp, dtype=np.float32)
            return D_neg[:, None] * (V.T @ V) * D_pos[None, :]
        if self.filter_type == "linear":
            return (V.T * sigma) @ V
        if self.filter_type == "power":
            return (V.T * np.power(sigma, self.beta, dtype=np.float32)) @ V
        if self.filter_type == "combined":
            D_neg = np.power(item_deg, -self.degree_exp, dtype=np.float32)
            D_pos = np.power(item_deg, +self.degree_exp, dtype=np.float32)
            sp    = np.power(sigma, self.beta, dtype=np.float32)
            return D_neg[:, None] * ((V.T * sp) @ V) * D_pos[None, :]
        raise ValueError(f"unknown filter_type: {self.filter_type}")

    # ------------------------------------------------------------------
    # GPU path  (everything on CUDA)
    # ------------------------------------------------------------------

    def _fit_gpu(self, R_hat: csr_matrix, item_deg: np.ndarray, K: int) -> np.ndarray:
        """
        Phase 1 — SVD + S_base on GPU, then S_base moved to CPU:
          R_hat_t  (n_users × n_items × 4 B)  +  S_base_t  (n_items² × 4 B)
          → peak = (n_users + n_items) × n_items × 4 B
          → free both GPU tensors after transferring S_base to numpy

        Phase 2 — S_svd GEMM on GPU, combine on CPU:
          S_svd_t  (n_items² × 4 B)  only
          → transfer to numpy, free GPU tensor

        Persistent after fit: V_t + sigma_t + item_d  ≈ 39 MB
        """
        import torch

        dev = self.device

        R_hat_t = torch.from_numpy(R_hat.toarray()).to(dev)   # (n_users, n_items)
        del R_hat

        # torch.svd_lowrank returns Vh of shape (n_items, K) — transpose to (K, n_items)
        _, sigma_t, Vt_t = torch.svd_lowrank(R_hat_t, q=K, niter=4)
        order   = torch.argsort(sigma_t, descending=True)
        sigma_t = sigma_t[order]
        V_t     = Vt_t.T[order].contiguous()                  # (K, n_items)
        del Vt_t

        needs_base = self.filter_type in ("standard", "linear", "power", "combined")
        if needs_base:
            S_base_t = R_hat_t.T @ R_hat_t                    # (n_items, n_items) on GPU
            del R_hat_t
            torch.cuda.empty_cache()
            S_base = S_base_t.cpu().numpy()                    # move to CPU RAM
            del S_base_t
            torch.cuda.empty_cache()
        else:
            del R_hat_t
            torch.cuda.empty_cache()
            S_base = None

        item_d = torch.from_numpy(item_deg.astype(np.float32)).to(dev)

        # Cache: V_t/sigma_t/item_d stay in VRAM (~39 MB); S_base in CPU RAM
        self._cache = {
            "backend": "gpu",
            "V_t":     V_t,
            "sigma_t": sigma_t,
            "item_d":  item_d,
            "S_base":  S_base,   # numpy, not GPU
        }

        # S_svd GEMM on GPU, combine on CPU
        S_svd_t = self._S_svd_torch(V_t, sigma_t, item_d)
        S_svd_t.mul_(self.alpha)
        S_svd = S_svd_t.cpu().numpy()
        del S_svd_t
        torch.cuda.empty_cache()

        return (S_base + S_svd) if S_base is not None else S_svd

    def _S_svd_torch(self, V_t, sigma_t, item_d):
        import torch
        if self.filter_type in ("standard", "svd_only"):
            D_neg = torch.pow(item_d, -self.degree_exp)
            D_pos = torch.pow(item_d, +self.degree_exp)
            return D_neg[:, None] * (V_t.T @ V_t) * D_pos[None, :]
        if self.filter_type == "linear":
            return (V_t.T * sigma_t) @ V_t
        if self.filter_type == "power":
            return (V_t.T * torch.pow(sigma_t, self.beta)) @ V_t
        if self.filter_type == "combined":
            D_neg = torch.pow(item_d, -self.degree_exp)
            D_pos = torch.pow(item_d, +self.degree_exp)
            sp    = torch.pow(sigma_t, self.beta)
            return D_neg[:, None] * ((V_t.T * sp) @ V_t) * D_pos[None, :]
        raise ValueError(f"unknown filter_type: {self.filter_type}")

    # ------------------------------------------------------------------
    # Fast refit for Optuna tuning
    # ------------------------------------------------------------------

    def refit_S(self, alpha: float, beta: float, degree_exp: float) -> None:
        """Recompute S from cached components — skips URM normalization and SVD.

        GPU peak: one S_svd_t  (n_items² × 4 B) only — S_base stays in CPU RAM.
        Only valid after fit().  Intended for Optuna tuning where K and
        filter_type are fixed and only alpha/beta/degree_exp vary.
        """
        import torch

        if self._cache is None:
            raise RuntimeError("refit_S() called before fit()")

        self.alpha      = alpha
        self.beta       = beta
        self.degree_exp = degree_exp

        c = self._cache
        if c["backend"] == "gpu":
            S_svd_t = self._S_svd_torch(c["V_t"], c["sigma_t"], c["item_d"])
            S_svd_t.mul_(alpha)
            S_svd = S_svd_t.cpu().numpy()
            del S_svd_t
            torch.cuda.empty_cache()
            self.S = (c["S_base"] + S_svd) if c["S_base"] is not None else S_svd
        else:
            S_svd = self._S_svd_np(c["V"], c["sigma"], c["item_deg"])
            S_svd *= alpha
            self.S = (c["S_base"] + S_svd) if c["S_base"] is not None else S_svd

        np.fill_diagonal(self.S, 0.0)

    # ------------------------------------------------------------------
    # inference + persistence
    # ------------------------------------------------------------------

    def _score_session_profile(self, profile: csr_matrix) -> np.ndarray:
        return np.asarray(profile @ self.S).flatten()

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st.update({k: getattr(self, k) for k in (
            "K", "alpha", "degree_exp", "beta", "filter_type", "device", "S"
        )})
        return st

    def _set_model_state(self, state: dict) -> None:
        super()._set_model_state(state)
        for k in ("K", "alpha", "degree_exp", "beta", "filter_type", "device", "S"):
            setattr(self, k, state[k])
