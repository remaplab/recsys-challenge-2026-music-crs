"""Identity + cosine + RoPE + Modality Multi-Head (per-modality L2 + per-modality
PCA-initialized projection + element-wise sum).

What changes vs the parent `identity_cosine_rope`:
  - The item encoder, currently a SINGLE Linear(total_dim, hidden_size) fitted
    once on the joint PCA of the concatenated features, is replaced with a
    PER-MODALITY pipeline:
        for each modality m:
            x_m = features[m]                 # (n, dim_m)
            x_m = L2-normalize(x_m, axis=-1)  # remove modality-specific scale
            e_m = Linear_m(x_m)               # (n, hidden_size), PCA-init from x_m
        e   = sum_m e_m                       # (n, hidden_size)

Why:
  - Joint concat-PCA on raw features gives the biggest-variance modality (most
    likely Qwen3-1024) most of the principal components → CF/CLAP are
    structurally under-represented at init.
  - L2-normalizing each modality before projection puts them on the same scale
    (unit-norm), so the per-modality PCA fits each modality's intrinsic
    geometry, not its absolute magnitude.
  - Element-wise SUM keeps the parameter count identical to a single
    Linear(total_dim, hidden_size) (same total weight matrix elements) but
    structures it as block-diagonal: each modality contributes independently.
    No final fusion Linear → the model can't learn cross-modality interactions
    at the encoder, but it can at the transformer level. Less expressive than
    concat+fuse but avoids a poorly-initialised final layer.

Inherits training/inference plumbing from the RoPE parent; we override only:
  - `_build_feature_matrix_per_modality` (new helper, also returns per-modality dims)
  - `_pca_init_encoder` (per-modality fits instead of joint)
  - `_fit_model` (loads per-modality feature matrices and passes dims to the model)
  - the model class (replaces `item_encoder` with `_ModalityMultiHead`)
  - `_set_model_state` (loads with the right `modality_dims`)
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from tqdm import tqdm

from .feature_bert4rec import (
    PAD_TOKEN,
    _FeatureBert4RecDataset,
)
from .feature_bert4rec_identity_cosine_rope import (
    FeatureBert4RecIdentityCosineRoPERecommender,
    _FeatureBert4RecIdentityCosineRoPEModel,
)


# ---------------------------------------------------------------------------
# Per-modality feature loader
# ---------------------------------------------------------------------------

def _build_feature_matrix_per_modality(
    parquet_paths: list[str | Path],
    modalities: list[str],
    id_map,
) -> tuple[np.ndarray, list[int]]:
    """Same semantics as `_build_feature_matrix` but also returns per-modality dims.

    Returns:
      full_matrix     : (n_tracks, sum(dims)) float32 — same as parent
      modality_dims   : list of per-modality dims, in order of `modalities`
    """
    from .feature_bert4rec import _build_feature_matrix
    chunks: list[np.ndarray] = []
    dims: list[int] = []
    for mod in modalities:
        single = _build_feature_matrix(parquet_paths, [mod], id_map)
        chunks.append(single)
        dims.append(single.shape[1])
    full = np.concatenate(chunks, axis=1)
    return full, dims


# ---------------------------------------------------------------------------
# Modality Multi-Head encoder
# ---------------------------------------------------------------------------

class _ModalityMultiHead(nn.Module):
    """Per-modality L2 → per-modality Linear → element-wise sum.

    Input x has shape (..., sum(modality_dims)). The output has shape
    (..., hidden_size). Each modality contributes its PCA-initialised
    projection, summed.
    """

    def __init__(self, modality_dims: list[int], hidden_size: int) -> None:
        super().__init__()
        self.modality_dims = list(modality_dims)
        self.hidden_size = hidden_size
        self.heads = nn.ModuleList(
            [nn.Linear(d, hidden_size, bias=True) for d in modality_dims]
        )
        # Slice boundaries, stored as plain Python (not buffers) — used only in forward.
        bounds: list[tuple[int, int]] = []
        s = 0
        for d in modality_dims:
            bounds.append((s, s + d))
            s += d
        self.boundaries = bounds

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor | None = None
        for (a, b), head in zip(self.boundaries, self.heads):
            sl = x[..., a:b]
            sl = F.normalize(sl, dim=-1)              # per-modality L2
            proj = head(sl)
            out = proj if out is None else out + proj  # element-wise sum
        assert out is not None
        return out


# ---------------------------------------------------------------------------
# Model: same as RoPE but with multi-head encoder swapped in
# ---------------------------------------------------------------------------

class _FeatureBert4RecIdentityCosineRoPEMMHModel(_FeatureBert4RecIdentityCosineRoPEModel):
    def __init__(
        self,
        warm_feature_matrix: np.ndarray,
        hidden_size: int,
        max_seq_len: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        init_tau: float,
        modality_dims: list[int],
    ) -> None:
        super().__init__(
            warm_feature_matrix, hidden_size, max_seq_len,
            n_layers, n_heads, dropout, init_tau=init_tau,
        )
        assert sum(modality_dims) == warm_feature_matrix.shape[1], (
            f"modality_dims sum {sum(modality_dims)} != feature_matrix dim "
            f"{warm_feature_matrix.shape[1]}"
        )
        self.modality_dims = list(modality_dims)
        # Replace the single Linear item_encoder
        self.item_encoder = _ModalityMultiHead(self.modality_dims, hidden_size)


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

class FeatureBert4RecIdentityCosineRoPEMMHRecommender(FeatureBert4RecIdentityCosineRoPERecommender):
    """Identity + cosine + RoPE + per-modality multi-head encoder."""

    RECOMMENDER_NAME = "FeatureBert4RecIdentityCosineRoPEMMH"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._modality_dims: list[int] = []

    def _build_modality_feature_matrix(self) -> tuple[np.ndarray, list[int]]:
        """Returns (full_matrix, modality_dims) for the model's item_encoder.

        Default impl loads embedding modalities from `self.feature_emb_paths`.
        Variants that inject additional modalities (e.g. ICM via TruncatedSVD)
        override this to concat extra columns and append their dims.
        """
        return _build_feature_matrix_per_modality(
            self.feature_emb_paths, self.feature_modalities, self.id_map
        )

    def _make_model(self, warm_feature_matrix: np.ndarray, modality_dims: list[int]):
        """Build the torch model. Variants override this to swap the encoder
        architecture (e.g. MLP heads, cross-modality attention, per-modality
        weights, FiLM, ...) without re-writing _fit_model."""
        return _FeatureBert4RecIdentityCosineRoPEMMHModel(
            warm_feature_matrix, self.hidden_size, self.max_seq_len,
            self.n_layers, self.n_heads, self.dropout,
            init_tau=self.init_tau,
            modality_dims=modality_dims,
        )

    def _pca_init_encoder_per_modality(
        self,
        feature_matrix: np.ndarray,
        modality_dims: list[int],
    ) -> None:
        """PCA-init each per-modality head from its L2-normalised modality slice.

        Mirrors the model's forward semantics (per-modality L2 then Linear),
        so the head starts at the PCA basis of the (already normalised) modality.
        """
        offsets = []
        s = 0
        for d in modality_dims:
            offsets.append((s, s + d))
            s += d

        for i, (a, b) in enumerate(offsets):
            mod = feature_matrix[:, a:b]
            mod_dim = mod.shape[1]
            n_components = min(self.hidden_size, mod_dim)
            # L2-normalise per row (match what the encoder does at forward time)
            norms = np.linalg.norm(mod, axis=1, keepdims=True).clip(min=1e-10)
            mod_n = mod / norms

            print(f"[{self.RECOMMENDER_NAME}] PCA({n_components}) on modality {i} "
                  f"(L2-normed {mod.shape[0]} × {mod_dim})...")
            pca = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
            pca.fit(mod_n)

            W = torch.from_numpy(pca.components_.astype(np.float32))            # (n_comp, mod_dim)
            mean = torch.from_numpy(pca.mean_.astype(np.float32))               # (mod_dim,)
            bias = -(mean @ W.T)                                                # (n_comp,)

            head = self.model.item_encoder.heads[i]                              # nn.Linear(mod_dim, hidden_size)
            with torch.no_grad():
                if n_components < self.hidden_size:
                    # Modality dim < hidden_size: PCA can only fill the first n_components rows;
                    # leave the rest at their (random) init so they can learn arbitrary directions.
                    head.weight[:n_components].copy_(W)
                    head.bias[:n_components].copy_(bias)
                    # Scale the random portion down to avoid early-step dominance.
                    head.weight[n_components:].mul_(0.1)
                    head.bias[n_components:].zero_()
                    print(f"  modality {i}: PCA-init first {n_components}/{self.hidden_size} rows; "
                          f"remaining {self.hidden_size - n_components} rows kept (scaled ×0.1)")
                else:
                    head.weight.copy_(W)
                    head.bias.copy_(bias)
            explained = float(pca.explained_variance_ratio_.sum())
            print(f"  modality {i}: PCA explained variance = {explained:.3f}")

    def _fit_model(self, urm: csr_matrix) -> None:
        assert self.id_map is not None and self._train_long is not None

        warm_track_ids: set[str] = set(self._train_long["track_id"].to_list())
        warm_track_ids &= set(self.id_map.track_to_idx.keys())
        self._warm_global_indices = sorted(
            self.id_map.track_to_idx[t] for t in warm_track_ids
        )
        self._cold_global_indices = sorted(
            idx for t, idx in self.id_map.track_to_idx.items()
            if t not in warm_track_ids
        )
        self._global_to_warm_local = {g: l for l, g in enumerate(self._warm_global_indices)}
        self._global_to_cold_local = {g: l for l, g in enumerate(self._cold_global_indices)}

        n_warm = len(self._warm_global_indices)
        n_cold = len(self._cold_global_indices)

        print(f"[{self.RECOMMENDER_NAME}] Loading per-modality features: {self.feature_modalities}")
        full_matrix, modality_dims = self._build_modality_feature_matrix()
        self._feature_dim = full_matrix.shape[1]
        self._modality_dims = modality_dims
        print(f"  modality_dims = {modality_dims}, total = {self._feature_dim}")

        warm_feature_matrix = full_matrix[self._warm_global_indices]
        self._cold_feature_matrix = full_matrix[self._cold_global_indices]

        train_sequences, val_sequences = self._build_train_val_sequences()
        print(
            f"[{self.RECOMMENDER_NAME}] warm={n_warm}, cold={n_cold}, "
            f"train_seqs={len(train_sequences)}, val_seqs={len(val_sequences)}, "
            f"device={self.device_}, init_tau={self.init_tau}"
        )

        pin = self.device_.type == "cuda"
        train_loader = DataLoader(
            _FeatureBert4RecDataset(train_sequences, n_warm, self.max_seq_len, self.mask_prob),
            batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=pin,
        )
        val_loader = DataLoader(
            _FeatureBert4RecDataset(val_sequences, n_warm, self.max_seq_len, self.mask_prob),
            batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=pin,
        )

        self.model = self._make_model(warm_feature_matrix, modality_dims)
        self._pca_init_encoder_per_modality(warm_feature_matrix, modality_dims)
        self.model.to(self.device_)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        total_steps = self.epochs * len(train_loader)
        warmup_steps = max(1, int(total_steps * self.warmup_ratio))

        _lr_lambda = self._make_cosine_lr_lambda(total_steps, warmup_steps)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

        epoch_bar = tqdm(
            range(1, self.epochs + 1),
            desc=f"[{self.RECOMMENDER_NAME}]",
            unit="ep", dynamic_ncols=True, file=sys.stdout,
        )

        best_val = float("inf")
        best_epoch = 0
        best_state: dict | None = None
        patience_left = self.early_stop_patience

        for epoch in epoch_bar:
            self.model.train()
            total_loss = 0.0
            for masked_seq, labels in tqdm(train_loader, desc=f"  ep {epoch:3d}", leave=False,
                                            unit="batch", dynamic_ncols=True, file=sys.stdout):
                masked_seq = masked_seq.to(self.device_)
                labels     = labels.to(self.device_)
                logits = self.model(masked_seq)
                loss = F.cross_entropy(
                    logits.view(-1, n_warm), labels.view(-1), ignore_index=-100
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            train_avg = total_loss / len(train_loader)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for masked_seq, labels in val_loader:
                    masked_seq = masked_seq.to(self.device_)
                    labels     = labels.to(self.device_)
                    logits = self.model(masked_seq)
                    val_loss += F.cross_entropy(
                        logits.view(-1, n_warm), labels.view(-1), ignore_index=-100
                    ).item()
            val_avg = val_loss / len(val_loader)

            improved = val_avg < best_val
            if improved:
                best_val = val_avg
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience_left = self.early_stop_patience
            else:
                patience_left -= 1

            epoch_bar.set_postfix(
                loss=f"{train_avg:.4f}", val=f"{val_avg:.4f}",
                best=f"{best_val:.4f}@{best_epoch}",
                tau=f"{self.model.tau.item():.4f}",
                patience=patience_left,
            )

            if patience_left <= 0:
                print(f"\n[{self.RECOMMENDER_NAME}] early stopping at epoch {epoch}; "
                      f"best val={best_val:.4f} at epoch {best_epoch}, "
                      f"tau={self.model.tau.item():.4f}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"[{self.RECOMMENDER_NAME}] restored best (val={best_val:.4f}, "
                  f"epoch={best_epoch}, tau={self.model.tau.item():.4f})")

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st["modality_dims"] = self._modality_dims
        return st

    def _set_model_state(self, state: dict) -> None:
        from .session_base import SessionRecommender
        SessionRecommender._set_model_state(self, state)
        self._train_long = None
        self.feature_emb_paths = state["feature_emb_paths"]
        self.feature_modalities = state.get("feature_modalities", ["metadata-qwen3_embedding_0.6b"])
        self.max_seq_len = state["max_seq_len"]
        self.hidden_size = state["hidden_size"]
        self.n_layers = state["n_layers"]
        self.n_heads = state["n_heads"]
        self.dropout = state["dropout"]
        self.mask_prob = state.get("mask_prob", 0.4)
        self.epochs = state["epochs"]
        self.batch_size = state["batch_size"]
        self.lr = state["lr"]
        self.weight_decay = state["weight_decay"]
        self.warmup_ratio = state.get("warmup_ratio", 0.1)
        self.val_ratio = state.get("val_ratio", 0.1)
        self.early_stop_patience = state.get("early_stop_patience", 10)
        self.init_tau = state.get("init_tau", 0.1)
        self.device_ = torch.device(state.get("device", "cpu"))
        self._feature_dim = state.get("feature_dim")
        self._modality_dims = state.get("modality_dims", [])
        self._warm_global_indices = state.get("warm_global_indices", [])
        self._cold_global_indices = state.get("cold_global_indices", [])
        self._cold_feature_matrix = state.get("cold_feature_matrix")
        self._global_to_warm_local = state.get("global_to_warm_local", {})
        self._global_to_cold_local = state.get(
            "global_to_cold_local",
            {g: l for l, g in enumerate(self._cold_global_indices)},
        )

        sd = state.get("model_state_dict")
        if sd is not None and self.id_map is not None and self._feature_dim is not None and self._modality_dims:
            n_warm = len(self._warm_global_indices)
            dummy = np.zeros((n_warm, self._feature_dim), dtype=np.float32)
            self.model = self._make_model(dummy, self._modality_dims)
            self.model.load_state_dict(sd)
            self.model.to(self.device_)
