from __future__ import annotations

import random
import sys
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
    ITEM_OFFSET,
    MASK_TOKEN,
    PAD_TOKEN,
    FeatureBert4RecRecommender,
    _build_feature_matrix,
    _FeatureBert4RecDataset,
    _FeatureBert4RecModel,
)

class _FeatureBert4RecIdentityModel(_FeatureBert4RecModel):

    def __init__(
        self,
        warm_feature_matrix: np.ndarray,
        hidden_size: int,
        max_seq_len: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__(
            warm_feature_matrix, hidden_size, max_seq_len, n_layers, n_heads, dropout
        )

        feature_dim = warm_feature_matrix.shape[1]
        self.item_encoder = nn.Linear(feature_dim, hidden_size, bias=True)

class FeatureBert4RecIdentityRecommender(FeatureBert4RecRecommender):

    RECOMMENDER_NAME = "FeatureBert4RecIdentity"

    def _pca_init_encoder(self, feature_matrix: np.ndarray) -> None:
        n_components = self.hidden_size
        print(f"[{self.RECOMMENDER_NAME}] Fitting PCA({n_components}) on warm feature matrix "
              f"({feature_matrix.shape[0]} × {feature_matrix.shape[1]})...")
        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
        pca.fit(feature_matrix)

        W = torch.from_numpy(pca.components_.astype(np.float32))
        mean = torch.from_numpy(pca.mean_.astype(np.float32))
        b = -(mean @ W.T)

        with torch.no_grad():
            self.model.item_encoder.weight.copy_(W)
            self.model.item_encoder.bias.copy_(b)

        explained = float(pca.explained_variance_ratio_.sum())
        print(f"  PCA explained variance: {explained:.3f} "
              f"({n_components}/{feature_matrix.shape[1]} components)")

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

        sequences = self._build_sequences()

        n_warm = len(self._warm_global_indices)
        n_cold = len(self._cold_global_indices)
        print(
            f"[{self.RECOMMENDER_NAME}] warm={n_warm}, cold={n_cold}, "
            f"sequences={len(sequences)}, device={self.device_}"
        )

        print(f"[{self.RECOMMENDER_NAME}] Loading feature embeddings: {self.feature_modalities}")
        full_matrix = _build_feature_matrix(
            self.feature_emb_paths, self.feature_modalities, self.id_map
        )
        self._feature_dim = full_matrix.shape[1]
        warm_feature_matrix = full_matrix[self._warm_global_indices]
        self._cold_feature_matrix = full_matrix[self._cold_global_indices]
        print(f"  warm_features={warm_feature_matrix.shape}, cold_features={self._cold_feature_matrix.shape}")

        random.shuffle(sequences)
        n_val = max(1, int(len(sequences) * self.val_ratio))
        val_sequences   = sequences[:n_val]
        train_sequences = sequences[n_val:]
        print(f"  train_seqs={len(train_sequences)}, val_seqs={len(val_sequences)}")

        pin = self.device_.type == "cuda"
        train_loader = DataLoader(
            _FeatureBert4RecDataset(train_sequences, n_warm, self.max_seq_len, self.mask_prob),
            batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=pin,
        )
        val_loader = DataLoader(
            _FeatureBert4RecDataset(val_sequences, n_warm, self.max_seq_len, self.mask_prob),
            batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=pin,
        )

        self.model = _FeatureBert4RecIdentityModel(
            warm_feature_matrix, self.hidden_size, self.max_seq_len,
            self.n_layers, self.n_heads, self.dropout,
        )

        self._pca_init_encoder(warm_feature_matrix)
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
            unit="ep",
            dynamic_ncols=True,
            file=sys.stdout,
        )

        best_val: float = float("inf")
        best_epoch: int = 0
        best_state: dict | None = None
        patience_left: int = self.early_stop_patience

        for epoch in epoch_bar:
            self.model.train()
            total_loss = 0.0
            batch_bar = tqdm(train_loader, desc=f"  ep {epoch:3d}", leave=False,
                             unit="batch", dynamic_ncols=True, file=sys.stdout)
            for masked_seq, labels in batch_bar:
                masked_seq = masked_seq.to(self.device_)
                labels     = labels.to(self.device_)

                logits = self.model(masked_seq)
                loss = F.cross_entropy(
                    logits.view(-1, n_warm),
                    labels.view(-1),
                    ignore_index=-100,
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                batch_bar.set_postfix(loss=f"{loss.item():.4f}")

            train_avg = total_loss / len(train_loader)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for masked_seq, labels in val_loader:
                    masked_seq = masked_seq.to(self.device_)
                    labels     = labels.to(self.device_)
                    logits = self.model(masked_seq)
                    val_loss += F.cross_entropy(
                        logits.view(-1, n_warm),
                        labels.view(-1),
                        ignore_index=-100,
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
                loss=f"{train_avg:.4f}",
                val=f"{val_avg:.4f}",
                best=f"{best_val:.4f}@{best_epoch}",
                patience=patience_left,
            )

            if patience_left <= 0:
                print(f"\n[{self.RECOMMENDER_NAME}] early stopping at epoch {epoch} "
                      f"(no val improvement for {self.early_stop_patience} epochs); "
                      f"best val={best_val:.4f} at epoch {best_epoch}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"[{self.RECOMMENDER_NAME}] restored best checkpoint "
                  f"(val={best_val:.4f}, epoch={best_epoch})")

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
        self.device_ = torch.device(state.get("device", "cpu"))
        self._feature_dim = state.get("feature_dim")
        self._warm_global_indices = state.get("warm_global_indices", [])
        self._cold_global_indices = state.get("cold_global_indices", [])
        self._cold_feature_matrix = state.get("cold_feature_matrix")
        self._global_to_warm_local = state.get("global_to_warm_local", {})
        self._global_to_cold_local = state.get(
            "global_to_cold_local",
            {g: l for l, g in enumerate(self._cold_global_indices)},
        )

        sd = state.get("model_state_dict")
        if sd is not None and self.id_map is not None and self._feature_dim is not None:
            n_warm = len(self._warm_global_indices)
            dummy = np.zeros((n_warm, self._feature_dim), dtype=np.float32)
            self.model = _FeatureBert4RecIdentityModel(
                dummy, self.hidden_size, self.max_seq_len,
                self.n_layers, self.n_heads, self.dropout,
            )
            self.model.load_state_dict(sd)
            self.model.to(self.device_)
