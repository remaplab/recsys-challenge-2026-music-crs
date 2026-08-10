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
from torch.utils.data import DataLoader
from tqdm import tqdm

from .feature_bert4rec import (
    ITEM_OFFSET,
    MASK_TOKEN,
    PAD_TOKEN,
    _build_feature_matrix,
    _FeatureBert4RecDataset,
)
from .feature_bert4rec_identity import (
    FeatureBert4RecIdentityRecommender,
    _FeatureBert4RecIdentityModel,
)

class _FeatureBert4RecIdentityCosineModel(_FeatureBert4RecIdentityModel):

    def __init__(self, *args: Any, init_tau: float = 0.1, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.log_tau = nn.Parameter(torch.tensor(float(np.log(init_tau))))

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        warm_embs = self.item_encoder(self.feature_matrix)
        emb, pad_mask = self._build_seq_emb(x, warm_embs)
        out = self.encoder(emb, src_key_padding_mask=pad_mask)
        out = self.output_norm(out)
        out_n  = F.normalize(out,      dim=-1)
        warm_n = F.normalize(warm_embs, dim=-1)
        return (out_n @ warm_n.T) / self.tau

class FeatureBert4RecIdentityCosineRecommender(FeatureBert4RecIdentityRecommender):

    RECOMMENDER_NAME = "FeatureBert4RecIdentityCosine"

    def __init__(self, *args: Any, init_tau: float = 0.1, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.init_tau = init_tau

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
            f"sequences={len(sequences)}, device={self.device_}, init_tau={self.init_tau}"
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

        self.model = _FeatureBert4RecIdentityCosineModel(
            warm_feature_matrix, self.hidden_size, self.max_seq_len,
            self.n_layers, self.n_heads, self.dropout,
            init_tau=self.init_tau,
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
                tau=f"{self.model.tau.item():.4f}",
                patience=patience_left,
            )

            if patience_left <= 0:
                print(f"\n[{self.RECOMMENDER_NAME}] early stopping at epoch {epoch} "
                      f"(no val improvement for {self.early_stop_patience} epochs); "
                      f"best val={best_val:.4f} at epoch {best_epoch}, "
                      f"tau={self.model.tau.item():.4f}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"[{self.RECOMMENDER_NAME}] restored best checkpoint "
                  f"(val={best_val:.4f}, epoch={best_epoch}, tau={self.model.tau.item():.4f})")

    def _score_session_sequence(
        self,
        prior: list[str],
        warm_embs: torch.Tensor,
        cold_embs: torch.Tensor | None,
        all_embs: torch.Tensor,
    ) -> np.ndarray:
        assert self.model is not None and self.id_map is not None
        n_warm = warm_embs.shape[0]

        tokens: list[int] = []
        for t in prior:
            if t not in self.id_map.track_to_idx:
                continue
            g = self.id_map.track_to_idx[t]
            if g in self._global_to_warm_local:
                tokens.append(self._global_to_warm_local[g] + ITEM_OFFSET)
            elif g in self._global_to_cold_local:
                tokens.append(self._global_to_cold_local[g] + n_warm + ITEM_OFFSET)

        tokens = tokens + [MASK_TOKEN]
        tokens = tokens[-self.max_seq_len:]
        pad_len = self.max_seq_len - len(tokens)
        tokens = [PAD_TOKEN] * pad_len + tokens

        x = torch.tensor([tokens], dtype=torch.long, device=self.device_)

        with torch.no_grad():
            hidden = self.model.encode_hidden(x, items_table=all_embs)
            h = hidden[0, -1, :]
            h_n  = F.normalize(h, dim=-1)
            tau  = self.model.tau
            warm_n = F.normalize(warm_embs, dim=-1)
            warm_scores = ((h_n @ warm_n.T) / tau).cpu().numpy()
            if cold_embs is not None:
                cold_n = F.normalize(cold_embs, dim=-1)
                cold_scores = ((h_n @ cold_n.T) / tau).cpu().numpy()
            else:
                cold_scores = None

        scores = np.full(self.id_map.n_tracks, -np.inf, dtype=np.float32)
        scores[self._warm_global_indices] = warm_scores
        if cold_scores is not None:
            scores[self._cold_global_indices] = cold_scores
        return scores

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st["init_tau"] = self.init_tau
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
            self.model = _FeatureBert4RecIdentityCosineModel(
                dummy, self.hidden_size, self.max_seq_len,
                self.n_layers, self.n_heads, self.dropout,
                init_tau=self.init_tau,
            )
            self.model.load_state_dict(sd)
            self.model.to(self.device_)
