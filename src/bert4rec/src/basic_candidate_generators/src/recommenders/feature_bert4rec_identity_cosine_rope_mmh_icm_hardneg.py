from __future__ import annotations

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

from .feature_bert4rec import _FeatureBert4RecDataset
from .feature_bert4rec_identity_cosine_rope_mmh_icm import (
    FeatureBert4RecIdentityCosineRoPEMMHICMRecommender,
)

def _build_hard_negatives(
    warm_features_l2: np.ndarray,
    k: int,
    batch: int = 1024,
) -> np.ndarray:
    n_warm = warm_features_l2.shape[0]
    out = np.zeros((n_warm, k), dtype=np.int32)
    for start in tqdm(range(0, n_warm, batch), desc="hard_negs", file=sys.stdout):
        end = min(start + batch, n_warm)
        sims = warm_features_l2[start:end] @ warm_features_l2.T

        for i in range(end - start):
            sims[i, start + i] = -np.inf

        top = np.argpartition(-sims, k, axis=1)[:, :k]
        for i in range(end - start):
            vals = sims[i, top[i]]
            order = np.argsort(-vals)
            out[start + i] = top[i][order]
    return out

class FeatureBert4RecIdentityCosineRoPEMMHICMHardNegRecommender(FeatureBert4RecIdentityCosineRoPEMMHICMRecommender):

    RECOMMENDER_NAME = "FeatureBert4RecIdentityCosineRoPEMMHICMHardNeg"

    def __init__(
        self,
        *args: Any,
        hardneg_k: int = 32,
        hardneg_weight: float = 0.3,
        hardneg_tau: float = 0.1,
        ema_decay: float = 0.0,
        ema_start_epoch: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.hardneg_k = int(hardneg_k)
        self.hardneg_weight = float(hardneg_weight)
        self.hardneg_tau = float(hardneg_tau)

        self.ema_decay = float(ema_decay)
        self.ema_start_epoch = int(ema_start_epoch)

    def _get_hardneg_weight(self, epoch: int) -> float:
        return self.hardneg_weight

    def _make_dataset(
        self,
        sequences: list[list[int]],
        n_warm: int,
        max_seq_len: int,
        mask_prob: float,
        is_train: bool = True,
    ):
        return _FeatureBert4RecDataset(sequences, n_warm, max_seq_len, mask_prob,
                                       deterministic=self._val_deterministic(is_train))

    def _fit_model(self, urm: csr_matrix) -> None:

        assert self.id_map is not None and self._train_long is not None

        self._set_seeds()

        warm_track_ids: set[str] = set(self._train_long["track_id"].to_list())
        warm_track_ids &= set(self.id_map.track_to_idx.keys())
        self._warm_global_indices = sorted(self.id_map.track_to_idx[t] for t in warm_track_ids)
        self._cold_global_indices = sorted(
            idx for t, idx in self.id_map.track_to_idx.items() if t not in warm_track_ids
        )
        self._global_to_warm_local = {g: l for l, g in enumerate(self._warm_global_indices)}
        self._global_to_cold_local = {g: l for l, g in enumerate(self._cold_global_indices)}

        n_warm = len(self._warm_global_indices)
        n_cold = len(self._cold_global_indices)

        print(f"[{self.RECOMMENDER_NAME}] Loading per-modality features (with ICM)...")
        full_matrix, modality_dims = self._build_modality_feature_matrix()
        self._feature_dim = full_matrix.shape[1]
        self._modality_dims = modality_dims
        warm_feature_matrix = full_matrix[self._warm_global_indices]
        self._cold_feature_matrix = full_matrix[self._cold_global_indices]

        train_sequences, val_sequences = self._build_train_val_sequences()
        print(
            f"[{self.RECOMMENDER_NAME}] warm={n_warm}, cold={n_cold}, "
            f"train_seqs={len(train_sequences)}, val_seqs={len(val_sequences)}, "
            f"device={self.device_}, init_tau={self.init_tau}, "
            f"hardneg_k={self.hardneg_k}, hardneg_weight={self.hardneg_weight}, "
            f"hardneg_tau={self.hardneg_tau}"
        )

        print(f"[{self.RECOMMENDER_NAME}] Pre-computing hard negatives (k={self.hardneg_k}) on per-modality-L2 features...")
        feat_l2_chunks = []
        s = 0
        for d in modality_dims:
            chunk = warm_feature_matrix[:, s:s + d]
            norms = np.linalg.norm(chunk, axis=1, keepdims=True).clip(min=1e-10)
            feat_l2_chunks.append(chunk / norms)
            s += d
        feat_l2 = np.concatenate(feat_l2_chunks, axis=1).astype(np.float32)

        feat_l2 = feat_l2 / np.linalg.norm(feat_l2, axis=1, keepdims=True).clip(min=1e-10)
        hard_neg_idx = _build_hard_negatives(feat_l2, k=self.hardneg_k)
        hard_neg_tensor = torch.from_numpy(hard_neg_idx).to(self.device_)
        print(f"  hard_neg matrix shape: {hard_neg_tensor.shape}")

        pin = self.device_.type == "cuda"
        train_loader = DataLoader(
            self._make_dataset(train_sequences, n_warm, self.max_seq_len, self.mask_prob, is_train=True),
            batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=pin,
        )
        val_loader = DataLoader(
            self._make_dataset(val_sequences, n_warm, self.max_seq_len, self.mask_prob, is_train=False),
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

        epoch_bar = tqdm(range(1, self.epochs + 1), desc=f"[{self.RECOMMENDER_NAME}]",
                          unit="ep", dynamic_ncols=True, file=sys.stdout)

        best_val = float("inf")
        best_epoch = 0
        best_state: dict | None = None
        patience_left = self.early_stop_patience
        val_examples = (
            self._build_val_eval_examples(val_sequences, getattr(self, "_val_session_dates", None))
            if self.early_stop_metric in ("ndcg", "recall") else None
        ) or None

        ema_enabled = self.ema_decay > 0.0
        ema_params: dict[str, torch.Tensor] = {}
        ema_best_val = float("inf")
        ema_best_epoch = 0
        ema_best_state: dict | None = None
        if ema_enabled:
            for n, p in self.model.named_parameters():
                ema_params[n] = p.detach().clone()
            print(f"[{self.RECOMMENDER_NAME}] EMA enabled: decay={self.ema_decay}, "
                  f"start_epoch={self.ema_start_epoch}, n_params={len(ema_params)}")

        for epoch in epoch_bar:
            self.model.train()
            total_loss = 0.0
            total_mlm  = 0.0
            total_hn   = 0.0
            for masked_seq, labels in tqdm(train_loader, desc=f"  ep {epoch:3d}", leave=False,
                                            unit="batch", dynamic_ncols=True, file=sys.stdout):
                masked_seq = masked_seq.to(self.device_)
                labels     = labels.to(self.device_)
                logits = self.model(masked_seq)
                loss_mlm = F.cross_entropy(
                    logits.view(-1, n_warm), labels.view(-1), ignore_index=-100
                )

                with torch.no_grad():
                    flat_logits = logits.view(-1, n_warm)
                    flat_labels = labels.view(-1)
                    valid = flat_labels != -100
                tgt_pos = valid.nonzero(as_tuple=False).squeeze(-1)
                if tgt_pos.numel() > 0:
                    tgt_idx = flat_labels[tgt_pos]
                    anchor = flat_logits[tgt_pos, tgt_idx].unsqueeze(1)
                    hn = hard_neg_tensor[tgt_idx]
                    neg_logits = flat_logits[tgt_pos.unsqueeze(1).expand_as(hn), hn]
                    hn_logits = torch.cat([anchor, neg_logits], dim=1) / self.hardneg_tau
                    hn_labels = torch.zeros(hn_logits.size(0), dtype=torch.long, device=hn_logits.device)
                    loss_hn = F.cross_entropy(hn_logits, hn_labels)
                else:
                    loss_hn = torch.zeros((), device=self.device_)

                loss = loss_mlm + self._get_hardneg_weight(epoch) * loss_hn

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                total_mlm  += loss_mlm.item()
                total_hn   += loss_hn.item()

                if ema_enabled and epoch >= self.ema_start_epoch:
                    with torch.no_grad():
                        d = self.ema_decay
                        for n, p in self.model.named_parameters():
                            ema_params[n].mul_(d).add_(p.detach(), alpha=1.0 - d)

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
            train_avg = total_loss / len(train_loader)

            ema_val_avg: float | None = None
            ema_improved = False
            if ema_enabled and epoch >= self.ema_start_epoch:
                live_backup = {n: p.detach().clone() for n, p in self.model.named_parameters()}
                with torch.no_grad():
                    for n, p in self.model.named_parameters():
                        p.data.copy_(ema_params[n])
                    ema_val = 0.0
                    for masked_seq, labels in val_loader:
                        masked_seq = masked_seq.to(self.device_)
                        labels     = labels.to(self.device_)
                        logits = self.model(masked_seq)
                        ema_val += F.cross_entropy(
                            logits.view(-1, n_warm), labels.view(-1), ignore_index=-100
                        ).item()
                    ema_val_avg = ema_val / len(val_loader)

                    ema_metrics = self._compute_val_metrics(val_examples)
                    for n, p in self.model.named_parameters():
                        p.data.copy_(live_backup[n])
                ema_sel, _ = self._val_selection(ema_metrics, ema_val_avg)
                if ema_sel < ema_best_val:
                    ema_best_val = ema_sel
                    ema_best_epoch = epoch

                    snap = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    for n in ema_params:
                        snap[n] = ema_params[n].detach().cpu().clone()
                    ema_best_state = snap
                    ema_improved = True

            val_metrics = self._compute_val_metrics(val_examples)
            sel, mdisp = self._val_selection(val_metrics, val_avg)
            _active = val_metrics is not None

            improved = sel < best_val
            if improved:
                best_val = sel
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

            if improved or ema_improved:
                patience_left = self.early_stop_patience
            else:
                patience_left -= 1

            postfix = dict(
                loss=f"{train_avg:.4f}",
                mlm=f"{total_mlm/len(train_loader):.4f}",
                hn=f"{total_hn/len(train_loader):.4f}",
                val=f"{val_avg:.4f}",
                best=f"{(-best_val) if _active else best_val:.4f}@{best_epoch}",
                tau=f"{self.model.tau.item():.4f}",
                patience=patience_left,
            )
            postfix.update(mdisp)
            if ema_val_avg is not None:
                postfix["ema_val"]  = f"{ema_val_avg:.4f}"
                postfix["ema_best"] = f"{(-ema_best_val) if _active else ema_best_val:.4f}@{ema_best_epoch}"
            epoch_bar.set_postfix(**postfix)
            if patience_left <= 0:
                _lbl = self._val_metric_label(_active)
                _bl = f"{(-best_val) if _active else best_val:.4f}"
                _be = f"{(-ema_best_val) if _active else ema_best_val:.4f}"
                print(f"\n[{self.RECOMMENDER_NAME}] early stopping at epoch {epoch}; "
                      f"best live {_lbl}={_bl}@{best_epoch}, "
                      f"best ema {_lbl}={_be}@{ema_best_epoch}")
                break

        use_ema = (
            ema_enabled
            and ema_best_state is not None
            and ema_best_val < best_val
        )
        _active = val_examples is not None
        _m = self._val_metric_label(_active)
        _sgn = -1.0 if _active else 1.0
        if use_ema:
            self.model.load_state_dict(ema_best_state)
            print(f"[{self.RECOMMENDER_NAME}] restored EMA best "
                  f"(ema_{_m}={_sgn*ema_best_val:.4f}@{ema_best_epoch} better than "
                  f"live_{_m}={_sgn*best_val:.4f}@{best_epoch})")
        elif best_state is not None:
            self.model.load_state_dict(best_state)
            tag = "live best" if not ema_enabled else f"live best (EMA worse: {_sgn*ema_best_val:.4f})"
            print(f"[{self.RECOMMENDER_NAME}] restored {tag} ({_m}={_sgn*best_val:.4f}, epoch={best_epoch})")
