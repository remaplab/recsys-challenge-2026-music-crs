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
    ITEM_OFFSET,
    MASK_TOKEN,
    PAD_TOKEN,
    _FeatureBert4RecDataset,
)
from .feature_bert4rec_identity_cosine_rope import (
    FeatureBert4RecIdentityCosineRoPERecommender,
    _FeatureBert4RecIdentityCosineRoPEModel,
)

def _build_feature_matrix_per_modality(
    parquet_paths: list[str | Path],
    modalities: list[str],
    id_map,
) -> tuple[np.ndarray, list[int]]:
    from .feature_bert4rec import _build_feature_matrix
    chunks: list[np.ndarray] = []
    dims: list[int] = []
    for mod in modalities:
        single = _build_feature_matrix(parquet_paths, [mod], id_map)
        chunks.append(single)
        dims.append(single.shape[1])
    full = np.concatenate(chunks, axis=1)
    return full, dims

class _ModalityMultiHead(nn.Module):

    def __init__(self, modality_dims: list[int], hidden_size: int) -> None:
        super().__init__()
        self.modality_dims = list(modality_dims)
        self.hidden_size = hidden_size
        self.heads = nn.ModuleList(
            [nn.Linear(d, hidden_size, bias=True) for d in modality_dims]
        )

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
            sl = F.normalize(sl, dim=-1)
            proj = head(sl)
            out = proj if out is None else out + proj
        assert out is not None
        return out

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

        self.item_encoder = _ModalityMultiHead(self.modality_dims, hidden_size)

class FeatureBert4RecIdentityCosineRoPEMMHRecommender(FeatureBert4RecIdentityCosineRoPERecommender):

    RECOMMENDER_NAME = "FeatureBert4RecIdentityCosineRoPEMMH"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._modality_dims: list[int] = []

    def _build_modality_feature_matrix(self) -> tuple[np.ndarray, list[int]]:
        return _build_feature_matrix_per_modality(
            self.feature_emb_paths, self.feature_modalities, self.id_map
        )

    def _make_model(self, warm_feature_matrix: np.ndarray, modality_dims: list[int]):
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
        offsets = []
        s = 0
        for d in modality_dims:
            offsets.append((s, s + d))
            s += d

        for i, (a, b) in enumerate(offsets):
            mod = feature_matrix[:, a:b]
            mod_dim = mod.shape[1]
            n_components = min(self.hidden_size, mod_dim)

            norms = np.linalg.norm(mod, axis=1, keepdims=True).clip(min=1e-10)
            mod_n = mod / norms

            print(f"[{self.RECOMMENDER_NAME}] PCA({n_components}) on modality {i} "
                  f"(L2-normed {mod.shape[0]} × {mod_dim})...")
            pca = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
            pca.fit(mod_n)

            W = torch.from_numpy(pca.components_.astype(np.float32))
            mean = torch.from_numpy(pca.mean_.astype(np.float32))
            bias = -(mean @ W.T)

            head = self.model.item_encoder.heads[i]
            with torch.no_grad():
                if n_components < self.hidden_size:

                    head.weight[:n_components].copy_(W)
                    head.bias[:n_components].copy_(bias)

                    head.weight[n_components:].mul_(0.1)
                    head.bias[n_components:].zero_()
                    print(f"  modality {i}: PCA-init first {n_components}/{self.hidden_size} rows; "
                          f"remaining {self.hidden_size - n_components} rows kept (scaled ×0.1)")
                else:
                    head.weight.copy_(W)
                    head.bias.copy_(bias)
            explained = float(pca.explained_variance_ratio_.sum())
            print(f"  modality {i}: PCA explained variance = {explained:.3f}")

    def _build_val_eval_examples(self, val_sequences, val_dates=None) -> list[tuple]:
        out: list[tuple] = []
        min_turn = int(self.eval_min_turn)
        if val_dates is None:
            val_dates = [None] * len(val_sequences)
        for seq, sdate in zip(val_sequences, val_dates):
            for p in range(1, len(seq)):
                if (p + 1) < min_turn:
                    continue
                hist = list(seq[:p])
                out.append((hist, [0] * len(hist), int(seq[p]) - ITEM_OFFSET, 0, sdate))
        return out

    def _eval_encode_hidden(self, x: torch.Tensor, warm_embs: torch.Tensor,
                            q_idx_seq: torch.Tensor) -> torch.Tensor:
        return self.model.encode_hidden(x, items_table=warm_embs)

    def _eval_scores(self, hidden, q, warm_n, cold_n, tau):
        h_n = F.normalize(hidden[:, -1, :], dim=-1)
        warm_logits = (h_n @ warm_n.T) / tau
        cold_logits = (h_n @ cold_n.T) / tau if cold_n is not None else None
        return warm_logits, cold_logits

    def _compute_val_metrics(self, val_examples) -> dict[str, float] | None:
        if self.early_stop_metric not in ("ndcg", "recall") or not val_examples:
            return None
        if getattr(self, "_val_metric_disabled", False):
            return None
        try:
            return self._compute_val_metrics_impl(val_examples)
        except Exception as e:
            self._val_metric_disabled = True
            print(f"\n[{self.RECOMMENDER_NAME}] WARNING: could not compute the "
                  f"'{self.early_stop_metric}' early-stop metric "
                  f"({type(e).__name__}: {e}); FALLING BACK to val_loss for early "
                  f"stopping (this model's scoring is incompatible with the generic "
                  f"cosine+tau eval). Postfix will show 'val=' instead of "
                  f"'{self.early_stop_metric}@K='.")
            return None

    @torch.no_grad()
    def _compute_val_metrics_impl(self, val_examples) -> dict[str, float] | None:
        assert self.model is not None
        k_ndcg   = int(self.eval_ndcg_k)
        k_recall = int(self.eval_recall_k)
        L = self.max_seq_len
        self.model.eval()

        warm_embs = self.model.item_encoder(self.model.feature_matrix)
        warm_n = F.normalize(warm_embs, dim=-1)
        if self._cold_feature_matrix is not None and len(self._cold_global_indices) > 0:
            cold_feat = torch.from_numpy(self._cold_feature_matrix).to(self.device_)
            cold_n = F.normalize(self.model.item_encoder(cold_feat), dim=-1)
        else:
            cold_n = None
        tau = self.model.tau

        date_filter = (
            getattr(self, "track_release_dates", None) is not None
            and getattr(self, "max_future_years", None) is not None
        )
        if date_filter:
            warm_days = torch.from_numpy(
                self.track_release_dates[self._warm_global_indices]
                .astype("datetime64[D]").astype("int64")
            ).to(self.device_)
            cold_days = (
                torch.from_numpy(
                    self.track_release_dates[self._cold_global_indices]
                    .astype("datetime64[D]").astype("int64")
                ).to(self.device_)
                if cold_n is not None else None
            )
            horizon = int(float(self.max_future_years) * 365)
            NOMASK = int(np.iinfo(np.int64).max)

        ndcg_sum, recall_sum, n = 0.0, 0.0, 0
        for start in range(0, len(val_examples), self.batch_size):
            chunk = val_examples[start:start + self.batch_size]
            B = len(chunk)
            x  = torch.full((B, L), PAD_TOKEN, dtype=torch.long)
            q  = torch.zeros((B, L), dtype=torch.long)
            gt = torch.empty(B, dtype=torch.long)

            seen_r: list[int] = []
            seen_c: list[int] = []
            gt_seen = torch.zeros(B, dtype=torch.bool)
            cutoffs: list[int] = []

            append_mask = getattr(self, "_infer_append_mask", True)
            for i, (hist, hist_q, g, qidx, sdate) in enumerate(chunk):
                h_tok = list(hist[-(L - 1):])
                h_q   = list(hist_q[-(L - 1):])
                toks  = h_tok + ([MASK_TOKEN] if append_mask else [])
                qseq  = h_q + ([int(qidx)] if append_mask else [])
                off = L - len(toks)
                x[i, off:] = torch.tensor(toks, dtype=torch.long)
                q[i, off:] = torch.tensor(qseq, dtype=torch.long)
                gt[i] = g
                seen_local = {int(t) - ITEM_OFFSET for t in hist}
                seen_r.extend([i] * len(seen_local))
                seen_c.extend(seen_local)
                if g in seen_local:
                    gt_seen[i] = True
                if date_filter:
                    cutoffs.append(
                        NOMASK if sdate is None
                        else int(np.datetime64(sdate, "D").astype("int64")) + horizon
                    )
            x = x.to(self.device_); q = q.to(self.device_); gt = gt.to(self.device_)
            hidden = self._eval_encode_hidden(x, warm_embs, q)

            warm_logits, cold_logits = self._eval_scores(hidden, q, warm_n, cold_n, tau)
            gt_score = warm_logits.gather(1, gt.unsqueeze(1))
            if seen_r:
                warm_logits[seen_r, seen_c] = float("-inf")
            if date_filter:
                cutoff_t = torch.tensor(cutoffs, device=warm_logits.device).unsqueeze(1)
                warm_logits = warm_logits.masked_fill(warm_days.unsqueeze(0) > cutoff_t, float("-inf"))
            rank = (warm_logits > gt_score).sum(dim=1)
            if cold_logits is not None:
                if date_filter and cold_days is not None:
                    cold_logits = cold_logits.masked_fill(cold_days.unsqueeze(0) > cutoff_t, float("-inf"))
                rank = rank + (cold_logits > gt_score).sum(dim=1)
            recoverable = ~gt_seen.to(rank.device)
            ndcg_hit   = (rank < k_ndcg)   & recoverable
            recall_hit = (rank < k_recall) & recoverable
            ndcg_gain = torch.where(
                ndcg_hit,
                1.0 / torch.log2(rank.float() + 2.0),
                torch.zeros((), device=rank.device),
            )
            ndcg_sum   += ndcg_gain.sum().item()
            recall_sum += recall_hit.float().sum().item()
            n += B
        if not n:
            return None
        return {"ndcg": ndcg_sum / n, "recall": recall_sum / n}

    def _val_selection(self, val_metrics: dict[str, float] | None, val_avg: float):
        if val_metrics is None:
            return val_avg, {}
        sel = -float(val_metrics[self.early_stop_metric])
        disp = {
            f"ndcg@{self.eval_ndcg_k}":   f"{val_metrics['ndcg']:.4f}",
            f"recall@{self.eval_recall_k}": f"{val_metrics['recall']:.4f}",
        }
        return sel, disp

    def _val_metric_label(self, active: bool) -> str:
        if not active:
            return "val_loss"
        k = self.eval_recall_k if self.early_stop_metric == "recall" else self.eval_ndcg_k
        return f"{self.early_stop_metric}@{k}"

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
        val_examples = (
            self._build_val_eval_examples(val_sequences, getattr(self, "_val_session_dates", None))
            if self.early_stop_metric in ("ndcg", "recall") else None
        ) or None

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

            val_metrics = self._compute_val_metrics(val_examples)
            sel, mdisp = self._val_selection(val_metrics, val_avg)
            _active = val_metrics is not None

            improved = sel < best_val
            if improved:
                best_val = sel
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience_left = self.early_stop_patience
            else:
                patience_left -= 1

            postfix = dict(
                loss=f"{train_avg:.4f}", val=f"{val_avg:.4f}",
                tau=f"{self.model.tau.item():.4f}", patience=patience_left,
            )
            postfix.update(mdisp)
            postfix["best"] = f"{(-best_val) if _active else best_val:.4f}@{best_epoch}"
            epoch_bar.set_postfix(**postfix)

            if patience_left <= 0:
                _bv = (-best_val) if _active else best_val
                print(f"\n[{self.RECOMMENDER_NAME}] early stopping at epoch {epoch}; "
                      f"best {self._val_metric_label(_active)}={_bv:.4f} at epoch {best_epoch}, "
                      f"tau={self.model.tau.item():.4f}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            _active = val_examples is not None
            _bv = (-best_val) if _active else best_val
            print(f"[{self.RECOMMENDER_NAME}] restored best ({self._val_metric_label(_active)}={_bv:.4f}, "
                  f"epoch={best_epoch}, tau={self.model.tau.item():.4f})")

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
