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
    PAD_TOKEN,
    _build_feature_matrix,
    _FeatureBert4RecDataset,
)
from .feature_bert4rec_identity_cosine_rope import (
    FeatureBert4RecIdentityCosineRoPERecommender,
    _FeatureBert4RecIdentityCosineRoPEModel,
)

CATEGORY_TO_IDX: dict[str, int] = {
    "": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5,
    "F": 6, "G": 7, "H": 8, "I": 9, "J": 10, "K": 11,
}
N_CATEGORIES = 12

SPECIFICITY_TO_IDX: dict[str, int] = {"": 0, "LL": 1, "HL": 2, "LH": 3, "HH": 4}
N_SPECIFICITIES = 5

def encode_category(cat: str | None) -> int:
    return CATEGORY_TO_IDX.get(cat or "", 0)

def encode_specificity(spec: str | None) -> int:
    return SPECIFICITY_TO_IDX.get(spec or "", 0)

def build_session_goal_map(df: pl.DataFrame) -> dict[str, tuple[int, int]]:
    if "conversation_goal" in df.columns:
        cg = df.select(["session_id", "conversation_goal"]).unnest("conversation_goal")
        sids = cg["session_id"].to_list()
        cats = cg["category"].fill_null("").to_list()
        specs = cg["specificity"].fill_null("").to_list()
    else:
        sids = df["session_id"].to_list()
        cats = df["category"].fill_null("").to_list() if "category" in df.columns else [""] * len(sids)
        specs = df["specificity"].fill_null("").to_list() if "specificity" in df.columns else [""] * len(sids)
    return {sid: (encode_category(c), encode_specificity(s)) for sid, c, s in zip(sids, cats, specs)}

class _GoalDataset(_FeatureBert4RecDataset):

    def __init__(
        self,
        sequences: list[list[int]],
        cats: list[int],
        specs: list[int],
        n_warm: int,
        max_seq_len: int,
        mask_prob: float,
    ) -> None:
        assert len(sequences) == len(cats) == len(specs)
        super().__init__(sequences, n_warm, max_seq_len, mask_prob)
        self.cats = cats
        self.specs = specs

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        masked, labels = super().__getitem__(idx)
        return masked, labels, int(self.cats[idx]), int(self.specs[idx])

class _FeatureBert4RecIdentityCosineRoPEGoalModel(_FeatureBert4RecIdentityCosineRoPEModel):

    def __init__(
        self,
        warm_feature_matrix: np.ndarray,
        hidden_size: int,
        max_seq_len: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        init_tau: float = 0.1,
    ) -> None:

        super().__init__(
            warm_feature_matrix, hidden_size, max_seq_len + 2,
            n_layers, n_heads, dropout, init_tau=init_tau,
        )

        self._music_max_seq_len = max_seq_len

        self.cat_emb  = nn.Embedding(N_CATEGORIES,    hidden_size)
        self.spec_emb = nn.Embedding(N_SPECIFICITIES, hidden_size)

        with torch.no_grad():
            self.cat_emb.weight[0].zero_()
            self.spec_emb.weight[0].zero_()

    def _prepend_goal(
        self,
        emb: torch.Tensor,
        pad_mask: torch.Tensor,
        cat_idx: torch.Tensor,
        spec_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = emb.shape
        cat_e  = self.cat_emb(cat_idx).unsqueeze(1)
        spec_e = self.spec_emb(spec_idx).unsqueeze(1)
        emb_pref = torch.cat([cat_e, spec_e, emb], dim=1)
        prefix_mask = torch.zeros((B, 2), dtype=torch.bool, device=emb.device)
        pad_pref = torch.cat([prefix_mask, pad_mask], dim=1)
        return emb_pref, pad_pref

    def forward(
        self,
        x: torch.Tensor,
        cat_idx: torch.Tensor,
        spec_idx: torch.Tensor,
    ) -> torch.Tensor:
        warm_embs = self.item_encoder(self.feature_matrix)
        emb, pad_mask = self._build_seq_emb(x, warm_embs)
        emb_pref, pad_pref = self._prepend_goal(emb, pad_mask, cat_idx, spec_idx)
        out = self.encoder(emb_pref, src_key_padding_mask=pad_pref)
        out = self.output_norm(out)
        out = out[:, 2:, :]
        out_n  = F.normalize(out,      dim=-1)
        warm_n = F.normalize(warm_embs, dim=-1)
        return (out_n @ warm_n.T) / self.tau

    def encode_hidden(
        self,
        x: torch.Tensor,
        cat_idx: torch.Tensor,
        spec_idx: torch.Tensor,
        items_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if items_table is None:
            items_table = self.item_encoder(self.feature_matrix)
        emb, pad_mask = self._build_seq_emb(x, items_table)
        emb_pref, pad_pref = self._prepend_goal(emb, pad_mask, cat_idx, spec_idx)
        out = self.encoder(emb_pref, src_key_padding_mask=pad_pref)
        out = self.output_norm(out)
        return out[:, 2:, :]

class FeatureBert4RecIdentityCosineRoPEGoalRecommender(FeatureBert4RecIdentityCosineRoPERecommender):

    RECOMMENDER_NAME = "FeatureBert4RecIdentityCosineRoPEGoal"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self._session_goal_map: dict[str, tuple[int, int]] = {}

    def fit(self, train_df: pl.DataFrame, track_metadata=None, **kwargs: Any) -> None:
        self._session_goal_map.update(build_session_goal_map(train_df))
        super().fit(train_df, track_metadata=track_metadata, **kwargs)

    def register_session_goals(self, mapping: dict[str, tuple[int, int]]) -> None:
        self._session_goal_map.update(mapping)

    def _extra_score_kwargs_for_session(
        self, sess_id: str, user_id: str
    ) -> dict[str, Any]:
        cat_idx, spec_idx = self._session_goal_map.get(sess_id, (0, 0))
        return {"cat_idx": cat_idx, "spec_idx": spec_idx}

    def _build_sequences_with_goal(self) -> tuple[list[list[int]], list[int], list[int]]:
        from .feature_bert4rec import ITEM_OFFSET
        assert self.id_map is not None and self._train_long is not None
        seqs: list[list[int]] = []
        cats: list[int] = []
        specs: list[int] = []
        for sid_t, grp in (
            self._train_long
            .sort(["session_id", "turn_number"])
            .group_by("session_id", maintain_order=True)
        ):
            sid = sid_t[0] if isinstance(sid_t, tuple) else sid_t
            tokens = [
                self._global_to_warm_local[self.id_map.track_to_idx[t]] + ITEM_OFFSET
                for t in grp["track_id"].to_list()
                if t in self.id_map.track_to_idx
                and self.id_map.track_to_idx[t] in self._global_to_warm_local
            ]
            if len(tokens) < 2:
                continue
            cat_idx, spec_idx = self._session_goal_map.get(sid, (0, 0))
            seqs.append(tokens)
            cats.append(cat_idx)
            specs.append(spec_idx)
        return seqs, cats, specs

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

        sequences, cats, specs = self._build_sequences_with_goal()
        n_warm = len(self._warm_global_indices)
        n_cold = len(self._cold_global_indices)

        from collections import Counter
        cat_dist  = Counter(cats)
        spec_dist = Counter(specs)
        print(
            f"[{self.RECOMMENDER_NAME}] warm={n_warm}, cold={n_cold}, "
            f"sequences={len(sequences)}, device={self.device_}, init_tau={self.init_tau}"
        )
        print(f"  cat_dist (train, idx→count):  {dict(sorted(cat_dist.items()))}")
        print(f"  spec_dist (train, idx→count): {dict(sorted(spec_dist.items()))}")

        print(f"[{self.RECOMMENDER_NAME}] Loading feature embeddings: {self.feature_modalities}")
        full_matrix = _build_feature_matrix(
            self.feature_emb_paths, self.feature_modalities, self.id_map
        )
        self._feature_dim = full_matrix.shape[1]
        warm_feature_matrix = full_matrix[self._warm_global_indices]
        self._cold_feature_matrix = full_matrix[self._cold_global_indices]

        idx = list(range(len(sequences)))
        random.shuffle(idx)
        n_val = max(1, int(len(idx) * self.val_ratio))
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        train_seqs  = [sequences[i] for i in train_idx]
        train_cats  = [cats[i]      for i in train_idx]
        train_specs = [specs[i]     for i in train_idx]
        val_seqs    = [sequences[i] for i in val_idx]
        val_cats    = [cats[i]      for i in val_idx]
        val_specs   = [specs[i]     for i in val_idx]

        pin = self.device_.type == "cuda"
        train_loader = DataLoader(
            _GoalDataset(train_seqs, train_cats, train_specs, n_warm, self.max_seq_len, self.mask_prob),
            batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=pin,
        )
        val_loader = DataLoader(
            _GoalDataset(val_seqs, val_cats, val_specs, n_warm, self.max_seq_len, self.mask_prob),
            batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=pin,
        )

        self.model = _FeatureBert4RecIdentityCosineRoPEGoalModel(
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
            unit="ep", dynamic_ncols=True, file=sys.stdout,
        )

        best_val = float("inf")
        best_epoch = 0
        best_state: dict | None = None
        patience_left = self.early_stop_patience

        for epoch in epoch_bar:
            self.model.train()
            total_loss = 0.0
            for masked_seq, labels, cat_idx, spec_idx in tqdm(
                train_loader, desc=f"  ep {epoch:3d}", leave=False,
                unit="batch", dynamic_ncols=True, file=sys.stdout,
            ):
                masked_seq = masked_seq.to(self.device_)
                labels     = labels.to(self.device_)
                cat_idx    = cat_idx.to(self.device_)
                spec_idx   = spec_idx.to(self.device_)
                logits = self.model(masked_seq, cat_idx, spec_idx)
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
                for masked_seq, labels, cat_idx, spec_idx in val_loader:
                    masked_seq = masked_seq.to(self.device_)
                    labels     = labels.to(self.device_)
                    cat_idx    = cat_idx.to(self.device_)
                    spec_idx   = spec_idx.to(self.device_)
                    logits = self.model(masked_seq, cat_idx, spec_idx)
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

    def _score_session_sequence(
        self,
        prior: list[str],
        warm_embs: torch.Tensor,
        cold_embs: torch.Tensor | None,
        all_embs: torch.Tensor,
        cat_idx: int = 0,
        spec_idx: int = 0,
    ) -> np.ndarray:
        from .feature_bert4rec import ITEM_OFFSET, MASK_TOKEN
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
        c = torch.tensor([cat_idx],  dtype=torch.long, device=self.device_)
        s = torch.tensor([spec_idx], dtype=torch.long, device=self.device_)

        with torch.no_grad():
            hidden = self.model.encode_hidden(x, c, s, items_table=all_embs)
            h = hidden[0, -1, :]
            h_n    = F.normalize(h, dim=-1)
            tau    = self.model.tau
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
        st["session_goal_map"] = self._session_goal_map
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
        self._session_goal_map = state.get("session_goal_map", {})

        sd = state.get("model_state_dict")
        if sd is not None and self.id_map is not None and self._feature_dim is not None:
            n_warm = len(self._warm_global_indices)
            dummy = np.zeros((n_warm, self._feature_dim), dtype=np.float32)
            self.model = _FeatureBert4RecIdentityCosineRoPEGoalModel(
                dummy, self.hidden_size, self.max_seq_len,
                self.n_layers, self.n_heads, self.dropout,
                init_tau=self.init_tau,
            )
            self.model.load_state_dict(sd)
            self.model.to(self.device_)
