"""Base for the 4-way stack: split_hidim_xattn + hardneg + query_full + MHA + user-aware.

Combines four orthogonal mechanisms in a single recommender, configured via flags:

  1. split_hidim_xattn (inherited)        : xattn fusion over per-modality
                                            split-ICM features.
  2. hardneg loss (rewritten in _fit_model): InfoNCE auxiliary over feature-cosine
                                            top-K hard negatives.
  3. query_full (Design A)                : per-turn Qwen3 query injected at EVERY
                                            non-padding position (train + inference).
  4. MHA xattn                            : multi-head modality cross-attention
                                            (replaces single-query xattn item_encoder).
  5. User-aware (flag-controlled)         : FiLM modulation on CF user emb OR
                                            taste-fusion on history-mean.

Concrete variants:
  - `_userfilm.py`     : use_film=True
  - `_userhistmod.py`  : use_query_fusion=True, source="taste"

The combined model conflicts with multi-inheritance: query_full/query/hardneg's
`_fit_model` is incompatible with user_base's `_fit_model` (different dataset
tuple, different forward signature). This module rewrites `_fit_model` to
include every required behavior in one place.
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .feature_bert4rec import ITEM_OFFSET, MASK_TOKEN, PAD_TOKEN
from .feature_bert4rec_identity_cosine_rope_mmh_icm_hardneg import _build_hard_negatives
from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim_xattn_hardneg_query import (
    _load_query_emb_cache,
)
from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim_xattn_user_base import (
    _UserAwareSplitHiDimXAttnModel,
    _UserAwareSplitHiDimXAttnRecommenderBase,
)
from .feature_bert4rec_variants_blocks import _ModalityCrossAttnMH


# ---------------------------------------------------------------------------
# Model: user-aware base + MHA xattn item encoder + query injection table
# ---------------------------------------------------------------------------

class _QueryFullMHAUserModel(_UserAwareSplitHiDimXAttnModel):
    """User-aware model with MHA xattn item encoder and per-position query injection.

    Forward signature: (x, user_idx, query_idx_seq=None, labels=None).
      - query_idx_seq is (B, L) long; 0 means "no query" (zero vector).
      - labels is (B, L) long; -100 = masked-out (loss ignore); only required when
        the user-aware mechanism uses the leave-one-session-out taste (taste-fusion
        or bias-term variants).
    """

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
        n_users: int = 1,
        xattn_heads: int = 4,
        query_emb_table: np.ndarray | None = None,
        use_bias_term: bool = False,
        use_film: bool = False,
        use_query_fusion: bool = False,
        query_fusion_source: str = "taste",
        alpha_init: float = 0.1,
    ) -> None:
        super().__init__(
            warm_feature_matrix, hidden_size, max_seq_len,
            n_layers, n_heads, dropout,
            init_tau=init_tau, modality_dims=modality_dims,
            n_users=n_users,
            use_bias_term=use_bias_term,
            use_film=use_film,
            use_query_fusion=use_query_fusion,
            query_fusion_source=query_fusion_source,
            alpha_init=alpha_init,
        )
        # Swap the single-query xattn produced by the user-base init with the
        # multi-head version (same per-modality Linear heads, K parallel queries
        # over modalities).
        self.item_encoder = _ModalityCrossAttnMH(
            modality_dims, self.hidden_size, n_attn_heads=int(xattn_heads),
        )

        # Query injection table + projection (Design A: applied at every
        # non-padding position).
        assert query_emb_table is not None, "query_emb_table required"
        self.register_buffer(
            "query_table",
            torch.from_numpy(query_emb_table),
            persistent=False,
        )
        query_dim = query_emb_table.shape[1]
        self.query_proj = nn.Linear(query_dim, self.hidden_size, bias=True)
        # Zero-init: step 0 reproduces the no-query baseline; the model learns
        # WHEN to use the query rather than being forced from epoch 1.
        nn.init.zeros_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)

    def _apply_query(self, emb: torch.Tensor, query_idx_seq: torch.Tensor | None) -> torch.Tensor:
        if query_idx_seq is None:
            return emb
        q = self.query_table[query_idx_seq]            # (B, L, query_dim)
        return emb + self.query_proj(q)

    # ------------------------------------------------------------------
    # Session-prior taste (NEW alternative to LOSO).
    #
    # Per-sample taste from the CURRENT session's items that the model
    # actually sees as context (positions where `labels == -100` AND
    # `masked != PAD`). The to-mask positions (MASK / random / keep-original
    # but labeled) are EXCLUDED because they all carry `labels != -100`, so
    # there is no leak of the target through the taste channel.
    #
    # Train/test alignment: at inference the model sees ALL prior items as
    # context (zero masking) → taste = mean over all warm prior items, same
    # recipe. This is the fix for the splitK user-disjoint test setup where
    # the LOSO user taste collapses to 0 at inference (= train/test gap).
    # ------------------------------------------------------------------
    def _session_prior_taste(
        self,
        masked_seq: torch.Tensor,
        labels: torch.Tensor,
        item_emb_n: torch.Tensor,
    ) -> torch.Tensor:
        from .feature_bert4rec import PAD_TOKEN as _PAD  # local import to avoid cycles
        safe = (labels == -100) & (masked_seq != _PAD)           # (B, L)
        warm_local = masked_seq - ITEM_OFFSET                     # invalid where not safe
        B, L = masked_seq.shape
        H = item_emb_n.size(1)
        device = item_emb_n.device

        safe_flat = safe.view(-1)
        b_row = torch.arange(B, device=device).unsqueeze(1).expand(-1, L).reshape(-1)
        w_flat = warm_local.view(-1)
        b_safe = b_row[safe_flat]
        w_safe = w_flat[safe_flat]

        session_sum = torch.zeros(B, H, device=device, dtype=item_emb_n.dtype)
        if b_safe.numel() > 0:
            session_sum.index_add_(0, b_safe, item_emb_n[w_safe])

        session_count = torch.zeros(B, device=device, dtype=item_emb_n.dtype)
        if b_safe.numel() > 0:
            ones = torch.ones(b_safe.numel(), device=device, dtype=item_emb_n.dtype)
            session_count.index_add_(0, b_safe, ones)
        session_count = session_count.clamp_min(1.0).unsqueeze(-1)

        return F.normalize(session_sum / session_count, dim=-1)

    # Override parent's _query_fuse to accept the "session_taste" source
    # (parent only knows "taste" and "cf").
    def _query_fuse(
        self,
        out_n: torch.Tensor,
        user_idx: torch.Tensor,
        taste_vec: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.query_fusion_source in ("taste", "session_taste"):
            uv = taste_vec
        else:  # "cf"
            cf = self.cf_table[user_idx]
            uv = F.normalize(self.cf_query_proj(cf), dim=-1)
        fused = out_n + self.alpha_query * uv.unsqueeze(1)
        return F.normalize(fused, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        user_idx: torch.Tensor,
        query_idx_seq: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        warm_embs = self.item_encoder(self.feature_matrix)
        emb, pad_mask = self._build_seq_emb(x, warm_embs)
        emb = self._apply_query(emb, query_idx_seq)
        out = self.encoder(emb, src_key_padding_mask=pad_mask)
        out = self.output_norm(out)
        if self.use_film:
            out = self._apply_film(out, user_idx)

        out_n  = F.normalize(out, dim=-1)
        warm_n = F.normalize(warm_embs, dim=-1)

        taste_vec = None
        need_user_loso = self.use_bias_term or (
            self.use_query_fusion and self.query_fusion_source == "taste"
        )
        need_session_taste = (
            self.use_query_fusion and self.query_fusion_source == "session_taste"
        )
        if need_user_loso:
            assert labels is not None, "labels required for LOSO taste during training"
            unmasked = self._decode_unmasked(x, labels)
            taste_vec = self._loso_taste(user_idx, unmasked, warm_n)
        elif need_session_taste:
            assert labels is not None, "labels required to identify safe context"
            taste_vec = self._session_prior_taste(x, labels, warm_n)

        if self.use_query_fusion:
            out_n = self._query_fuse(out_n, user_idx, taste_vec)

        score = (out_n @ warm_n.T) / self.tau
        if self.use_bias_term:
            bias = self.alpha_bias * (taste_vec @ warm_n.T) / self.tau
            score = score + bias.unsqueeze(1)
        return score

    def encode_hidden(
        self,
        x: torch.Tensor,
        user_idx: torch.Tensor,
        items_table: torch.Tensor | None = None,
        query_idx_seq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if items_table is None:
            items_table = self.item_encoder(self.feature_matrix)
        emb, pad_mask = self._build_seq_emb(x, items_table)
        emb = self._apply_query(emb, query_idx_seq)
        out = self.encoder(emb, src_key_padding_mask=pad_mask)
        out = self.output_norm(out)
        if self.use_film:
            out = self._apply_film(out, user_idx)
        return out


# ---------------------------------------------------------------------------
# Dataset: emits (masked_seq, labels, user_idx, q_idx_seq) per sample.
# Query indices populated at ALL non-padding positions (Design A).
# ---------------------------------------------------------------------------

class _QueryFullUserDataset(Dataset):
    def __init__(
        self,
        sequences: list[list[tuple[int, int]]],
        user_idxs: list[int],
        n_warm: int,
        max_seq_len: int,
        mask_prob: float,
    ) -> None:
        assert len(sequences) == len(user_idxs)
        self.sequences   = sequences
        self.user_idxs   = user_idxs
        self.n_warm      = n_warm
        self.max_seq_len = max_seq_len
        self.mask_prob   = mask_prob

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        seq    = self.sequences[idx][-self.max_seq_len:]
        tokens = [tok for tok, _ in seq]
        q_idxs = [qi  for _,  qi in seq]
        masked = list(tokens)
        labels = [-100] * len(tokens)

        to_mask = [i for i in range(len(tokens)) if random.random() < self.mask_prob]
        if not to_mask:
            to_mask = [random.randrange(len(tokens))]

        for i in to_mask:
            labels[i] = tokens[i] - ITEM_OFFSET
            r = random.random()
            if r < 0.8:
                masked[i] = MASK_TOKEN
            elif r < 0.9:
                masked[i] = random.randint(0, self.n_warm - 1) + ITEM_OFFSET
            # else 10% keep original

        pad_len = self.max_seq_len - len(tokens)
        masked   = [PAD_TOKEN] * pad_len + masked
        labels   = [-100]      * pad_len + labels
        q_at_all = [0]         * pad_len + list(q_idxs)

        return (
            torch.tensor(masked,   dtype=torch.long),
            torch.tensor(labels,   dtype=torch.long),
            int(self.user_idxs[idx]),
            torch.tensor(q_at_all, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Recommender base: combined fit + score
# ---------------------------------------------------------------------------

class _QueryFullMHAUserRecommenderBase(_UserAwareSplitHiDimXAttnRecommenderBase):
    """Base recommender for the 4-way variants.

    Subclasses set the user-aware flags (use_film / use_query_fusion / ...)
    via __init__. Everything else (hardneg + query_full + MHA + EMA) is
    handled here.
    """

    RECOMMENDER_NAME = "FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullMHAUserBase"

    def __init__(
        self,
        *args: Any,
        # MHA
        xattn_heads: int = 4,
        # Query
        query_emb_dir: str = "models/query_emb_cache/qwen3_frozen",
        query_cache_splits: list[str] | None = None,
        # Hardneg
        hardneg_k: int = 32,
        hardneg_weight: float = 0.3,
        hardneg_tau: float = 0.1,
        # EMA
        ema_decay: float = 0.0,
        ema_start_epoch: int = 1,
        # When query_fusion_source="taste" (LOSO user history): for users
        # without train history (e.g. all test users under splitK), the train
        # taste collapses to 0 at inference. Setting this True substitutes
        # the mean of the user's inference-side warm tracks as a fallback.
        # NB: distributional shift vs train-time LOSO; may help or hurt.
        # Has no effect for query_fusion_source in ("session_taste", "cf").
        use_taste_inference_fallback: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.xattn_heads        = int(xattn_heads)
        self.query_emb_dir      = str(query_emb_dir)
        self.query_cache_splits = list(query_cache_splits) if query_cache_splits else ["train", "dev", "blind_a"]
        self._query_emb_table: np.ndarray | None = None
        self._query_lookup: dict[tuple[str, int], int] = {}
        # Populated by recommend() before the inference loop runs.
        self._session_target_turn: dict[str, int] = {}
        self._cur_prior_turn_list: dict[str, list[tuple[int, str]]] = {}
        self._inference_user_warm_globals: dict[str, set[int]] = {}
        # Hardneg / EMA
        self.hardneg_k       = int(hardneg_k)
        self.hardneg_weight  = float(hardneg_weight)
        self.hardneg_tau     = float(hardneg_tau)
        self.ema_decay       = float(ema_decay)
        self.ema_start_epoch = int(ema_start_epoch)
        self.use_taste_inference_fallback = bool(use_taste_inference_fallback)

    def _get_hardneg_weight(self, epoch: int) -> float:
        """Constant weight (same as parent hardneg)."""
        return self.hardneg_weight

    # ------------------------------------------------------------------
    # Query cache loader
    # ------------------------------------------------------------------

    def _resolve_query_emb_dir(self) -> Path:
        p = Path(self.query_emb_dir)
        if p.is_absolute():
            return p
        from launchers._predict_fold_common import repo_path
        return repo_path(self.query_emb_dir)

    def _load_query_cache(self) -> None:
        cache_dir = self._resolve_query_emb_dir()
        print(f"[{self.RECOMMENDER_NAME}] Loading query cache from {cache_dir}")
        emb, lookup = _load_query_emb_cache(cache_dir, self.query_cache_splits)
        self._query_emb_table = emb
        self._query_lookup    = lookup
        print(f"  query_emb_table: {emb.shape}  (row 0 is zero); {len(lookup)} (sess, turn) keys")

    # ------------------------------------------------------------------
    # Sequence build: user_idx + (token, q_idx) pairs per session
    # ------------------------------------------------------------------

    def _build_user_aware_query_sequences(self) -> tuple[
        list[list[tuple[int, int]]], list[int],
        list[list[tuple[int, int]]], list[int],
    ]:
        """(train_seqs, train_user_idxs, val_seqs, val_user_idxs).

        train/val seqs are lists of (warm-token, query_idx) pairs.
        Per-sequence random split with val_ratio (each session yields 1 sequence)."""
        assert self.id_map is not None and self._train_long is not None
        seqs: list[list[tuple[int, int]]]    = []
        user_idxs: list[int]                  = []
        n_missing = 0
        n_total   = 0
        for sid_t, grp in (
            self._train_long
            .sort(["session_id", "turn_number"])
            .group_by("session_id", maintain_order=True)
        ):
            sid = sid_t[0] if isinstance(sid_t, tuple) else sid_t
            uid = self._session_user_map.get(sid, "<unk>")
            user_idx = self._user_to_idx.get(uid, 0)

            tn_col = grp["turn_number"].to_list()
            tk_col = grp["track_id"].to_list()
            pairs: list[tuple[int, int]] = []
            for tn, tid in zip(tn_col, tk_col):
                gidx = self.id_map.track_to_idx.get(tid)
                if gidx is None or gidx not in self._global_to_warm_local:
                    continue
                token = self._global_to_warm_local[gidx] + ITEM_OFFSET
                qidx  = self._query_lookup.get((sid, int(tn)), 0)
                n_total += 1
                if qidx == 0:
                    n_missing += 1
                pairs.append((token, qidx))
            if len(pairs) >= 2:
                seqs.append(pairs)
                user_idxs.append(user_idx)

        cov = 1.0 - (n_missing / max(1, n_total))
        print(f"  built {len(seqs)} sequences; query coverage: "
              f"{n_total - n_missing}/{n_total} = {cov:.1%}")

        idx = list(range(len(seqs)))
        random.shuffle(idx)
        n_val = max(1, int(len(idx) * self.val_ratio))
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        return (
            [seqs[i] for i in train_idx],
            [user_idxs[i] for i in train_idx],
            [seqs[i] for i in val_idx],
            [user_idxs[i] for i in val_idx],
        )

    # ------------------------------------------------------------------
    # Override _make_model to produce the combined model
    # ------------------------------------------------------------------

    def _make_model(self, warm_feature_matrix: np.ndarray, modality_dims: list[int]):
        assert self._query_emb_table is not None, "_load_query_cache must run before _make_model"
        return _QueryFullMHAUserModel(
            warm_feature_matrix, self.hidden_size, self.max_seq_len,
            self.n_layers, self.n_heads, self.dropout,
            init_tau=self.init_tau,
            modality_dims=modality_dims,
            n_users=max(1, self._n_users),
            xattn_heads=self.xattn_heads,
            query_emb_table=self._query_emb_table,
            use_bias_term=self.use_bias_term,
            use_film=self.use_film,
            use_query_fusion=self.use_query_fusion,
            query_fusion_source=self.query_fusion_source,
            alpha_init=self.alpha_init,
        )

    # ------------------------------------------------------------------
    # Combined _fit_model: user-aware + query + hardneg + EMA
    # ------------------------------------------------------------------

    def _fit_model(self, urm: csr_matrix) -> None:
        assert self.id_map is not None and self._train_long is not None

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

        # 1. Query cache BEFORE building sequences.
        self._load_query_cache()

        print(f"[{self.RECOMMENDER_NAME}] Loading per-modality features (with split ICM)...")
        full_matrix, modality_dims = self._build_modality_feature_matrix()
        self._feature_dim = full_matrix.shape[1]
        self._modality_dims = modality_dims
        warm_feature_matrix = full_matrix[self._warm_global_indices]
        self._cold_feature_matrix = full_matrix[self._cold_global_indices]

        # 2. User infrastructure.
        self._build_user_index()
        self._build_user_played_global()

        train_seqs, train_uidx, val_seqs, val_uidx = self._build_user_aware_query_sequences()
        print(
            f"[{self.RECOMMENDER_NAME}] warm={n_warm}, cold={n_cold}, "
            f"train_seqs={len(train_seqs)}, val_seqs={len(val_seqs)}, "
            f"device={self.device_}, init_tau={self.init_tau}, "
            f"hardneg_k={self.hardneg_k}, hardneg_weight={self.hardneg_weight}, "
            f"hardneg_tau={self.hardneg_tau}, xattn_heads={self.xattn_heads}, "
            f"use_film={self.use_film}, use_query_fusion={self.use_query_fusion}, "
            f"query_source={self.query_fusion_source}"
        )

        # 3. Hardneg precompute (feature-cosine top-K).
        print(f"[{self.RECOMMENDER_NAME}] Pre-computing hard negatives (k={self.hardneg_k})...")
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
        hard_neg_tensor = torch.from_numpy(hard_neg_idx).to(self.device_).long()

        # 4. Loaders.
        pin = self.device_.type == "cuda"
        train_loader = DataLoader(
            _QueryFullUserDataset(train_seqs, train_uidx, n_warm, self.max_seq_len, self.mask_prob),
            batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=pin,
        )
        val_loader = DataLoader(
            _QueryFullUserDataset(val_seqs, val_uidx, n_warm, self.max_seq_len, self.mask_prob),
            batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=pin,
        )

        # 5. Model + PCA init + user tables.
        self.model = self._make_model(warm_feature_matrix, modality_dims)
        self._pca_init_encoder_per_modality(warm_feature_matrix, modality_dims)
        self.model.to(self.device_)
        self._load_cf_table()
        self._refresh_user_sums()

        n_qp = sum(p.numel() for p in self.model.query_proj.parameters())
        print(f"  query_proj params: {n_qp:,}")

        # 6. Optimizer / scheduler.
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        total_steps  = self.epochs * len(train_loader)
        warmup_steps = max(1, int(total_steps * self.warmup_ratio))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, self._make_cosine_lr_lambda(total_steps, warmup_steps)
        )

        epoch_bar = tqdm(range(1, self.epochs + 1), desc=f"[{self.RECOMMENDER_NAME}]",
                         unit="ep", dynamic_ncols=True, file=sys.stdout)

        best_val = float("inf")
        best_epoch = 0
        best_state: dict | None = None
        patience_left = self.early_stop_patience

        # EMA shadow.
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

        need_taste_in_loss = self.use_bias_term or (
            self.use_query_fusion and self.query_fusion_source in ("taste", "session_taste")
        )

        for epoch in epoch_bar:
            # Refresh user_sum/count once per epoch with the current encoder.
            self._refresh_user_sums()

            self.model.train()
            total_loss = 0.0
            total_mlm  = 0.0
            total_hn   = 0.0
            for masked_seq, labels, user_idx, q_idx in tqdm(
                train_loader, desc=f"  ep {epoch:3d}", leave=False,
                unit="batch", dynamic_ncols=True, file=sys.stdout,
            ):
                masked_seq = masked_seq.to(self.device_)
                labels     = labels.to(self.device_)
                user_idx   = user_idx.to(self.device_)
                q_idx      = q_idx.to(self.device_)

                # `labels` is forwarded so the user-aware model can compute the
                # leave-one-session-out taste during forward (LOSO subtracts the
                # current session's items from user_sum to avoid the train-time
                # leak).
                labels_kw = labels if need_taste_in_loss else None
                logits = self.model(masked_seq, user_idx,
                                    query_idx_seq=q_idx, labels=labels_kw)
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
                    hn_labels = torch.zeros(hn_logits.size(0), dtype=torch.long,
                                            device=hn_logits.device)
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

                # EMA update on parameters (not buffers).
                if ema_enabled and epoch >= self.ema_start_epoch:
                    with torch.no_grad():
                        d = self.ema_decay
                        for n, p in self.model.named_parameters():
                            ema_params[n].mul_(d).add_(p.detach(), alpha=1.0 - d)

            # Val pass (live params).
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for masked_seq, labels, user_idx, q_idx in val_loader:
                    masked_seq = masked_seq.to(self.device_)
                    labels     = labels.to(self.device_)
                    user_idx   = user_idx.to(self.device_)
                    q_idx      = q_idx.to(self.device_)
                    labels_kw = labels if need_taste_in_loss else None
                    logits = self.model(masked_seq, user_idx,
                                        query_idx_seq=q_idx, labels=labels_kw)
                    val_loss += F.cross_entropy(
                        logits.view(-1, n_warm), labels.view(-1), ignore_index=-100
                    ).item()
            val_avg   = val_loss / len(val_loader)
            train_avg = total_loss / len(train_loader)

            # EMA val: temporarily swap params, eval, swap back.
            ema_val_avg: float | None = None
            ema_improved = False
            if ema_enabled and epoch >= self.ema_start_epoch:
                live_backup = {n: p.detach().clone() for n, p in self.model.named_parameters()}
                with torch.no_grad():
                    for n, p in self.model.named_parameters():
                        p.data.copy_(ema_params[n])
                    ema_val = 0.0
                    for masked_seq, labels, user_idx, q_idx in val_loader:
                        masked_seq = masked_seq.to(self.device_)
                        labels     = labels.to(self.device_)
                        user_idx   = user_idx.to(self.device_)
                        q_idx      = q_idx.to(self.device_)
                        labels_kw = labels if need_taste_in_loss else None
                        logits = self.model(masked_seq, user_idx,
                                            query_idx_seq=q_idx, labels=labels_kw)
                        ema_val += F.cross_entropy(
                            logits.view(-1, n_warm), labels.view(-1), ignore_index=-100
                        ).item()
                    ema_val_avg = ema_val / len(val_loader)
                    for n, p in self.model.named_parameters():
                        p.data.copy_(live_backup[n])
                if ema_val_avg < ema_best_val:
                    ema_best_val = ema_val_avg
                    ema_best_epoch = epoch
                    snap = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    for n in ema_params:
                        snap[n] = ema_params[n].detach().cpu().clone()
                    ema_best_state = snap
                    ema_improved = True

            improved = val_avg < best_val
            if improved:
                best_val = val_avg
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            if improved or ema_improved:
                patience_left = self.early_stop_patience
            else:
                patience_left -= 1

            ab = float(self.model.alpha_bias.item())  if hasattr(self.model, "alpha_bias")  else -1
            aq = float(self.model.alpha_query.item()) if hasattr(self.model, "alpha_query") else -1
            postfix = dict(
                loss=f"{train_avg:.4f}",
                mlm=f"{total_mlm/len(train_loader):.4f}",
                hn=f"{total_hn/len(train_loader):.4f}",
                val=f"{val_avg:.4f}",
                best=f"{best_val:.4f}@{best_epoch}",
                tau=f"{self.model.tau.item():.4f}",
                patience=patience_left,
            )
            if ab >= 0:
                postfix["ab"] = f"{ab:.3f}"
            if aq >= 0:
                postfix["aq"] = f"{aq:.3f}"
            if ema_val_avg is not None:
                postfix["ema_val"]  = f"{ema_val_avg:.4f}"
                postfix["ema_best"] = f"{ema_best_val:.4f}@{ema_best_epoch}"
            epoch_bar.set_postfix(**postfix)
            if patience_left <= 0:
                print(f"\n[{self.RECOMMENDER_NAME}] early stopping at epoch {epoch}; "
                      f"best live val={best_val:.4f}@{best_epoch}, "
                      f"best ema val={ema_best_val:.4f}@{ema_best_epoch}")
                break

        use_ema = ema_enabled and ema_best_state is not None and ema_best_val < best_val
        if use_ema:
            self.model.load_state_dict(ema_best_state)
            print(f"[{self.RECOMMENDER_NAME}] restored EMA best "
                  f"(ema_val={ema_best_val:.4f}@{ema_best_epoch} < live_val={best_val:.4f}@{best_epoch})")
        elif best_state is not None:
            self.model.load_state_dict(best_state)
            tag = "live best" if not ema_enabled else f"live best (EMA worse: {ema_best_val:.4f})"
            print(f"[{self.RECOMMENDER_NAME}] restored {tag} "
                  f"(val={best_val:.4f}, epoch={best_epoch}, tau={self.model.tau.item():.4f})")

        # Final refresh AFTER restoring best so inference uses sums consistent
        # with the chosen weights.
        self._refresh_user_sums()

    # ------------------------------------------------------------------
    # Inference: populate target_turn + prior turn list (Design A) and
    # forward user_idx to score_session_sequence.
    # ------------------------------------------------------------------

    def recommend(self, context_df: pl.DataFrame, *args: Any, **kwargs: Any) -> pl.DataFrame:
        if "track_id" not in context_df.columns:
            from .interactions import explode_music_turns
            context_df = explode_music_turns(context_df)

        if "target_turn" in context_df.columns:
            tt_df = context_df.select(["session_id", "target_turn"]).unique(subset=["session_id"])
            self._session_target_turn = {
                row["session_id"]: int(row["target_turn"])
                for row in tt_df.iter_rows(named=True)
            }
        else:
            self._session_target_turn = {}

        self._cur_prior_turn_list = {}
        if "turn_number" in context_df.columns and context_df.height > 0:
            ctx_sorted = context_df.sort(["session_id", "turn_number"])
            for sid, grp in ctx_sorted.group_by("session_id", maintain_order=True):
                sid_str = sid[0] if isinstance(sid, tuple) else sid
                tns  = grp["turn_number"].to_list()
                tids = grp["track_id"].to_list()
                self._cur_prior_turn_list[sid_str] = [
                    (int(tn), tid) for tn, tid in zip(tns, tids)
                    if tid is not None and tn is not None
                ]

        if not self._cur_prior_turn_list and context_df.height > 0:
            n_with_track = context_df.filter(pl.col("track_id").is_not_null()).height
            if n_with_track > 0:
                print(
                    f"[{self.RECOMMENDER_NAME}] WARNING: prior turn map empty "
                    f"despite {n_with_track} context rows with track_id. "
                    f"All prior positions will get query=0 — train/test mismatch!"
                )

        n_prior_total = 0; n_prior_hit = 0
        for sid, prior_list in self._cur_prior_turn_list.items():
            for tn, _tid in prior_list:
                n_prior_total += 1
                if self._query_lookup.get((sid, tn), 0) != 0:
                    n_prior_hit += 1
        n_tgt_total = 0; n_tgt_hit = 0
        for sid, tt in self._session_target_turn.items():
            n_tgt_total += 1
            if self._query_lookup.get((sid, int(tt)), 0) != 0:
                n_tgt_hit += 1
        if n_prior_total or n_tgt_total:
            pcov = (n_prior_hit / n_prior_total * 100) if n_prior_total else 0.0
            tcov = (n_tgt_hit   / n_tgt_total   * 100) if n_tgt_total   else 0.0
            print(
                f"[{self.RECOMMENDER_NAME}] inference query coverage: "
                f"prior = {n_prior_hit}/{n_prior_total} ({pcov:.1f}%), "
                f"target = {n_tgt_hit}/{n_tgt_total} ({tcov:.1f}%) "
                f"  |  n_sessions={len(self._session_target_turn)}"
            )

        # Per-user inference warm-globals: union of warm track globals across
        # ALL the user's sessions in context_df. Used as the taste-fallback
        # source at score time when `_full_taste` returns 0 (user not in train
        # index = no train history). At inference there is no MLM leak: we're
        # predicting the next/target turn, so taste can include the current
        # session's prior items without faking the train-time LOSO recipe.
        self._inference_user_warm_globals: dict[str, set[int]] = {}
        if {"user_id", "track_id"}.issubset(context_df.columns) and self.id_map is not None:
            for r in context_df.iter_rows(named=True):
                uid = r.get("user_id"); tid = r.get("track_id")
                if uid is None or tid is None:
                    continue
                g = self.id_map.track_to_idx.get(tid)
                if g is None or g not in self._global_to_warm_local:
                    continue
                self._inference_user_warm_globals.setdefault(uid, set()).add(g)

        # Skip the user_base recommend (no override needed there) and the
        # parent's split chain — they all defer to the same SessionRecommender
        # base loop that calls _extra_score_kwargs_for_session + _score_session_sequence.
        return super().recommend(context_df, *args, **kwargs)

    def _extra_score_kwargs_for_session(
        self, sess_id: str, user_id: str,
    ) -> dict[str, Any]:
        # User idx (from user_base).
        uid = self._session_user_map.get(sess_id, user_id)
        user_idx = self._user_to_idx.get(uid, 0)
        # Target query idx (Design B-style).
        target_turn = self._session_target_turn.get(sess_id)
        target_query_idx = (
            self._query_lookup.get((sess_id, int(target_turn)), 0)
            if target_turn is not None else 0
        )
        # Prior query idxs (Design A).
        prior_qidxs: list[int] = []
        for tn, _tid in self._cur_prior_turn_list.get(sess_id, []):
            prior_qidxs.append(self._query_lookup.get((sess_id, tn), 0))
        return {
            "user_idx":          int(user_idx),
            "user_id":           str(uid),
            "target_query_idx":  int(target_query_idx),
            "prior_query_idxs":  prior_qidxs,
        }

    def _score_session_sequence(
        self,
        prior: list[str],
        warm_embs: torch.Tensor,
        cold_embs: torch.Tensor | None,
        all_embs: torch.Tensor,
        user_idx: int = 0,
        user_id: str | None = None,
        target_query_idx: int = 0,
        prior_query_idxs: list[int] | None = None,
    ) -> np.ndarray:
        assert self.model is not None and self.id_map is not None
        n_warm = warm_embs.shape[0]

        tokens: list[int]      = []
        kept_qidxs: list[int]  = []
        prior_qidxs = prior_query_idxs or [0] * len(prior)
        for j, t in enumerate(prior):
            if t not in self.id_map.track_to_idx:
                continue
            g = self.id_map.track_to_idx[t]
            qj = prior_qidxs[j] if j < len(prior_qidxs) else 0
            if g in self._global_to_warm_local:
                tokens.append(self._global_to_warm_local[g] + ITEM_OFFSET)
                kept_qidxs.append(qj)
            elif g in self._global_to_cold_local:
                tokens.append(self._global_to_cold_local[g] + n_warm + ITEM_OFFSET)
                kept_qidxs.append(qj)

        tokens.append(MASK_TOKEN)
        kept_qidxs.append(int(target_query_idx))

        tokens     = tokens[-self.max_seq_len:]
        kept_qidxs = kept_qidxs[-self.max_seq_len:]
        pad_len = self.max_seq_len - len(tokens)
        tokens     = [PAD_TOKEN] * pad_len + tokens
        q_idx_seq  = [0]         * pad_len + kept_qidxs

        x = torch.tensor([tokens],     dtype=torch.long, device=self.device_)
        q = torch.tensor([q_idx_seq],  dtype=torch.long, device=self.device_)
        u = torch.tensor([int(user_idx)], dtype=torch.long, device=self.device_)

        need_user_loso = self.use_bias_term or (
            self.use_query_fusion and self.query_fusion_source == "taste"
        )
        need_session_taste = (
            self.use_query_fusion and self.query_fusion_source == "session_taste"
        )

        with torch.no_grad():
            hidden = self.model.encode_hidden(x, u, items_table=all_embs, query_idx_seq=q)
            h = hidden[0, -1, :]
            h_n = F.normalize(h, dim=-1)

            taste = None
            if need_user_loso:
                # User-level taste from training (full_taste). For splitK
                # user-disjoint test users this returns 0 → optional
                # fallback substitutes the mean of the user's inference-side
                # warm tracks (current session prior + other sessions of
                # the same user, no LOSO). Distributionally different from
                # train-time LOSO; empirically can help OR hurt depending
                # on `alpha_query` calibration — opt in via the
                # `use_taste_inference_fallback` flag.
                taste = self.model._full_taste(u).squeeze(0)
                if (
                    self.use_taste_inference_fallback
                    and float(taste.abs().sum()) == 0.0
                    and user_id is not None
                ):
                    all_user_warm = self._inference_user_warm_globals.get(user_id, set())
                    if all_user_warm:
                        warm_local_idxs = [
                            self._global_to_warm_local[g] for g in all_user_warm
                        ]
                        idxs_t = torch.tensor(
                            warm_local_idxs, dtype=torch.long,
                            device=warm_embs.device,
                        )
                        embs_n = F.normalize(warm_embs[idxs_t], dim=-1)
                        taste = F.normalize(embs_n.mean(dim=0), dim=-1)
            elif need_session_taste:
                # Mirror the train recipe at inference: mean(L2(item_encoder))
                # over warm items in the current session's prior (no leak —
                # the target turn is not in prior). Same formula as
                # `_session_prior_taste` at train, where context = positions
                # the model SEES (labels == -100). Here, the entire prior is
                # context, no masking.
                warm_local_idxs: list[int] = []
                for t in prior:
                    if t not in self.id_map.track_to_idx:
                        continue
                    g = self.id_map.track_to_idx[t]
                    if g in self._global_to_warm_local:
                        warm_local_idxs.append(self._global_to_warm_local[g])
                if warm_local_idxs:
                    idxs_t = torch.tensor(
                        warm_local_idxs, dtype=torch.long,
                        device=warm_embs.device,
                    )
                    embs_n = F.normalize(warm_embs[idxs_t], dim=-1)
                    taste = F.normalize(embs_n.mean(dim=0), dim=-1)
                else:
                    taste = torch.zeros(
                        warm_embs.size(1),
                        device=warm_embs.device,
                        dtype=warm_embs.dtype,
                    )

            if self.use_query_fusion:
                if self.query_fusion_source in ("taste", "session_taste"):
                    uv = taste
                else:  # "cf"
                    cf = self.model.cf_table[int(user_idx)]
                    uv = F.normalize(self.model.cf_query_proj(cf), dim=-1)
                h_n = F.normalize(h_n + self.model.alpha_query * uv, dim=-1)

            tau = self.model.tau
            warm_n = F.normalize(warm_embs, dim=-1)
            warm_scores = ((h_n @ warm_n.T) / tau).cpu().numpy()
            if cold_embs is not None:
                cold_n = F.normalize(cold_embs, dim=-1)
                cold_scores = ((h_n @ cold_n.T) / tau).cpu().numpy()
            else:
                cold_scores = None

            if self.use_bias_term:
                ab_warm = (self.model.alpha_bias * (taste @ warm_n.T) / tau).cpu().numpy()
                warm_scores = warm_scores + ab_warm
                if cold_scores is not None:
                    ab_cold = (self.model.alpha_bias * (taste @ cold_n.T) / tau).cpu().numpy()
                    cold_scores = cold_scores + ab_cold

        scores = np.full(self.id_map.n_tracks, -np.inf, dtype=np.float32)
        scores[self._warm_global_indices] = warm_scores
        if cold_scores is not None:
            scores[self._cold_global_indices] = cold_scores
        return scores

    # ------------------------------------------------------------------
    # save / load: query_table is non-persistent; rebuild on load.
    # ------------------------------------------------------------------

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st["query_emb_dir"]      = self.query_emb_dir
        st["query_cache_splits"] = self.query_cache_splits
        return st

    def _set_model_state(self, state: dict) -> None:
        self.query_emb_dir      = state.get("query_emb_dir", self.query_emb_dir)
        self.query_cache_splits = state.get("query_cache_splits", self.query_cache_splits)
        self._load_query_cache()
        super()._set_model_state(state)
        if self.model is not None:
            self.model.query_table = torch.from_numpy(self._query_emb_table).to(self.device_)
