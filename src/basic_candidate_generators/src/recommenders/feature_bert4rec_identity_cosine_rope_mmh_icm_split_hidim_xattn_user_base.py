"""Base for user-aware variants on top of split_hidim_xattn.

Adds four independently-toggleable user-information injection mechanisms
to the current ndcg@20 winner (`split_hidim_xattn`). Five concrete
variants are defined in companion files:

  - `*_userbias.py`     (A)   : history-mean taste term added to score
  - `*_userfilm.py`     (B)   : CF user embedding → FiLM on output_norm
  - `*_userhistmod.py`  (C)   : history-mean fused into query side (5th mod)
  - `*_usermod.py`      (C')  : CF user embedding fused into query side
  - `*_userbias_userfilm.py`  : combo A + B

Each variant just flips the right flags in `__init__`; the model and
training loop live here.

Architectural notes:

- **History-based (A, C)**: taste_table[u] = mean of item_encoder(features
  of u's warm training tracks), L2-normalised. Refreshed at the START of
  each epoch (during training the previous-epoch snapshot is used so the
  taste is part of the autograd graph for the *previous* epoch's
  gradients but not the current ones — cheap, no per-batch recompute).
  Cold users get a zero vector.

- **CF-based (B, C')**: cf_table[u] = pre-loaded `cf-bpr` 128d vector
  from `TalkPlayData-Challenge-User-Embeddings/*.parquet`. Cold users
  get zero (matches the official cold-user behaviour). Used through a
  trainable projection.

- All tables stored as non-learnable **buffers** sized to `n_users`.
  Index 0 is reserved for "unknown user" (zero vector). Devset / blind-A
  sessions whose user is not in training also map to index 0 (fully cold).

- The 4 mechanisms are orthogonal: any combination can be enabled.
"""

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

from .feature_bert4rec import ITEM_OFFSET, MASK_TOKEN, PAD_TOKEN
from .feature_bert4rec_identity_cosine_rope_mmh import (
    _FeatureBert4RecIdentityCosineRoPEMMHModel,
)
from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim_xattn import (
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnRecommender,
)
from .feature_bert4rec_identity_cosine_rope_mmh_icm_xattn import _ModalityCrossAttn
from .feature_bert4rec_user_helpers import (
    USER_CF_DIM,
    _UserAwareDataset,
    build_session_user_map,
    compute_user_taste_embeddings,
    load_user_cf_embeddings,
)


_DEFAULT_USER_EMB_PATHS = (
    "data/talkpl-ai/TalkPlayData-Challenge-User-Embeddings/data/train-00000-of-00001.parquet",
    "data/talkpl-ai/TalkPlayData-Challenge-User-Embeddings/data/test_warm-00000-of-00001.parquet",
    "data/talkpl-ai/TalkPlayData-Challenge-User-Embeddings/data/test_cold-00000-of-00001.parquet",
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class _UserAwareSplitHiDimXAttnModel(_FeatureBert4RecIdentityCosineRoPEMMHModel):
    """split_hidim_xattn architecture (xattn fusion over modalities) plus
    user-info injection mechanisms.

    Mechanisms (independently flagged):
      - `use_bias_term`     (A): score += alpha_bias * (taste_vec @ item_emb)
      - `use_film`          (B): out  = output_norm(out) * (1+γ(u)) + β(u)
      - `use_query_fusion`  (C/C'): h = L2(h) + alpha_query * L2(user_query_vec)
        with user_query_vec from `taste_table` (taste) or `cf_table` projection (cf).
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
        use_bias_term: bool = False,
        use_film: bool = False,
        use_query_fusion: bool = False,
        query_fusion_source: str = "taste",   # "taste" | "cf"
        alpha_init: float = 0.1,
    ) -> None:
        super().__init__(
            warm_feature_matrix, hidden_size, max_seq_len,
            n_layers, n_heads, dropout, init_tau=init_tau,
            modality_dims=modality_dims,
        )
        # XAttn fusion (inherited from MMHICMXAttnRecommender via the
        # SplitHiDimXAttn parent through MRO of the recommender — here
        # we replicate the same swap explicitly).
        self.item_encoder = _ModalityCrossAttn(modality_dims, self.hidden_size)

        self.use_bias_term       = bool(use_bias_term)
        self.use_film            = bool(use_film)
        self.use_query_fusion    = bool(use_query_fusion)
        self.query_fusion_source = str(query_fusion_source)

        need_taste = self.use_bias_term or (self.use_query_fusion and self.query_fusion_source == "taste")
        need_cf    = self.use_film       or (self.use_query_fusion and self.query_fusion_source == "cf")

        # Static lookup buffers (filled by the recommender at each epoch start).
        #
        # For taste-based variants we store SUM and COUNT of L2-normalised item
        # embeddings over each user's training tracks, so the forward pass can
        # subtract the current session's contribution and compute a leak-free
        # leave-one-session-out (LOSO) taste:
        #   loso_taste(u, sess) = L2_normalize(
        #       (user_sum[u] - sum_{i in sess} item_emb_n[i])
        #       / max(user_count[u] - |sess|, 1)
        #   )
        # At inference (no current training session) we use the full mean.
        if need_taste:
            self.register_buffer(
                "user_sum",
                torch.zeros(max(1, n_users), hidden_size, dtype=torch.float32),
            )
            self.register_buffer(
                "user_count",
                torch.zeros(max(1, n_users), dtype=torch.float32),
            )
        if need_cf:
            self.register_buffer(
                "cf_table",
                torch.zeros(max(1, n_users), USER_CF_DIM, dtype=torch.float32),
            )

        # FiLM heads.
        if self.use_film:
            self.film_user_proj = nn.Linear(USER_CF_DIM, hidden_size, bias=True)
            self.film_gamma     = nn.Linear(hidden_size, hidden_size, bias=True)
            self.film_beta      = nn.Linear(hidden_size, hidden_size, bias=True)
            with torch.no_grad():
                self.film_gamma.weight.zero_(); self.film_gamma.bias.zero_()
                self.film_beta.weight.zero_();  self.film_beta.bias.zero_()

        # Query-fusion CF projection (only used if query mode = cf).
        if self.use_query_fusion and self.query_fusion_source == "cf":
            self.cf_query_proj = nn.Linear(USER_CF_DIM, hidden_size, bias=True)

        # Mixing scalars.
        if self.use_bias_term:
            self.alpha_bias = nn.Parameter(torch.tensor(float(alpha_init)))
        if self.use_query_fusion:
            self.alpha_query = nn.Parameter(torch.tensor(float(alpha_init)))

    # ------------------------------------------------------------------
    # Forward (training): logits = (B, L, n_warm)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # FiLM and query-fusion helpers (mechanism-side, taste-agnostic)
    # ------------------------------------------------------------------

    def _apply_film(self, out: torch.Tensor, user_idx: torch.Tensor) -> torch.Tensor:
        cf = self.cf_table[user_idx]
        user_proj = self.film_user_proj(cf)
        gamma = self.film_gamma(user_proj).unsqueeze(1)
        beta  = self.film_beta(user_proj).unsqueeze(1)
        return out * (1.0 + gamma) + beta

    def _query_fuse(self, out_n: torch.Tensor, user_idx: torch.Tensor,
                     taste_vec: torch.Tensor | None) -> torch.Tensor:
        if self.query_fusion_source == "taste":
            uv = taste_vec
        else:
            cf = self.cf_table[user_idx]
            uv = F.normalize(self.cf_query_proj(cf), dim=-1)
        fused = out_n + self.alpha_query * uv.unsqueeze(1)
        return F.normalize(fused, dim=-1)

    # ------------------------------------------------------------------
    # Leave-one-session-out taste (fix for the training leak in
    # history-based variants).
    #
    # The leak: user_sum[u] includes every track in u's training history.
    # During training, the masked targets are drawn from that same set, so
    # an unmodified mean taste contains the answer. The model collapses to
    # "answer ~ taste" and fails catastrophically at inference where the
    # target items are NOT in the training history.
    #
    # Fix: per-sample, subtract the current session's items from user_sum.
    # Both `forward` (train) and `_score_session_sequence` (inference) need
    # taste, but only training has a current-session leak; inference uses
    # the unmodified mean (full_taste).
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_unmasked(masked_seq: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Recover original warm-local indices per position from (masked, labels).

        labels[b,i] != -100 means position i was masked → warm-local = labels.
        Else position kept its original token → warm-local = masked - ITEM_OFFSET.
        PAD positions (masked == 0) are returned as -1 so callers can filter.
        """
        # labels already in warm-local space at masked positions.
        kept = masked_seq - ITEM_OFFSET                              # valid where token >= ITEM_OFFSET
        unmasked = torch.where(labels != -100, labels, kept)
        pad = masked_seq == PAD_TOKEN
        return torch.where(pad, torch.full_like(unmasked, -1), unmasked)

    def _loso_taste(
        self,
        user_idx: torch.Tensor,
        unmasked: torch.Tensor,
        item_emb_n: torch.Tensor,
    ) -> torch.Tensor:
        """Leave-one-session-out taste, L2-normalised, per sample.

        user_idx     : (B,) long
        unmasked     : (B, L) warm-local indices or -1 for PAD
        item_emb_n   : (n_warm, H), L2-normalised

        Returns (B, H), L2-normalised. Cold users (user_count == 0)
        get a zero vector.
        """
        B, L = unmasked.shape
        H = item_emb_n.size(1)
        device = item_emb_n.device

        valid = unmasked >= 0                                        # (B, L)
        b_row = torch.arange(B, device=device).unsqueeze(1).expand(-1, L)
        b_flat = b_row[valid]                                        # (n_valid,)
        w_flat = unmasked[valid]                                     # (n_valid,)

        session_sum = torch.zeros(B, H, device=device, dtype=item_emb_n.dtype)
        if b_flat.numel() > 0:
            session_sum.index_add_(0, b_flat, item_emb_n[w_flat])

        session_count = torch.zeros(B, device=device, dtype=item_emb_n.dtype)
        if b_flat.numel() > 0:
            ones = torch.ones(b_flat.numel(), device=device, dtype=item_emb_n.dtype)
            session_count.index_add_(0, b_flat, ones)

        user_sum_b   = self.user_sum[user_idx]                       # (B, H)
        user_count_b = self.user_count[user_idx]                     # (B,)
        loso_sum   = user_sum_b - session_sum
        loso_count = (user_count_b - session_count).clamp_min(1.0).unsqueeze(-1)
        return F.normalize(loso_sum / loso_count, dim=-1)

    def _full_taste(self, user_idx: torch.Tensor) -> torch.Tensor:
        """Mean taste (no exclusion). Used at inference."""
        cnt = self.user_count[user_idx].clamp_min(1.0).unsqueeze(-1)
        return F.normalize(self.user_sum[user_idx] / cnt, dim=-1)

    # ------------------------------------------------------------------
    # Forward (training): labels are REQUIRED so we can run LOSO.
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        user_idx: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        warm_embs = self.item_encoder(self.feature_matrix)
        emb, pad_mask = self._build_seq_emb(x, warm_embs)
        out = self.encoder(emb, src_key_padding_mask=pad_mask)
        out = self.output_norm(out)
        if self.use_film:
            out = self._apply_film(out, user_idx)

        out_n  = F.normalize(out, dim=-1)
        warm_n = F.normalize(warm_embs, dim=-1)

        # Build LOSO taste only if either mechanism needs it.
        taste_vec = None
        need_taste = self.use_bias_term or (self.use_query_fusion and self.query_fusion_source == "taste")
        if need_taste:
            assert labels is not None, "labels required for LOSO taste during training"
            unmasked = self._decode_unmasked(x, labels)
            item_emb_n = F.normalize(warm_embs, dim=-1)              # (n_warm, H)
            taste_vec = self._loso_taste(user_idx, unmasked, item_emb_n)   # (B, H)

        if self.use_query_fusion:
            out_n = self._query_fuse(out_n, user_idx, taste_vec)

        score = (out_n @ warm_n.T) / self.tau
        if self.use_bias_term:
            bias = self.alpha_bias * (taste_vec @ warm_n.T) / self.tau   # (B, n_warm)
            score = score + bias.unsqueeze(1)
        return score

    def encode_hidden(
        self,
        x: torch.Tensor,
        user_idx: torch.Tensor,
        items_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Inference path — returns hidden post-FiLM; the recommender's
        `_score_session_sequence` applies bias and query-fusion using
        `_full_taste` (no LOSO at inference)."""
        if items_table is None:
            items_table = self.item_encoder(self.feature_matrix)
        emb, pad_mask = self._build_seq_emb(x, items_table)
        out = self.encoder(emb, src_key_padding_mask=pad_mask)
        out = self.output_norm(out)
        if self.use_film:
            out = self._apply_film(out, user_idx)
        return out


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

class _UserAwareSplitHiDimXAttnRecommenderBase(
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnRecommender
):
    """Base recommender that adds the user-aware training/inference path.

    Subclasses flip the four flags via `__init__`; the rest of the family
    inheritance (split, hidim, xattn) is preserved.
    """

    RECOMMENDER_NAME = "FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnUser"

    def __init__(
        self,
        *args: Any,
        use_bias_term: bool = False,
        use_film: bool = False,
        use_query_fusion: bool = False,
        query_fusion_source: str = "taste",
        alpha_init: float = 0.1,
        user_emb_paths: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.use_bias_term       = bool(use_bias_term)
        self.use_film            = bool(use_film)
        self.use_query_fusion    = bool(use_query_fusion)
        self.query_fusion_source = str(query_fusion_source)
        self.alpha_init          = float(alpha_init)
        self.user_emb_paths      = list(user_emb_paths) if user_emb_paths else list(_DEFAULT_USER_EMB_PATHS)

        # Populated in _fit_model:
        self._session_user_map: dict[str, str] = {}
        self._user_to_idx: dict[str, int] = {}    # user_id -> row in tables (0 reserved for cold)
        self._n_users: int = 1
        self._user_played_global: dict[str, list[int]] = {}
        # Populated in recommend() (inference taste fallback for cold users).
        self._inference_user_warm_globals: dict[str, set[int]] = {}

    # ------------------------------------------------------------------
    # Public hook for the launcher: tell the recommender about devset/blind
    # session -> user map so inference can pick the right user_idx.
    # ------------------------------------------------------------------

    def register_session_user_map(self, mapping: dict[str, str]) -> None:
        self._session_user_map.update(mapping)

    # ------------------------------------------------------------------
    # Hook for the inference pipeline: resolve user_idx per session so the
    # parent's .recommend() forwards it to _score_session_sequence.
    #
    # Note: under splitK the cg_train/cg_val users are disjoint by
    # construction (user-coherent split). Validation users were never seen
    # at fit time, so they are NOT in self._user_to_idx and they fall back
    # to idx 0 = <unk>. That's the correct behaviour given the train/val
    # boundary — personalisation lights up at final-submission time when
    # blind-A users have been trained on.
    # ------------------------------------------------------------------
    def _extra_score_kwargs_for_session(
        self, sess_id: str, user_id: str
    ) -> dict[str, Any]:
        uid = self._session_user_map.get(sess_id, user_id)
        return {
            "user_idx": self._user_to_idx.get(uid, 0),
            "user_id":  str(uid),
        }

    # ------------------------------------------------------------------
    # recommend hook — pre-build the user→warm-globals inference map for
    # the taste fallback (cold users have no train history; we substitute
    # with the mean of warm tracks they're seen with across all their
    # inference sessions).
    # ------------------------------------------------------------------

    def recommend(self, context_df: pl.DataFrame, *args: Any, **kwargs: Any) -> pl.DataFrame:
        if "track_id" not in context_df.columns:
            from .interactions import explode_music_turns
            context_df = explode_music_turns(context_df)
        self._inference_user_warm_globals = {}
        if (
            {"user_id", "track_id"}.issubset(context_df.columns)
            and self.id_map is not None
        ):
            for r in context_df.iter_rows(named=True):
                uid = r.get("user_id"); tid = r.get("track_id")
                if uid is None or tid is None:
                    continue
                g = self.id_map.track_to_idx.get(tid)
                if g is None or g not in self._global_to_warm_local:
                    continue
                self._inference_user_warm_globals.setdefault(uid, set()).add(g)
        return super().recommend(context_df, *args, **kwargs)

    # ------------------------------------------------------------------
    # fit hook — capture session->user from raw train_df (before explode)
    # ------------------------------------------------------------------

    def fit(self, train_df: pl.DataFrame, track_metadata=None, **kwargs: Any) -> None:
        self._session_user_map.update(build_session_user_map(train_df))
        super().fit(train_df, track_metadata=track_metadata, **kwargs)

    # ------------------------------------------------------------------
    # Override _make_model to use the user-aware variant
    # ------------------------------------------------------------------

    def _make_model(self, warm_feature_matrix: np.ndarray, modality_dims: list[int]):
        return _UserAwareSplitHiDimXAttnModel(
            warm_feature_matrix, self.hidden_size, self.max_seq_len,
            self.n_layers, self.n_heads, self.dropout,
            init_tau=self.init_tau,
            modality_dims=modality_dims,
            n_users=max(1, self._n_users),
            use_bias_term=self.use_bias_term,
            use_film=self.use_film,
            use_query_fusion=self.use_query_fusion,
            query_fusion_source=self.query_fusion_source,
            alpha_init=self.alpha_init,
        )

    # ------------------------------------------------------------------
    # User-vector tables: built before the model, populated through it
    # ------------------------------------------------------------------

    def _build_user_index(self) -> None:
        """Assign idx=0 to "unknown"; enumerate every user we've seen in train
        sessions PLUS every user for whom we have a CF embedding (test users
        included).

        Why this matters: CF embeddings cover train + test_warm users. Without
        the test users in `_user_to_idx`, they fall back to idx 0 = <unk> at
        inference and cf_table[0] is the zero vector → FiLM / cf-query become
        identity → the user signal we KNOW about is silently dropped.

        Taste-based variants (use_bias_term / source="taste") only fill
        user_sum/user_count for users with TRAIN history (built from
        `_user_played_global`), so test users still get a zero taste — that
        is structurally unavoidable without an inference-time history."""
        train_users: set[str] = set(self._session_user_map.values())
        cf_users: set[str] = set()
        need_cf = self.use_film or (self.use_query_fusion and self.query_fusion_source == "cf")
        if need_cf:
            # Same loader the cf_table population uses; we just need the keys
            # here, so the extra parquet I/O is one-shot and ~MBs.
            from launchers_crossvalidation._cv_utils import repo_path  # local import
            paths = [str(repo_path(p)) for p in self.user_emb_paths]
            cf_dict, _ = load_user_cf_embeddings(paths)
            cf_users = set(cf_dict.keys())
        all_users = train_users | cf_users
        ordered = ["<unk>"] + sorted(all_users)
        self._user_to_idx = {u: i for i, u in enumerate(ordered)}
        self._n_users = len(ordered)
        extra = len(cf_users - train_users)
        print(f"[{self.RECOMMENDER_NAME}] users in index: {self._n_users} "
              f"(incl. <unk> @ idx 0; train={len(train_users)}, "
              f"CF-only test={extra})")

    def _build_user_played_global(self) -> None:
        """user_id -> list of global track indices played in TRAIN (warm only)."""
        assert self._train_long is not None and self.id_map is not None
        warm_ids = set(self._warm_global_indices)
        long = self._train_long
        out: dict[str, list[int]] = {}
        for r in long.iter_rows(named=True):
            uid = r["user_id"]; tid = r["track_id"]
            g = self.id_map.track_to_idx.get(tid)
            if g is None or g not in warm_ids:
                continue
            out.setdefault(uid, []).append(g)
        self._user_played_global = out

    @torch.no_grad()
    def _refresh_user_sums(self) -> None:
        """Recompute (user_sum, user_count) on the model's buffers.

        user_sum[u]   = sum over plays of u of L2_normed(item_encoder(features[t]))
        user_count[u] = number of plays of u (warm tracks only)

        Refreshed at the START of every epoch with the current item_encoder
        state. The forward pass then derives a leak-free per-sample taste
        via leave-one-session-out subtraction.
        """
        assert self.model is not None
        if not (self.use_bias_term or (self.use_query_fusion and self.query_fusion_source == "taste")):
            return

        feat_t = self.model.feature_matrix
        item_emb_n = F.normalize(self.model.item_encoder(feat_t), dim=-1)  # (n_warm, H)

        usum = self.model.user_sum
        ucnt = self.model.user_count
        usum.zero_()
        ucnt.zero_()

        for uid, global_idxs in self._user_played_global.items():
            local = [self._global_to_warm_local[g] for g in global_idxs
                     if g in self._global_to_warm_local]
            if not local:
                continue
            idx = self._user_to_idx.get(uid)
            if idx is None or idx == 0:
                continue
            t = torch.tensor(local, dtype=torch.long, device=item_emb_n.device)
            # Sum counts duplicate plays of the same track multiple times
            # — matches how `_loso_taste` subtracts repeated plays.
            usum[idx] = item_emb_n[t].sum(dim=0)
            ucnt[idx] = float(len(local))

    def _load_cf_table(self) -> None:
        """Load `cf-bpr` user embeddings into model.cf_table."""
        assert self.model is not None
        if not (self.use_film or (self.use_query_fusion and self.query_fusion_source == "cf")):
            return
        from launchers_crossvalidation._cv_utils import repo_path  # local import
        paths = [str(repo_path(p)) for p in self.user_emb_paths]
        uid_to_cf, _ = load_user_cf_embeddings(paths)
        table = self.model.cf_table
        absorbed = 0
        with torch.no_grad():
            table.zero_()
            for uid, vec in uid_to_cf.items():
                idx = self._user_to_idx.get(uid)
                if idx is None or idx == 0:
                    continue
                table[idx] = torch.from_numpy(vec).to(table.device)
                absorbed += 1
        print(f"[{self.RECOMMENDER_NAME}] CF user embeddings loaded: {absorbed}/{self._n_users - 1}")

    # ------------------------------------------------------------------
    # Build (sequences, user_idxs) — replaces _build_train_val_sequences
    # ------------------------------------------------------------------

    def _build_user_aware_sequences(self) -> tuple[
        list[list[int]], list[int], list[list[int]], list[int]
    ]:
        """Return (train_seqs, train_user_idxs, val_seqs, val_user_idxs).

        Mirrors `_build_train_val_sequences` (random per-sequence split with
        val_ratio) but also yields, for each sequence, the integer user_idx
        of the session's user (0 if unknown)."""
        assert self.id_map is not None and self._train_long is not None
        from .feature_bert4rec import ITEM_OFFSET as _OFF
        seqs: list[list[int]] = []
        user_idxs: list[int] = []
        for sid_t, grp in (
            self._train_long
            .sort(["session_id", "turn_number"])
            .group_by("session_id", maintain_order=True)
        ):
            sid = sid_t[0] if isinstance(sid_t, tuple) else sid_t
            uid = self._session_user_map.get(sid, "<unk>")
            user_idx = self._user_to_idx.get(uid, 0)
            tokens = [
                self._global_to_warm_local[self.id_map.track_to_idx[t]] + _OFF
                for t in grp["track_id"].to_list()
                if t in self.id_map.track_to_idx
                and self.id_map.track_to_idx[t] in self._global_to_warm_local
            ]
            if len(tokens) >= 2:
                seqs.append(tokens)
                user_idxs.append(user_idx)
        # Random per-sequence split (1 sequence per session here so equivalent to session-level)
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
    # _fit_model — full override (similar to MMH's but user-aware)
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

        print(f"[{self.RECOMMENDER_NAME}] Loading per-modality features (with split ICM)...")
        full_matrix, modality_dims = self._build_modality_feature_matrix()
        self._feature_dim = full_matrix.shape[1]
        self._modality_dims = modality_dims
        warm_feature_matrix = full_matrix[self._warm_global_indices]
        self._cold_feature_matrix = full_matrix[self._cold_global_indices]

        # User infrastructure
        self._build_user_index()
        self._build_user_played_global()

        train_seqs, train_uidx, val_seqs, val_uidx = self._build_user_aware_sequences()
        print(
            f"[{self.RECOMMENDER_NAME}] warm={n_warm}, cold={n_cold}, "
            f"train_seqs={len(train_seqs)}, val_seqs={len(val_seqs)}, "
            f"device={self.device_}, init_tau={self.init_tau}, "
            f"use_bias_term={self.use_bias_term}, use_film={self.use_film}, "
            f"use_query_fusion={self.use_query_fusion}, query_source={self.query_fusion_source}"
        )

        pin = self.device_.type == "cuda"
        train_loader = DataLoader(
            _UserAwareDataset(train_seqs, train_uidx, n_warm, self.max_seq_len, self.mask_prob),
            batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=pin,
        )
        val_loader = DataLoader(
            _UserAwareDataset(val_seqs, val_uidx, n_warm, self.max_seq_len, self.mask_prob),
            batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=pin,
        )

        self.model = self._make_model(warm_feature_matrix, modality_dims)
        self._pca_init_encoder_per_modality(warm_feature_matrix, modality_dims)
        self.model.to(self.device_)

        # Pre-populate cf_table (static) and the initial user-sum/count
        # tables (PCA-init encoder state).
        self._load_cf_table()
        self._refresh_user_sums()

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        total_steps = self.epochs * len(train_loader)
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

        for epoch in epoch_bar:
            # Refresh user_sum/count once per epoch (using current item_encoder weights).
            self._refresh_user_sums()

            self.model.train()
            total_loss = 0.0
            for masked_seq, labels, user_idx in tqdm(
                train_loader, desc=f"  ep {epoch:3d}", leave=False,
                unit="batch", dynamic_ncols=True, file=sys.stdout,
            ):
                masked_seq = masked_seq.to(self.device_)
                labels     = labels.to(self.device_)
                user_idx   = user_idx.to(self.device_)
                logits = self.model(masked_seq, user_idx, labels=labels)
                loss = F.cross_entropy(
                    logits.view(-1, n_warm), labels.view(-1), ignore_index=-100
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for masked_seq, labels, user_idx in val_loader:
                    masked_seq = masked_seq.to(self.device_)
                    labels     = labels.to(self.device_)
                    user_idx   = user_idx.to(self.device_)
                    logits = self.model(masked_seq, user_idx, labels=labels)
                    val_loss += F.cross_entropy(
                        logits.view(-1, n_warm), labels.view(-1), ignore_index=-100
                    ).item()
            val_avg   = val_loss / len(val_loader)
            train_avg = total_loss / len(train_loader)

            improved = val_avg < best_val
            if improved:
                best_val = val_avg
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience_left = self.early_stop_patience
            else:
                patience_left -= 1

            ab = float(self.model.alpha_bias.item())  if hasattr(self.model, "alpha_bias")  else -1
            aq = float(self.model.alpha_query.item()) if hasattr(self.model, "alpha_query") else -1
            epoch_bar.set_postfix(
                loss=f"{train_avg:.4f}", val=f"{val_avg:.4f}",
                best=f"{best_val:.4f}@{best_epoch}",
                tau=f"{self.model.tau.item():.4f}",
                ab=f"{ab:.3f}" if ab >= 0 else "off",
                aq=f"{aq:.3f}" if aq >= 0 else "off",
                patience=patience_left,
            )
            if patience_left <= 0:
                print(f"\n[{self.RECOMMENDER_NAME}] early stopping at epoch {epoch}; "
                      f"best val={best_val:.4f} at epoch {best_epoch}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"[{self.RECOMMENDER_NAME}] restored best (val={best_val:.4f}, epoch={best_epoch})")

        # Final refresh of user_sum/count AFTER restoring the best checkpoint
        # so inference uses sums consistent with the chosen weights.
        self._refresh_user_sums()

    # ------------------------------------------------------------------
    # Inference — extended _score_session_sequence signature
    # ------------------------------------------------------------------

    def _score_session_sequence(
        self,
        prior: list[str],
        warm_embs: torch.Tensor,
        cold_embs: torch.Tensor | None,
        all_embs: torch.Tensor,
        user_idx: int = 0,
        user_id: str | None = None,
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
        uidx = torch.tensor([int(user_idx)], dtype=torch.long, device=self.device_)

        # Pre-compute full (non-LOSO) taste for this user when needed.
        # At inference the devset session's target items are NOT in the
        # training history, so no leak — we use the unmodified user mean.
        need_taste = self.use_bias_term or (
            self.use_query_fusion and self.query_fusion_source == "taste"
        )

        with torch.no_grad():
            hidden = self.model.encode_hidden(x, uidx, items_table=all_embs)
            h = hidden[0, -1, :]                                # (H,)

            taste = None
            if need_taste:
                taste = self.model._full_taste(uidx).squeeze(0)  # (H,), L2-normed
                # NB: under splitK (user-disjoint train↔test) every test user
                # falls to user_count[uidx]=0 → taste=0 for the entire eval.
                # A per-inference-session fallback was tried (mean of the
                # user's inference items) but the distribution mismatch with
                # train-time LOSO taste made ndcg worse, not better. For the
                # splitK setup use a session-prior taste variant (computed
                # the same way at train and test) instead — see the 4-way
                # base for that mechanism.

            h_n = F.normalize(h, dim=-1)
            if self.use_query_fusion:
                if self.query_fusion_source == "taste":
                    uv = taste
                else:
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
