"""HSTU (Hierarchical Sequential Transduction Unit) — Meta 2024.

Paper: "Actions Speak Louder than Words: Trillion-Parameter Sequential
Transducers for Generative Recommendations" (Zhai et al., arXiv:2402.17152).

Architectural overview
----------------------
Replaces the Transformer attention block with a custom HSTU layer:

    Y_in = norm(X)
    U, V, Q, K = SiLU(Linear_UVQK(Y_in))   # 4 projections, all silu-activated
    A = phi(QK^T) * RAB                    # phi = SiLU; RAB = relative position bias
    Y_out = norm(A V) * U                  # element-wise GATING with U (key trick)
    X = X + Linear_out(Y_out)              # residual

The gating with U gives explicit forget/keep behaviour that vanilla attention
lacks. The SiLU-normalized scores replace softmax, removing the bottleneck and
making the unit efficient on long sequences.

Training paradigm
-----------------
Causal next-item prediction (autoregressive). For sequence [t1, t2, t3, t4],
position i predicts t_{i+1}. No MASK token, no random masking — fully causal.

Feature-based items
-------------------
Item embeddings come from a shared MLP encoder over multi-modal features
(qwen3 + clap + cf-bpr, 1664d concat). This handles cold tracks naturally
and is consistent with our other variants. Inference also exposes cold
items via the cold-in-prior + warm/cold split-scoring pattern.
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .feature_bert4rec import (
    ITEM_OFFSET,
    PAD_TOKEN,
    FeatureBert4RecRecommender,
    _build_feature_matrix,
)
from .session_base import SessionRecommender


# ---------------------------------------------------------------------------
# HSTU layer
# ---------------------------------------------------------------------------

class _HSTULayer(nn.Module):
    """Single HSTU transduction unit (paper Eq. 1-3, Zhai et al. 2024).

    Implements the three sub-layers of a single HSTU block:
      Eq. 1   U, V, Q, K = SiLU(f1(LN(X)))       — pre-norm + UVQK projection
      Eq. 2   A = SiLU(QK^T + RAB)               — pointwise attention (no softmax)
      Eq. 3   Y = f2(LN(A V) ⊙ U)                — norm-then-gate, then output proj

    Crucial vs my MVP:
      - Norm comes BEFORE the U-gate, not after (paper Eq 3)
      - Scores are SiLU(QK^T + RAB), NOT SiLU(QK^T / L)
      - Relative Attention Bias (per-head, learnable, indexed by j-i)
    """

    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert hidden_size % n_heads == 0, f"hidden_size {hidden_size} not divisible by n_heads {n_heads}"
        self.hidden_size = hidden_size
        self.n_heads     = n_heads
        self.head_dim    = hidden_size // n_heads
        self.max_seq_len = max_seq_len

        self.uvqk_proj = nn.Linear(hidden_size, 4 * hidden_size)
        self.out_proj  = nn.Linear(hidden_size, hidden_size)
        self.norm_in   = nn.LayerNorm(hidden_size)   # pre-norm (Eq 1 input)
        self.norm_attn = nn.LayerNorm(hidden_size)   # Norm inside Eq 3
        self.dropout   = nn.Dropout(dropout)

        # Relative Attention Bias: per-head learnable scalar for each relative
        # offset in [-max_seq_len+1, max_seq_len-1]. Shape (2L-1, n_heads).
        # Indexed at forward time by (j - i + max_seq_len - 1).
        self.rel_pos_bias = nn.Parameter(torch.zeros(2 * max_seq_len - 1, n_heads))

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D), causal_mask: (L, L) bool where True = blocked position.
        """
        B, L, D = x.shape
        residual = x
        x = self.norm_in(x)                              # pre-norm

        # Eq 1: U, V, Q, K = SiLU(f1(X))
        uvqk = self.uvqk_proj(x)                         # (B, L, 4D)
        u, v, q, k = uvqk.chunk(4, dim=-1)               # each (B, L, D)
        u = F.silu(u); v = F.silu(v)
        q = F.silu(q); k = F.silu(k)

        # Multi-head split for Q, K, V (U stays full-D for gating after concat)
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, L, d)
        q_h, k_h, v_h = map(split_heads, (q, k, v))

        # Build per-head RAB for the current sequence length L
        positions = torch.arange(L, device=x.device)
        rel_pos   = positions[None, :] - positions[:, None] + (self.max_seq_len - 1)  # (L, L)
        rab       = self.rel_pos_bias[rel_pos]                              # (L, L, H)
        rab       = rab.permute(2, 0, 1).unsqueeze(0)                       # (1, H, L, L)

        # Eq 2: A = SiLU(QK^T + RAB) — no softmax, no /L scaling
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))                   # (B, H, L, L)
        scores = scores + rab
        scores = F.silu(scores)

        # Causal mask: zero out blocked positions (post-SiLU, 0 = neutral)
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask[None, None, :, :], 0.0)

        attn_out = torch.matmul(scores, v_h)                                # (B, H, L, d)
        # Concat heads → (B, L, D)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)

        # Eq 3: Y = f2(LN(A V) ⊙ U) — norm FIRST, then gate, then linear
        attn_out = self.norm_attn(attn_out) * u                             # norm-then-gate
        return residual + self.dropout(self.out_proj(attn_out))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class _HSTUModel(nn.Module):
    """Causal autoregressive recommender with HSTU layers + feature-based items.

    Token vocabulary (same convention as feature_bert4rec):
      0       = PAD
      1..n+1  = items (warm_local + ITEM_OFFSET=1 for HSTU since we have no MASK)

    Note ITEM_OFFSET overridden locally to 1 (PAD=0 only, no MASK token in causal model).
    """

    def __init__(
        self,
        warm_feature_matrix: np.ndarray,
        hidden_size: int,
        max_seq_len: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        n_warm, feature_dim = warm_feature_matrix.shape
        self.hidden_size = hidden_size
        self.n_warm      = n_warm
        self.max_seq_len = max_seq_len

        self.register_buffer("feature_matrix", torch.from_numpy(warm_feature_matrix))

        # Shared item encoder (multimodal features → hidden)
        self.item_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )

        # Only PAD is special (no MASK in causal autoregressive)
        self.pad_emb = nn.Embedding(1, hidden_size)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)

        self.emb_norm    = nn.LayerNorm(hidden_size)
        self.emb_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            _HSTULayer(hidden_size, n_heads, max_seq_len, dropout) for _ in range(n_layers)
        ])
        self.output_norm = nn.LayerNorm(hidden_size)

        # Pre-compute causal mask (upper-triangular = blocked positions)
        causal = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal)

    def _build_seq_emb(self, x: torch.Tensor, items_table: torch.Tensor) -> torch.Tensor:
        """Token IDs → sequence embedding.

        Token 0 → pad_emb, tokens 1..n → items_table[token-1].
        items_table = warm_embs (training) OR concat(warm, cold) (inference).
        """
        # Unified table: row 0 = pad, rows 1..n = items
        full_table = torch.cat([self.pad_emb.weight, items_table], dim=0)
        emb = full_table[x]  # (B, L, D)
        positions = torch.arange(emb.shape[1], device=x.device).unsqueeze(0)
        emb = emb + self.pos_emb(positions)
        return self.emb_dropout(self.emb_norm(emb))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Training forward — causal logits over warm items at every position.

        Returns (B, L, n_warm). Position i predicts item at position i+1
        (so labels are typically shift by 1).
        """
        warm_embs = self.item_encoder(self.feature_matrix)  # (n_warm, D)
        emb = self._build_seq_emb(x, warm_embs)

        for layer in self.layers:
            emb = layer(emb, self.causal_mask)

        out = self.output_norm(emb)
        return out @ warm_embs.T  # (B, L, n_warm)

    def encode_hidden(self, x: torch.Tensor, items_table: torch.Tensor | None = None) -> torch.Tensor:
        """Inference: returns hidden states (B, L, D). Caller does scoring."""
        if items_table is None:
            items_table = self.item_encoder(self.feature_matrix)
        emb = self._build_seq_emb(x, items_table)
        for layer in self.layers:
            emb = layer(emb, self.causal_mask)
        return self.output_norm(emb)


# ---------------------------------------------------------------------------
# Dataset: causal next-item, one (prefix, next) per prefix per sequence
# ---------------------------------------------------------------------------

# Item tokens use offset 1 (PAD=0 only). No MASK token in causal models.
_HSTU_ITEM_OFFSET = 1


class _HSTUDataset(Dataset):
    """For each sequence [t1, ..., tN], emits one training row with:
        input = [t1, ..., t_{N-1}]   (padded to max_seq_len)
        labels = [t2-1, ..., tN-1, -100]  (shifted by 1; warm-local target at each position)

    Each position i in the input predicts the item at position i+1. Positions
    beyond the sequence end have label=-100 (no loss).
    """

    def __init__(self, sequences: list[list[int]], max_seq_len: int) -> None:
        # Sequences already use warm-local indices + _HSTU_ITEM_OFFSET (we'll re-offset)
        # but to keep consistency with the rest of the codebase, sequences come in
        # with warm_local + ITEM_OFFSET (=2). We strip that and re-add 1 (no MASK).
        self.sequences = []
        for seq in sequences:
            # Re-index: token in [2, n+1] → [1, n]
            self.sequences.append([t - ITEM_OFFSET + _HSTU_ITEM_OFFSET for t in seq])
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx][-self.max_seq_len:]
        # Input: seq[:-1], Target: seq[1:]
        # If seq has length 1, no valid (input, target) pair — pad with single token & no label
        if len(seq) < 2:
            input_seq = seq + [PAD_TOKEN] * (self.max_seq_len - 1)
            labels    = [-100] * self.max_seq_len
        else:
            inp = seq[:-1]
            tgt = seq[1:]
            # warm-local labels: token - 1 (since offset is 1)
            tgt_labels = [t - _HSTU_ITEM_OFFSET for t in tgt]

            # Right-pad to max_seq_len
            pad_in  = self.max_seq_len - len(inp)
            pad_lbl = self.max_seq_len - len(tgt_labels)
            input_seq = inp        + [PAD_TOKEN] * pad_in
            labels    = tgt_labels + [-100]      * pad_lbl

        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(labels,    dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

class HSTURecommender(FeatureBert4RecRecommender):
    """HSTU-based sequential recommender.

    Reuses feature_bert4rec's fit() scaffolding (id_map, warm/cold split,
    feature loading, save/load) but the model architecture is HSTU and training
    is causal next-item instead of BERT-style random masking.
    """

    RECOMMENDER_NAME = "HSTU"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # _build_sequences: same as parent BUT we don't strip MASK since we
    # don't use it. Tokens stay in [ITEM_OFFSET, n+ITEM_OFFSET-1] convention;
    # the dataset re-offsets to [1, n].
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

        sequences = self._build_sequences()
        n_warm = len(self._warm_global_indices)
        n_cold = len(self._cold_global_indices)
        print(f"[{self.RECOMMENDER_NAME}] warm={n_warm}, cold={n_cold}, "
              f"sequences={len(sequences)}, device={self.device_}")

        print(f"[{self.RECOMMENDER_NAME}] Loading feature embeddings: {self.feature_modalities}")
        full_matrix = _build_feature_matrix(
            self.feature_emb_paths, self.feature_modalities, self.id_map
        )
        self._feature_dim = full_matrix.shape[1]
        warm_feature_matrix       = full_matrix[self._warm_global_indices]
        self._cold_feature_matrix = full_matrix[self._cold_global_indices]
        print(f"  warm_features={warm_feature_matrix.shape}, cold_features={self._cold_feature_matrix.shape}")

        random.shuffle(sequences)
        n_val = max(1, int(len(sequences) * self.val_ratio))
        val_sequences   = sequences[:n_val]
        train_sequences = sequences[n_val:]
        print(f"  train_seqs={len(train_sequences)}, val_seqs={len(val_sequences)}")

        pin = self.device_.type == "cuda"
        train_loader = DataLoader(
            _HSTUDataset(train_sequences, self.max_seq_len),
            batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=pin,
        )
        val_loader = DataLoader(
            _HSTUDataset(val_sequences, self.max_seq_len),
            batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=pin,
        )

        self.model = _HSTUModel(
            warm_feature_matrix, self.hidden_size, self.max_seq_len,
            self.n_layers, self.n_heads, self.dropout,
        )
        self.model.to(self.device_)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        total_steps  = self.epochs * len(train_loader)
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

        best_val:      float       = float("inf")
        best_epoch:    int         = 0
        best_state:    dict | None = None
        patience_left: int         = self.early_stop_patience

        for epoch in epoch_bar:
            self.model.train()
            total_loss = 0.0
            batch_bar = tqdm(train_loader, desc=f"  ep {epoch:3d}", leave=False,
                             unit="batch", dynamic_ncols=True, file=sys.stdout)
            for input_seq, labels in batch_bar:
                input_seq = input_seq.to(self.device_)
                labels    = labels.to(self.device_)

                logits = self.model(input_seq)  # (B, L, n_warm)
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
                for input_seq, labels in val_loader:
                    input_seq = input_seq.to(self.device_)
                    labels    = labels.to(self.device_)
                    logits = self.model(input_seq)
                    val_loss += F.cross_entropy(
                        logits.view(-1, n_warm),
                        labels.view(-1),
                        ignore_index=-100,
                    ).item()
            val_avg = val_loss / len(val_loader)

            if val_avg < best_val:
                best_val   = val_avg
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
                print(f"\n[{self.RECOMMENDER_NAME}] early stopping at epoch {epoch}, "
                      f"best val={best_val:.4f} at epoch {best_epoch}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"[{self.RECOMMENDER_NAME}] restored best checkpoint (val={best_val:.4f}, epoch={best_epoch})")

    # ------------------------------------------------------------------
    # Inference scoring: use the LAST position (no MASK appended — causal)
    # ------------------------------------------------------------------

    def _score_session_sequence(
        self,
        prior: list[str],
        warm_embs: torch.Tensor,
        cold_embs: torch.Tensor | None,
        all_embs: torch.Tensor,
    ) -> np.ndarray:
        assert self.model is not None and self.id_map is not None
        n_warm = warm_embs.shape[0]

        # Build tokens with HSTU offset (=1, no MASK)
        tokens: list[int] = []
        for t in prior:
            if t not in self.id_map.track_to_idx:
                continue
            g = self.id_map.track_to_idx[t]
            if g in self._global_to_warm_local:
                tokens.append(self._global_to_warm_local[g] + _HSTU_ITEM_OFFSET)
            elif g in self._global_to_cold_local:
                tokens.append(self._global_to_cold_local[g] + n_warm + _HSTU_ITEM_OFFSET)

        # No MASK — last position predicts the next
        tokens = tokens[-self.max_seq_len:]
        if not tokens:
            tokens = [PAD_TOKEN]
        pad_len = self.max_seq_len - len(tokens)
        tokens  = tokens + [PAD_TOKEN] * pad_len  # right-pad (consistent with training)

        last_real_idx = next((i for i in reversed(range(self.max_seq_len)) if tokens[i] != PAD_TOKEN), 0)

        x = torch.tensor([tokens], dtype=torch.long, device=self.device_)
        with torch.no_grad():
            hidden = self.model.encode_hidden(x, items_table=all_embs)
            h = hidden[0, last_real_idx, :]  # read at last real token (predicts next)
            warm_scores = (h @ warm_embs.T).cpu().numpy()
            cold_scores = (h @ cold_embs.T).cpu().numpy() if cold_embs is not None else None

        scores = np.full(self.id_map.n_tracks, -np.inf, dtype=np.float32)
        scores[self._warm_global_indices] = warm_scores
        if cold_scores is not None:
            scores[self._cold_global_indices] = cold_scores
        return scores

    # ------------------------------------------------------------------
    # save / load — instantiate the HSTU model class
    # ------------------------------------------------------------------

    def _set_model_state(self, state: dict) -> None:
        SessionRecommender._set_model_state(self, state)
        self._train_long = None
        self.feature_emb_paths   = state["feature_emb_paths"]
        self.feature_modalities  = state.get("feature_modalities", ["metadata-qwen3_embedding_0.6b"])
        self.max_seq_len         = state["max_seq_len"]
        self.hidden_size         = state["hidden_size"]
        self.n_layers            = state["n_layers"]
        self.n_heads             = state["n_heads"]
        self.dropout             = state["dropout"]
        self.mask_prob           = state.get("mask_prob", 0.4)
        self.epochs              = state["epochs"]
        self.batch_size          = state["batch_size"]
        self.lr                  = state["lr"]
        self.weight_decay        = state["weight_decay"]
        self.warmup_ratio        = state.get("warmup_ratio", 0.1)
        self.val_ratio           = state.get("val_ratio", 0.1)
        self.early_stop_patience = state.get("early_stop_patience", 10)
        self.device_             = torch.device(state.get("device", "cpu"))
        self._feature_dim          = state.get("feature_dim")
        self._warm_global_indices  = state.get("warm_global_indices", [])
        self._cold_global_indices  = state.get("cold_global_indices", [])
        self._cold_feature_matrix  = state.get("cold_feature_matrix")
        self._global_to_warm_local = state.get("global_to_warm_local", {})
        self._global_to_cold_local = state.get(
            "global_to_cold_local",
            {g: l for l, g in enumerate(self._cold_global_indices)},
        )

        sd = state.get("model_state_dict")
        if sd is not None and self.id_map is not None and self._feature_dim is not None:
            n_warm = len(self._warm_global_indices)
            dummy = np.zeros((n_warm, self._feature_dim), dtype=np.float32)
            self.model = _HSTUModel(
                dummy, self.hidden_size, self.max_seq_len,
                self.n_layers, self.n_heads, self.dropout,
            )
            self.model.load_state_dict(sd)
            self.model.to(self.device_)
