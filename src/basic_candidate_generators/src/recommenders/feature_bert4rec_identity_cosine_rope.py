"""Identity (PCA-init linear encoder) + cosine + RoPE (rotary position emb).

Same backbone as `feature_bert4rec_identity_cosine` but the additive absolute
positional embedding is replaced with rotary positional encoding (RoPE)
applied to Q/K inside self-attention.

Why RoPE here:
  - Absolute pos emb forces the model to memorise a per-position vector. With
    only ~30K sessions it overfits early and stops generalising.
  - RoPE encodes RELATIVE position via Q/K rotation: position information is
    baked into the attention score itself, so no extra params and no per-
    position memorisation. Better inductive bias when sequences are short
    (avg ~3-4 turns) and varied.

Implementation:
  - Custom self-attention layer with manual RoPE applied to q and k.
  - Custom pre-norm transformer encoder layer (mirrors nn.TransformerEncoderLayer
    with norm_first=True, which is what the parent uses).
  - Drop pos_emb from the input embedding sum (overridden _build_seq_emb).
  - Output scoring stays cosine/tau (inherited from IdentityCosine).
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

from .feature_bert4rec import (
    PAD_TOKEN,
    _build_feature_matrix,
    _FeatureBert4RecDataset,
)
from .feature_bert4rec_identity_cosine import (
    FeatureBert4RecIdentityCosineRecommender,
    _FeatureBert4RecIdentityCosineModel,
)


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------

def _precompute_rope_freqs(head_dim: int, max_seq_len: int, base: float = 10000.0
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (cos, sin) of shape (max_seq_len, head_dim/2)."""
    assert head_dim % 2 == 0, f"RoPE needs even head_dim, got {head_dim}"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)              # (max_seq_len, head_dim/2)
    return freqs.cos(), freqs.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE rotation.

    x:   (B, H, L, head_dim) — query or key tensor split into heads
    cos: (1, 1, L, head_dim/2)
    sin: (1, 1, L, head_dim/2)
    Returns: same shape as x.
    """
    x1 = x[..., 0::2]                              # (..., L, head_dim/2)
    x2 = x[..., 1::2]
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    # Interleave back to (..., L, head_dim)
    out = torch.stack([rot1, rot2], dim=-1)        # (..., L, head_dim/2, 2)
    return out.flatten(-2)


# ---------------------------------------------------------------------------
# Custom transformer blocks with RoPE
# ---------------------------------------------------------------------------

class _RoPESelfAttention(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        assert hidden_size % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.dropout_p = dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                   # each (B, L, H, head_dim)
        q = q.transpose(1, 2)                          # (B, H, L, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos_l = cos[:, :, :L, :]
        sin_l = sin[:, :, :L, :]
        q = _apply_rope(q, cos_l, sin_l)
        k = _apply_rope(k, cos_l, sin_l)

        if key_padding_mask is not None:
            # SDPA bool mask: True = attend → invert key_padding_mask (True at pad).
            attn_mask = ~key_padding_mask[:, None, None, :]   # (B, 1, 1, L)
        else:
            attn_mask = None

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )                                              # (B, H, L, head_dim)
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.proj(out)


class _RoPEEncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer — mirrors nn.TransformerEncoderLayer
    with norm_first=True, GELU FFN with dim_feedforward = 4 * hidden_size.
    Replaces the standard self-attention with a RoPE-equipped variant.
    """
    def __init__(self, hidden_size: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn  = _RoPESelfAttention(hidden_size, n_heads, dropout)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, cos, sin, key_padding_mask=None):
        x = x + self.drop1(self.attn(self.norm1(x), cos, sin, key_padding_mask=key_padding_mask))
        x = x + self.drop2(self.ff(self.norm2(x)))
        return x


class _RoPEEncoder(nn.Module):
    """Stack of RoPE encoder layers. Matches the call signature
    `encoder(x, src_key_padding_mask=...)` of nn.TransformerEncoder
    so it slots in without changing the parent's forward/encode_hidden.
    """
    def __init__(self, n_layers: int, hidden_size: int, n_heads: int,
                  dropout: float, max_seq_len: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_RoPEEncoderLayer(hidden_size, n_heads, dropout) for _ in range(n_layers)]
        )
        head_dim = hidden_size // n_heads
        cos, sin = _precompute_rope_freqs(head_dim, max_seq_len)
        # Shape buffers as (1, 1, max_seq_len, head_dim/2) for broadcasting.
        self.register_buffer("cos", cos[None, None, :, :])
        self.register_buffer("sin", sin[None, None, :, :])

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None
                 ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, self.cos, self.sin, key_padding_mask=src_key_padding_mask)
        return x


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class _FeatureBert4RecIdentityCosineRoPEModel(_FeatureBert4RecIdentityCosineModel):
    """IdentityCosine model with RoPE encoder (no additive pos_emb)."""

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
            warm_feature_matrix, hidden_size, max_seq_len, n_layers, n_heads, dropout,
            init_tau=init_tau,
        )
        # Swap the additive pos_emb for a frozen-zero layer (it's still on the
        # module so save/load works, but contributes nothing to the input sum).
        with torch.no_grad():
            self.pos_emb.weight.zero_()
        self.pos_emb.weight.requires_grad = False

        # Replace the standard transformer encoder with our RoPE variant.
        self.encoder = _RoPEEncoder(n_layers, hidden_size, n_heads, dropout, max_seq_len)


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

class FeatureBert4RecIdentityCosineRoPERecommender(FeatureBert4RecIdentityCosineRecommender):
    """Identity + cosine + RoPE."""

    RECOMMENDER_NAME = "FeatureBert4RecIdentityCosineRoPE"

    def _build_train_val_sequences(self) -> tuple[list[list[int]], list[list[int]]]:
        """Build all sequences then split into (train, val).

        Default impl: random per-sequence split (1 sequence per session, so
        sequence-level == session-level). Subclasses that emit MULTIPLE
        sequences per session (e.g. sliding-window augmentation) must
        override this to split at SESSION level — otherwise sub-sequences
        of the same session leak between sides.
        """
        sequences = self._build_sequences()
        random.shuffle(sequences)
        n_val = max(1, int(len(sequences) * self.val_ratio))
        return sequences[n_val:], sequences[:n_val]

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

        print(f"[{self.RECOMMENDER_NAME}] Loading feature embeddings: {self.feature_modalities}")
        full_matrix = _build_feature_matrix(
            self.feature_emb_paths, self.feature_modalities, self.id_map
        )
        self._feature_dim = full_matrix.shape[1]
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

        self.model = _FeatureBert4RecIdentityCosineRoPEModel(
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
            self.model = _FeatureBert4RecIdentityCosineRoPEModel(
                dummy, self.hidden_size, self.max_seq_len,
                self.n_layers, self.n_heads, self.dropout,
                init_tau=self.init_tau,
            )
            self.model.load_state_dict(sd)
            self.model.to(self.device_)
