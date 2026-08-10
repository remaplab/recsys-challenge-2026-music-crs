"""Reusable building blocks for the architectural variants explored on top of
the `split_hidim_xattn(_hardneg)` baseline.

  - `_SwiGLURoPEEncoderLayer` / `_SwiGLURoPEEncoder` : RoPE encoder with the
    standard GELU FFN replaced by SwiGLU (LLaMA/Mistral-style). Same param
    budget as the GELU baseline (gate + value share the 4*H hidden, so we
    use 8/3 * H per branch ≈ 2.67H to match params).

  - `apply_modality_block_dropout(x, boundaries, p)`: during training,
    zeroes random modality blocks of `x` (a (..., total_dim) feature matrix)
    with probability `p` per block, INDEPENDENTLY per sample. At inference,
    returns `x` unchanged.

  - `_ModalityCrossAttnMH` : multi-head version of `_ModalityCrossAttn`.
    4 heads attending over the K modality projections (head_dim = H/n_heads).

  - `_ModalityCrossAttnGLU` : GLU-style fusion. Per-modality scalar gate
    via sigmoid(q · m_k / sqrt(H)) → multiply → sum (no softmax, no
    normalization constraint). Modalities can collectively contribute more
    or less to the output (softmax force-couples them at sum=1).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_bert4rec_identity_cosine_rope import (
    _RoPESelfAttention,
    _precompute_rope_freqs,
)
from .feature_bert4rec_identity_cosine_rope_mmh import _ModalityMultiHead


# ---------------------------------------------------------------------------
# SwiGLU FFN encoder
# ---------------------------------------------------------------------------

class _SwiGLUFeedForward(nn.Module):
    """SwiGLU = (SiLU(W_gate x) * W_val x) → W_out.

    To match a standard FFN's param budget at hidden_dim=4H, we shrink the
    SwiGLU hidden to (8/3)*H (LLaMA/Mistral convention). With H=hidden_size:
      standard:  H*(4H) + 4H*H = 8 H^2
      swiglu  :  2 * H*(d) + d*H = 3 H d, with d = (8/3) H → 8 H^2 ✓
    """
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        d = int((hidden_size * 8) // 3)
        # Round up to a multiple of 8 for tensor-core friendliness.
        d = ((d + 7) // 8) * 8
        self.w_gate = nn.Linear(hidden_size, d, bias=False)
        self.w_val  = nn.Linear(hidden_size, d, bias=False)
        self.w_out  = nn.Linear(d, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(self.dropout(F.silu(self.w_gate(x)) * self.w_val(x)))


class _SwiGLURoPEEncoderLayer(nn.Module):
    """RoPE encoder layer with SwiGLU FFN (replaces GELU-Linear-Linear)."""
    def __init__(self, hidden_size: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn  = _RoPESelfAttention(hidden_size, n_heads, dropout)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ff    = _SwiGLUFeedForward(hidden_size, dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, cos, sin, key_padding_mask=None):
        x = x + self.drop1(self.attn(self.norm1(x), cos, sin, key_padding_mask=key_padding_mask))
        x = x + self.drop2(self.ff(self.norm2(x)))
        return x


class _SwiGLURoPEEncoder(nn.Module):
    """RoPE encoder using SwiGLU layers."""
    def __init__(self, n_layers: int, hidden_size: int, n_heads: int,
                  dropout: float, max_seq_len: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_SwiGLURoPEEncoderLayer(hidden_size, n_heads, dropout) for _ in range(n_layers)]
        )
        head_dim = hidden_size // n_heads
        cos, sin = _precompute_rope_freqs(head_dim, max_seq_len)
        self.register_buffer("cos", cos[None, None, :, :])
        self.register_buffer("sin", sin[None, None, :, :])

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None
                 ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, self.cos, self.sin, key_padding_mask=src_key_padding_mask)
        return x


# ---------------------------------------------------------------------------
# Modality block dropout
# ---------------------------------------------------------------------------

def apply_modality_block_dropout(
    x: torch.Tensor,
    boundaries: list[tuple[int, int]],
    p: float,
) -> torch.Tensor:
    """Zero entire modality blocks of `x` with probability `p` per block.

    `x` has shape (..., total_dim) where total_dim == sum(b-a for a,b in
    boundaries). At training, per item (each leading-dim element), each
    modality block is zeroed independently with probability `p`.

    Implements per-item modality dropout — equivalent to randomly hiding
    modalities for each item per forward pass. Encourages the model to
    not over-rely on any single modality. At inference (p==0 or eval mode
    handled by the caller), this is the identity.
    """
    if p <= 0.0:
        return x
    # Build a (..., total_dim) mask where each modality slice is either
    # all-ones (kept) or all-zeros (dropped). Sample one keep/drop per
    # leading-dim element per modality.
    leading_shape = x.shape[:-1]
    K = len(boundaries)
    # (..., K) Bernoulli keeps
    keep = (torch.rand(*leading_shape, K, device=x.device) >= p).to(x.dtype)
    mask = torch.empty_like(x)
    for k, (a, b) in enumerate(boundaries):
        mask[..., a:b] = keep[..., k:k + 1]
    return x * mask


# ---------------------------------------------------------------------------
# Multi-head cross-attention over modalities (replacement for single-query)
# ---------------------------------------------------------------------------

class _ModalityCrossAttnMH(_ModalityMultiHead):
    """Multi-head cross-attention fusion over per-modality projections.

      m_k = Linear_k(L2(x_k))                    # (..., H)
      M   = stack([m_1, ..., m_K])               # (..., K, H)
      For head h ∈ [n_heads]:
        q_h  ∈ R^{H/n_heads}                     # learned
        Kh_k = M_k[:, h*hd:(h+1)*hd]             # head-projected key/value
        attn = softmax(q_h · Kh.T / sqrt(hd))    # over K modalities
        out_h = attn @ Kh
      out = concat(out_h for h)                  # (..., H)
    """
    def __init__(self, modality_dims: list[int], hidden_size: int,
                  n_attn_heads: int = 4) -> None:
        super().__init__(modality_dims, hidden_size)
        assert hidden_size % n_attn_heads == 0, (
            f"hidden_size {hidden_size} not divisible by n_attn_heads {n_attn_heads}"
        )
        self.n_attn_heads = int(n_attn_heads)
        self.head_dim = hidden_size // n_attn_heads
        # One learned query per head.
        self.query = nn.Parameter(torch.empty(n_attn_heads, self.head_dim))
        nn.init.normal_(self.query, std=0.02)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projs: list[torch.Tensor] = []
        for (a, b), head in zip(self.boundaries, self.heads):
            sl = F.normalize(x[..., a:b], dim=-1)
            projs.append(head(sl))                              # (..., H)
        m = torch.stack(projs, dim=-2)                           # (..., K, H)
        # Reshape (..., K, H) → (..., K, n_heads, head_dim) → (..., n_heads, K, head_dim)
        leading = m.shape[:-2]
        K = m.shape[-2]
        mh = m.view(*leading, K, self.n_attn_heads, self.head_dim)
        mh = mh.transpose(-3, -2)                                # (..., n_heads, K, head_dim)
        # attn_logits[..., h, k] = q_h · mh[..., h, k, :] / sqrt(hd)
        # query: (n_heads, head_dim) → (n_heads, 1, head_dim) to broadcast over K.
        attn = (mh * self.query.unsqueeze(-2)).sum(dim=-1) * self.scale  # (..., n_heads, K)
        w = F.softmax(attn, dim=-1).unsqueeze(-1)                # (..., n_heads, K, 1)
        out = (w * mh).sum(dim=-2)                               # (..., n_heads, head_dim)
        return out.reshape(*leading, self.hidden_size)            # (..., H)


# ---------------------------------------------------------------------------
# GLU-style modality fusion (sigmoid gates instead of softmax)
# ---------------------------------------------------------------------------

class _ModalityCrossAttnGLU(_ModalityMultiHead):
    """Per-modality sigmoid gate × projection → sum (no softmax).

      m_k = Linear_k(L2(x_k))                    # (..., H)
      gate_k = sigmoid(q · m_k / sqrt(H))        # (..., 1) scalar per modality
      out = sum_k(gate_k * m_k)                  # (..., H)

    Unlike softmax (which forces gates to sum to 1 per item — coupling
    modalities), sigmoid gates are independent. The model can lean on
    multiple modalities simultaneously or downweight all of them, which
    matches the intuition that the right number of "active" modalities
    varies per item.
    """
    def __init__(self, modality_dims: list[int], hidden_size: int) -> None:
        super().__init__(modality_dims, hidden_size)
        self.query = nn.Parameter(torch.empty(hidden_size))
        nn.init.normal_(self.query, std=0.02)
        self.scale = hidden_size ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projs: list[torch.Tensor] = []
        for (a, b), head in zip(self.boundaries, self.heads):
            sl = F.normalize(x[..., a:b], dim=-1)
            projs.append(head(sl))                              # (..., H)
        m = torch.stack(projs, dim=-2)                           # (..., K, H)
        gate_logits = (m @ self.query) * self.scale              # (..., K)
        gates = torch.sigmoid(gate_logits).unsqueeze(-1)         # (..., K, 1)
        return (gates * m).sum(dim=-2)                            # (..., H)
