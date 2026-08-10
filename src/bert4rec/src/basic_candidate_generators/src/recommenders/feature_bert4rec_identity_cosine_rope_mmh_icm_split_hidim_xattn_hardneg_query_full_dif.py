from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_bert4rec_identity_cosine_rope import _apply_rope
from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim_xattn_hardneg_query import (
    _MMHICMXAttnQueryModel,
)
from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim_xattn_hardneg_query_full import (
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullRecommender,
)

class _DIFRoPESelfAttention(nn.Module):

    def __init__(self, hidden_size: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        assert hidden_size % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)

        self.q_query = nn.Linear(hidden_size, hidden_size)
        self.k_query = nn.Linear(hidden_size, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.dropout_p = dropout

        nn.init.zeros_(self.q_query.weight)
        nn.init.zeros_(self.q_query.bias)

    def _heads(self, t: torch.Tensor, B: int, L: int) -> torch.Tensor:
        return t.reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        xv: torch.Tensor,
        qside: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, _ = xv.shape
        qkv = self.qkv(xv).reshape(B, L, 3, self.n_heads, self.head_dim)
        q_i, k_i, v = qkv.unbind(dim=2)
        q_i = _apply_rope(q_i.transpose(1, 2), cos[:, :, :L, :], sin[:, :, :L, :])
        k_i = _apply_rope(k_i.transpose(1, 2), cos[:, :, :L, :], sin[:, :, :L, :])
        v = v.transpose(1, 2)

        attn_mask = None
        if qside is not None:
            q_q = _apply_rope(self._heads(self.q_query(qside), B, L), cos[:, :, :L, :], sin[:, :, :L, :])
            k_q = _apply_rope(self._heads(self.k_query(qside), B, L), cos[:, :, :L, :], sin[:, :, :L, :])
            att_query = (q_q @ k_q.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn_mask = att_query
        if key_padding_mask is not None:
            pad = torch.zeros(B, 1, 1, L, dtype=xv.dtype, device=xv.device)
            pad = pad.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
            attn_mask = pad if attn_mask is None else attn_mask + pad

        out = F.scaled_dot_product_attention(
            q_i, k_i, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.proj(out)

class _DIFRoPEEncoderLayer(nn.Module):

    def __init__(self, hidden_size: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn  = _DIFRoPESelfAttention(hidden_size, n_heads, dropout)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, qside, cos, sin, key_padding_mask=None):
        xv = self.norm1(x)
        x = x + self.drop1(self.attn(xv, qside, cos, sin, key_padding_mask=key_padding_mask))
        x = x + self.drop2(self.ff(self.norm2(x)))
        return x

class _DIFRoPEEncoder(nn.Module):

    def __init__(self, n_layers: int, hidden_size: int, n_heads: int,
                 dropout: float, cos: torch.Tensor, sin: torch.Tensor) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_DIFRoPEEncoderLayer(hidden_size, n_heads, dropout) for _ in range(n_layers)]
        )
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x: torch.Tensor, qside: torch.Tensor | None = None,
                src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, qside, self.cos, self.sin, key_padding_mask=src_key_padding_mask)
        return x

class _MMHICMXAttnQueryDIFModel(_MMHICMXAttnQueryModel):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        old = self.encoder
        n_heads = old.layers[0].attn.n_heads
        dropout = old.layers[0].attn.dropout_p
        self.encoder = _DIFRoPEEncoder(
            len(old.layers), self.hidden_size, n_heads, dropout,
            cos=old.cos.clone(), sin=old.sin.clone(),
        )

        nn.init.normal_(self.query_proj.weight, std=0.02)
        nn.init.zeros_(self.query_proj.bias)

    def _query_side(self, query_idx_seq: torch.Tensor | None) -> torch.Tensor | None:
        if query_idx_seq is None:
            return None
        q = self.query_table[query_idx_seq]
        return self.query_proj(q)

    def forward(self, x: torch.Tensor, query_idx_seq: torch.Tensor | None = None) -> torch.Tensor:
        warm_embs = self.item_encoder(self.feature_matrix)
        emb, pad_mask = self._build_seq_emb(x, warm_embs)
        qside = self._query_side(query_idx_seq)
        out = self.encoder(emb, qside, src_key_padding_mask=pad_mask)
        out = self.output_norm(out)
        out_n  = F.normalize(out,       dim=-1)
        warm_n = F.normalize(warm_embs, dim=-1)
        return (out_n @ warm_n.T) / self.tau

    def encode_hidden(
        self,
        x: torch.Tensor,
        items_table: torch.Tensor | None = None,
        query_idx_seq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if items_table is None:
            items_table = self.item_encoder(self.feature_matrix)
        emb, pad_mask = self._build_seq_emb(x, items_table)
        qside = self._query_side(query_idx_seq)
        out = self.encoder(emb, qside, src_key_padding_mask=pad_mask)
        return self.output_norm(out)

class FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullDIFRecommender(
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullRecommender,
):

    RECOMMENDER_NAME = "FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullDIF"

    def _make_model(self, warm_feature_matrix: np.ndarray, modality_dims: list[int]):
        assert self._query_emb_table is not None, "_load_query_cache must run before _make_model"
        return _MMHICMXAttnQueryDIFModel(
            warm_feature_matrix, self.hidden_size, self.max_seq_len,
            self.n_layers, self.n_heads, self.dropout,
            init_tau=self.init_tau,
            modality_dims=modality_dims,
            query_emb_table=self._query_emb_table,
        )
