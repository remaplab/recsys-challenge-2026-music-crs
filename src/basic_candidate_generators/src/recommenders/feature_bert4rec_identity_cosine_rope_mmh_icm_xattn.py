"""mmh_icm with cross-modality attention instead of element-wise sum.

Per-modality Linear projections are stacked as a (n_modalities, hidden)
sequence per item; a single learnable query attends over them with scaled
dot-product attention, producing the item embedding. The model can therefore
weight modalities DIFFERENTLY for different items (e.g., for a track with
rich CF history, lean on CF; for a brand-new track, lean on Qwen3 / ICM).

  m_k = Linear_k(L2(x_k))               # per modality
  M   = stack([m_1, ..., m_K])          # (B, K, H) per item
  e   = softmax(q · M^T / sqrt(H)) · M  # (B, H)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_bert4rec_identity_cosine_rope_mmh import (
    _FeatureBert4RecIdentityCosineRoPEMMHModel,
    _ModalityMultiHead,
)
from .feature_bert4rec_identity_cosine_rope_mmh_icm import (
    FeatureBert4RecIdentityCosineRoPEMMHICMRecommender,
)


class _ModalityCrossAttn(_ModalityMultiHead):
    """Per-modality Linear → stack → single-query cross-attention → output."""

    def __init__(self, modality_dims: list[int], hidden_size: int) -> None:
        super().__init__(modality_dims, hidden_size)
        # Single learnable query (1 attention head, no multi-head split).
        self.query = nn.Parameter(torch.empty(hidden_size))
        nn.init.normal_(self.query, std=0.02)
        self.scale = hidden_size ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projs: list[torch.Tensor] = []
        for (a, b), head in zip(self.boundaries, self.heads):
            sl = F.normalize(x[..., a:b], dim=-1)
            projs.append(head(sl))                            # (..., H)
        m = torch.stack(projs, dim=-2)                         # (..., K, H)
        # Attention: q · M^T → (..., K)
        attn = (m @ self.query) * self.scale                   # (..., K)
        w = F.softmax(attn, dim=-1).unsqueeze(-1)              # (..., K, 1)
        return (w * m).sum(dim=-2)                             # (..., H)


class _MMHICMXAttnModel(_FeatureBert4RecIdentityCosineRoPEMMHModel):
    def __init__(self, *args: Any, modality_dims: list[int], **kwargs: Any) -> None:
        super().__init__(*args, modality_dims=modality_dims, **kwargs)
        self.item_encoder = _ModalityCrossAttn(modality_dims, self.hidden_size)


class FeatureBert4RecIdentityCosineRoPEMMHICMXAttnRecommender(FeatureBert4RecIdentityCosineRoPEMMHICMRecommender):
    """mmh_icm with cross-modality attention fusion."""

    RECOMMENDER_NAME = "FeatureBert4RecIdentityCosineRoPEMMHICMXAttn"

    def _make_model(self, warm_feature_matrix: np.ndarray, modality_dims: list[int]):
        return _MMHICMXAttnModel(
            warm_feature_matrix, self.hidden_size, self.max_seq_len,
            self.n_layers, self.n_heads, self.dropout,
            init_tau=self.init_tau,
            modality_dims=modality_dims,
        )
