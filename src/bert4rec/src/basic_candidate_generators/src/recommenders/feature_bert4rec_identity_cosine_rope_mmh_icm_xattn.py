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

    def __init__(self, modality_dims: list[int], hidden_size: int) -> None:
        super().__init__(modality_dims, hidden_size)

        self.query = nn.Parameter(torch.empty(hidden_size))
        nn.init.normal_(self.query, std=0.02)
        self.scale = hidden_size ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projs: list[torch.Tensor] = []
        for (a, b), head in zip(self.boundaries, self.heads):
            sl = F.normalize(x[..., a:b], dim=-1)
            projs.append(head(sl))
        m = torch.stack(projs, dim=-2)

        attn = (m @ self.query) * self.scale
        w = F.softmax(attn, dim=-1).unsqueeze(-1)
        return (w * m).sum(dim=-2)

class _MMHICMXAttnModel(_FeatureBert4RecIdentityCosineRoPEMMHModel):
    def __init__(self, *args: Any, modality_dims: list[int], **kwargs: Any) -> None:
        super().__init__(*args, modality_dims=modality_dims, **kwargs)
        self.item_encoder = _ModalityCrossAttn(modality_dims, self.hidden_size)

class FeatureBert4RecIdentityCosineRoPEMMHICMXAttnRecommender(FeatureBert4RecIdentityCosineRoPEMMHICMRecommender):

    RECOMMENDER_NAME = "FeatureBert4RecIdentityCosineRoPEMMHICMXAttn"

    def _make_model(self, warm_feature_matrix: np.ndarray, modality_dims: list[int]):
        return _MMHICMXAttnModel(
            warm_feature_matrix, self.hidden_size, self.max_seq_len,
            self.n_layers, self.n_heads, self.dropout,
            init_tau=self.init_tau,
            modality_dims=modality_dims,
        )
