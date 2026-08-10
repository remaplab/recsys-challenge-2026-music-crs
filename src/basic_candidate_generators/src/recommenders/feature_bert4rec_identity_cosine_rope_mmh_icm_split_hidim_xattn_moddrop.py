"""split_hidim_xattn + per-item modality block dropout (no FFN change).

Isolates the modality-dropout regularizer from the SwiGLU FFN bundled with
it in the `_swiglu` variant. The encoder stays the GELU RoPE encoder of the
parent (`_MMHICMXAttnModel`) — only `self.feature_matrix` is passed through
`apply_modality_block_dropout` before the per-modality projections during
training. Inference is identity.

This lets us attribute any delta vs the `_swiglu` variant to mod_drop alone
rather than the FFN swap.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim_xattn import (
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnRecommender,
)
from .feature_bert4rec_identity_cosine_rope_mmh_icm_xattn import _MMHICMXAttnModel
from .feature_bert4rec_variants_blocks import apply_modality_block_dropout


class _ModDropXAttnModel(_MMHICMXAttnModel):
    """XAttn model with per-item modality block dropout, GELU FFN unchanged."""

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
        mod_drop_p: float,
    ) -> None:
        super().__init__(
            warm_feature_matrix, hidden_size, max_seq_len,
            n_layers, n_heads, dropout,
            init_tau=init_tau, modality_dims=modality_dims,
        )
        self.mod_drop_p = float(mod_drop_p)

    def _maybe_drop_modalities(self, feat: torch.Tensor) -> torch.Tensor:
        if not self.training or self.mod_drop_p <= 0.0:
            return feat
        return apply_modality_block_dropout(
            feat, self.item_encoder.boundaries, self.mod_drop_p
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self._maybe_drop_modalities(self.feature_matrix)
        warm_embs = self.item_encoder(feat)
        emb, pad_mask = self._build_seq_emb(x, warm_embs)
        out = self.encoder(emb, src_key_padding_mask=pad_mask)
        out = self.output_norm(out)
        out_n  = F.normalize(out,      dim=-1)
        warm_n = F.normalize(warm_embs, dim=-1)
        return (out_n @ warm_n.T) / self.tau


class FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnModDropRecommender(
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnRecommender,
):
    """split_hidim_xattn + modality dropout (GELU FFN preserved)."""

    RECOMMENDER_NAME = "FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnModDrop"

    def __init__(self, *args: Any, mod_drop_p: float = 0.1, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.mod_drop_p = float(mod_drop_p)

    def _make_model(self, warm_feature_matrix: np.ndarray, modality_dims: list[int]):
        return _ModDropXAttnModel(
            warm_feature_matrix, self.hidden_size, self.max_seq_len,
            self.n_layers, self.n_heads, self.dropout,
            init_tau=self.init_tau,
            modality_dims=modality_dims,
            mod_drop_p=self.mod_drop_p,
        )
