"""split_hidim_xattn with GLU (sigmoid-gate) modality fusion instead of softmax.

Softmax over K modalities forces the gate weights to sum to 1 per item —
the modalities compete for a fixed mass. GLU uses an independent sigmoid
gate per modality, allowing the model to use multiple modalities at full
weight or downweight all of them for noisy items. No probability simplex
constraint.

  m_k = Linear_k(L2(x_k))                    # (..., H)
  gate_k = sigmoid(q · m_k / sqrt(H))         # (..., 1) scalar per modality
  out = sum_k(gate_k * m_k)                  # (..., H)
"""

from __future__ import annotations

import numpy as np

from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim_xattn import (
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnRecommender,
)
from .feature_bert4rec_identity_cosine_rope_mmh_icm_xattn import _MMHICMXAttnModel
from .feature_bert4rec_variants_blocks import _ModalityCrossAttnGLU


class _GLUFusionXAttnModel(_MMHICMXAttnModel):
    """XAttn model with sigmoid-gate (GLU-style) modality fusion."""

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
    ) -> None:
        super().__init__(
            warm_feature_matrix, hidden_size, max_seq_len,
            n_layers, n_heads, dropout,
            init_tau=init_tau, modality_dims=modality_dims,
        )
        # Replace softmax xattn with sigmoid-gate fusion.
        self.item_encoder = _ModalityCrossAttnGLU(modality_dims, hidden_size)


class FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnGLURecommender(
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnRecommender,
):
    """split_hidim_xattn with GLU-style modality fusion."""

    RECOMMENDER_NAME = "FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnGLU"

    def _make_model(self, warm_feature_matrix: np.ndarray, modality_dims: list[int]):
        return _GLUFusionXAttnModel(
            warm_feature_matrix, self.hidden_size, self.max_seq_len,
            self.n_layers, self.n_heads, self.dropout,
            init_tau=self.init_tau,
            modality_dims=modality_dims,
        )
