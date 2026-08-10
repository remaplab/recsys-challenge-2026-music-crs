"""split_hidim_xattn_hardneg with GLU (sigmoid-gate) modality fusion."""

from __future__ import annotations

from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim_xattn_glu import (
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnGLURecommender,
)
from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim_xattn_hardneg import (
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegRecommender,
)


class FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegGLURecommender(
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnGLURecommender,
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegRecommender,
):
    """GLU fusion + hardneg loss."""
    RECOMMENDER_NAME = "FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegGLU"
