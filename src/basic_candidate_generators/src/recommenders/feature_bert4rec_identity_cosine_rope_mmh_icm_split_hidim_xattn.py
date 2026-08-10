"""mmh_icm_split_hidim + cross-modality attention fusion.

split_hidim is the current dominant CG (wins all 6 official metrics);
xattn was the runner-up (Δndcg@20 = 0.0002 only). Combining them gives the
attention mechanism MORE modalities with HIGHER information density per
modality to fuse over. Multiple inheritance: feature loading from
SplitHiDim, fusion model from XAttn.

Expected: small but compounding gain on top of split_hidim (~+0.001/+0.005
ndcg@20). Trade-off is mostly ranking sharpness within top-K — for
recall@100 specifically use the `_hardneg` variant.
"""

from __future__ import annotations

from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim import (
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimRecommender,
)
from .feature_bert4rec_identity_cosine_rope_mmh_icm_xattn import (
    FeatureBert4RecIdentityCosineRoPEMMHICMXAttnRecommender,
)


class FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnRecommender(
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimRecommender,
    FeatureBert4RecIdentityCosineRoPEMMHICMXAttnRecommender,
):
    """split_hidim feature loader + xattn fusion model."""

    RECOMMENDER_NAME = "FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttn"
