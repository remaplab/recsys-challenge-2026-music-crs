"""mmh_icm_split_hidim + xattn fusion + hard-negative aux loss.

Triple compound of the three best confirmed improvements on split:
  - SplitHiDim : feature loading (more granular ICM SVD)
  - XAttn      : fusion (attention over modalities)
  - HardNeg    : training loop (hardneg InfoNCE aux)

Multiple inheritance pulls the three from independent layers (features
/ fusion / loss) and Python's MRO resolves each method to the right
parent. If the three are truly orthogonal, this should be the new best
overall.
"""

from __future__ import annotations

from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim import (
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimRecommender,
)
from .feature_bert4rec_identity_cosine_rope_mmh_icm_xattn import (
    FeatureBert4RecIdentityCosineRoPEMMHICMXAttnRecommender,
)
from .feature_bert4rec_identity_cosine_rope_mmh_icm_hardneg import (
    FeatureBert4RecIdentityCosineRoPEMMHICMHardNegRecommender,
)


class FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegRecommender(
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimRecommender,
    FeatureBert4RecIdentityCosineRoPEMMHICMXAttnRecommender,
    FeatureBert4RecIdentityCosineRoPEMMHICMHardNegRecommender,
):
    """split_hidim features + xattn fusion + hardneg loss."""

    RECOMMENDER_NAME = "FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNeg"
