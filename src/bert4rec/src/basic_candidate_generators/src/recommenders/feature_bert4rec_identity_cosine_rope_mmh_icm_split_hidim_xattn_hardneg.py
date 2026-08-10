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

    RECOMMENDER_NAME = "FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNeg"
