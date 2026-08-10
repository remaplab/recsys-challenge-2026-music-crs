from __future__ import annotations

from .feature_bert4rec_identity_cosine_rope_mmh_icm_split import (
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitRecommender,
)

class FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimRecommender(FeatureBert4RecIdentityCosineRoPEMMHICMSplitRecommender):

    RECOMMENDER_NAME = "FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDim"
    DEFAULT_BLOCK_DIMS = {
        "artist": 256,
        "album":  256,
        "tag":    128,
        "decade": 0,
        "popularity": 0,
        "interaction_popularity": 0,
        "duration": 0,
    }
