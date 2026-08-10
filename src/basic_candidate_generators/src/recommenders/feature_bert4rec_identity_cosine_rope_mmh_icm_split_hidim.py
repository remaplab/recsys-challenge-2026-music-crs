"""mmh_icm_split with HIGHER per-block SVD dims for the cardinal categorical features.

In `split` we compress artist (~5K unique) / album (~10K) / tag (~5K) sparse
one-hot blocks to 128/128/64 dim respectively. With this many unique values,
those compressions are aggressive and may discard relevant variance.

This variant raises them to 256/256/128.
"""

from __future__ import annotations

from .feature_bert4rec_identity_cosine_rope_mmh_icm_split import (
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitRecommender,
)


class FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimRecommender(FeatureBert4RecIdentityCosineRoPEMMHICMSplitRecommender):
    """split with artist/album/tag SVD dims doubled."""

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
