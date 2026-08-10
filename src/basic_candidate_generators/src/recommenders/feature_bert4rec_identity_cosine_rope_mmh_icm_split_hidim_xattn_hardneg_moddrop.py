"""split_hidim_xattn_hardneg + per-item modality block dropout (no FFN change).

Multiple inheritance: ModDrop recommender provides `_make_model` returning
the ModDrop XAttn model (GELU FFN preserved); HardNeg recommender provides
`_fit_model` with the hardneg InfoNCE auxiliary loss.
"""

from __future__ import annotations

from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim_xattn_moddrop import (
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnModDropRecommender,
)
from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim_xattn_hardneg import (
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegRecommender,
)


class FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegModDropRecommender(
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnModDropRecommender,
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegRecommender,
):
    """split_hidim_xattn + ModDrop (GELU FFN) + hardneg loss.

    MRO: ModDrop._make_model wins (first parent), HardNeg._fit_model wins
    (only parent that defines it).
    """
    RECOMMENDER_NAME = "FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegModDrop"
