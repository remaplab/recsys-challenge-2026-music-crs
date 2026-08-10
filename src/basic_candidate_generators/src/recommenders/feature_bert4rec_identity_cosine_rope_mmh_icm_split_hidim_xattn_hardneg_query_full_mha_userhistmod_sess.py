"""4-way stack: hardneg + query_full (Design A) + MHA xattn + SESSION-PRIOR taste.

Variant of `..._userhistmod` that swaps the user-LOSO taste for a per-sample
SESSION-PRIOR taste:
  train     : mean(L2(item_encoder(positions where labels == -100))) — i.e.
              only the context positions the model SEES (mask/random/keep-
              original-but-labeled positions all carry labels != -100, so
              the target leak is impossible).
  inference : mean(L2(item_encoder(prior warm items))) — entire prior is
              context (no MLM masking at inference), same formula.

Train↔test alignment of the taste channel is exact (same recipe, same
distribution), which is the supposed structural fix for the splitK
user-disjoint setup. Whether the resulting signal is informative enough
above the BERT4Rec sequence + Qwen3 query is empirical — tune to find out.
"""

from __future__ import annotations

from typing import Any

from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim_xattn_hardneg_query_full_mha_user_base import (
    _QueryFullMHAUserRecommenderBase,
)


class FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullMHAUserHistModSessRecommender(
    _QueryFullMHAUserRecommenderBase,
):
    RECOMMENDER_NAME = "FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullMHAUserHistModSess"

    def __init__(self, *args: Any, alpha_init: float = 0.1, **kwargs: Any) -> None:
        super().__init__(
            *args,
            use_bias_term=False,
            use_film=False,
            use_query_fusion=True,
            query_fusion_source="session_taste",
            alpha_init=alpha_init,
            **kwargs,
        )
