"""Session-based recommenders built on top of BaseRecommender.

Treats every session as the "user" of classical CF. Each model fits on a
sessions × tracks URM built from train conversations (role == 'music').

At inference time, for each test (session, turn) the recommender:
  - builds a session profile from prior music turns of that session
  - scores all tracks via the model-specific routine
  - falls back to a configurable cold-start strategy when profile is empty

Output schema (DataFrame returned by recommend):
    session_id     str
    turn           int
    track_ids      list[str]    most relevant first
    scores         list[float]  same order as track_ids
    fallback_used  list[int]    0/1 per track (1 if produced by fallback)
"""

from .user_base import UserRecommender
from .fallback import AbstractFallback, PopularityFallback
from .item_knn import ItemKNNRecommender
from .user_knn import UserKNNRecommender
from .top_pop import TopPopularRecommender
from .ease import EASERecommender
from .rp3beta import RP3BetaRecommender
from .slim_bpr import SLIMBPRRecommender
from .slim_elasticnet import SLIMElasticNetRecommender
from .multvae import MultVAERecommender
from .gfcf import GFCFRecommender
from .feature_bert4rec import FeatureBert4RecRecommender
from .feature_bert4rec_identity import FeatureBert4RecIdentityRecommender
from .feature_bert4rec_identity_cosine import FeatureBert4RecIdentityCosineRecommender
from .feature_bert4rec_identity_cosine_rope import FeatureBert4RecIdentityCosineRoPERecommender
from .feature_bert4rec_identity_cosine_rope_mmh import FeatureBert4RecIdentityCosineRoPEMMHRecommender
from .feature_bert4rec_identity_cosine_rope_mmh_icm import FeatureBert4RecIdentityCosineRoPEMMHICMRecommender
from .feature_bert4rec_identity_cosine_rope_mmh_icm_split import FeatureBert4RecIdentityCosineRoPEMMHICMSplitRecommender
from .feature_bert4rec_identity_cosine_rope_mmh_icm_xattn import FeatureBert4RecIdentityCosineRoPEMMHICMXAttnRecommender
from .feature_bert4rec_identity_cosine_rope_mmh_icm_hardneg import FeatureBert4RecIdentityCosineRoPEMMHICMHardNegRecommender
from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim import FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimRecommender
from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim_xattn import FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnRecommender
from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim_xattn_hardneg import FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegRecommender
from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim_xattn_hardneg_query import FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryRecommender
from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim_xattn_hardneg_query_full import FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullRecommender
from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim_xattn_hardneg_query_full_latefusion import FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullLateFusionRecommender
from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim_xattn_hardneg_query_full_trainfusion import FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullTrainFusionRecommender
from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim_xattn_hardneg_query_full_mha_userhistmod_sess import FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullMHAUserHistModSessRecommender
from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim_xattn_moddrop import FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnModDropRecommender
from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim_xattn_hardneg_moddrop import FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegModDropRecommender
from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim_xattn_glu import FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnGLURecommender
from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim_xattn_hardneg_glu import FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegGLURecommender
from .lightgcn import LightGCNRecommender
from .lightgcn_inductive import LightGCNInductiveRecommender
from .two_tower_multimodal import TwoTowerMultimodalRecommender
from .hstu import HSTURecommender
from .lightfm_rec import LightFMRecommender
from .two_tower import TwoTowerRecommender
from .query_retrieval import QueryRetrievalRecommender
from .lightfm_icm import LightFMICMRecommender
from .two_tower_v2 import TwoTowerV2Recommender
from .two_tower_v2_enhanced import TwoTowerV2EnhancedRecommender
from .session_knn import SessionKNNRecommender
from .sequential_rules import SequentialRulesRecommender
from .pure_svd import PureSVDRecommender
from .ials import IALSRecommender
from .prod2vec import Prod2VecRecommender
from .recvae import RecVAERecommender
from .text_cg import TextCGRecommender, TFIDFTextCG, BM25TextCG
# from .gp_cg import GPCGRecommender  # module missing from main (referenced
# in __init__.py but `gp_cg.py` was never committed). Re-enable when the file
# is added back.

__all__ = [
    "UserRecommender",
    "AbstractFallback",
    "PopularityFallback",
    "ItemKNNRecommender",
    "UserKNNRecommender",
    "TopPopularRecommender",
    "EASERecommender",
    "RP3BetaRecommender",
    "SLIMBPRRecommender",
    "SLIMElasticNetRecommender",
    "MultVAERecommender",
    "GFCFRecommender",
    "FeatureBert4RecRecommender",
    "FeatureBert4RecIdentityRecommender",
    "FeatureBert4RecIdentityCosineRecommender",
    "FeatureBert4RecIdentityCosineRoPERecommender",
    "FeatureBert4RecIdentityCosineRoPEMMHRecommender",
    "FeatureBert4RecIdentityCosineRoPEMMHICMRecommender",
    "FeatureBert4RecIdentityCosineRoPEMMHICMSplitRecommender",
    "FeatureBert4RecIdentityCosineRoPEMMHICMXAttnRecommender",
    "FeatureBert4RecIdentityCosineRoPEMMHICMHardNegRecommender",
    "FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimRecommender",
    "FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnRecommender",
    "FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegRecommender",
    "FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryRecommender",
    "FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullRecommender",
    "FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullLateFusionRecommender",
    "FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullTrainFusionRecommender",
    "FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullMHAUserHistModSessRecommender",
    "FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnModDropRecommender",
    "FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegModDropRecommender",
    "FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnGLURecommender",
    "FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegGLURecommender",
    "LightGCNRecommender",
    "LightGCNInductiveRecommender",
    "TwoTowerMultimodalRecommender",
    "HSTURecommender",
    "LightFMRecommender",
    "TwoTowerRecommender",
    "QueryRetrievalRecommender",
    "LightFMICMRecommender",
    "TwoTowerV2Recommender",
    "TwoTowerV2EnhancedRecommender",
    "SessionKNNRecommender",
    "SequentialRulesRecommender",
    "PureSVDRecommender",
    "IALSRecommender",
    "Prod2VecRecommender",
    "RecVAERecommender",
    "TextCGRecommender",
    "TFIDFTextCG",
    "BM25TextCG",
    # "GPCGRecommender",  # commented out — see import above
]
