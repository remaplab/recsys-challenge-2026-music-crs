"""Embedding-based candidate generators (frozen Qwen3 towers).

Training-free CGs built on the precomputed Qwen3-Embedding track/query caches
(L2-normed → dot = cosine). See SPEC.md.

  - emb_matrix       : W_emb builder + disk IO + id_map remap / CBF adapter
  - EmbeddingItemKNN : content item-item KNN (profile @ W_emb), pipeline-native
  - DenseQueryCG     : query→track retrieval from the per-turn query cache
"""

from .embedding_item_knn import EmbeddingItemKNN
from .dense_query_cg import DenseQueryCG
from .hybrid_cg import HybridCG

__all__ = ["EmbeddingItemKNN", "DenseQueryCG", "HybridCG"]
