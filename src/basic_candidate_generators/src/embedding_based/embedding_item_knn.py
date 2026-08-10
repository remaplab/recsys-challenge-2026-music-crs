"""EmbeddingItemKNN — content item-item KNN over frozen Qwen3 track embeddings.

Pipeline-native: subclasses `UserRecommender`, so cold fallback, future-track
masking, seen removal, multiturn inference and save/load all come for free. The
only model-specific work is loading `W_emb` and remapping it to the fold id_map.

W_emb is built once per Qwen size by `build_emb_sim.py` and cached in tower-id
space; `_fit_model` loads + remaps it (fold-safe). If no cache path is given it
is built on the fly from the track tower (slower — warns).

Compatible with the standard CG pipeline: instantiated as
`EmbeddingItemKNN(urm_mode=..., **params)` with path params resolved via
`_cv_utils.resolve_param_paths`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from recommenders.interactions import build_item_cbf_similarity_fast
from recommenders.item_knn import build_cf_similarity
from recommenders.user_base import UserRecommender

from .emb_matrix import (
    build_emb_item_sim,
    load_emb_sim,
    load_image_tower,
    load_track_tower,
    remap_sim_to_idmap,
    save_emb_sim,
    sparsify_topk,
)


class EmbeddingItemKNN(UserRecommender):
    """Content embedding KNN, optionally blended with CF + tag-CBF signals.

    Final item-item matrix (all three in id_map index space):
        W = W_emb  +  cf_weight * W_cf  +  icm_weight * W_cbf

    W_emb is the embedding base (frozen Qwen track-tower cosine, top-k). W_cf is
    the collaborative item-item cosine over the URM (shared `build_cf_similarity`
    from item_knn). W_cbf is the tag content similarity. Weights are additive so
    the model degrades to pure content (both 0) or becomes CF/tag-heavy as the
    weights grow; the reranker/marginal-recall gate decides the mix.
    """

    RECOMMENDER_NAME = "EmbeddingItemKNN"

    def __init__(
        self,
        emb_sim_path: str | None = None,     # cached W_emb .npz (tower-id space)
        track_emb_dir: str | None = None,    # dense_tracks_* dir; build if no cache
        model_size: str = "qwen3_8b",        # bookkeeping / cache selection
        k: int = 150,                        # effective W_emb neighbours (load-time trim)
        cache_k: int = 500,                  # neighbours stored when building W_emb cache
        cf_weight: float = 0.0,              # collaborative W_cf blend (off by default)
        cf_k: int = 150,                     # CF neighbours per item
        shrink: float = 10.0,                # CF cosine shrink
        icm_weight: float = 0.0,             # tag-CBF blend (off by default)
        k_icm: int = 500,                    # effective tag-CBF neighbours (load-time trim)
        icm_sim_path: str | None = None,     # cached W_cbf .npz stem (per-fold, hash-keyed)
        icm_cache_k: int = 500,              # neighbours stored when building W_cbf cache
        img_sim_path: str | None = None,     # cached W_img .npz (tower-id space)
        img_parquet_glob: str | None = None, # SigLIP2 parquets; built if cache absent
        img_weight: float = 0.0,             # image SigLIP2 blend (off by default)
        img_k: int = 150,                    # effective image neighbours (load-time trim)
        img_cache_k: int = 500,              # neighbours stored when building the cache
        use_gpu: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.emb_sim_path = emb_sim_path
        self.track_emb_dir = track_emb_dir
        self.model_size = model_size
        self.k = k
        self.cache_k = cache_k
        self.cf_weight = cf_weight
        self.cf_k = cf_k
        self.shrink = shrink
        self.icm_weight = icm_weight
        self.k_icm = k_icm
        self.icm_sim_path = icm_sim_path
        self.icm_cache_k = icm_cache_k
        self.img_sim_path = img_sim_path
        self.img_parquet_glob = img_parquet_glob
        self.img_weight = img_weight
        self.img_k = img_k
        self.img_cache_k = img_cache_k
        self.use_gpu = use_gpu
        self.W: csr_matrix | None = None

    def _fit_model(self, urm: csr_matrix) -> None:
        t0 = time.time()
        W_tower, tower_ids = self._load_or_build_emb_sim()
        W_tower = sparsify_topk(W_tower, self.k)   # trim cached/built K down to k
        W = remap_sim_to_idmap(W_tower, tower_ids, self.id_map)
        covered = len(np.unique(W.tocoo().row))
        print(f"[{self.RECOMMENDER_NAME}] W_emb remapped in {time.time()-t0:.1f}s, "
              f"nnz={W.nnz}, items-with-neighbours={covered}/{self.id_map.n_tracks}")

        if self.cf_weight > 0:
            t1 = time.time()
            W_cf = build_cf_similarity(urm, self.cf_k, self.shrink,
                                       desc=self.RECOMMENDER_NAME)
            print(f"[{self.RECOMMENDER_NAME}] CF fit in {time.time()-t1:.1f}s, "
                  f"nnz={W_cf.nnz}")
            W = W + self.cf_weight * W_cf

        if self.icm is not None and self.icm_weight > 0:
            t1 = time.time()
            W_cbf = self._load_or_build_icm_sim()
            W_cbf = sparsify_topk(W_cbf, self.k_icm)
            print(f"[{self.RECOMMENDER_NAME}] tag-CBF fit in {time.time()-t1:.1f}s, "
                  f"nnz={W_cbf.nnz}")
            W = W + self.icm_weight * W_cbf

        if self.img_weight > 0:
            t1 = time.time()
            W_img_tower, img_ids = self._load_or_build_img_sim()
            W_img_tower = sparsify_topk(W_img_tower, self.img_k)
            W_img = remap_sim_to_idmap(W_img_tower, img_ids, self.id_map)
            print(f"[{self.RECOMMENDER_NAME}] W_img blended in {time.time()-t1:.1f}s, "
                  f"nnz={W_img.nnz}")
            W = W + self.img_weight * W_img

        self.W = W.tocsr()

    def _load_or_build_emb_sim(self) -> tuple[csr_matrix, np.ndarray]:
        """Load cached Qwen W_emb, else build it from the track tower and cache."""
        if self.emb_sim_path is not None:
            cache = self.emb_sim_path
            if not cache.endswith(".npz"):
                cache = cache + ".npz"
            if Path(cache).exists():
                return load_emb_sim(self.emb_sim_path)
        if self.track_emb_dir is None:
            raise ValueError(
                f"[{self.RECOMMENDER_NAME}] no cached emb_sim_path and no "
                f"track_emb_dir to build W_emb from"
            )
        print(f"[{self.RECOMMENDER_NAME}] no W_emb cache — building from "
              f"{self.track_emb_dir} (cache_k={self.cache_k})…")
        tower_ids, emb = load_track_tower(self.track_emb_dir)
        W_tower = build_emb_item_sim(emb, k=self.cache_k, use_gpu=self.use_gpu)
        if self.emb_sim_path is not None:
            save_emb_sim(W_tower, tower_ids, self.emb_sim_path)
            print(f"[{self.RECOMMENDER_NAME}] cached W_emb → {self.emb_sim_path}")
        return W_tower, tower_ids

    def _icm_hash(self) -> str:
        """Content hash of the per-fold ICM — keys its W_cbf cache fold-safely.

        The tag-CBF similarity is fold-dependent (the interaction-popularity
        feature group is built from the fold's interactions), so its cache is
        keyed by the ICM contents rather than shared across folds.
        """
        import hashlib

        icm = self.icm.tocsr()
        h = hashlib.sha1()
        h.update(np.asarray((*icm.shape, icm.nnz), dtype=np.int64).tobytes())
        h.update(icm.indptr.tobytes())
        h.update(icm.indices.tobytes())
        h.update(np.ascontiguousarray(icm.data).tobytes())
        return h.hexdigest()[:16]

    def _load_or_build_icm_sim(self) -> csr_matrix:
        """Load cached tag-CBF W_cbf (per-fold, hash-keyed), else build + cache.

        W_cbf is already in id_map index space (the ICM is id_map-aligned), so
        no remap is needed — the hash + stored idx_to_track guard against a
        stale/colliding cache.
        """
        cache_path = None
        if self.icm_sim_path is not None:
            stem = self.icm_sim_path
            if stem.endswith(".npz"):
                stem = stem[:-4]
            cache_path = f"{stem}__{self._icm_hash()}.npz"
            if Path(cache_path).exists():
                W, ids = load_emb_sim(cache_path)
                if list(ids) == list(self.id_map.idx_to_track):
                    return W
                print(f"[{self.RECOMMENDER_NAME}] W_cbf cache id mismatch "
                      f"({cache_path}) — rebuilding")

        W_cbf = build_item_cbf_similarity_fast(self.icm, self.icm_cache_k, use_gpu=False)
        if cache_path is not None:
            save_emb_sim(W_cbf,
                         np.asarray(self.id_map.idx_to_track, dtype=object),
                         cache_path)
            print(f"[{self.RECOMMENDER_NAME}] cached W_cbf → {cache_path}")
        return W_cbf

    def _load_or_build_img_sim(self) -> tuple[csr_matrix, np.ndarray]:
        """Load cached SigLIP2 W_img, else build it from the parquets and cache."""
        if self.img_sim_path is not None:
            cache = self.img_sim_path
            if not cache.endswith(".npz"):
                cache = cache + ".npz"
            if Path(cache).exists():
                return load_emb_sim(self.img_sim_path)
        if self.img_parquet_glob is None:
            raise ValueError(
                f"[{self.RECOMMENDER_NAME}] img_weight>0 but no cached "
                f"img_sim_path and no img_parquet_glob to build from"
            )
        print(f"[{self.RECOMMENDER_NAME}] no W_img cache — building from "
              f"{self.img_parquet_glob} (cache_k={self.img_cache_k})…")
        img_ids, emb = load_image_tower(self.img_parquet_glob)
        W_img_tower = build_emb_item_sim(emb, k=self.img_cache_k, use_gpu=self.use_gpu)
        if self.img_sim_path is not None:
            save_emb_sim(W_img_tower, img_ids, self.img_sim_path)
            print(f"[{self.RECOMMENDER_NAME}] cached W_img → {self.img_sim_path}")
        return W_img_tower, img_ids

    def _score_session_profile(self, profile: csr_matrix) -> np.ndarray:
        return np.asarray(profile.dot(self.W).todense()).flatten()

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st.update({
            "emb_sim_path": self.emb_sim_path, "track_emb_dir": self.track_emb_dir,
            "model_size": self.model_size, "k": self.k, "cache_k": self.cache_k,
            "cf_weight": self.cf_weight, "cf_k": self.cf_k, "shrink": self.shrink,
            "icm_weight": self.icm_weight, "k_icm": self.k_icm,
            "icm_sim_path": self.icm_sim_path, "icm_cache_k": self.icm_cache_k,
            "img_sim_path": self.img_sim_path,
            "img_parquet_glob": self.img_parquet_glob,
            "img_weight": self.img_weight, "img_k": self.img_k,
            "img_cache_k": self.img_cache_k,
            "use_gpu": self.use_gpu, "W": self.W,
        })
        return st

    def _set_model_state(self, state: dict) -> None:
        super()._set_model_state(state)
        self.emb_sim_path = state.get("emb_sim_path")
        self.track_emb_dir = state.get("track_emb_dir")
        self.model_size = state.get("model_size", "qwen3_8b")
        self.k = state.get("k", 150)
        self.cache_k = state.get("cache_k", 500)
        self.cf_weight = state.get("cf_weight", 0.0)
        self.cf_k = state.get("cf_k", 150)
        self.shrink = state.get("shrink", 10.0)
        self.icm_weight = state.get("icm_weight", 0.0)
        self.k_icm = state.get("k_icm", 150)
        self.icm_sim_path = state.get("icm_sim_path")
        self.icm_cache_k = state.get("icm_cache_k", 500)
        self.img_sim_path = state.get("img_sim_path")
        self.img_parquet_glob = state.get("img_parquet_glob")
        self.img_weight = state.get("img_weight", 0.0)
        self.img_k = state.get("img_k", 150)
        self.img_cache_k = state.get("img_cache_k", 500)
        self.use_gpu = state.get("use_gpu", True)
        self.W = state.get("W")
