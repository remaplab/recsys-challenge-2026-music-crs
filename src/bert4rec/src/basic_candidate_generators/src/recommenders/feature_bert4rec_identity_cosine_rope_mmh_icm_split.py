from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize as sk_normalize

from .feature_bert4rec_identity_cosine_rope_mmh import (
    _build_feature_matrix_per_modality,
)
from .feature_bert4rec_identity_cosine_rope_mmh_icm import (
    FeatureBert4RecIdentityCosineRoPEMMHICMRecommender,
)
from .interactions import build_icm_blocks

_BLOCK_DIMS: dict[str, int] = {
    "artist": 128,
    "album":  128,
    "tag":    64,
    "decade": 0,
    "popularity": 0,
    "interaction_popularity": 0,
    "duration": 0,
}

class FeatureBert4RecIdentityCosineRoPEMMHICMSplitRecommender(FeatureBert4RecIdentityCosineRoPEMMHICMRecommender):

    RECOMMENDER_NAME = "FeatureBert4RecIdentityCosineRoPEMMHICMSplit"
    DEFAULT_BLOCK_DIMS: dict[str, int] = _BLOCK_DIMS

    def __init__(
        self,
        *args,
        block_dims: dict[str, int] | None = None,
        artist_dim: int | None = None,
        album_dim: int | None = None,
        tag_dim: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.block_dims: dict[str, int] = (
            dict(block_dims) if block_dims is not None else dict(self.DEFAULT_BLOCK_DIMS)
        )

        if artist_dim is not None:
            self.block_dims["artist"] = int(artist_dim)
        if album_dim is not None:
            self.block_dims["album"] = int(album_dim)
        if tag_dim is not None:
            self.block_dims["tag"] = int(tag_dim)

    def _build_modality_feature_matrix(self) -> tuple[np.ndarray, list[int]]:

        full_matrix, modality_dims = _build_feature_matrix_per_modality(
            self.feature_emb_paths, self.feature_modalities, self.id_map
        )

        assert self._track_metadata_cache is not None
        assert self._train_long is not None
        blocks = build_icm_blocks(
            self._track_metadata_cache, self.id_map, interactions=self._train_long
        )
        print(f"[{self.RECOMMENDER_NAME}] ICM blocks: "
              + ", ".join(f"{k}={v.shape[1]}" for k, v in blocks.items()))

        added_chunks: list[np.ndarray] = []
        added_dims: list[int] = []
        added_names: list[str] = []
        for name, target_dim in self.block_dims.items():
            block = blocks[name]
            if block.shape[1] <= 1:

                print(f"  {name}: all-zero block, skipping")
                continue
            added_names.append(name)
            if target_dim == 0 or block.shape[1] <= target_dim + 1:

                dense = sk_normalize(block, norm="l2", axis=1).toarray().astype(np.float32)
                added_chunks.append(dense)
                added_dims.append(dense.shape[1])
                print(f"  {name}: raw dense ({dense.shape[1]}d)")
            else:

                normed = sk_normalize(block, norm="l2", axis=1)
                n_comp = min(target_dim, block.shape[1] - 1)
                svd = TruncatedSVD(n_components=n_comp, random_state=0, algorithm="randomized")
                comp = svd.fit_transform(normed).astype(np.float32)
                explained = float(svd.explained_variance_ratio_.sum())
                added_chunks.append(comp)
                added_dims.append(n_comp)
                print(f"  {name}: SVD({n_comp}), explained={explained:.3f}")

        full_with_icm = np.concatenate([full_matrix] + added_chunks, axis=1)
        dims_with_icm = modality_dims + added_dims

        self._modality_names = list(self.feature_modalities) + [f"icm-{n}" for n in added_names]
        print(f"  modality_dims after ICM split: {dims_with_icm}, total = {full_with_icm.shape[1]}")
        return full_with_icm, dims_with_icm
