"""mmh_icm + ICM split into separate sub-modalities (artist/album/tag/decade/...).

In mmh_icm, all ICM features (artist + album + tag + decade + popularity bins
+ duration) live in a single bundle that gets one TruncatedSVD compression and
one Linear projection. Splitting them lets the model learn weights per
*semantic axis*: "same artist" might count more than "same tag", and that
distinction is invisible when they share a projection.

Per-group treatment:
  - artist  → TruncatedSVD(128) → modality
  - album   → TruncatedSVD(128) → modality
  - tag     → TruncatedSVD(64)  → modality
  - decade  → use raw (≤ ~10d, no SVD)
  - dataset_popularity_bin    → use raw 5d
  - interaction_popularity_bin→ use raw 5d
  - duration_bin              → use raw 5d

The dense embedding modalities (qwen3/cf/clap) are kept as in mmh_icm so the
parent's behaviour is preserved on top of them.
"""

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


# Per-block target dim after compression. 0 = use raw sparse columns as dense.
# Class-level default; subclasses can override via DEFAULT_BLOCK_DIMS or per-instance
# via the `block_dims` constructor kwarg.
_BLOCK_DIMS: dict[str, int] = {
    "artist": 128,
    "album":  128,
    "tag":    64,
    "decade": 0,                   # ≤ ~10 cols
    "popularity": 0,               # 5 cols
    "interaction_popularity": 0,   # 5 cols
    "duration": 0,                 # 5 cols
}


class FeatureBert4RecIdentityCosineRoPEMMHICMSplitRecommender(FeatureBert4RecIdentityCosineRoPEMMHICMRecommender):
    """mmh_icm with ICM split into per-group sub-modalities."""

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
        # `block_dims` overrides DEFAULT_BLOCK_DIMS if provided.
        self.block_dims: dict[str, int] = (
            dict(block_dims) if block_dims is not None else dict(self.DEFAULT_BLOCK_DIMS)
        )
        # Per-block scalar overrides (used by Optuna search space — three
        # tunable scalars are simpler to express in YAML than a dict).
        if artist_dim is not None:
            self.block_dims["artist"] = int(artist_dim)
        if album_dim is not None:
            self.block_dims["album"] = int(album_dim)
        if tag_dim is not None:
            self.block_dims["tag"] = int(tag_dim)

    def _build_modality_feature_matrix(self) -> tuple[np.ndarray, list[int]]:
        # Start from the 3 dense modalities (qwen3 + cf + clap).
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
        for name, target_dim in self.block_dims.items():
            block = blocks[name]
            if block.shape[1] <= 1:
                # All-zero block (e.g. no data for this feature group) — skip.
                print(f"  {name}: all-zero block, skipping")
                continue
            if target_dim == 0 or block.shape[1] <= target_dim + 1:
                # Use raw dense (also L2-normed per row for consistency).
                dense = sk_normalize(block, norm="l2", axis=1).toarray().astype(np.float32)
                added_chunks.append(dense)
                added_dims.append(dense.shape[1])
                print(f"  {name}: raw dense ({dense.shape[1]}d)")
            else:
                # L2-norm rows then TruncatedSVD-compress.
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
        print(f"  modality_dims after ICM split: {dims_with_icm}, total = {full_with_icm.shape[1]}")
        return full_with_icm, dims_with_icm
