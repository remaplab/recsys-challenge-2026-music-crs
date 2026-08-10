from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize as sk_normalize

from .feature_bert4rec_identity_cosine_rope_mmh import (
    FeatureBert4RecIdentityCosineRoPEMMHRecommender,
    _build_feature_matrix_per_modality,
)
from .interactions import build_icm

class FeatureBert4RecIdentityCosineRoPEMMHICMRecommender(FeatureBert4RecIdentityCosineRoPEMMHRecommender):

    RECOMMENDER_NAME = "FeatureBert4RecIdentityCosineRoPEMMHICM"

    def __init__(
        self,
        *args: Any,
        icm_compressed_dim: int = 256,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.icm_compressed_dim = int(icm_compressed_dim)

        self._track_metadata_cache: pl.DataFrame | None = None

    def fit(self, train_df: pl.DataFrame, track_metadata: pl.DataFrame | None = None,
             **kwargs: Any) -> None:
        if track_metadata is None:
            raise ValueError(
                f"[{self.RECOMMENDER_NAME}] requires track_metadata to build ICM "
                "(artist/album/tag/decade/popularity/duration). Pass it via the launcher's "
                "configs.feature_bert4rec.yaml `track_metadata_path` or `track_metadata_paths`."
            )
        self._track_metadata_cache = track_metadata
        super().fit(train_df, track_metadata=track_metadata, **kwargs)

    def _build_modality_feature_matrix(self) -> tuple[np.ndarray, list[int]]:

        full_matrix, modality_dims = _build_feature_matrix_per_modality(
            self.feature_emb_paths, self.feature_modalities, self.id_map
        )

        assert self._track_metadata_cache is not None
        assert self._train_long is not None
        icm_sparse: csr_matrix = build_icm(
            self._track_metadata_cache, self.id_map, interactions=self._train_long
        )
        print(f"[{self.RECOMMENDER_NAME}] raw ICM: {icm_sparse.shape}, nnz={icm_sparse.nnz}")

        icm_normed = sk_normalize(icm_sparse, norm="l2", axis=1)

        n_components = min(self.icm_compressed_dim, icm_sparse.shape[1] - 1)
        if n_components < self.icm_compressed_dim:
            print(f"  ICM has only {icm_sparse.shape[1]} features; SVD capped to {n_components}")
        print(f"[{self.RECOMMENDER_NAME}] TruncatedSVD({n_components}) on L2-normed ICM...")
        svd = TruncatedSVD(n_components=n_components, random_state=0, algorithm="randomized")
        icm_dense = svd.fit_transform(icm_normed).astype(np.float32)
        explained = float(svd.explained_variance_ratio_.sum())
        print(f"  ICM SVD explained variance: {explained:.3f}")

        full_with_icm = np.concatenate([full_matrix, icm_dense], axis=1)
        dims_with_icm = modality_dims + [icm_dense.shape[1]]
        print(f"  modality_dims after ICM: {dims_with_icm}, total = {full_with_icm.shape[1]}")
        return full_with_icm, dims_with_icm

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st["icm_compressed_dim"] = self.icm_compressed_dim
        return st

    def _set_model_state(self, state: dict) -> None:
        super()._set_model_state(state)
        self.icm_compressed_dim = state.get("icm_compressed_dim", 256)
