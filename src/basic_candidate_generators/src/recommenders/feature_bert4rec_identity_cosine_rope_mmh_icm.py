"""Identity + cosine + RoPE + MMH (3 dense modalities) + ICM (4th sparse-derived modality).

Adds ICM (Item Content Matrix) — artist / album / tag / release_decade /
popularity-bin / duration-bin features — as a fourth modality on top of the
mmh winner. ICM is intrinsically sparse and high-dim (~20K cols of one-hots),
so it's TruncatedSVD-compressed to a fixed dense width before joining the
per-modality stack.

Hypothesis:
  - The dense modalities we already use (Qwen3 / CF-BPR / CLAP) carry semantic
    similarity (text / behavioural / audio). They miss the *categorical* axis:
    "same artist", "same album", "same decade", "popular vs niche".
  - ICM gives that axis directly. Cold tracks have ICM features too (they're
    derived from track_metadata, not interactions), so this should help the
    biggest remaining lever — the cold-target gap (warm r@200 = 0.525 vs
    cold r@200 = 0.398 on mmh).

Pipeline:
  1. Build ICM via `interactions.build_icm` (artist/album/tag/decade/popularity/duration).
  2. L2-normalise rows of the sparse ICM (tracks with more active features
     don't dominate the SVD).
  3. TruncatedSVD compress to `icm_compressed_dim` (default 256) — sklearn's
     TruncatedSVD works directly on CSR.
  4. Concat compressed ICM as a 4th modality (with its own per-modality
     L2-norm + Linear + PCA-init in the model).
"""

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
    """MMH + ICM as a 4th modality (TruncatedSVD-compressed)."""

    RECOMMENDER_NAME = "FeatureBert4RecIdentityCosineRoPEMMHICM"

    def __init__(
        self,
        *args: Any,
        icm_compressed_dim: int = 256,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.icm_compressed_dim = int(icm_compressed_dim)
        # Need track_metadata at _fit_model time to build ICM; captured in fit().
        self._track_metadata_cache: pl.DataFrame | None = None

    # ------------------------------------------------------------------
    # fit hook — capture track_metadata for ICM construction
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Feature matrix: embeddings + ICM compressed
    # ------------------------------------------------------------------

    def _build_modality_feature_matrix(self) -> tuple[np.ndarray, list[int]]:
        # Load the 3 dense embedding modalities as in the mmh parent.
        full_matrix, modality_dims = _build_feature_matrix_per_modality(
            self.feature_emb_paths, self.feature_modalities, self.id_map
        )

        # Build raw ICM (sparse CSR, (n_tracks, ~20K)) using all training plays
        # so the play-count bin feature is populated.
        assert self._track_metadata_cache is not None
        assert self._train_long is not None
        icm_sparse: csr_matrix = build_icm(
            self._track_metadata_cache, self.id_map, interactions=self._train_long
        )
        print(f"[{self.RECOMMENDER_NAME}] raw ICM: {icm_sparse.shape}, nnz={icm_sparse.nnz}")

        # L2-norm rows of the sparse ICM. Tracks with many active categorical
        # features (lots of tags, multiple artists/albums) get their contribution
        # rescaled so they don't dominate the SVD components.
        icm_normed = sk_normalize(icm_sparse, norm="l2", axis=1)

        # TruncatedSVD compress. n_components must be < n_features.
        n_components = min(self.icm_compressed_dim, icm_sparse.shape[1] - 1)
        if n_components < self.icm_compressed_dim:
            print(f"  ICM has only {icm_sparse.shape[1]} features; SVD capped to {n_components}")
        print(f"[{self.RECOMMENDER_NAME}] TruncatedSVD({n_components}) on L2-normed ICM...")
        svd = TruncatedSVD(n_components=n_components, random_state=0, algorithm="randomized")
        icm_dense = svd.fit_transform(icm_normed).astype(np.float32)
        explained = float(svd.explained_variance_ratio_.sum())
        print(f"  ICM SVD explained variance: {explained:.3f}")

        # Concat ICM as a new modality.
        full_with_icm = np.concatenate([full_matrix, icm_dense], axis=1)
        dims_with_icm = modality_dims + [icm_dense.shape[1]]
        print(f"  modality_dims after ICM: {dims_with_icm}, total = {full_with_icm.shape[1]}")
        return full_with_icm, dims_with_icm

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st["icm_compressed_dim"] = self.icm_compressed_dim
        return st

    def _set_model_state(self, state: dict) -> None:
        super()._set_model_state(state)
        self.icm_compressed_dim = state.get("icm_compressed_dim", 256)
