"""DenseModalityCG — untrained query->track retrieval over a parquet modality.

`dense_text_{8b,4b,0.6b}` reuse the existing `DenseQueryCG` directly (query and
track live in the SAME Qwen3 dense_tracks/query caches → cosine is meaningful).

`dense_text_comp` instead scores the query against a track-embedding column from
the competition Track-Embeddings parquet (e.g. `metadata-qwen3_embedding_0.6b`).
This subclass swaps only the catalogue source: `load_modality_tower(glob, column)`
in place of `load_track_tower(dir)`. Everything else — query-cache loading, tiled
GEMM scoring, future masking, seen removal, persistence — is inherited verbatim.

NOTE: untrained cosine assumes query and track share an embedding space. The
competition columns are produced by a different encoding pipeline than our query
caches, so this signal may be weak; its RRF weight self-prunes if so. If a
projection is wanted, promote it to a learned tower instead.
"""
from __future__ import annotations

import time
from typing import Any

import polars as pl

from embedding_based.dense_query_cg import DenseQueryCG
from embedding_based.emb_matrix import load_modality_tower


class DenseModalityCG(DenseQueryCG):
    RECOMMENDER_NAME = "DenseModalityCG"

    def __init__(
        self,
        *args: Any,
        track_parquet_glob: str | None = None,
        track_column: str | None = None,
        **kwargs: Any,
    ) -> None:
        # track_emb_dir is unused for this CG; the catalogue comes from the
        # parquet column. Pass a placeholder so the parent's guard is satisfied.
        kwargs.setdefault("track_emb_dir", "__modality__")
        super().__init__(*args, **kwargs)
        if track_parquet_glob is None or track_column is None:
            raise ValueError(
                f"[{self.RECOMMENDER_NAME}] track_parquet_glob and track_column required")
        self.track_parquet_glob = track_parquet_glob
        self.track_column = track_column

    def fit(self, train_df, track_metadata: pl.DataFrame | None = None, **kwargs: Any) -> None:
        t0 = time.time()
        self.track_ids, self.track_emb = load_modality_tower(
            self.track_parquet_glob, self.track_column)
        self.track_to_idx = {t: i for i, t in enumerate(self.track_ids)}
        if track_metadata is not None:
            from embedding_based.dense_query_cg import _build_release_dates
            self.release_dates = _build_release_dates(self.track_ids, track_metadata)
        self._load_query_caches()
        print(f"[{self.RECOMMENDER_NAME}] fit in {time.time()-t0:.1f}s — "
              f"{len(self.track_ids)} tracks ({self.track_column}), "
              f"{len(self.query_key_to_row)} cached queries")

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st.update({"track_parquet_glob": self.track_parquet_glob,
                   "track_column": self.track_column})
        return st

    def _set_model_state(self, state: dict) -> None:
        super()._set_model_state(state)
        self.track_parquet_glob = state.get("track_parquet_glob")
        self.track_column = state.get("track_column")
