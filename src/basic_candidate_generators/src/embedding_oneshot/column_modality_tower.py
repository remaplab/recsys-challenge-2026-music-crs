"""ColumnModalityTowerCG — SingleTowerCG over a Track-Embeddings parquet column.

Shared base for the cross-modal towers (B/C/audioclap), which all read their
frozen track side from one column of the competition Track-Embeddings parquet
(image-siglip2 / cf-bpr / audio-laion_clap). Subclasses fix `COLUMN`; the parquet
glob comes from config. Query side stays Qwen3-8B (handled by SingleTowerCG).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from embedding_based.emb_matrix import load_modality_tower

from .single_tower_base import SingleTowerCG


class ColumnModalityTowerCG(SingleTowerCG):
    COLUMN = ""   # subclass sets the parquet column

    def __init__(self, track_parquet_glob: str | None = None,
                 track_column: str | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        if track_parquet_glob is None:
            raise ValueError(f"[{self.RECOMMENDER_NAME}] track_parquet_glob required")
        self.track_parquet_glob = track_parquet_glob
        self.track_column = track_column or self.COLUMN

    def _load_modality(self) -> tuple[np.ndarray, np.ndarray]:
        return load_modality_tower(self.track_parquet_glob, self.track_column)

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st.update({"track_parquet_glob": self.track_parquet_glob,
                   "track_column": self.track_column})
        return st
