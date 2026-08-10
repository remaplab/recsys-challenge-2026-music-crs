"""TowerA — query8B × track-8B-text two-tower (the parent ensemble's tower A)."""
from __future__ import annotations

from typing import Any

import numpy as np

from embedding_based.emb_matrix import load_track_tower

from .single_tower_base import SingleTowerCG


class TowerA(SingleTowerCG):
    RECOMMENDER_NAME = "TowerA"
    TOWER = "A"

    def __init__(self, track_emb_dir: str | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        if track_emb_dir is None:
            raise ValueError(f"[{self.RECOMMENDER_NAME}] track_emb_dir required")
        self.track_emb_dir = track_emb_dir

    def _load_modality(self) -> tuple[np.ndarray, np.ndarray]:
        return load_track_tower(self.track_emb_dir)

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st["track_emb_dir"] = self.track_emb_dir
        return st
