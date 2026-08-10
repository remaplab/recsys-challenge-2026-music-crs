"""TowerC — query8B × CF-BPR two-tower (parent ensemble's tower C; cold-user CF)."""
from __future__ import annotations

from .column_modality_tower import ColumnModalityTowerCG


class TowerC(ColumnModalityTowerCG):
    RECOMMENDER_NAME = "TowerC"
    TOWER = "C"
    COLUMN = "cf-bpr"
