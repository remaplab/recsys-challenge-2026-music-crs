"""TowerB — query8B × SigLIP2 image two-tower (parent ensemble's tower B)."""
from __future__ import annotations

from .column_modality_tower import ColumnModalityTowerCG


class TowerB(ColumnModalityTowerCG):
    RECOMMENDER_NAME = "TowerB"
    TOWER = "B"
    COLUMN = "image-siglip2"
