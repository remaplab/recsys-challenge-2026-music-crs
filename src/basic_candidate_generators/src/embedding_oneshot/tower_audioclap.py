"""TowerAudioClap — query8B × LAION-CLAP audio two-tower."""
from __future__ import annotations

from .column_modality_tower import ColumnModalityTowerCG


class TowerAudioClap(ColumnModalityTowerCG):
    RECOMMENDER_NAME = "TowerAudioClap"
    TOWER = "D"
    COLUMN = "audio-laion_clap"
