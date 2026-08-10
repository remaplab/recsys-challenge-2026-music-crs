"""Canonical paths for the LBO package."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

# Raw competition data
DATA = ROOT / "data" / "talkpl-ai"
TRAIN_PARQUET = DATA / "TalkPlayData-Challenge-Dataset/data/train-00000-of-00001.parquet"
TEST_PARQUET = DATA / "TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet"
BLIND_PARQUET = DATA / "TalkPlayData-Challenge-Blind-A/data/test-00000-of-00001.parquet"
TRACKS_META = DATA / "TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet"

# Cached embeddings
QUERY_EMB_DIR = ROOT / "models" / "query_emb_cache" / "qwen3_frozen"
TRACK_EMB_DIR = ROOT / "models" / "retrieval_text_towers" / "Qwen__Qwen3-Embedding-0.6B" / "dense_tracks_len256_poollast"

# Outputs
MODELS_OUT = ROOT / "models" / "LowerBoundOptimization"
SHIFT_OUT = MODELS_OUT / "distribution_shift"
SHIFT_PLOTS = SHIFT_OUT / "plots"
SHIFT_V2_OUT = MODELS_OUT / "distribution_shift_v2"
SHIFT_V2_PLOTS = SHIFT_V2_OUT / "plots"

# Blind-A n_music_turns PMF — drives augmentation.
# `max_turn` = number of completed (user, music, assistant) cycles BEFORE the
# prediction query. The prediction query itself sits at turn `max_turn + 1`.
# 20/80 blind sessions are pure cold-query (max_turn=0).
BLIND_MUSIC_TURNS_PMF = {0: 20, 1: 15, 2: 10, 3: 5, 4: 8, 5: 9, 6: 8, 7: 5}
