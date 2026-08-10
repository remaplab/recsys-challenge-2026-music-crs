"""Central knobs for the Qwen3 query/track encoding (adapted from the
submission `config.py` to the embedding-lab layout).

Both queries and track-metadata are encoded with the SAME Qwen3 model and the
SAME last-token pooling, so dot products are cosine similarity in one space:

    query : instruction = `query_instruction` (default "catalog"), len 512
    track : instruction = `track_instruction` (default "none"),    len 256

Everything is rooted at the embedding-lab project directory so paths line up
with the rest of the pipeline (./data, ./models).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# src/qwen/config.py -> parents[2] == embedding-lab/
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class QwenConfig:
    # ── model ────────────────────────────────────────────────────────────
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    query_instruction: str = "catalog"   # key into QWEN_INSTRUCTIONS
    track_instruction: str = "none"      # tracks are encoded with no prefix
    query_max_length: int = 512
    track_max_length: int = 256
    query_batch_size: int = 16
    track_batch_size: int = 16

    device: str = "auto"                 # "auto" | "cuda" | "cpu" | "mps"
    dtype: str = "auto"                  # "auto" | "float16" | "bfloat16" | "float32"
    # Set local_files_only=True only if the model is already on disk (offline).
    local_files_only: bool = False
    trust_remote_code: bool = False

    # ── data ─────────────────────────────────────────────────────────────
    track_meta_path: Path = (
        ROOT / "data/talkpl-ai/TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet"
    )
    # Canonical track ordering (matches organizer tower + mined hard-negative indices)
    embedding_shards_dir: Path = ROOT / "data/talkpl-ai/TalkPlayData-Challenge-Track-Embeddings/data"

    # ── outputs ──────────────────────────────────────────────────────────
    # TrackTower-compatible cache (track_ids.npy + <modality>.npy + <modality>__mask.npy)
    track_tower_cache_dir: Path = ROOT / "models/track_tower_qwen_cache"
    modality_name: str = "metadata-qwen3-native"