from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from emblib.retrieval.paths import (
    BLIND_A_PATH,
    GAMBLING_PURE_RESPONSE11_PATH,
    GAMBLING_QWEN_ALL_FALLBACKS_PATH,
    QWEN_TRACK_CACHE_DIR,
    TRACK_METADATA_PATH,
    TRAIN_PATH,
)

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BestSubmissionConfig:
    blind_path: Path = BLIND_A_PATH
    train_path: Path = TRAIN_PATH
    track_metadata_path: Path = TRACK_METADATA_PATH
    base_response_path: Path = GAMBLING_PURE_RESPONSE11_PATH
    output_path: Path = GAMBLING_QWEN_ALL_FALLBACKS_PATH
    qwen_track_cache_dir: Path = QWEN_TRACK_CACHE_DIR

    qwen_model: str = "Qwen/Qwen3-Embedding-0.6B"
    qwen_instruction: str = "catalog"
    qwen_query_max_length: int = 512
    qwen_track_max_length: int = 256
    qwen_query_batch_size: int = 16
    qwen_track_batch_size: int = 16
    score_batch_size: int = 64
    top_k: int = 20

    device: str = "auto"
    dtype: str = "auto"
    local_files_only: bool = True
    trust_remote_code: bool = False
    filter_future_releases_for_qwen: bool = True
