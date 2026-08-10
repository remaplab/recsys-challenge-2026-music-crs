from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # src/heuristic — package root (imports, exp/ outputs)
REPO_ROOT = ROOT.parents[1]                   # repo root — shared data + model caches live here
DATA_ROOT = REPO_ROOT / "data/talkpl-ai"

BLIND_A_PATH = DATA_ROOT / "TalkPlayData-Challenge-Blind-A/data/test-00000-of-00001.parquet"
TRAIN_PATH = DATA_ROOT / "TalkPlayData-Challenge-Dataset/data/train-00000-of-00001.parquet"
TRACK_METADATA_PATH = DATA_ROOT / "TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet"

BLIND_A_INFERENCE_DIR = ROOT / "exp/inference/blind_a"
GAMBLING_DIR = BLIND_A_INFERENCE_DIR / "gambling"
HEURISTICS_DIR = BLIND_A_INFERENCE_DIR / "heuristics"
TEXT_RETRIEVAL_ALL_DIR = BLIND_A_INFERENCE_DIR / "text_retrieval_all"

GAMBLING_PURE_RESPONSE11_PATH = GAMBLING_DIR / "gambling_prediction_response11.json"
GAMBLING_QWEN_FIRST_TURN_PATH = GAMBLING_DIR / "gambling_prediction_qwen_first_turn_response11.json"
GAMBLING_TFIDF_FIRST_TURN_PATH = GAMBLING_DIR / "gambling_prediction_tfidf_first_turn_response11.json"
GAMBLING_BM25_FIRST_TURN_PATH = GAMBLING_DIR / "gambling_prediction_bm25_first_turn_response11.json"
GAMBLING_SPLADE_FIRST_TURN_PATH = GAMBLING_DIR / "gambling_prediction_splade_first_turn_response11.json"
GAMBLING_QWEN_ALL_FALLBACKS_PATH = GAMBLING_DIR / "gambling_prediction_qwen_all_fallbacks_response11.json"

HEURISTIC_ONLY_BLIND_A_PATH = HEURISTICS_DIR / "heuristic_v2_blindA_submission.json"

RETRIEVAL_TOWERS_DIR = REPO_ROOT / "models/retrieval_text_towers"
QWEN_TRACK_CACHE_DIR = RETRIEVAL_TOWERS_DIR / "Qwen__Qwen3-Embedding-0.6B/dense_tracks_len256_poollast"
SPLADE_TRACK_CACHE_DIR = (
    RETRIEVAL_TOWERS_DIR / "naver__splade-cocondenser-ensembledistil/splade_tracks_len256_top128"
)
