"""End-to-end recall + heatmap evaluation for one encoder.

──────────────────────────────────────────────────────────────────────────────
WHAT CHANGED FOR THE QWEN3-NATIVE PIPELINE
──────────────────────────────────────────────────────────────────────────────
1. New folds `val` and `test` evaluate on the Blind-A-matched split
   (scripts/01b_rebuild_split.py). Their sessions come from the organizer TRAIN
   conversation parquet, so their query embeddings live under the "train" cache
   key — we load that, then SUBSET to exactly the pinned predict-turn rows for
   the fold (one per session). `dev` and `blind_a` behave as before.

2. Track-tower routing sends the Qwen3-native encoders to OUR Qwen3-native tower
   (models/track_tower_qwen_cache, 1024-d):
       qwen3_native_frozen, qwen3_lora__*   → metadata-qwen3-native (NEW)
       qwen3_frozen, keyword_*_qwen3        → organizer metadata-qwen3 (legacy)
       bert_*, sbert_*, modernbert_*        → per-backbone text tower (768-d)

3. Reports are written to models/eval_results/<encoder>/<fold>/recall.json so
   scripts/07_compare_all.py --split val (or test) picks them up unchanged.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emblib.data.split import (
    load_blinda_split, turn_pairs_for_fold, select_pair_indices,
)
from emblib.evaluation.recall import (
    DEFAULT_KS, compute_topk, compute_recall, save_heatmaps, save_report,
)
from emblib.tracks.organizer_track_loader import load_track_tower
from emblib.tracks.qwen_track_loader import load_qwen_track_tower, QWEN_MODALITY
from emblib.tracks.text_track_loader import (
    load_text_track_tower, get_modality_name as get_text_modality,
)


DATA = Path("./data/talkpl-ai")
TOWER_CACHE = Path("./models/track_tower_cache")
QWEN_TOWER_CACHE = Path("./models/track_tower_qwen_cache")
TEXT_TOWER_CACHE = Path("./models/track_tower_text_cache")
QUERY_CACHE = Path("./models/query_emb_cache")
EVAL_OUT = Path("./models/eval_results")
SPLITS_PATH = Path("./models/splits/train_val_test_blinda_matched.parquet")

LEGACY_QWEN3_MODALITY = "metadata-qwen3_embedding_0.6b"

# Which cached conversation split each eval fold reads its query embeddings from.
# val/test sessions are organizer-train-derived, so their queries are encoded
# under the "train" key (and then subset to predict-turn rows).
FOLD_TO_CONV_SPLIT = {"val": "train", "test": "train", "dev": "dev", "blind_a": "blind_a"}


def load_tower_for_encoder(encoder_name: str):
    """Choose the track tower based on the encoder cache name's prefix.

    Returns (track_emb, track_mask, id_to_idx).
    """
    # *_native_frozen and *_proj_* HF encoders match against the backbone's
    # own native-space text tower.
    if encoder_name.startswith("bert_native_frozen") or encoder_name.startswith("bert_proj_"):
        backbone = "bert"
    elif encoder_name.startswith("sbert_native_frozen") or encoder_name.startswith("sbert_proj_"):
        backbone = "sbert"
    elif encoder_name.startswith("modernbert_native_frozen") or encoder_name.startswith("modernbert_proj_"):
        backbone = "modernbert"
    elif encoder_name.startswith("qwen3_native_frozen") or encoder_name.startswith("qwen3_lora"):
        # NEW: query mirror of OUR Qwen3-native track tower.
        print(f"[tower] {encoder_name!r} → Qwen3-native metadata tower (1024-d)")
        tower = load_qwen_track_tower(QWEN_TOWER_CACHE, QWEN_MODALITY)
        return (
            tower.embeddings[QWEN_MODALITY],
            tower.masks[QWEN_MODALITY],
            tower.id_to_idx,
        )
    else:
        # Legacy: qwen3_frozen, keyword_*_qwen3 output 1024-d organizer Qwen3 vectors.
        print(f"[tower] {encoder_name!r} → organizer Qwen3 metadata tower (1024-d, legacy)")
        tower = load_track_tower(
            shards_dir=DATA / "TalkPlayData-Challenge-Track-Embeddings/data",
            modalities=[LEGACY_QWEN3_MODALITY], cache_dir=TOWER_CACHE,
        )
        return (
            tower.embeddings[LEGACY_QWEN3_MODALITY],
            tower.masks[LEGACY_QWEN3_MODALITY],
            tower.id_to_idx,
        )

    print(f"[tower] {encoder_name!r} → {backbone!r} text track tower (768-d)")
    tower = load_text_track_tower(backbone, cache_dir=TEXT_TOWER_CACHE)
    modality = get_text_modality(backbone)
    return tower.embeddings[modality], tower.masks[modality], tower.id_to_idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", required=True,
                   help="Cache name under models/query_emb_cache/, e.g. "
                        "qwen3_native_frozen, qwen3_lora__qwen3_lora_routing, "
                        "bert_native_frozen, keyword_modernbert_qwen3.")
    p.add_argument("--split", default="val",
                   choices=["val", "test", "dev", "blind_a"])
    p.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    p.add_argument("--no-mask-played", action="store_true")
    args = p.parse_args()

    conv_split = FOLD_TO_CONV_SPLIT[args.split]
    qcache = QUERY_CACHE / args.encoder
    q_emb_path = qcache / f"{conv_split}.npy"
    meta_path = qcache / f"{conv_split}_meta.parquet"
    if not q_emb_path.exists():
        raise FileNotFoundError(
            f"No cached queries at {q_emb_path}. "
            f"Run 03_encode_queries.py --encoder {args.encoder} --splits {conv_split} first."
        )
    q_emb = np.load(q_emb_path)
    meta_df = pl.read_parquet(meta_path)
    print(f"queries: {q_emb.shape}, meta rows: {meta_df.shape[0]}  "
          f"(fold={args.split}, conv_split={conv_split})")

    # ── For val/test: subset to the pinned predict-turn rows of the fold ──
    if args.split in ("val", "test"):
        if not SPLITS_PATH.exists():
            raise FileNotFoundError(
                f"{SPLITS_PATH} not found. Run scripts/01b_rebuild_split.py first."
            )
        split_df = load_blinda_split(SPLITS_PATH)
        pairs = turn_pairs_for_fold(split_df, args.split)
        idx = select_pair_indices(meta_df, pairs)
        if not idx:
            raise ValueError(
                f"No rows in {meta_path} match the {args.split} (session_id, turn) pairs. "
                f"Check that 03_encode_queries.py was run on the 'train' split and that "
                f"predict_turn_number aligns with meta turn_number."
            )
        idx_arr = np.asarray(idx, dtype=np.int64)
        q_emb = q_emb[idx_arr]
        meta_df = (
            meta_df.with_row_index(name="__row")
            .filter(pl.col("__row").is_in(idx))
            .drop("__row")
        )
        print(f"  subset to {q_emb.shape[0]} predict-turn rows for fold {args.split!r}")

    track_emb, track_mask, id_to_idx = load_tower_for_encoder(args.encoder)
    print(f"track tower: {track_emb.shape}  mask coverage {track_mask.mean():.3f}")

    if track_emb.shape[1] != q_emb.shape[1]:
        raise ValueError(
            f"DIM MISMATCH: queries D={q_emb.shape[1]} vs track tower D={track_emb.shape[1]}.\n"
            f"  encoder cache: {args.encoder}\n"
            f"  This usually means the encoder was trained/encoded against a different "
            f"tower than the one being loaded now. Re-encode with 03_encode_queries.py "
            f"and (for LoRA) re-train with 05_train_encoder.py."
        )

    max_k = max(args.ks)
    topk = compute_topk(q_emb, track_emb, track_mask, meta_df, id_to_idx,
                        max_k=max_k, mask_played=not args.no_mask_played)

    report = compute_recall(topk, meta_df, id_to_idx, ks=tuple(args.ks))
    out_dir = EVAL_OUT / args.encoder / args.split
    save_report(report, out_dir, encoder_name=args.encoder)
    save_heatmaps(report, out_dir, encoder_name=args.encoder)


if __name__ == "__main__":
    main()