"""Fine-tune any encoder against the matching frozen track tower.

──────────────────────────────────────────────────────────────────────────────
WHAT CHANGED FOR THE QWEN3-NATIVE PIPELINE
──────────────────────────────────────────────────────────────────────────────
1. Track-tower routing for the Qwen3 family now points at OUR Qwen3-native
   tower (models/track_tower_qwen_cache, modality "metadata-qwen3-native",
   1024-d) built by scripts/04b_encode_qwen_tracks.py — the exact mirror of the
   query encoder. The BERT/SBERT/ModernBERT projected families are unchanged.

2. The split is the Blind-A-matched train/val/test parquet produced by
   scripts/01b_rebuild_split.py (columns: session_id, split, predict_turn_number).
       • TRAIN sessions contribute ALL their (session, turn) rows.
       • VAL sessions are pinned to exactly ONE predict-turn row each (the GT
         turn that makes the fold match Blind A on K). We therefore pre-filter
         the val meta to those (session_id, turn_number) pairs.

3. Query meta + GT bookkeeping come from the qwen3_native_frozen cache
   (models/query_emb_cache/qwen3_native_frozen/train_meta.parquet), and hard
   negatives are the ones mined in Qwen3-native space (scripts/02). Because the
   hard-neg array is aligned row-for-row with that FULL train meta, we pass the
   full meta (LoRATrainDataset internally keeps only train sessions).

──────────────────────────────────────────────────────────────────────────────
TRACK-TOWER ROUTING
──────────────────────────────────────────────────────────────────────────────
    qwen3_lora_*           → metadata-qwen3-native (our tower, 1024-d)
    bert_proj_*            → BERT-encoded text tower (768-d)
    sbert_proj_*           → SBERT-encoded text tower (768-d)
    modernbert_proj_*      → ModernBERT-encoded text tower (768-d)

Hard negatives mined in Qwen3-native space are reused unchanged: the negative
INDICES address the canonical track-id ordering shared across all towers.

Examples:
  python scripts/05_train_encoder.py --exp-name qwen3_lora_routing    --kind qwen3_lora_routing
  python scripts/05_train_encoder.py --exp-name qwen3_lora_no_routing --kind qwen3_lora_no_routing
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emblib.data.split import (
    load_blinda_split, sessions_for_fold, turn_pairs_for_fold, filter_meta_to_pairs,
)
from emblib.data.user_features import load_user_features
from emblib.tracks.qwen_track_loader import load_qwen_track_tower, QWEN_MODALITY
from emblib.tracks.text_track_loader import (
    load_text_track_tower, get_modality_name as get_text_modality,
)
from emblib.training.lora_trainer import (
    LoRATrainConfig, LoRATrainDataset, train_lora_encoder,
)


DATA = Path("./data/talkpl-ai")
USER_CACHE = Path("./models/user_features_cache")
QWEN_TOWER_CACHE = Path("./models/track_tower_qwen_cache")
TEXT_TOWER_CACHE = Path("./models/track_tower_text_cache")
SPLITS_PATH = Path("./models/splits/train_val_test_blinda_matched.parquet")
HARD_NEGS = Path("./models/hard_negatives/train_negs.npy")
TRAIN_CONV = DATA / "TalkPlayData-Challenge-Dataset/data/train-00000-of-00001.parquet"
TRACK_META = DATA / "TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet"

# Query meta + GT bookkeeping reused from the frozen Qwen3 query cache.
QUERY_META = Path("./models/query_emb_cache/qwen3_native_frozen/train_meta.parquet")


def _backbone_from_kind(kind: str) -> str:
    """Pick the text-tower backbone name from a *_proj_* kind string."""
    if kind.startswith("bert_proj_"):
        return "bert"
    if kind.startswith("sbert_proj_"):
        return "sbert"
    if kind.startswith("modernbert_proj_"):
        return "modernbert"
    raise ValueError(f"kind {kind!r} is not a *_proj_* kind")


def load_tower_for_kind(kind: str):
    """Return (track_emb, track_mask, id_to_idx) for the encoder kind being trained.

    The kind string also dictates the output dimension of the encoder, so the
    matching tower must have the same dim. We assert this at call sites by
    checking encoder.output_dim == track_emb.shape[1].
    """
    if kind.startswith("qwen3_"):
        print(f"[tower] {kind!r} → Qwen3-native metadata tower (1024-d)")
        tower = load_qwen_track_tower(QWEN_TOWER_CACHE, QWEN_MODALITY)
        return (
            tower.embeddings[QWEN_MODALITY],
            tower.masks[QWEN_MODALITY],
            tower.id_to_idx,
        )

    if kind.startswith(("bert_proj_", "sbert_proj_", "modernbert_proj_")):
        backbone = _backbone_from_kind(kind)
        print(f"[tower] {kind!r} → {backbone!r} text track tower (768-d)")
        tower = load_text_track_tower(backbone, cache_dir=TEXT_TOWER_CACHE)
        modality = get_text_modality(backbone)
        return (
            tower.embeddings[modality],
            tower.masks[modality],
            tower.id_to_idx,
        )

    raise ValueError(f"unknown kind {kind!r}")


def build_encoder_for_training(kind: str):
    if kind in ("qwen3_lora_routing", "qwen3_lora_no_routing"):
        from emblib.encoders.qwen3_lora import LoRAQueryEncoderConfig, Qwen3LoRAQueryEncoder
        return Qwen3LoRAQueryEncoder(LoRAQueryEncoderConfig(
            use_routing=(kind == "qwen3_lora_routing"),
        ))
    if kind.startswith(("bert_proj_", "sbert_proj_", "modernbert_proj_")):
        from scripts._encoders_factory import _build_untrained_hf_proj
        return _build_untrained_hf_proj(kind)
    raise ValueError(f"unknown training kind {kind!r}")


def load_track_lookup(track_meta_path: Path) -> dict[str, dict]:
    """track_id -> metadata-row dict for v2's [PLAYED] line resolution."""
    print(f"Loading track lookup from {track_meta_path}")
    md = pl.read_parquet(track_meta_path)
    lookup = {row["track_id"]: row for row in md.to_dicts()}
    print(f"  {len(lookup)} tracks indexed")
    return lookup


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", required=True)
    p.add_argument("--kind", required=True,
                   help="qwen3_lora_routing | qwen3_lora_no_routing | "
                        "{bert,sbert,modernbert}_proj_{frozen,lora}_{routing,no_routing}")
    p.add_argument("--max-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n-hard-negs", type=int, default=16)
    p.add_argument("--patience", type=int, default=1)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.02)
    p.add_argument("--eval-subsample", type=int, default=2000)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Blind-A-matched split ────────────────────────────────────────────
    if not SPLITS_PATH.exists():
        raise FileNotFoundError(
            f"{SPLITS_PATH} not found. Run scripts/01b_rebuild_split.py first."
        )
    split_df = load_blinda_split(SPLITS_PATH)
    train_sessions = sessions_for_fold(split_df, "train")
    val_pairs = turn_pairs_for_fold(split_df, "val")        # one (sid, turn) per val session
    val_sessions = sessions_for_fold(split_df, "val")
    print(f"  train sessions: {len(train_sessions)}, "
          f"val sessions: {len(val_sessions)} (val predict-turn rows: {len(val_pairs)})")

    # ── pick the right track tower for this encoder kind ─────────────────
    track_emb, track_mask, id_to_idx = load_tower_for_kind(args.kind)

    users = load_user_features(
        user_meta_path=DATA / "TalkPlayData-Challenge-User-Metadata/data/all_users-00000-of-00001.parquet",
        user_emb_train_path=DATA / "TalkPlayData-Challenge-User-Embeddings/data/train-00000-of-00001.parquet",
        user_emb_warm_path=DATA / "TalkPlayData-Challenge-User-Embeddings/data/test_warm-00000-of-00001.parquet",
        user_emb_cold_path=DATA / "TalkPlayData-Challenge-User-Embeddings/data/test_cold-00000-of-00001.parquet",
        cache_dir=USER_CACHE,
    )

    # Track lookup is shared by both train and val datasets; loaded once.
    track_lookup = load_track_lookup(TRACK_META)

    assert QUERY_META.exists(), (
        f"{QUERY_META} not found. Run "
        "`python scripts/03_encode_queries.py --encoder qwen3_native_frozen --splits train` "
        "first — we reuse its meta parquet for GT track ids and session bookkeeping."
    )
    train_meta = pl.read_parquet(QUERY_META)

    # VAL meta = only the pinned predict-turn rows (one per val session). This is
    # what makes the in-training recall match the Blind-A-matched K distribution.
    val_meta = filter_meta_to_pairs(train_meta, val_pairs)
    print(f"  val meta rows after predict-turn filter: {val_meta.shape[0]}")

    print("\nBuilding train dataset (all turns of train sessions)...")
    train_ds = LoRATrainDataset(
        meta_df=train_meta, conv_parquet_path=TRAIN_CONV,
        id_to_idx=id_to_idx, tower_mask=track_mask,
        users=users, keep_session_ids=train_sessions,
        hard_negs_path=HARD_NEGS,        # aligned row-for-row with the FULL train_meta
        track_lookup=track_lookup,
    )
    print("Building val dataset (pinned predict-turn rows, no hard negs)...")
    val_ds = LoRATrainDataset(
        meta_df=val_meta, conv_parquet_path=TRAIN_CONV,
        id_to_idx=id_to_idx, tower_mask=track_mask,
        users=users, keep_session_ids=val_sessions, hard_negs_path=None,
        track_lookup=track_lookup,
    )

    encoder = build_encoder_for_training(args.kind).to(device)
    print(f"Encoder: {args.kind}  trainable params: {encoder.n_trainable():,}")

    # Sanity: encoder output dim must match the track tower dim, otherwise
    # the dot-product scores below are nonsense.
    if encoder.output_dim != track_emb.shape[1]:
        raise ValueError(
            f"DIM MISMATCH: encoder outputs {encoder.output_dim}-d but track "
            f"tower is {track_emb.shape[1]}-d. Check that the right tower "
            f"is being loaded for kind={args.kind!r}."
        )
    print(f"  output_dim={encoder.output_dim}  matches  track tower D={track_emb.shape[1]}  ✓")

    cfg = LoRATrainConfig(
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        max_epochs=args.max_epochs,
        patience=args.patience,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        temperature=args.temperature,
        n_hard_negatives=args.n_hard_negs,
        eval_subsample=args.eval_subsample,
    )
    out_dir = Path("./models/checkpoints") / args.exp_name
    train_lora_encoder(encoder, train_ds, val_ds, track_emb, track_mask,
                       cfg, device, out_dir)

    print(f"\nDone. Adapter saved to {out_dir}")
    inf_kind = "qwen3_lora" if args.kind.startswith("qwen3_") else args.kind
    print(f"\nNext:")
    print(f"  python scripts/03_encode_queries.py --encoder {inf_kind} --adapter {out_dir} --splits train")
    cache_tag = f"{inf_kind}__{out_dir.name}"
    print(f"  python scripts/06_evaluate.py --encoder {cache_tag} --split val")


if __name__ == "__main__":
    main()