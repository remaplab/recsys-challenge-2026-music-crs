"""Encode the track-metadata tower for one backbone (BERT / SBERT / ModernBERT).

This script is the missing piece that makes the encoder-comparison study
fair. The organizer ships a Qwen3-encoded track tower; using it as the
retrieval target for a BERT-based query encoder forces BERT to learn a
projection INTO Qwen3 space — an asymmetric handicap. With this script,
each backbone gets its own native-space track tower, and downstream
training + evaluation match queries against the matching tower
automatically (see updates in scripts/05_train_encoder.py and
scripts/06_evaluate.py).

Run on a GPU node, once per backbone:

    python scripts/04_encode_tracks.py --backbone bert
    python scripts/04_encode_tracks.py --backbone sbert
    python scripts/04_encode_tracks.py --backbone modernbert

Output: models/track_tower_text_cache/<backbone>/

Each tower has the same track_ids in the same order as the organizer's
track tower, so previously-mined hard-negative indices remain valid
across all towers.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emblib.tracks.text_track_loader import BACKBONE_MODELS, encode_text_track_tower


DATA = Path("./data/talkpl-ai")
TRACK_META = DATA / "TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet"
EMBED_SHARDS = DATA / "TalkPlayData-Challenge-Track-Embeddings/data"
CACHE = Path("./models/track_tower_text_cache")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True, choices=list(BACKBONE_MODELS),
                   help="Which HF model to encode the track-metadata text with.")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Encoding batch size. Lower if you run out of GPU memory.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if not torch.cuda.is_available():
        print("[warn] no GPU detected — encoding 47k tracks on CPU will be slow.")

    encode_text_track_tower(
        backbone=args.backbone,
        track_meta_path=TRACK_META,
        embedding_shards_dir=EMBED_SHARDS,
        cache_dir=CACHE,
        batch_size=args.batch_size,
        device=device,
    )

    print(f"\nDone. Tower cached under {CACHE / args.backbone}")


if __name__ == "__main__":
    main()