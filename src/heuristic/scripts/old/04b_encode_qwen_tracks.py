"""Encode the track-metadata tower with Qwen3 — the track-side mirror of the
new query encoder.

WHY THIS EXISTS
===============
The organizer ships a precomputed `metadata-qwen3_embedding_0.6b` track tower,
but it was built with the organizer's own text rendering + pooling. The new
pipeline encodes BOTH sides — queries and tracks — with the SAME Qwen3 model,
the SAME `track_metadata_text` rendering for tracks / `Instruct:`-prefixed
queries, and the SAME last-token pooling (`pool_last_token`). That makes the
dot product exact cosine similarity in one shared space, with no projection or
format mismatch in between.

This script encodes every track's `track_metadata_text` (Title/Artist/Album/
Tags/Year) with instruction "none", max-length 256, last-token pool, and caches
a `TrackTower`-compatible tower at `models/track_tower_qwen_cache/`.

CANONICAL ORDER
===============
Tracks are emitted in the SAME order as the organizer embedding shards, so the
integer indices in previously-mined hard negatives remain valid against this
tower (mining is re-run in this space anyway by scripts/02, but the invariant
is preserved regardless).

Run on a GPU node (CPU works but is slow for ~47k tracks):

    uv run python scripts/04b_encode_qwen_tracks.py --batch-size 64

Output: models/track_tower_qwen_cache/{track_ids.npy, metadata-qwen3-native.npy,
        metadata-qwen3-native__mask.npy}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emblib.qwen.config import QwenConfig
from emblib.tracks.qwen_track_loader import build_qwen_track_tower, QWEN_MODALITY


def main():
    cfg = QwenConfig()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch-size", type=int, default=cfg.track_batch_size,
                   help="Encoding batch size. 64 is a good default on a 16GB+ GPU; "
                        "lower it if you hit OOM. Default comes from QwenConfig "
                        f"({cfg.track_batch_size}).")
    p.add_argument("--cache-dir", type=Path, default=cfg.track_tower_cache_dir,
                   help="Where to write the Qwen3 track tower.")
    p.add_argument("--track-meta", type=Path, default=cfg.track_meta_path)
    p.add_argument("--embedding-shards", type=Path, default=cfg.embedding_shards_dir,
                   help="Organizer all_tracks-*.parquet dir — defines canonical order.")
    p.add_argument("--model", type=str, default=cfg.model)
    p.add_argument("--max-length", type=int, default=cfg.track_max_length)
    p.add_argument("--device", type=str, default=cfg.device,
                   choices=["auto", "cuda", "cpu", "mps"])
    p.add_argument("--dtype", type=str, default=cfg.dtype,
                   choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--local-files-only", action="store_true", default=cfg.local_files_only,
                   help="Only use a locally-cached HF model (offline).")
    args = p.parse_args()

    print("#" * 80)
    print("  ENCODE QWEN3 TRACK TOWER (track-side mirror of the query encoder)")
    print("#" * 80)
    print(f"  model       : {args.model}")
    print(f"  instruction : {cfg.track_instruction!r}  (tracks get no prefix)")
    print(f"  max_length  : {args.max_length}")
    print(f"  batch_size  : {args.batch_size}")
    print(f"  device/dtype: {args.device}/{args.dtype}")
    print(f"  cache_dir   : {args.cache_dir}")
    print(f"  modality    : {QWEN_MODALITY}")

    build_qwen_track_tower(
        track_meta_path=args.track_meta,
        embedding_shards_dir=args.embedding_shards,
        cache_dir=args.cache_dir,
        model_name=args.model,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device_arg=args.device,
        dtype_arg=args.dtype,
        local_files_only=args.local_files_only,
        trust_remote_code=cfg.trust_remote_code,
        instruction_name=cfg.track_instruction,
        modality=QWEN_MODALITY,
    )

    print(f"\nDone. Tower cached under {args.cache_dir}")
    print("NEXT: encode queries (03), mine negatives (02), then evaluate/train.")


if __name__ == "__main__":
    main()