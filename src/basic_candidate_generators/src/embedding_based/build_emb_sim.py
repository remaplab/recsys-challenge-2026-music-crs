"""Build + cache W_emb item-item similarity from a Qwen3 track tower.

Run once per (model_size, k). The matrix is stored in stable tower-id space and
remapped to each fold's id_map at recommender fit time.

Usage:
    cd src/basic_candidate_generators
    uv run python -m src.embedding_based.build_emb_sim \\
        --track_emb_dir models/retrieval_text_towers/Qwen__Qwen3-Embedding-8B/dense_tracks_len256_poollast \\
        --out models/embedding_sim/W_emb_qwen3_8b_k150.npz --k 150
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .emb_matrix import build_emb_item_sim, load_track_tower, save_emb_sim


def main() -> None:
    p = argparse.ArgumentParser(description="Build cached W_emb item-item similarity.")
    p.add_argument("--track_emb_dir", required=True, help="dense_tracks_* tower cache dir")
    p.add_argument("--out", required=True, help="output .npz path")
    p.add_argument("--k", type=int, default=500,
                   help="neighbours per item to cache (trimmed down at load via sparsify_topk)")
    p.add_argument("--block", type=int, default=4096, help="row-block for tiled GEMM")
    p.add_argument("--no_gpu", action="store_true")
    args = p.parse_args()

    track_ids, emb = load_track_tower(args.track_emb_dir)
    print(f"[build_emb_sim] {emb.shape[0]} tracks x {emb.shape[1]} dim, k={args.k}")
    t0 = time.time()
    W = build_emb_item_sim(emb, k=args.k, block=args.block, use_gpu=not args.no_gpu)
    print(f"[build_emb_sim] built in {time.time()-t0:.1f}s, nnz={W.nnz}")
    save_emb_sim(W, track_ids, Path(args.out))
    print(f"[build_emb_sim] saved → {args.out}")


if __name__ == "__main__":
    main()
