"""scripts/09_rerank_jina_v3.py

Second-stage reranking with jina-reranker-v3 on top of any bi-encoder produced by
scripts/08_try_embedding_models.py.

WHY A SEPARATE SCRIPT
---------------------
jina-reranker-v3 is NOT a bi-encoder. It is a listwise cross-encoder: it puts the
query and ALL candidate documents in one context window and scores them jointly
("last but not late" interaction). There is no fixed per-track vector to dot a
query against, so it cannot replace the embedding model in script 08. The correct
use is two-stage retrieval, which is also the highest-leverage move for NDCG@20:

    1. retrieve top-N candidates with a bi-encoder (script 08's cached val tower);
    2. rerank those N with jina-reranker-v3 using the real track-metadata text;
    3. recompute macro-by-turn NDCG@20 on the reranked order.

Because NDCG@20 only needs the top 20, reranking the bi-encoder's top-N (default 64,
the model's single-pass document budget) is sufficient and cheap (0.6B model).

USAGE
-----
  uv run python scripts/09_rerank_jina_v3.py --base qwen3_4b
  uv run python scripts/09_rerank_jina_v3.py --base qwen3_8b --candidate-n 64

Reads:
  models/track_tower_generic_cache/<base>/{track_ids,emb,mask}.npy
  models/query_emb_generic_cache/<base>/val.npy + val_meta.parquet
  data/.../all_tracks-...parquet         (for the document text to rerank)
Writes:
  models/eval_results/<base>__jinav3_rerank/val/ndcg.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import time

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emblib.qwen.qwen_embeddings import track_metadata_text

DATA        = Path("./data/talkpl-ai")
TRACK_META  = DATA / "TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet"
TRACK_CACHE = Path("./models/track_tower_generic_cache")
QUERY_CACHE = Path("./models/query_emb_generic_cache")
EVAL_OUT    = Path("./models/eval_results")
NDCG_K      = 20


def load_base(base: str):
    c = TRACK_CACHE / base
    emb  = np.load(c / "emb.npy")
    mask = np.load(c / "mask.npy")
    ids  = [str(t) for t in np.load(c / "track_ids.npy", allow_pickle=True).tolist()]
    q    = np.load(QUERY_CACHE / base / "val.npy")
    meta = pl.read_parquet(QUERY_CACHE / base / "val_meta.parquet").to_dicts()
    return emb, mask, ids, q, meta


def topn_candidates(q_row, track_emb, track_mask, prior_idx, gt_idx, n):
    scores = track_emb @ q_row
    scores[~track_mask] = -np.inf
    for j in prior_idx:
        scores[j] = -np.inf
    # ensure GT can still be retrieved even if it was a prior play
    if gt_idx is not None:
        scores[gt_idx] = max(scores[gt_idx], -1e30)
    n = min(n, int(np.isfinite(scores).sum()))
    cand = np.argpartition(-scores, n - 1)[:n]
    return cand[np.argsort(-scores[cand])]


def ndcg_from_rank(rank: int) -> float:
    return (1.0 / np.log2(1 + rank)) if rank <= NDCG_K else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="bi-encoder key from script 08 (e.g. qwen3_4b)")
    p.add_argument("--candidate-n", type=int, default=128,
                   help="candidates retrieved per query then reranked (<=64 = 1 jina pass)")
    p.add_argument("--model-id", default="jinaai/jina-reranker-v3")
    args = p.parse_args()

    import torch
    from transformers import AutoModel

    emb, mask, ids, q, meta = load_base(args.base)
    id_to_idx = {t: i for i, t in enumerate(ids)}
    md_by_id  = {str(r["track_id"]): r for r in pl.read_parquet(TRACK_META).to_dicts()}
    print(f"base={args.base}  val={q.shape}  tower={emb.shape}  candidate_n={args.candidate_n}")

    print(f"loading reranker {args.model_id} ...")
    # model = AutoModel.from_pretrained(args.model_id, dtype="auto", trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,  # IMPORTANT
        trust_remote_code=True,
      #  attn_implementation="flash_attention_2"
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    start_time = time.time()

  #  print(model.config.attn_implementation)
    print("\n=== RERANK CONFIG ===")
    print(f"num_queries={len(meta)}")
    print(f"candidate_n={args.candidate_n}")
    print(f"model={args.model_id}")
    print(f"device={next(model.parameters()).device}")
    print("=====================\n")

    per_turn = defaultdict(list)
    n_scor = 0
    for i, r in enumerate(meta):

        if i % 50 == 0 and i > 0:
            elapsed = time.time() - start_time
            avg = elapsed / i
            remaining = avg * (len(meta) - i)

            print(
                f"[progress] {i}/{len(meta)} | "
                f"avg={avg:.3f}s/query | "
                f"elapsed={elapsed / 60:.1f} min | "
                f"ETA={remaining / 60:.1f} min"
            )

        gid = r.get("gt_track_id")
        if gid is None or gid not in id_to_idx:
            continue
        gt_idx = id_to_idx[gid]
        if not mask[gt_idx]:
            continue
        prior_idx = [id_to_idx[t] for t in (r.get("prior_track_ids") or []) if t in id_to_idx]
        cand = topn_candidates(q[i], emb, mask, prior_idx, gt_idx, args.candidate_n)
        if gt_idx not in cand:                       # GT not retrieved -> NDCG 0
            per_turn[int(r["turn_number"])].append(0.0); n_scor += 1
            continue

        cand_ids  = [ids[j] for j in cand]
        cand_docs = [track_metadata_text(md_by_id[t]) if t in md_by_id else t for t in cand_ids]

        # real user text persisted by script 08 (v2 query string); fall back to
        # session_id only if an older cache without query_text is used.
        query_text = r.get("query_text") or r.get("session_id")
        with torch.no_grad():
            results = model.rerank(query_text, cand_docs, top_n=len(cand_docs))
        order = [res["index"] for res in results]    # indices into cand_docs, best first
        reranked_ids = [cand_ids[o] for o in order]
        rank = reranked_ids.index(gid) + 1
        per_turn[int(r["turn_number"])].append(ndcg_from_rank(rank)); n_scor += 1

    ptm = {k: float(np.mean(v)) for k, v in sorted(per_turn.items())}
    total = time.time() - start_time
    print(f"\nTOTAL TIME: {total / 60:.2f} minutes")

    macro = float(np.mean(list(ptm.values()))) if ptm else 0.0
    micro = float(np.mean([v for vs in per_turn.values() for v in vs])) if per_turn else 0.0
    print(f"\n[{args.base} + jina-v3] NDCG@{NDCG_K} macro-by-turn={macro:.4f} micro={micro:.4f} (n={n_scor})")
    for k in sorted(ptm):
        print(f"  turn {k:>2}: {ptm[k]:.4f}")

    key = f"{args.base}__jinav3_rerank"
    out = EVAL_OUT / key / "val"; out.mkdir(parents=True, exist_ok=True)
    (out / "ndcg.json").write_text(json.dumps(
        {"encoder": key, "fold": "val", "metric": f"ndcg@{NDCG_K}",
         "macro_by_turn": macro, "micro": micro, "n_scorable": n_scor,
         "per_turn_mean": ptm, "candidate_n": args.candidate_n}, indent=2))
    print(f"wrote {out / 'ndcg.json'}")


if __name__ == "__main__":
    main()