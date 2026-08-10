"""scripts/13_precompute_reranked_qwen_recs.py

Run a second-stage reranker (jina_v3 listwise, or a pointwise qwen3_reranker_* /
bge_v2_m3) ONCE over all Blind-A turns and save the reranked Qwen recommendations
(top --save-k per turn) to disk. gambling_updated then consumes this file via
--reranked-recs-path and needs no model and no embeddings -- it just feeds these
reranked Qwen recs into the heuristic's fallback slots.

NO DRIFT BY CONSTRUCTION
------------------------
This script does not reimplement the dense-scoring / rerank / ranking pipeline. It
imports scripts/launchers/gambling_updated.py and calls its EXACT
build_all_qwen_recs_cached(..., reranker=<key>) with top_k = --save-k. The recs saved
here are therefore byte-for-byte what gambling_updated --reranker <key> would compute
live -- just precomputed and truncated to --qwen-fill-k at load time.

WHAT IT WRITES
--------------
  <out> (default: exp/inference/blind_a/gambling/reranked_<reranker>_qwen_recs_<keytag>.json)
  {
    "meta": { model, reranker, candidate_n, save_k, rerank_doc, q_max_tokens,
              track_metadata_path, query_cache_dir, query_cache_mtime, generated_at },
    "rows": [ { session_id, user_id, turn_number, qwen_reranked_track_ids: [...save_k] }, ... ]
  }

USAGE (GPU node, from repo root)
--------------------------------
  # Qwen3-Reranker-0.6B (pointwise) over the Qwen 0.6B embedding candidates:
  uv run python scripts/13_precompute_reranked_qwen_recs.py --model 0.6 --reranker qwen3_reranker_0p6b
  # jina-reranker-v3 (listwise):
  uv run python scripts/13_precompute_reranked_qwen_recs.py --model 0.6 --reranker jina_v3
  # then, fast repeated runs (CPU, no model):
  uv run python scripts/launchers/gambling_updated.py --model 0.6 \
      --reranked-recs-path exp/inference/blind_a/gambling/reranked_qwen3_reranker_0p6b_qwen_recs_Qwen3-Embedding-0.6B.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Import the launcher module by path and REUSE its exact pipeline (single source of
# truth -- guarantees the precomputed recs equal gambling_updated --reranker <key>).
_LAUNCHER = REPO / "scripts/launchers/gambling_updated.py"
if not _LAUNCHER.exists():
    sys.stderr.write(f"FATAL: cannot find launcher at {_LAUNCHER}\n")
    raise SystemExit(2)
_spec = importlib.util.spec_from_file_location("gambling_updated", _LAUNCHER)
gu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gu)

from emblib.retrieval.rerankers import RERANKERS  # noqa: E402  (registry only, no torch)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["0.6", "4", "8"], default="0.6",
                   help="which Qwen embedding model's caches to score+rerank with")
    p.add_argument("--reranker", choices=sorted(RERANKERS), default="qwen3_reranker_0p6b",
                   help="second-stage reranker to apply")
    p.add_argument("--save-k", type=int, default=64,
                   help="how many reranked recs to save per turn (>= the gambling_updated "
                        "--qwen-fill-k you intend to use; default 64 gives headroom)")
    p.add_argument("--candidate-n", type=int, default=64,
                   help="dense candidates reranked per turn (jina_v3 caps at 64/pass)")
    p.add_argument("--rerank-doc", choices=["neural", "metadata"], default="neural")
    p.add_argument("--q-max-tokens", type=int, default=224)
    p.add_argument("--rerank-batch-size", type=int, default=None)
    p.add_argument("--rerank-max-length", type=int, default=None)
    p.add_argument("--rerank-attn", default=None, choices=["flash_attention_2", "sdpa", "eager"])
    p.add_argument("--rerank-device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--rerank-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    p.add_argument("--rerank-allow-downloads", dest="rerank_local_files_only",
                   action="store_false", default=True)
    p.add_argument("--blind-path", type=Path, default=gu.BLIND_A_PATH)
    p.add_argument("--train-path", type=Path, default=gu.TRAIN_PATH)
    p.add_argument("--track-metadata-path", type=Path, default=gu.TRACK_METADATA_PATH)
    p.add_argument("--qwen-track-cache-dir", type=Path, default=None)
    p.add_argument("--query-cache-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None, help="output JSON path")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if a.save_k < 1:
        raise SystemExit("--save-k must be >= 1")

    track_cache = a.qwen_track_cache_dir or gu.model_track_cache_dir(a.model)
    query_cache = a.query_cache_dir or gu.model_query_cache_dir(a.model)
    key_tag = gu.MODEL_FOLDER[a.model].split("__")[-1]            # e.g. Qwen3-Embedding-0.6B
    out = a.out or (gu.BLIND_A_INFERENCE_DIR / "gambling"
                    / f"reranked_{a.reranker}_qwen_recs_{key_tag}.json")

    print("precompute reranked Qwen recs")
    print(f"  reusing pipeline from : {gu.__file__}")
    print(f"  emblib.retrieval.core    : {gu._core.__file__}")
    print(f"  embedding model       : {a.model}  ({gu.MODEL_FOLDER[a.model]})")
    print(f"  reranker              : {a.reranker}  ({RERANKERS[a.reranker]['model_id']})")
    print(f"  track cache dir       : {track_cache}")
    print(f"  query cache dir       : {query_cache}  (mtime {gu._mtime(query_cache / 'query_meta.parquet')})")
    print(f"  candidate_n={a.candidate_n}  save_k={a.save_k}  doc={a.rerank_doc}  q_max_tokens={a.q_max_tokens}")
    print(f"  output                : {out}")

    # Build an args object exactly like gambling_updated.parse_args() would produce,
    # but with the reranker forced on and top_k = save_k so we persist a deep list.
    gargs = SimpleNamespace(
        model=a.model,
        blind_path=a.blind_path,
        train_path=a.train_path,
        track_metadata_path=a.track_metadata_path,
        qwen_track_cache_dir=track_cache,
        query_cache_dir=query_cache,
        reranker=a.reranker,
        candidate_n=a.candidate_n,
        rerank_doc=a.rerank_doc,
        q_max_tokens=a.q_max_tokens,
        rerank_batch_size=a.rerank_batch_size,
        rerank_max_length=a.rerank_max_length,
        rerank_attn=a.rerank_attn,
        rerank_device=a.rerank_device,
        rerank_dtype=a.rerank_dtype,
        rerank_local_files_only=a.rerank_local_files_only,
        top_k=a.save_k,
    )

    task_rows = gu.load_blind_task_rows(a.blind_path)
    print(f"\nrunning {a.reranker} over {len(task_rows)} turns ...")
    t0 = time.time()
    recs_by_key = gu.build_all_qwen_recs_cached(gargs, task_rows)   # <- identical pipeline
    print(f"done in {(time.time() - t0) / 60:.2f} min")

    rows = []
    for task in task_rows:
        k = gu.key(task)
        ids = recs_by_key[k]
        if len(ids) != a.save_k:
            raise RuntimeError(f"turn {k} produced {len(ids)} recs, expected save_k={a.save_k}")
        rows.append({
            "session_id": k[0],
            "user_id": k[1],
            "turn_number": k[2],
            "qwen_reranked_track_ids": ids,
        })

    meta = {
        "model": a.model,
        "reranker": a.reranker,
        "candidate_n": a.candidate_n,
        "save_k": a.save_k,
        "rerank_doc": a.rerank_doc,
        "q_max_tokens": a.q_max_tokens,
        "track_metadata_path": str(a.track_metadata_path),
        "query_cache_dir": str(query_cache),
        "query_cache_mtime": gu._mtime(query_cache / "query_meta.parquet"),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved {len(rows)} turns x top-{a.save_k} {a.reranker}-reranked recs -> {out}")
    print(f"use it with: gambling_updated.py --model {a.model} --reranked-recs-path {out}")


if __name__ == "__main__":
    main()