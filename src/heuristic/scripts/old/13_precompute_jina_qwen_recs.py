"""scripts/13_precompute_jina_qwen_recs.py

Run jina-reranker-v3 ONCE over all Blind-A turns and save the jina-reranked Qwen
recommendations (top --save-k per turn) to disk, so the heavy rerank pass is done
a single time. gambling_updated then consumes this file via --jina-recs-path and
needs no model and no embeddings -- it just feeds these reranked Qwen recs into the
heuristic's fallback slots.

NO DRIFT BY CONSTRUCTION
------------------------
This script does not reimplement the dense-scoring / rerank / ranking pipeline. It
imports scripts/launchers/gambling_updated.py and calls its EXACT
build_all_qwen_recs_cached(..., rerank="jina") with top_k = --save-k. The recs saved
here are therefore byte-for-byte what gambling_updated --rerank jina would compute
live -- just precomputed and truncated to --qwen-fill-k at load time.

WHAT IT WRITES
--------------
  <out> (default: exp/inference/blind_a/gambling/jina_qwen_recs_<keytag>.json)
  {
    "meta": { model, candidate_n, save_k, rerank_doc, rerank_model_id,
              track_metadata_path, query_cache_dir, query_cache_mtime, generated_at },
    "rows": [ { session_id, user_id, turn_number, qwen_reranked_track_ids: [...save_k] }, ... ]
  }

USAGE (GPU node, from repo root)
--------------------------------
  uv run python scripts/13_precompute_jina_qwen_recs.py --model 0.6
  uv run python scripts/13_precompute_jina_qwen_recs.py --model 0.6 --save-k 64 --candidate-n 64
  uv run python scripts/13_precompute_jina_qwen_recs.py --model 0.6 --rerank-doc metadata
  # then, fast repeated runs (CPU, no model):
  uv run python scripts/launchers/gambling_updated.py --model 0.6 \
      --jina-recs-path exp/inference/blind_a/gambling/jina_qwen_recs_Qwen3-Embedding-0.6B.json
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
# truth -- guarantees the precomputed recs equal gambling_updated --rerank jina).
_LAUNCHER = REPO / "scripts/launchers/gambling_updated.py"
if not _LAUNCHER.exists():
    sys.stderr.write(f"FATAL: cannot find launcher at {_LAUNCHER}\n")
    raise SystemExit(2)
_spec = importlib.util.spec_from_file_location("gambling_updated", _LAUNCHER)
gu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gu)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["0.6", "4", "8"], default="0.6",
                   help="which Qwen model's caches to score+rerank with")
    p.add_argument("--save-k", type=int, default=64,
                   help="how many jina-reranked recs to save per turn (>= the gambling_updated "
                        "--qwen-fill-k you intend to use; default 64 gives headroom)")
    p.add_argument("--candidate-n", type=int, default=64,
                   help="dense candidates reranked per turn (<=64 = one jina pass)")
    p.add_argument("--rerank-doc", choices=["neural", "metadata"], default="neural")
    p.add_argument("--rerank-model-id", default="jinaai/jina-reranker-v3")
    p.add_argument("--rerank-device", default="auto")
    p.add_argument("--rerank-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
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
    out = a.out or (gu.BLIND_A_INFERENCE_DIR / "gambling" / f"jina_qwen_recs_{key_tag}.json")

    print("precompute jina-reranked Qwen recs")
    print(f"  reusing pipeline from : {gu.__file__}")
    print(f"  emblib.retrieval.core    : {gu._core.__file__}")
    print(f"  model                 : {a.model}  ({gu.MODEL_FOLDER[a.model]})")
    print(f"  track cache dir       : {track_cache}")
    print(f"  query cache dir       : {query_cache}  (mtime {gu._mtime(query_cache / 'query_meta.parquet')})")
    print(f"  rerank                : {a.rerank_model_id}  candidate_n={a.candidate_n}  doc={a.rerank_doc}  save_k={a.save_k}")
    print(f"  output                : {out}")

    # Build an args object exactly like gambling_updated.parse_args() would produce,
    # but with rerank forced to jina and top_k = save_k so we persist a deep list.
    gargs = SimpleNamespace(
        model=a.model,
        blind_path=a.blind_path,
        train_path=a.train_path,
        track_metadata_path=a.track_metadata_path,
        qwen_track_cache_dir=track_cache,
        query_cache_dir=query_cache,
        rerank="jina",
        candidate_n=a.candidate_n,
        rerank_model_id=a.rerank_model_id,
        rerank_doc=a.rerank_doc,
        rerank_device=a.rerank_device,
        rerank_dtype=a.rerank_dtype,
        rerank_local_files_only=a.rerank_local_files_only,
        top_k=a.save_k,
    )

    task_rows = gu.load_blind_task_rows(a.blind_path)
    print(f"\nrunning jina over {len(task_rows)} turns ...")
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
        "candidate_n": a.candidate_n,
        "save_k": a.save_k,
        "rerank_doc": a.rerank_doc,
        "rerank_model_id": a.rerank_model_id,
        "track_metadata_path": str(a.track_metadata_path),
        "query_cache_dir": str(query_cache),
        "query_cache_mtime": gu._mtime(query_cache / "query_meta.parquet"),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved {len(rows)} turns x top-{a.save_k} jina-reranked recs -> {out}")
    print("use it with: gambling_updated.py --model "
          f"{a.model} --jina-recs-path {out}")


if __name__ == "__main__":
    main()