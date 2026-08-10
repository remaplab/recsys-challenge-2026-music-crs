"""scripts/11_fuse_candidates.py

Reciprocal Rank Fusion (RRF) of several bi-encoders' cached candidate lists.

Each retriever (script 08 -> script 10) caches a per-row ranked candidate list
at models/rerank_candidates/<base>_cand_<N>.npz. Those lists index the SAME
canonical track order and the rows are the SAME val (session, turn) pairs in the
SAME order across bases, so we can fuse them by rank:

    fused_score(track) = sum_over_retrievers 1 / (k + rank_in_that_retriever)

A track that ranks high in several retrievers floats to the top. RRF needs no
score calibration across the different embedding spaces (it uses ranks only),
which is exactly why it's robust for fusing 1024-/2560-/4096-d Qwen towers.

The fused list usually has HIGHER recall@N than any single retriever (the three
towers miss partly different ground truths), which raises the ceiling for both
the bi-encoder baseline and the reranker.

OUTPUT (a self-contained "fused base" that scripts/10_rerank.py consumes via
--base <out-name>):
  models/rerank_candidates/<out>_cand_<N>.npz        fused candidates
  models/track_tower_generic_cache/<out>/track_ids.npy   (copied from a ref base)
  models/query_emb_generic_cache/<out>/val_meta.parquet  (copied from a ref base)
  models/eval_results/<out>/val/ndcg.json            fused-retriever NDCG@20 baseline

Runs on CPU in seconds — no GPU needed.

USAGE
  uv run python scripts/11_fuse_candidates.py \
      --bases qwen3_0p6b qwen3_4b qwen3_8b --built-n 500 --k 60 --out-name rrf_qwen
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

CAND_CACHE  = Path("./models/rerank_candidates")
TRACK_CACHE = Path("./models/track_tower_generic_cache")
QUERY_CACHE = Path("./models/query_emb_generic_cache")
EVAL_OUT    = Path("./models/eval_results")
NDCG_K      = 20


def load_cand(base, built_n):
    p = CAND_CACHE / f"{base}_cand_{built_n}.npz"
    if not p.exists():
        # accept a larger cached file and slice
        larger = []
        for f in CAND_CACHE.glob(f"{base}_cand_*.npz"):
            try:
                M = int(f.stem.split("_cand_")[1])
            except (IndexError, ValueError):
                continue
            if M >= built_n:
                larger.append((M, f))
        if not larger:
            raise FileNotFoundError(
                f"No candidate cache for {base!r} at >= {built_n}. Run "
                f"`scripts/10_rerank.py --base {base} --recall-only` first.")
        _, p = min(larger)
    z = np.load(p)
    return z["cand"][:, :built_n].copy(), z["gt_idx"], z["turns"], z["scorable"], p


def recall_at(cand, gt_idx, scorable, ns):
    out = {}
    for N in ns:
        hit = tot = 0
        for i in range(cand.shape[0]):
            if not scorable[i]:
                continue
            tot += 1
            if gt_idx[i] in cand[i, :N]:
                hit += 1
        out[N] = hit / max(tot, 1)
    return out


def fused_retriever_ndcg(cand, gt_idx, turns, scorable):
    inv = 1.0 / np.log2(np.arange(2, NDCG_K + 2))
    per_turn = defaultdict(list)
    for i in range(cand.shape[0]):
        if not scorable[i]:
            continue
        pos = np.where(cand[i] == gt_idx[i])[0]
        rank = int(pos[0]) + 1 if len(pos) else NDCG_K + 99
        per_turn[int(turns[i])].append(inv[rank - 1] if rank <= NDCG_K else 0.0)
    ptm = {k: float(np.mean(v)) for k, v in sorted(per_turn.items())}
    macro = float(np.mean(list(ptm.values()))) if ptm else 0.0
    micro = float(np.mean([v for vs in per_turn.values() for v in vs])) if per_turn else 0.0
    return macro, micro, ptm


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bases", nargs="+", required=True,
                   help="bi-encoder keys whose candidates to fuse (e.g. qwen3_0p6b qwen3_4b qwen3_8b)")
    p.add_argument("--built-n", type=int, default=500,
                   help="candidate depth read from each base AND written for the fused list")
    p.add_argument("--k", type=float, default=60.0, help="RRF constant (standard default 60)")
    p.add_argument("--out-name", default="rrf_qwen")
    p.add_argument("--recall-ns", type=int, nargs="+", default=[20, 50, 100, 200, 300, 500])
    args = p.parse_args()

    assert len(args.bases) >= 2, "fuse at least 2 bases"
    print("#" * 74)
    print(f"  RRF fusion  bases={args.bases}  built_n={args.built_n}  k={args.k}")
    print("#" * 74)

    loaded = {b: load_cand(b, args.built_n) for b in args.bases}
    for b, (_, _, _, _, src) in loaded.items():
        print(f"  {b:<14} <- {src}")

    ref = args.bases[0]
    cand_ref, gt_idx, turns, scorable_ref, _ = loaded[ref]
    n = cand_ref.shape[0]

    # Sanity: all bases must share the same rows (same gt / turns where scorable),
    # which holds because every base encodes the same sorted (session,turn) pairs.
    scorable = scorable_ref.copy()
    for b in args.bases[1:]:
        cand_b, gt_b, turns_b, scor_b, _ = loaded[b]
        if cand_b.shape[0] != n:
            raise ValueError(f"row count mismatch: {ref}={n} vs {b}={cand_b.shape[0]}")
        agree = (gt_b == gt_idx) | ~(scorable & scor_b)
        if not agree.all():
            n_bad = int((~agree).sum())
            print(f"  [warn] {b}: {n_bad} rows disagree on gt_idx — excluding those rows")
            scorable &= (gt_b == gt_idx)
        scorable &= scor_b
    print(f"  fusing {n} rows  ({int(scorable.sum())} scorable in all bases)")

    # ── RRF fuse ─────────────────────────────────────────────────────────
    cand_list = [loaded[b][0] for b in args.bases]
    fused = np.full((n, args.built_n), -1, dtype=np.int32)
    for i in range(n):
        score: dict[int, float] = defaultdict(float)
        for cand_b in cand_list:
            row = cand_b[i]
            for rank, tid in enumerate(row):
                if tid < 0:
                    break
                score[int(tid)] += 1.0 / (args.k + rank + 1)
        if not score:
            continue
        ranked = sorted(score.keys(), key=lambda t: -score[t])[:args.built_n]
        fused[i, :len(ranked)] = ranked

    # ── save fused candidates + the bits script 10 needs ─────────────────
    CAND_CACHE.mkdir(parents=True, exist_ok=True)
    out_npz = CAND_CACHE / f"{args.out_name}_cand_{args.built_n}.npz"
    np.savez(out_npz, cand=fused, gt_idx=gt_idx, turns=turns, scorable=scorable)
    print(f"\n  wrote {out_npz}  cand={fused.shape}")

    (TRACK_CACHE / args.out_name).mkdir(parents=True, exist_ok=True)
    shutil.copy(TRACK_CACHE / ref / "track_ids.npy",
                TRACK_CACHE / args.out_name / "track_ids.npy")
    (QUERY_CACHE / args.out_name).mkdir(parents=True, exist_ok=True)
    shutil.copy(QUERY_CACHE / ref / "val_meta.parquet",
                QUERY_CACHE / args.out_name / "val_meta.parquet")
    print(f"  copied track_ids.npy + val_meta.parquet from {ref!r} into fused base "
          f"{args.out_name!r}")

    # ── recall comparison: each base vs fused ────────────────────────────
    ns = sorted(set(args.recall_ns))
    print(f"\n  recall@N  (scorable rows = {int(scorable.sum())}):")
    header = "    " + f"{'N':<8}" + "".join(f"{b:>14}" for b in args.bases) + f"{'RRF':>14}"
    print(header)
    base_rec = {b: recall_at(loaded[b][0], gt_idx, scorable, ns) for b in args.bases}
    fused_rec = recall_at(fused, gt_idx, scorable, ns)
    for N in ns:
        line = f"    {N:<8}" + "".join(f"{base_rec[b][N]:>14.4f}" for b in args.bases)
        line += f"{fused_rec[N]:>14.4f}"
        print(line)

    # ── fused-retriever NDCG@20 baseline (so script 10 shows rerank delta) ─
    macro, micro, ptm = fused_retriever_ndcg(fused, gt_idx, turns, scorable)
    print(f"\n  FUSED retriever NDCG@{NDCG_K} (top-{args.built_n} order) "
          f"macro-by-turn = {macro:.4f}  micro = {micro:.4f}")
    out_eval = EVAL_OUT / args.out_name / "val"
    out_eval.mkdir(parents=True, exist_ok=True)
    (out_eval / "ndcg.json").write_text(json.dumps({
        "encoder": args.out_name, "fold": "val", "metric": f"ndcg@{NDCG_K}",
        "macro_by_turn": macro, "micro": micro,
        "n_scorable": int(scorable.sum()),
        "fused_from": args.bases, "rrf_k": args.k,
        "recall_at_n": fused_rec, "per_base_recall_at_n": base_rec,
    }, indent=2))
    print(f"  wrote {out_eval / 'ndcg.json'}")
    print(f"\nNEXT: rerank the fused base, e.g.\n"
          f"  uv run python scripts/10_rerank.py --base {args.out_name} "
          f"--reranker qwen3_reranker_0p6b --sweep 50 100 200 --subsample 800")


if __name__ == "__main__":
    main()