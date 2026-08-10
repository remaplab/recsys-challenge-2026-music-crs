"""Score encoders on macro-by-turn NDCG@20 — the metric the challenge actually
ranks on — instead of (or alongside) recall.

WHY A SEPARATE SCRIPT
=====================
`06_evaluate.py` computes recall via `src/evaluation/recall.py`. Rather than
touch that module, this script reads the SAME cached artifacts everything else
uses and computes NDCG@20 itself, so it can't perturb the recall path:

    query embeddings : models/query_emb_cache/<encoder>/<conv_split>.npy
    query meta       : models/query_emb_cache/<encoder>/<conv_split>_meta.parquet
    track tower      : routed by encoder name (Qwen3-native / organizer / text)
    fold definition  : models/splits/train_val_test_blinda_matched.parquet

WHAT "macro-by-turn NDCG@20" MEANS HERE
=======================================
Each query has exactly ONE ground-truth track (`gt_track_id`), so for a single
relevant item the ideal DCG is 1 and

    NDCG@20(row) = 1 / log2(1 + rank)   if the GT is within the top 20
                 = 0                     otherwise

where `rank` is the GT's 1-indexed position after scoring all valid tracks and
EXCLUDING the row's already-played `prior_track_ids` (same candidate masking
`06_evaluate.py` applies for recall, so the two metrics are comparable).

"macro-by-turn" = average NDCG@20 within each predict-turn bucket K, then average
those per-K means with equal weight. This matches how the leaderboard aggregates
(and why the Blind-A-matched split balances the K distribution). We also report
the plain micro mean (average over all rows) for reference.

FOLDS
=====
val / test  : Blind-A-matched folds. Queries are cached under the "train" conv
              split, then subset to the fold's pinned (session_id, turn) rows.
dev / blind_a : evaluate every cached row (no predict-turn subsetting); the
              "by-turn" macro still groups by each row's `turn_number`.

USAGE
=====
    uv run python scripts/06b_evaluate_ndcg.py --encoder qwen3_native_frozen --split val
    uv run python scripts/06b_evaluate_ndcg.py --encoder qwen3_lora__qwen3_lora_no_routing --split val

    # compare several at once → writes a table + per-encoder JSON
    uv run python scripts/06b_evaluate_ndcg.py --split val --compare \
        qwen3_native_frozen qwen3_lora__qwen3_lora_no_routing

Output per encoder: models/eval_results/<encoder>/<fold>/ndcg.json
Comparison table  : models/eval_results/ndcg_comparison_<fold>.csv
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emblib.data.split import load_blinda_split, turn_pairs_for_fold, select_pair_indices
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
FOLD_TO_CONV_SPLIT = {"val": "train", "test": "train", "dev": "dev", "blind_a": "blind_a"}

NDCG_K = 20
SCORE_CHUNK = 512   # rows per matmul block (keeps the score matrix small)


def load_tower_for_encoder(encoder_name: str):
    """Pick the track tower by encoder-name prefix. Mirrors 06_evaluate.py."""
    if encoder_name.startswith("bert_native_frozen") or encoder_name.startswith("bert_proj_"):
        backbone = "bert"
    elif encoder_name.startswith("sbert_native_frozen") or encoder_name.startswith("sbert_proj_"):
        backbone = "sbert"
    elif encoder_name.startswith("modernbert_native_frozen") or encoder_name.startswith("modernbert_proj_"):
        backbone = "modernbert"
    elif encoder_name.startswith("qwen3_native_frozen") or encoder_name.startswith("qwen3_lora"):
        print(f"[tower] {encoder_name!r} → Qwen3-native metadata tower (1024-d)")
        tower = load_qwen_track_tower(QWEN_TOWER_CACHE, QWEN_MODALITY)
        return tower.embeddings[QWEN_MODALITY], tower.masks[QWEN_MODALITY], tower.id_to_idx
    else:
        print(f"[tower] {encoder_name!r} → organizer Qwen3 metadata tower (1024-d, legacy)")
        tower = load_track_tower(
            shards_dir=DATA / "TalkPlayData-Challenge-Track-Embeddings/data",
            modalities=[LEGACY_QWEN3_MODALITY], cache_dir=TOWER_CACHE,
        )
        return (tower.embeddings[LEGACY_QWEN3_MODALITY],
                tower.masks[LEGACY_QWEN3_MODALITY], tower.id_to_idx)

    print(f"[tower] {encoder_name!r} → {backbone!r} text track tower (768-d)")
    tower = load_text_track_tower(backbone, cache_dir=TEXT_TOWER_CACHE)
    modality = get_text_modality(backbone)
    return tower.embeddings[modality], tower.masks[modality], tower.id_to_idx


def _load_queries_and_meta(encoder: str, fold: str):
    conv_split = FOLD_TO_CONV_SPLIT[fold]
    qcache = QUERY_CACHE / encoder
    q_path = qcache / f"{conv_split}.npy"
    meta_path = qcache / f"{conv_split}_meta.parquet"
    if not q_path.exists():
        raise FileNotFoundError(
            f"No cached queries at {q_path}. Run "
            f"`03_encode_queries.py --encoder {encoder} --splits {conv_split}` first."
        )
    q_emb = np.load(q_path)
    meta = pl.read_parquet(meta_path)

    if fold in ("val", "test"):
        if not SPLITS_PATH.exists():
            raise FileNotFoundError(f"{SPLITS_PATH} not found. Run 01b_rebuild_split.py first.")
        split_df = load_blinda_split(SPLITS_PATH)
        pairs = turn_pairs_for_fold(split_df, fold)
        idx = select_pair_indices(meta, pairs)
        if not idx:
            raise ValueError(
                f"No rows match the {fold} (session_id, turn) pairs. Was 03 run on 'train', "
                f"and does predict_turn_number align with meta turn_number?"
            )
        q_emb = q_emb[np.asarray(idx, dtype=np.int64)]
        meta = (meta.with_row_index(name="__row")
                    .filter(pl.col("__row").is_in(idx)).drop("__row"))
    return q_emb, meta, conv_split


def compute_ndcg_per_row(q_emb, track_emb, track_mask, meta, id_to_idx, mask_played=True):
    """Return (ndcg_values, turn_numbers, n_skipped). ndcg in [0,1] per scorable row."""
    if track_emb.shape[1] != q_emb.shape[1]:
        raise ValueError(
            f"DIM MISMATCH: queries D={q_emb.shape[1]} vs track tower D={track_emb.shape[1]}. "
            f"Encoder cache and tower disagree — re-encode with 03 (and re-train for LoRA)."
        )
    inv_log = 1.0 / np.log2(np.arange(2, NDCG_K + 2))   # discounts for ranks 1..K

    rows = meta.to_dicts()
    n = len(rows)
    ndcg = np.zeros(n, dtype=np.float64)
    turns = np.full(n, -1, dtype=np.int64)
    scorable = np.zeros(n, dtype=bool)
    n_skipped = 0

    masked_cols = ~track_mask
    for start in range(0, n, SCORE_CHUNK):
        end = min(start + SCORE_CHUNK, n)
        scores = q_emb[start:end] @ track_emb.T          # (b, n_tracks)
        scores[:, masked_cols] = -np.inf
        for bi in range(end - start):
            ri = start + bi
            r = rows[ri]
            tn = r.get("turn_number")
            if tn is not None:
                turns[ri] = int(tn)
            gt_id = r.get("gt_track_id")
            if gt_id is None or gt_id not in id_to_idx:
                n_skipped += 1
                continue
            gt_idx = id_to_idx[gt_id]
            row_scores = scores[bi]
            gt_score = row_scores[gt_idx]
            if not np.isfinite(gt_score):
                # GT itself is masked out of the catalog — unrankable.
                n_skipped += 1
                continue
            # Exclude already-played tracks from the candidate set (same as recall).
            if mask_played:
                for tid in (r.get("prior_track_ids") or []):
                    j = id_to_idx.get(tid)
                    if j is not None:
                        row_scores[j] = -np.inf
                row_scores[gt_idx] = gt_score   # never mask the GT, even if replayed
            # rank = 1 + (#tracks strictly better than GT). Ties: GT ranked after them.
            rank = 1 + int(np.sum(row_scores > gt_score))
            scorable[ri] = True
            if rank <= NDCG_K:
                ndcg[ri] = inv_log[rank - 1]
            # else stays 0.0
    return ndcg, turns, scorable, n_skipped


def macro_by_turn(ndcg, turns, scorable):
    """Per-K mean NDCG, the macro mean over K, and the micro mean."""
    per_turn: dict[int, list[float]] = defaultdict(list)
    vals = []
    for v, k, ok in zip(ndcg.tolist(), turns.tolist(), scorable.tolist()):
        if not ok:
            continue
        vals.append(v)
        per_turn[int(k)].append(v)
    per_turn_mean = {k: float(np.mean(v)) for k, v in sorted(per_turn.items())}
    per_turn_n = {k: len(v) for k, v in sorted(per_turn.items())}
    macro = float(np.mean(list(per_turn_mean.values()))) if per_turn_mean else 0.0
    micro = float(np.mean(vals)) if vals else 0.0
    return macro, micro, per_turn_mean, per_turn_n


def evaluate_one(encoder: str, fold: str, mask_played: bool):
    q_emb, meta, conv_split = _load_queries_and_meta(encoder, fold)
    print(f"queries: {q_emb.shape}, meta rows: {meta.shape[0]}  "
          f"(fold={fold}, conv_split={conv_split})")
    track_emb, track_mask, id_to_idx = load_tower_for_encoder(encoder)
    print(f"track tower: {track_emb.shape}  mask coverage {track_mask.mean():.3f}")

    ndcg, turns, scorable, n_skipped = compute_ndcg_per_row(
        q_emb, track_emb, track_mask, meta, id_to_idx, mask_played=mask_played)
    macro, micro, per_turn_mean, per_turn_n = macro_by_turn(ndcg, turns, scorable)

    n_scorable = int(scorable.sum())
    print(f"\n  scorable rows: {n_scorable}/{meta.shape[0]}  (skipped {n_skipped})")
    print(f"  NDCG@{NDCG_K}  macro-by-turn = {macro:.4f}   micro = {micro:.4f}")
    print(f"  per-turn NDCG@{NDCG_K}:")
    for k in sorted(per_turn_mean):
        print(f"    turn {k:>2}: {per_turn_mean[k]:.4f}   (n={per_turn_n[k]})")

    report = {
        "encoder": encoder, "fold": fold, "metric": f"ndcg@{NDCG_K}",
        "macro_by_turn": macro, "micro": micro,
        "n_scorable": n_scorable, "n_skipped": int(n_skipped),
        "mask_played": bool(mask_played),
        "per_turn_mean": per_turn_mean, "per_turn_n": per_turn_n,
    }
    out_dir = EVAL_OUT / encoder / fold
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ndcg.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  wrote {out_dir / 'ndcg.json'}")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoder", help="Single encoder cache name to score.")
    p.add_argument("--compare", nargs="+", metavar="ENCODER",
                   help="Score several encoders and write a comparison table.")
    p.add_argument("--split", default="val", choices=["val", "test", "dev", "blind_a"])
    p.add_argument("--no-mask-played", action="store_true",
                   help="Do NOT exclude already-played prior tracks from ranking.")
    args = p.parse_args()

    if not args.encoder and not args.compare:
        p.error("give --encoder NAME or --compare NAME [NAME ...]")
    encoders = args.compare if args.compare else [args.encoder]
    mask_played = not args.no_mask_played

    reports = []
    for enc in encoders:
        print("\n" + "=" * 78)
        print(f"  NDCG@{NDCG_K}  |  encoder={enc}  fold={args.split}")
        print("=" * 78)
        reports.append(evaluate_one(enc, args.split, mask_played))

    if len(reports) > 1:
        print("\n" + "#" * 78)
        print(f"  COMPARISON — NDCG@{NDCG_K} on '{args.split}'")
        print("#" * 78)
        print(f"  {'encoder':<42}{'macro-by-turn':>16}{'micro':>10}")
        for r in sorted(reports, key=lambda r: -r["macro_by_turn"]):
            print(f"  {r['encoder']:<42}{r['macro_by_turn']:>16.4f}{r['micro']:>10.4f}")
        best = max(reports, key=lambda r: r["macro_by_turn"])
        print(f"\n  best (macro-by-turn): {best['encoder']}  =  {best['macro_by_turn']:.4f}")

        table = pl.DataFrame([
            {"encoder": r["encoder"], "fold": r["fold"],
             f"ndcg@{NDCG_K}_macro_by_turn": r["macro_by_turn"],
             f"ndcg@{NDCG_K}_micro": r["micro"],
             "n_scorable": r["n_scorable"]}
            for r in sorted(reports, key=lambda r: -r["macro_by_turn"])
        ])
        EVAL_OUT.mkdir(parents=True, exist_ok=True)
        csv_path = EVAL_OUT / f"ndcg_comparison_{args.split}.csv"
        table.write_csv(csv_path)
        print(f"\n  wrote {csv_path}")


if __name__ == "__main__":
    main()