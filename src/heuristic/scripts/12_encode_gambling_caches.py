"""scripts/12_encode_gambling_caches.py  (v4 — adds blindb_all per-turn stage)

Build the EXACT Qwen embeddings the submission/diagnosis pipeline loads — for
Qwen3 0.6B / 4B / 8B — the TRACK tower plus query caches for any subset of:

    tracks    : track tower (metadata-parquet order, instruction "none", len 256)
    blind     : Blind-A queries, ONE row per session at the final user turn   — submission
    blindb_all: Blind-B queries, ONE ROW PER USER TURN (all turns), gt visible
                where the music answer is shown, '' at the masked predict turn  — NEW v4
    val       : Blind-A-matched VAL fold, pinned predict-turn rows (with GT)
    test      : Blind-A-matched TEST fold, pinned predict-turn rows (with GT)
    train     : Blind-A-matched TRAIN fold, ONE ROW PER MUSIC TURN (with GT)
    splitk    : splitK 5-fold CV buckets, ONE ROW PER MUSIC TURN (with GT), per bucket

NEW IN v4: BLINDB_ALL
=====================
Encodes EVERY user turn of the Blind-B parquet (not just the final turn like the
`blind` stage). Each row's query is the visible conversation prefix up to that
turn; gt_track_id is the shown music answer when present, '' at the masked final
(predict) turn. prior_track_ids = music tracks shown at earlier turns. Uses the
SAME query-text + instruction machinery as every other stage. conversation_goal
is present in Blind-B, so Level-2 instructions are leakage-safe.

NEW IN v3: PER-BUCKET QUERY INSTRUCTIONS (Level 2)
==================================================
The Qwen query instruction ("Instruct: ...\\nQuery: ...") is resolved PER ROW from
the session's conversation_goal via emblib.retrieval.query_instructions and the
prompts file --instruction-prompts (default: emblib/retrieval/
instruction_prompts.json; 'none' = legacy fixed "catalog" instruction). The
APPLIED instruction is stored as an "instruction" column in every
query_meta.parquet; the launchers validate that column.

>> EDITING the prompts JSON stales every query cache exactly like editing
>> core.py does — re-encode, the loaders will refuse a mismatched cache.

IMPORT IS MANDATORY: every text/encoding/instruction function is imported from
emblib.retrieval — the SAME modules gambling_updated imports.

NO LEAKAGE: for every (session, K) row, only the visible prefix forms the query.

USAGE (from the package root, GPU node)
=======================================
    uv run python scripts/12_encode_gambling_caches.py --stages blindb_all --models qwen3_8b \
        --instruction-prompts none
    uv run python scripts/12_encode_gambling_caches.py --stages blindb_all \
        --models qwen3_0p6b qwen3_8b --instruction-prompts none
"""
from __future__ import annotations

import argparse
import os as _os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


# --- paths + constants -------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]            # package root (src/heuristic) — for imports
REPO_ROOT = REPO.parents[1]                           # repo root — data + model outputs live here
DATA = REPO_ROOT / "data/talkpl-ai"

TRACK_METADATA_PATH = DATA / "TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet"
BLIND_A_PATH = DATA / "TalkPlayData-Challenge-Blind-A/data/test-00000-of-00001.parquet"
BLIND_B_PATH = DATA / "TalkPlayData-Challenge-Blind-B/data/test-00000-of-00001.parquet"
TRAIN_CONV_PATH = DATA / "TalkPlayData-Challenge-Dataset/data/train-00000-of-00001.parquet"
TEST_CONV_PATH = DATA / "TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet"
SPLIT_PATH = REPO_ROOT / "models/splits/train_val_test_blinda_matched.parquet"
SPLITK_ASSIGNMENT_PATH = REPO_ROOT / "data/splitK/splitK_assignment.parquet"
OUT_ROOT = REPO_ROOT / "models/retrieval_text_towers"

TRACK_SUBDIR = "dense_tracks_len256_poollast"
QUERY_SUBDIR = "dense_blinda_query_len512_poollast"
BLINDB_ALL_SUBDIR = "dense_blindb_all_query_len512_poollast"
TRAIN_QUERY_SUBDIR = "dense_train_query_len512_poollast"
VAL_QUERY_SUBDIR = "dense_val_query_len512_poollast"
TEST_QUERY_SUBDIR = "dense_test_query_len512_poollast"
SPLITK_SUBDIR_FMT = "dense_splitk_{bucket}_query_len512_poollast"

QWEN_TRACK_MAX_LENGTH = 256
QWEN_QUERY_MAX_LENGTH = 512
LOCAL_FILES_ONLY = True
TRUST_REMOTE_CODE = False

MODEL_IDS: dict[str, str] = {
    "qwen3_0p6b": _os.environ.get("QWEN_0P6B_PATH", "Qwen/Qwen3-Embedding-0.6B"),
    "qwen3_4b":   _os.environ.get("QWEN_4B_PATH",   "Qwen/Qwen3-Embedding-4B"),
    "qwen3_8b":   _os.environ.get("QWEN_8B_PATH",   "Qwen/Qwen3-Embedding-8B"),
}
# folder name must stay canonical even when MODEL_IDS is overridden to a local path
_CANON_ID = {"qwen3_0p6b": "Qwen/Qwen3-Embedding-0.6B",
             "qwen3_4b":   "Qwen/Qwen3-Embedding-4B",
             "qwen3_8b":   "Qwen/Qwen3-Embedding-8B"}
MODEL_FOLDER: dict[str, str] = {key: cid.replace("/", "__") for key, cid in _CANON_ID.items()}
DEFAULT_TRACK_BATCH: dict[str, int] = {"qwen3_0p6b": 16, "qwen3_4b": 8, "qwen3_8b": 4}
DEFAULT_QUERY_BATCH: dict[str, int] = {"qwen3_0p6b": 16, "qwen3_4b": 8, "qwen3_8b": 4}

FOLD_STAGES = ("train", "val", "test")
ALL_STAGES = ("tracks", "blind", "blindb_all") + FOLD_STAGES + ("splitk",)


# =============================================================================
# MANDATORY import from the real emblib.retrieval
# =============================================================================
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

try:
    import emblib.retrieval.core as _core
    import emblib.retrieval.qwen_embeddings as _qwen
    import emblib.retrieval.query_instructions as _qinstr
    from emblib.retrieval.core import (
        build_variant_query,
        load_blind_task_rows,
        load_neural_track_metadata,
        make_tasks_from_queries,
        parse_date,
        visible_conversation_prefix,
    )
    from emblib.retrieval.qwen_embeddings import (
        encode_qwen_texts,
        load_or_generate_track_embeddings,
    )
    from emblib.retrieval.query_instructions import (
        describe as describe_prompts,
        load_prompts,
        resolve_instructions,
    )
except Exception as exc:  # noqa: BLE001
    sys.stderr.write(
        "\nFATAL: could not import the real emblib.retrieval functions.\n"
        f"  reason: {exc!r}\n"
        f"  package root inferred as: {REPO}\n"
        "  Fix: run from the package root (cd src/embeddings-package) and make sure\n"
        "  emblib/retrieval/{core.py,qwen_embeddings.py,query_instructions.py} exist.\n"
    )
    raise SystemExit(2)


# =============================================================================
# cache-dir helpers
# =============================================================================
def track_cache_dir(out_root: Path, key: str) -> Path:
    return out_root / MODEL_FOLDER[key] / TRACK_SUBDIR


def query_cache_dir(out_root: Path, key: str) -> Path:
    return out_root / MODEL_FOLDER[key] / QUERY_SUBDIR


def blindb_all_cache_dir(out_root: Path, key: str) -> Path:
    return out_root / MODEL_FOLDER[key] / BLINDB_ALL_SUBDIR


def fold_query_cache_dir(out_root: Path, key: str, fold: str) -> Path:
    sub = {"train": TRAIN_QUERY_SUBDIR, "val": VAL_QUERY_SUBDIR, "test": TEST_QUERY_SUBDIR}[fold]
    return out_root / MODEL_FOLDER[key] / sub


def splitk_query_cache_dir(out_root: Path, key: str, bucket: str) -> Path:
    return out_root / MODEL_FOLDER[key] / SPLITK_SUBDIR_FMT.format(bucket=bucket)


# =============================================================================
# encoders / writers
# =============================================================================
def encode_tracks(key: str, out_root: Path, track_ids, track_docs, batch_size, args) -> Path:
    cache_dir = track_cache_dir(out_root, key)
    print(f"  track cache: {cache_dir}")
    load_or_generate_track_embeddings(
        track_ids=track_ids,
        track_docs=track_docs,
        cache_dir=cache_dir,
        model_name=MODEL_IDS[key],
        max_length=QWEN_TRACK_MAX_LENGTH,
        batch_size=batch_size,
        device_arg=args.device,
        dtype_arg=args.dtype,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    return cache_dir


def _encode_query_cache(
    cache_dir: Path,
    key: str,
    tasks: list[dict[str, Any]],
    query_texts: list[str],
    instructions: list[str],
    batch_size: int,
    args,
    extra_columns: dict[str, list] | None = None,
    skip_if_cached: bool = True,
) -> Path:
    """Encode query_texts (each with ITS row's instruction) and write
    query_embeddings.npy + query_meta.parquet (incl. the 'instruction' column).
    skip_if_cached: skipping requires BOTH query_text AND instruction to match."""
    if len(instructions) != len(query_texts):
        raise ValueError(f"{len(instructions)} instructions for {len(query_texts)} queries")
    emb_path = cache_dir / "query_embeddings.npy"
    meta_path = cache_dir / "query_meta.parquet"
    if skip_if_cached and emb_path.exists() and meta_path.exists():
        try:
            persisted = pl.read_parquet(meta_path)
            same_text = persisted["query_text"].to_list() == query_texts
            same_instr = ("instruction" in persisted.columns
                          and persisted["instruction"].to_list() == instructions)
            if same_text and same_instr and np.load(emb_path, mmap_mode="r").shape[0] == len(query_texts):
                print(f"  query cache up to date (text + instruction), skipping: {cache_dir}")
                return cache_dir
            print(f"  cached query cache stale/mismatched, re-encoding: {cache_dir}")
        except Exception as exc:  # noqa: BLE001
            print(f"  cached query cache unreadable ({exc!r}), re-encoding: {cache_dir}")

    print(f"  query cache: {cache_dir}  ({len(query_texts)} rows)")
    query_embeddings = encode_qwen_texts(
        model_name=MODEL_IDS[key],
        texts=query_texts,
        is_query=True,
        instruction_name=instructions,           # per-row Level-2 instructions
        max_length=QWEN_QUERY_MAX_LENGTH,
        batch_size=batch_size,
        device_arg=args.device,
        dtype_arg=args.dtype,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, query_embeddings.astype(np.float32, copy=False))

    cols: dict[str, list] = {
        "session_id": [t["session_id"] for t in tasks],
        "user_id": [t["user_id"] for t in tasks],
        "turn_number": [int(t["turn_number"]) for t in tasks],
        "query_text": list(query_texts),
        "instruction": list(instructions),
    }
    schema: dict[str, Any] = {
        "session_id": pl.String,
        "user_id": pl.String,
        "turn_number": pl.Int64,
        "query_text": pl.String,
        "instruction": pl.String,
    }
    if extra_columns:
        for name, values in extra_columns.items():
            cols[name] = values
            schema[name] = pl.List(pl.String) if values and isinstance(values[0], list) else pl.String
    pl.DataFrame(cols, schema=schema).write_parquet(meta_path)
    print(f"  wrote {query_embeddings.shape} -> {emb_path}")

    persisted = pl.read_parquet(meta_path)
    bad_text = [i for i, (a, b) in enumerate(zip(persisted["query_text"].to_list(), query_texts)) if a != b]
    bad_instr = [i for i, (a, b) in enumerate(zip(persisted["instruction"].to_list(), instructions)) if a != b]
    if bad_text or bad_instr or persisted.height != len(query_texts):
        raise RuntimeError(
            f"query cache read-back mismatch in {cache_dir}: "
            f"{len(bad_text)} text rows / {len(bad_instr)} instruction rows differ."
        )
    print(f"  read-back OK: {persisted.height} rows (query_text + instruction) match what was built")
    return cache_dir


def encode_blind(key, out_root, tasks, query_texts, instrs, batch_size, args) -> Path:
    return _encode_query_cache(query_cache_dir(out_root, key), key, tasks, query_texts, instrs,
                               batch_size, args, skip_if_cached=False)


def _gt_extra_columns(tasks: list[dict[str, Any]], gts: list[str]) -> dict[str, list]:
    return {
        "gt_track_id": [str(g) for g in gts],
        "prior_track_ids": [[str(t) for t in task["seen_tracks"]] for task in tasks],
    }


def encode_fold(key, out_root, fold, tasks, query_texts, gts, instrs, batch_size, args) -> Path:
    return _encode_query_cache(fold_query_cache_dir(out_root, key, fold), key, tasks, query_texts,
                               instrs, batch_size, args,
                               extra_columns=_gt_extra_columns(tasks, gts), skip_if_cached=False)


def encode_blindb_all(key, out_root, tasks, query_texts, gts, instrs, batch_size, args) -> Path:
    return _encode_query_cache(blindb_all_cache_dir(out_root, key), key, tasks, query_texts,
                               instrs, batch_size, args,
                               extra_columns=_gt_extra_columns(tasks, gts), skip_if_cached=False)


def encode_splitk_bucket(key, out_root, bucket, tasks, query_texts, gts, instrs,
                         batch_size, args) -> Path:
    return _encode_query_cache(splitk_query_cache_dir(out_root, key, bucket), key, tasks,
                               query_texts, instrs, batch_size, args,
                               extra_columns=_gt_extra_columns(tasks, gts),
                               skip_if_cached=not args.splitk_overwrite)


# =============================================================================
# Task-row construction (shared by the fold stages and splitK)
# =============================================================================
def _index_conversations(paths: list[Path]) -> dict[str, dict]:
    idx: dict[str, dict] = {}
    for p in paths:
        if p and Path(p).exists():
            for r in pl.read_parquet(p).iter_rows(named=True):
                idx[str(r["session_id"])] = dict(r)
    return idx


def _music_turns_with_user(convs: list[dict[str, Any]]) -> list[int]:
    by_turn: dict[int, set[str]] = defaultdict(set)
    for t in convs or []:
        tn = t.get("turn_number")
        role = t.get("role")
        if tn is None or role is None:
            continue
        if role == "music" and not t.get("content"):
            continue
        by_turn[int(tn)].add(str(role))
    return sorted(tn for tn, roles in by_turn.items() if {"music", "user"} <= roles)


def _build_one_task_row(sid: str, K: int, row: dict, track_text_by_id: dict[str, str]):
    row = dict(row)
    convs = row.get("conversations") or []
    music_at_k = [t for t in convs
                  if t.get("role") == "music" and t.get("content")
                  and int(t.get("turn_number", -1)) == K]
    user_at_k = [t for t in convs
                 if t.get("role") == "user" and int(t.get("turn_number", -1)) == K]
    if not music_at_k or not user_at_k:
        return None
    gt = str(music_at_k[0]["content"])
    prefix_tracks = [
        str(t.get("content")) for t in convs
        if t.get("role") == "music" and t.get("content")
        and int(t.get("turn_number", -1)) < K
    ]
    row["conversation_prefix"] = visible_conversation_prefix(convs, K)
    row["prefix_track_ids"] = prefix_tracks
    task_row = {
        "session_id": sid,
        "user_id": str(row.get("user_id")),
        "turn_number": int(K),
        "session_date": parse_date(row.get("session_date")),
        "prefix_track_ids": prefix_tracks,
        "row": row,
    }
    return task_row, gt


# ---- Blind-B per-turn builder (includes the masked final/predict turn) -------
def _user_turns(convs: list[dict[str, Any]]) -> list[int]:
    """All turn numbers that have a USER message (these are the points to encode a query at)."""
    turns = set()
    for t in convs or []:
        if t.get("role") == "user" and t.get("turn_number") is not None:
            turns.add(int(t["turn_number"]))
    return sorted(turns)


def _build_blindb_task_row(sid: str, K: int, row: dict, track_text_by_id: dict[str, str]):
    """One row per USER turn K. gt = the music answer shown AT K if present (visible
    turns), else '' (the masked predict turn). prior = music tracks shown at turns < K.
    Always returns a row as long as a user message exists at K."""
    row = dict(row)
    convs = row.get("conversations") or []
    user_at_k = [t for t in convs
                 if t.get("role") == "user" and int(t.get("turn_number", -1)) == K]
    if not user_at_k:
        return None
    music_at_k = [t for t in convs
                  if t.get("role") == "music" and t.get("content")
                  and int(t.get("turn_number", -1)) == K]
    gt = str(music_at_k[0]["content"]) if music_at_k else ""   # '' at masked predict turn
    prefix_tracks = [
        str(t.get("content")) for t in convs
        if t.get("role") == "music" and t.get("content")
        and int(t.get("turn_number", -1)) < K
    ]
    row["conversation_prefix"] = visible_conversation_prefix(convs, K)
    row["prefix_track_ids"] = prefix_tracks
    task_row = {
        "session_id": sid,
        "user_id": ("" if (row.get("user_id") is None
                              or str(row.get("user_id")).strip() in ("", "None"))
                     else str(row.get("user_id"))),
        "turn_number": int(K),
        "session_date": parse_date(row.get("session_date")),
        "prefix_track_ids": prefix_tracks,
        "row": row,
    }
    return task_row, gt


def build_blindb_all_task_rows(blind_b_path: Path, track_text_by_id, prompts_cfg):
    """Every user turn of every Blind-B session -> (tasks, query_texts, gts, instructions)."""
    if not blind_b_path.exists():
        raise SystemExit(f"FATAL: Blind-B parquet not found at {blind_b_path}")
    df = pl.read_parquet(blind_b_path)
    print(f"  Blind-B sessions: {df.height}")
    task_rows: list[dict[str, Any]] = []
    gts: list[str] = []
    n_masked = 0
    n_visible = 0
    for r in df.iter_rows(named=True):
        sid = str(r.get("session_id"))
        convs = r.get("conversations") or []
        for K in _user_turns(convs):
            built = _build_blindb_task_row(sid, int(K), dict(r), track_text_by_id)
            if built is None:
                continue
            task_row, gt = built
            task_rows.append(task_row)
            gts.append(gt)
            if gt == "":
                n_masked += 1
            else:
                n_visible += 1
    print(f"  Blind-B per-turn rows: {len(task_rows)}  "
          f"(visible-music turns={n_visible}, masked/predict turns={n_masked})")
    queries_and_seen = [
        build_variant_query(t["row"], "default", track_text_by_id, include_thoughts=False)
        for t in task_rows
    ]
    tasks, query_texts = make_tasks_from_queries(task_rows, queries_and_seen)
    instrs = resolve_instructions(task_rows, prompts_cfg)
    return tasks, query_texts, gts, instrs


def _rows_to_tasks(sid_turn_pairs, conv_idx, track_text_by_id, label, prompts_cfg):
    """(session, K) pairs -> (tasks, query_texts, gts, instructions)."""
    task_rows: list[dict[str, Any]] = []
    gts: list[str] = []
    miss_session = miss_turn = 0
    for sid, K in sid_turn_pairs:
        row = conv_idx.get(sid)
        if row is None:
            miss_session += 1
            continue
        built = _build_one_task_row(sid, int(K), row, track_text_by_id)
        if built is None:
            miss_turn += 1
            continue
        task_row, gt = built
        task_rows.append(task_row)
        gts.append(gt)
    print(f"  [{label}] usable task rows: {len(task_rows)}  "
          f"(missing session={miss_session}, missing user/music@K={miss_turn})")
    queries_and_seen = [
        build_variant_query(t["row"], "default", track_text_by_id, include_thoughts=False)
        for t in task_rows
    ]
    tasks, query_texts = make_tasks_from_queries(task_rows, queries_and_seen)
    instrs = resolve_instructions(task_rows, prompts_cfg)
    return tasks, query_texts, gts, instrs


def _load_split_pairs(split_path: Path, fold: str) -> list[tuple[str, int]]:
    df = pl.read_parquet(split_path)
    if "split" not in df.columns or "predict_turn_number" not in df.columns:
        raise ValueError(
            f"{split_path} must have columns 'split' and 'predict_turn_number' "
            f"(produced by scripts/01b_rebuild_split.py). Got {df.columns}."
        )
    df = df.filter(pl.col("split") == fold)
    return [(str(s), int(k)) for s, k in zip(df["session_id"].to_list(),
                                             df["predict_turn_number"].to_list())]


def build_fold_task_rows(split_path, fold, conv_idx, track_text_by_id, prompts_cfg):
    pairs = _load_split_pairs(split_path, fold)
    print(f"  {fold} sessions in split: {len(pairs)}  | indexed conv sessions: {len(conv_idx)}")
    if fold == "train":
        expanded: list[tuple[str, int]] = []
        miss = 0
        for sid, _K in pairs:
            row = conv_idx.get(sid)
            if row is None:
                miss += 1
                continue
            for K in _music_turns_with_user(row.get("conversations") or []):
                expanded.append((sid, K))
        print(f"  train: expanded {len(pairs)} sessions -> {len(expanded)} per-turn rows "
              f"(missing sessions={miss})")
        pairs = expanded
    return _rows_to_tasks(sorted(pairs), conv_idx, track_text_by_id, fold, prompts_cfg)


def load_splitk_buckets(assignment_path: Path) -> dict[str, list[str]]:
    df = pl.read_parquet(assignment_path)
    if set(df.columns) != {"session_id", "bucket"}:
        raise ValueError(
            f"{assignment_path} has unexpected columns {df.columns}; expected "
            f"['session_id', 'bucket']."
        )
    out: dict[str, list[str]] = defaultdict(list)
    for sid, bucket in zip(df["session_id"].to_list(), df["bucket"].to_list()):
        out[str(bucket)].append(str(sid))
    return dict(out)


def build_splitk_bucket_task_rows(bucket, bucket_sids, conv_idx, track_text_by_id, prompts_cfg):
    pairs: list[tuple[str, int]] = []
    miss = 0
    for sid in bucket_sids:
        row = conv_idx.get(sid)
        if row is None:
            miss += 1
            continue
        for K in _music_turns_with_user(row.get("conversations") or []):
            pairs.append((sid, K))
    print(f"  [{bucket}] {len(bucket_sids)} sessions -> {len(pairs)} per-turn rows "
          f"(missing sessions={miss})")
    return _rows_to_tasks(sorted(pairs), conv_idx, track_text_by_id, bucket, prompts_cfg)


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(MODEL_IDS), choices=list(MODEL_IDS))
    parser.add_argument("--stages", nargs="+", default=["all"], choices=["all", *ALL_STAGES])
    parser.add_argument("--instruction-prompts", default="default",
                        help="path to the Level-2 per-bucket prompts JSON; 'default' = "
                             "emblib/retrieval/instruction_prompts.json (legacy fixed "
                             "instruction if that file is absent); 'none' = legacy.")
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--blind-path", type=Path, default=BLIND_A_PATH)
    parser.add_argument("--blind-b-path", type=Path, default=BLIND_B_PATH)
    parser.add_argument("--track-metadata-path", type=Path, default=TRACK_METADATA_PATH)
    parser.add_argument("--split-path", type=Path, default=SPLIT_PATH)
    parser.add_argument("--train-conv", type=Path, default=TRAIN_CONV_PATH)
    parser.add_argument("--test-conv", type=Path, default=TEST_CONV_PATH)
    parser.add_argument("--splitk-assignment", type=Path, default=SPLITK_ASSIGNMENT_PATH)
    parser.add_argument("--splitk-buckets", nargs="+", default=None)
    parser.add_argument("--splitk-overwrite", action="store_true")
    parser.add_argument("--track-batch-size", type=int, default=None)
    parser.add_argument("--query-batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--allow-downloads", dest="local_files_only", action="store_false", default=True)
    parser.add_argument("--trust-remote-code", action="store_true", default=TRUST_REMOTE_CODE)
    args = parser.parse_args()
    stages = set(args.stages)
    if "all" in stages:
        stages = set(ALL_STAGES)
    args.stages = stages
    return args


def main() -> None:
    args = parse_args()
    prompts_cfg = load_prompts(args.instruction_prompts)

    print("text/encoder functions: emblib.retrieval (imported -- query text + instruction "
          "guaranteed identical to gambling_updated)")
    print(f"  emblib.retrieval.core               : {_core.__file__}")
    print(f"  emblib.retrieval.qwen_embeddings    : {_qwen.__file__}")
    print(f"  emblib.retrieval.query_instructions : {_qinstr.__file__}")
    print(f"  instruction prompts                 : {describe_prompts(prompts_cfg)}")
    print(f"  stages: {sorted(args.stages)}")

    want = lambda s: s in args.stages  # noqa: E731

    track_ids, track_docs, track_text_by_id, _release_dates, _ = \
        load_neural_track_metadata(args.track_metadata_path)
    print(f"track metadata: {len(track_ids)} tracks (metadata-parquet order)")

    need_conv_idx = bool(args.stages & ({"splitk"} | set(FOLD_STAGES)))
    conv_idx: dict[str, dict] = {}
    if need_conv_idx:
        print("indexing conversations (train + test + blind)...")
        conv_idx = _index_conversations([args.train_conv, args.test_conv, args.blind_path])
        print(f"  indexed {len(conv_idx)} sessions")

    # --- blind (Blind-A, one row per session) ----------------------------------
    blind_tasks: list[dict[str, Any]] = []
    blind_query_texts: list[str] = []
    blind_instrs: list[str] = []
    if want("blind"):
        task_rows = load_blind_task_rows(args.blind_path)
        queries_and_seen = [
            build_variant_query(task_row["row"], "default", track_text_by_id, include_thoughts=False)
            for task_row in task_rows
        ]
        blind_tasks, blind_query_texts = make_tasks_from_queries(task_rows, queries_and_seen)
        blind_instrs = resolve_instructions(task_rows, prompts_cfg)
        n_nondefault = sum(1 for i in blind_instrs
                           if prompts_cfg and i != prompts_cfg["base"])
        print(f"blind queries: {len(blind_tasks)} tasks "
              f"({n_nondefault} rows get a per-bucket instruction)")

    # --- blindb_all (Blind-B, ALL user turns) ----------------------------------
    blindb_payload = None
    if want("blindb_all"):
        print("building Blind-B per-turn task rows (ALL turns)...")
        blindb_payload = build_blindb_all_task_rows(args.blind_b_path, track_text_by_id, prompts_cfg)
        print(f"blindb_all queries: {len(blindb_payload[0])} tasks")

    # --- train / val / test -----------------------------------------------------
    fold_payloads: dict[str, tuple[list, list, list, list]] = {}
    requested_folds = [f for f in FOLD_STAGES if want(f)]
    if requested_folds:
        if not args.split_path.exists():
            raise SystemExit(f"FATAL: split not found at {args.split_path}. Run scripts/01b first.")
        for fold in requested_folds:
            print(f"building {fold} task rows from {args.split_path}")
            fold_payloads[fold] = build_fold_task_rows(
                args.split_path, fold, conv_idx, track_text_by_id, prompts_cfg,
            )
            print(f"{fold} queries: {len(fold_payloads[fold][0])} tasks")

    # --- splitK ----------------------------------------------------------------
    splitk_payloads: dict[str, tuple[list, list, list, list]] = {}
    if want("splitk"):
        if not args.splitk_assignment.exists():
            print(f"\n[skip] splitk: {args.splitk_assignment} not found.")
        else:
            buckets = load_splitk_buckets(args.splitk_assignment)
            selected = args.splitk_buckets or sorted(buckets)
            unknown = [b for b in selected if b not in buckets]
            if unknown:
                raise SystemExit(f"FATAL: unknown splitK bucket(s) {unknown}")
            print(f"splitK: encoding {len(selected)} bucket(s): {selected}")
            for bucket in selected:
                splitk_payloads[bucket] = build_splitk_bucket_task_rows(
                    bucket, buckets[bucket], conv_idx, track_text_by_id, prompts_cfg,
                )

    # --- per-model encoding ------------------------------------------------------
    written: list[Path] = []
    for key in args.models:
        track_bs = args.track_batch_size or DEFAULT_TRACK_BATCH[key]
        query_bs = args.query_batch_size or DEFAULT_QUERY_BATCH[key]
        print("\n" + "#" * 72)
        print(f"  {key}  ({MODEL_IDS[key]})  track_bs={track_bs}  query_bs={query_bs}")
        print("#" * 72)
        if want("tracks"):
            written.append(encode_tracks(key, args.out_root, track_ids, track_docs, track_bs, args))
        if want("blind"):
            written.append(encode_blind(key, args.out_root, blind_tasks, blind_query_texts,
                                        blind_instrs, query_bs, args))
        if want("blindb_all") and blindb_payload is not None:
            bt, bq, bg, bi = blindb_payload
            print(f"\n-- encoding BLINDB_ALL queries ({len(bt)} rows) --")
            written.append(encode_blindb_all(key, args.out_root, bt, bq, bg, bi, query_bs, args))
        for fold, (tasks, qtexts, gts, instrs) in fold_payloads.items():
            print(f"\n-- encoding {fold.upper()} queries ({len(tasks)} rows) --")
            written.append(encode_fold(key, args.out_root, fold, tasks, qtexts, gts, instrs,
                                       query_bs, args))
        for bucket, (tasks, qtexts, gts, instrs) in splitk_payloads.items():
            print(f"\n-- encoding splitK bucket {bucket} ({len(tasks)} rows) --")
            written.append(encode_splitk_bucket(key, args.out_root, bucket, tasks, qtexts, gts,
                                                instrs, query_bs, args))

    print("\nDone. Caches written/verified:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()