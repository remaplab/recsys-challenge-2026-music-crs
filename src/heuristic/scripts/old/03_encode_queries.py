"""Encode queries with the chosen encoder and cache to disk.

Usage:
  python scripts/03_encode_queries.py --encoder qwen3_frozen --splits train dev blind_a
  python scripts/03_encode_queries.py --encoder qwen3_lora            --adapter <ckpt> --splits dev
  python scripts/03_encode_queries.py --encoder bert_proj_lora_routing --adapter <ckpt> --splits dev
  python scripts/03_encode_queries.py --encoder bert_native_frozen --splits dev
  python scripts/03_encode_queries.py --encoder keyword_bert_qwen3 --splits dev

Cache layout (per encoder + adapter):
    models/query_emb_cache/<name>/<split>.npy
    models/query_emb_cache/<name>/<split>_meta.parquet
where <name> = encoder if no adapter, else encoder__adapter_basename.

Query text: built by `build_query_text_v2`, which drops noise fields (raw age,
gender, country_name, category/specificity enum labels, goal_progress
assessments, assistant content) and resolves prior-track ids to lines that
look like track-tower entries via the `track_lookup` we load below.
"""
from __future__ import annotations
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._encoders_factory import build_encoder
from emblib.data.parsing import build_query_text_v2
from emblib.data.user_features import load_user_features
from emblib.encoders.base import encode_corpus


DATA = Path("./data/talkpl-ai")
OUT_BASE = Path("./models/query_emb_cache")
USER_CACHE = Path("./models/user_features_cache")
TRACK_META = DATA / "TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet"

SPLIT_TO_PARQUET = {
    "train":   DATA / "TalkPlayData-Challenge-Dataset/data/train-00000-of-00001.parquet",
    "dev":     DATA / "TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet",
    "blind_a": DATA / "TalkPlayData-Challenge-Blind-A/data/test-00000-of-00001.parquet",
}

META_SCHEMA = {
    "session_id": pl.String, "user_id": pl.String, "turn_number": pl.Int64,
    "gt_track_id": pl.String, "category": pl.String, "specificity": pl.String,
    "prior_track_ids": pl.List(pl.String),
}


def load_track_lookup(track_meta_path: Path) -> dict[str, dict]:
    """Build a track_id -> metadata-row dict for v2's [PLAYED] line resolution.

    The lookup is keyed by track_id; each value is the full track-metadata
    parquet row (track_name, artist_name, tag_list, release_date, ...). It
    is read once per script invocation (~47k entries, ~50 MB of dict overhead).
    """
    print(f"Loading track lookup from {track_meta_path}")
    md = pl.read_parquet(track_meta_path)
    lookup = {row["track_id"]: row for row in md.to_dicts()}
    print(f"  {len(lookup)} tracks indexed")
    return lookup


def parse_session(item: dict, track_lookup: dict[str, dict]):
    """Convert one raw session row into one query-row per assistant turn.

    The query text is built by `build_query_text_v2`, which:
      - drops raw age, gender, country_name, category/specificity enum labels,
        goal_progress assessments, and assistant `content`;
      - keeps preferred_musical_culture, age_group, listener_goal, chat history
        thoughts, and the current user message;
      - injects `[PLAYED]` lines (resolved name+artist+tags+year via
        track_lookup) and a `[SESSION] year=YYYY` line;
      - lays out the prompt so the always-survives block (profile, goal,
        session, current-user) is at the bottom and survives left-truncation.
    """
    convs = item["conversations"]
    user_profile = item.get("user_profile") or {}
    conv_goal = item.get("conversation_goal") or {}
    cat = (conv_goal.get("category") or "")
    spec = (conv_goal.get("specificity") or "")
    session_date = item.get("session_date")
    if session_date is not None:
        session_date = str(session_date)

    turns_by_number = defaultdict(list)
    for t in convs:
        turns_by_number[t["turn_number"]].append(t)

    rows = []
    for tn in sorted(turns_by_number.keys()):
        user_msgs = [t for t in turns_by_number[tn] if t["role"] == "user"]
        if not user_msgs:
            continue
        music_msgs = [t for t in turns_by_number[tn] if t["role"] == "music"]
        gt = music_msgs[0]["content"] if music_msgs else None

        chat_history, prior_ids = [], []
        for t_ in sorted(turns_by_number.keys()):
            if t_ >= tn:
                break
            for prev in turns_by_number[t_]:
                chat_history.append({
                    "role": prev.get("role"),
                    "content": prev.get("content") or "",
                    "thought": prev.get("thought") or "",
                })
                if prev.get("role") == "music" and prev.get("content"):
                    prior_ids.append(prev["content"])

        text = build_query_text_v2(
            chat_history=chat_history,
            user_query=user_msgs[0]["content"],
            user_profile=user_profile,
            conversation_goal=conv_goal,
            session_date=session_date,
            track_lookup=track_lookup,
            use_thoughts=True,
        )
        rows.append({
            "session_id": item["session_id"], "user_id": item["user_id"],
            "turn_number": tn, "gt_track_id": gt,
            "category": cat, "specificity": spec,
            "prior_track_ids": prior_ids, "query_text": text,
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", required=True)
    p.add_argument("--adapter", type=Path, default=None,
                   help="Required for qwen3_lora and *_proj_* encoders.")
    p.add_argument("--splits", nargs="+", default=["train", "dev", "blind_a"])
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    cache_name = args.encoder
    if args.adapter is not None:
        cache_name = f"{args.encoder}__{args.adapter.name}"
    out_dir = OUT_BASE / cache_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = build_encoder(args.encoder, args.adapter).to(device)

    users = load_user_features(
        user_meta_path=DATA / "TalkPlayData-Challenge-User-Metadata/data/all_users-00000-of-00001.parquet",
        user_emb_train_path=DATA / "TalkPlayData-Challenge-User-Embeddings/data/train-00000-of-00001.parquet",
        user_emb_warm_path=DATA / "TalkPlayData-Challenge-User-Embeddings/data/test_warm-00000-of-00001.parquet",
        user_emb_cold_path=DATA / "TalkPlayData-Challenge-User-Embeddings/data/test_cold-00000-of-00001.parquet",
        cache_dir=USER_CACHE,
    )

    track_lookup = load_track_lookup(TRACK_META)

    for split in args.splits:
        path = SPLIT_TO_PARQUET.get(split)
        if path is None or not path.exists():
            print(f"[skip] {split}: {path} missing"); continue
        print(f"\n=== {split} ===")
        df = pl.read_parquet(path)
        rows = []
        for item in tqdm(df.to_dicts(), desc=f"parse {split}"):
            rows.extend(parse_session(item, track_lookup))
        print(f"  {len(rows)} (session, turn) rows")

        n = len(rows)
        u_cf = np.zeros((n, users.cf.shape[1]), dtype=np.float32)
        is_cold = np.ones(n, dtype=bool)
        for i, r in enumerate(rows):
            uid = r["user_id"]
            if uid in users.id_to_idx:
                u_idx = users.id_to_idx[uid]
                u_cf[i] = users.cf[u_idx]
                is_cold[i] = bool(users.is_cold[u_idx])

        emb = encode_corpus(encoder, [r["query_text"] for r in rows],
                            u_cf, is_cold, batch_size=args.batch_size)
        np.save(out_dir / f"{split}.npy", emb)

        meta_df = pl.DataFrame({
            "session_id": [r["session_id"] for r in rows],
            "user_id": [r["user_id"] for r in rows],
            "turn_number": [r["turn_number"] for r in rows],
            "gt_track_id": [r["gt_track_id"] for r in rows],
            "category": [r["category"] for r in rows],
            "specificity": [r["specificity"] for r in rows],
            "prior_track_ids": [r["prior_track_ids"] for r in rows],
        }, schema=META_SCHEMA)
        meta_df.write_parquet(out_dir / f"{split}_meta.parquet")
        print(f"  wrote {emb.shape} -> {out_dir}")


if __name__ == "__main__":
    main()