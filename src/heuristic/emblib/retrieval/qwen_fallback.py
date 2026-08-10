from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from emblib.retrieval.config import BestSubmissionConfig
from emblib.retrieval.qwen_embeddings import (
    encode_qwen_texts,
    load_or_generate_track_embeddings,
    load_track_metadata,
    parse_date,
    unwrap_text,
)


def profile_text(row: dict[str, Any]) -> str:
    profile = row.get("user_profile") or {}
    fields = [
        ("age group", profile.get("age_group")),
        ("country", profile.get("country_code")),
        ("language", profile.get("preferred_language")),
        ("musical culture", profile.get("preferred_musical_culture")),
    ]
    return "; ".join(f"{name}: {unwrap_text(value)}" for name, value in fields if unwrap_text(value))


def goal_text(row: dict[str, Any]) -> str:
    goal = row.get("conversation_goal") or {}
    fields = [
        ("listener goal", goal.get("listener_goal")),
        ("category", goal.get("category")),
        ("specificity", goal.get("specificity")),
    ]
    return "; ".join(f"{name}: {unwrap_text(value)}" for name, value in fields if unwrap_text(value))


def visible_conversation_prefix(conversation: list[dict[str, Any]], target_turn: int) -> list[dict[str, Any]]:
    return [
        turn
        for turn in conversation
        if int(turn.get("turn_number", -1)) < target_turn
        or (int(turn.get("turn_number", -1)) == target_turn and turn.get("role") == "user")
    ]


def target_turn_and_prefix(row: dict[str, Any]) -> tuple[int, list[str]]:
    conversation = row.get("conversations") or []
    user_turns = [turn for turn in conversation if turn.get("role") == "user"]
    if not user_turns:
        raise ValueError(f"row has no user turns: {row.get('session_id')}")
    target_turn = max(int(turn.get("turn_number", -1)) for turn in user_turns)
    prefix_tracks = [
        str(turn.get("content"))
        for turn in conversation
        if turn.get("role") == "music"
        and turn.get("content")
        and int(turn.get("turn_number", -1)) < target_turn
    ]
    return target_turn, prefix_tracks


def load_blind_task_rows(blind_path: Path) -> list[dict[str, Any]]:
    df = pl.read_parquet(blind_path)
    task_rows: list[dict[str, Any]] = []
    for raw_row in df.iter_rows(named=True):
        row = dict(raw_row)
        target_turn, prefix_tracks = target_turn_and_prefix(row)
        row["conversation_prefix"] = visible_conversation_prefix(row.get("conversations") or [], target_turn)
        row["prefix_track_ids"] = prefix_tracks
        task_rows.append(
            {
                "session_id": str(row["session_id"]),
                "user_id": str(row["user_id"]),
                "turn_number": int(target_turn),
                "session_date": parse_date(row.get("session_date")),
                "prefix_track_ids": prefix_tracks,
                "row": row,
            }
        )
    return task_rows


def build_query_text(row: dict[str, Any], track_text_by_id: dict[str, str]) -> tuple[str, list[str]]:
    conversation = row.get("conversation_prefix") or row.get("conversations") or []
    user_turns = [turn for turn in conversation if turn.get("role") == "user"]
    last_user = user_turns[-1] if user_turns else {}
    last_turn_number = int(last_user.get("turn_number", -1)) if last_user else -1

    seen_tracks = [str(track_id) for track_id in (row.get("prefix_track_ids") or [])]
    parts: list[str] = []

    ptext = profile_text(row)
    if ptext:
        parts.append(f"User profile: {ptext}")
    gtext = goal_text(row)
    if gtext:
        parts.append(f"Conversation goal: {gtext}")
    if row.get("session_date"):
        parts.append(f"Session date: {row['session_date']}")

    history_parts: list[str] = []
    for turn in conversation:
        role = turn.get("role")
        turn_number = int(turn.get("turn_number", -1))
        if turn_number >= last_turn_number:
            continue
        if role == "user":
            text = unwrap_text(turn.get("content"))
            if text:
                history_parts.append(f"Previous user request: {text}")
        elif role == "music":
            track_id = str(turn.get("content"))
            if track_id and track_id not in seen_tracks:
                seen_tracks.append(track_id)
            track_text = track_text_by_id.get(track_id, "")
            if track_text:
                history_parts.append(f"Previously played track:\n{track_text}")
    if history_parts:
        parts.append("History:\n" + "\n".join(history_parts))

    current = unwrap_text(last_user.get("content"))
    if current:
        parts.append(f"Current user request: {current}")
    return "\n\n".join(parts), seen_tracks


def build_train_popularity(train_path: Path, track_ids: list[str]) -> list[str]:
    df = pl.read_parquet(train_path)
    if "track_ids" in df.columns:
        counts = Counter(track_id for seq in df["track_ids"] for track_id in seq)
    else:
        counts = Counter(
            str(turn["content"])
            for conversation in df["conversations"]
            for turn in conversation
            if turn.get("role") == "music"
        )
    popular_items = [track_id for track_id, _ in counts.most_common()]
    seen = set(popular_items)
    popular_items.extend(track_id for track_id in track_ids if track_id not in seen)
    return popular_items


def release_ordinals(track_ids: list[str], release_dates: dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            release_dates[track_id].toordinal() if release_dates.get(track_id) is not None else 0
            for track_id in track_ids
        ],
        dtype=np.int32,
    )


def filter_and_rank(
    *,
    scores: np.ndarray,
    seen_tracks: list[str],
    session_date: Any,
    track_ids: list[str],
    track_index: dict[str, int],
    release_ord: np.ndarray,
    release_dates: dict[str, Any],
    popular_items: list[str],
    top_k: int,
    filter_future_releases: bool,
) -> list[str]:
    row_scores = scores.copy()
    blocked = set(seen_tracks)
    for track_id in blocked:
        idx = track_index.get(track_id)
        if idx is not None:
            row_scores[idx] = -np.inf
    if filter_future_releases and session_date is not None:
        row_scores[release_ord > session_date.toordinal()] = -np.inf

    recs: list[str] = []
    finite_count = int(np.isfinite(row_scores).sum())
    if finite_count > 0:
        k = min(max(top_k * 10, top_k), finite_count)
        top_idx = np.argpartition(-row_scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-row_scores[top_idx])]
        for idx in top_idx:
            track_id = track_ids[int(idx)]
            if track_id not in blocked and track_id not in recs:
                recs.append(track_id)
            if len(recs) >= top_k:
                break

    for track_id in popular_items:
        if track_id in blocked or track_id in recs:
            continue
        if filter_future_releases and session_date is not None:
            release_date = release_dates.get(track_id)
            if release_date is not None and release_date > session_date:
                continue
        recs.append(track_id)
        if len(recs) >= top_k:
            break
    return recs[:top_k]


def qwen_score_rows(
    *,
    query_embeddings: np.ndarray,
    track_embeddings: np.ndarray,
    batch_size: int,
) -> list[np.ndarray]:
    rows: list[np.ndarray] = []
    tracks = track_embeddings.astype(np.float32, copy=False)
    for start in range(0, len(query_embeddings), batch_size):
        end = min(start + batch_size, len(query_embeddings))
        scores = query_embeddings[start:end].astype(np.float32, copy=False) @ tracks.T
        rows.extend([row.astype(np.float32, copy=False) for row in scores])
    return rows


def generate_qwen_fallback_recs(
    config: BestSubmissionConfig,
    task_rows: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str, int], list[str]], dict[str, Any]]:
    track_ids, track_docs, track_text_by_id, release_dates = load_track_metadata(config.track_metadata_path)
    track_embeddings = load_or_generate_track_embeddings(
        track_ids=track_ids,
        track_docs=track_docs,
        cache_dir=config.qwen_track_cache_dir,
        model_name=config.qwen_model,
        max_length=config.qwen_track_max_length,
        batch_size=config.qwen_track_batch_size,
        device_arg=config.device,
        dtype_arg=config.dtype,
        local_files_only=config.local_files_only,
        trust_remote_code=config.trust_remote_code,
    )

    query_texts: list[str] = []
    seen_by_row: list[list[str]] = []
    for task in task_rows:
        query, seen_tracks = build_query_text(task["row"], track_text_by_id)
        query_texts.append(query)
        seen_by_row.append(seen_tracks)

    query_embeddings = encode_qwen_texts(
        model_name=config.qwen_model,
        texts=query_texts,
        is_query=True,
        instruction_name=config.qwen_instruction,
        max_length=config.qwen_query_max_length,
        batch_size=config.qwen_query_batch_size,
        device_arg=config.device,
        dtype_arg=config.dtype,
        local_files_only=config.local_files_only,
        trust_remote_code=config.trust_remote_code,
    )
    score_rows = qwen_score_rows(
        query_embeddings=query_embeddings,
        track_embeddings=track_embeddings,
        batch_size=config.score_batch_size,
    )

    track_index = {track_id: idx for idx, track_id in enumerate(track_ids)}
    release_ord = release_ordinals(track_ids, release_dates)
    popular_items = build_train_popularity(config.train_path, track_ids)

    recs_by_key: dict[tuple[str, str, int], list[str]] = {}
    for row_idx, task in enumerate(task_rows):
        key = (task["session_id"], task["user_id"], int(task["turn_number"]))
        recs_by_key[key] = filter_and_rank(
            scores=score_rows[row_idx],
            seen_tracks=seen_by_row[row_idx],
            session_date=task["session_date"],
            track_ids=track_ids,
            track_index=track_index,
            release_ord=release_ord,
            release_dates=release_dates,
            popular_items=popular_items,
            top_k=config.top_k,
            filter_future_releases=config.filter_future_releases_for_qwen,
        )

    return recs_by_key, {
        "track_ids": track_ids,
        "release_dates": release_dates,
        "qwen_rows": len(recs_by_key),
        "qwen_track_cache_dir": str(config.qwen_track_cache_dir),
    }
