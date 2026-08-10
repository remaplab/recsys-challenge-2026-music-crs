from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import sparse

from emblib.retrieval.paths import BLIND_A_PATH, TRACK_METADATA_PATH, TRAIN_PATH

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_TRACK_METADATA_PATH = TRACK_METADATA_PATH
DEFAULT_TRAIN_PATH = TRAIN_PATH
DEFAULT_BLIND_PATH = BLIND_A_PATH
TOKEN_RE = re.compile(r"(?u)\b\w+\b")
PROFILE_FIELDS = (
    "age_group",
    "country_code",
    "country_name",
    "gender",
    "preferred_language",
    "preferred_musical_culture",
    "user_split",
)
GOAL_FIELDS = ("category", "specificity", "listener_goal")


def unwrap_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(unwrap_text(item) for item in value if item is not None)
    return str(value).strip()


def parse_date(value: Any) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def track_doc(row: dict[str, Any]) -> str:
    title = unwrap_text(row.get("track_name"))
    artist = unwrap_text(row.get("artist_name"))
    album = unwrap_text(row.get("album_name"))
    tags = unwrap_text(row.get("tag_list"))
    # Title and artist are repeated because user requests often name them directly.
    return " ".join(
        part
        for part in [
            title,
            title,
            title,
            title,
            artist,
            artist,
            artist,
            artist,
            artist,
            album,
            album,
            tags,
            tags,
        ]
        if part
    )


def text_terms(text: str) -> list[str]:
    tokens = [token.lower() for token in TOKEN_RE.findall(text)]
    if not tokens:
        return []
    bigrams = [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]
    return tokens + bigrams


def fit_tfidf(texts: list[str], max_features: int) -> tuple[dict[str, int], np.ndarray, sparse.csr_matrix]:
    doc_term_counts: list[Counter[str]] = []
    document_frequency: Counter[str] = Counter()
    total_frequency: Counter[str] = Counter()
    for text in texts:
        counts = Counter(text_terms(text))
        doc_term_counts.append(counts)
        document_frequency.update(counts.keys())
        total_frequency.update(counts)

    terms = [
        term
        for term, _ in total_frequency.most_common(max_features)
        if document_frequency[term] > 0
    ]
    vocabulary = {term: index for index, term in enumerate(terms)}
    idf = np.array(
        [math.log((1 + len(texts)) / (1 + document_frequency[term])) + 1.0 for term in terms],
        dtype=np.float32,
    )
    matrix = transform_tfidf_from_counts(doc_term_counts, vocabulary, idf)
    return vocabulary, idf, matrix


def transform_tfidf(texts: list[str], vocabulary: dict[str, int], idf: np.ndarray) -> sparse.csr_matrix:
    counts = [Counter(text_terms(text)) for text in texts]
    return transform_tfidf_from_counts(counts, vocabulary, idf)


def transform_tfidf_from_counts(
    counts_by_row: list[Counter[str]],
    vocabulary: dict[str, int],
    idf: np.ndarray,
) -> sparse.csr_matrix:
    row_indices: list[int] = []
    col_indices: list[int] = []
    values: list[float] = []
    for row_index, counts in enumerate(counts_by_row):
        for term, count in counts.items():
            col_index = vocabulary.get(term)
            if col_index is None:
                continue
            row_indices.append(row_index)
            col_indices.append(col_index)
            values.append((1.0 + math.log(float(count))) * float(idf[col_index]))
    matrix = sparse.csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(counts_by_row), len(vocabulary)),
        dtype=np.float32,
    )
    row_norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
    nonzero = row_norms > 0
    if nonzero.any():
        inv_norms = np.zeros_like(row_norms, dtype=np.float32)
        inv_norms[nonzero] = 1.0 / row_norms[nonzero]
        matrix = sparse.diags(inv_norms).dot(matrix).tocsr()
    return matrix


def load_track_metadata(path: Path) -> tuple[pl.DataFrame, list[str], list[str], dict[str, str], dict[str, date | None]]:
    df = pl.read_parquet(path)
    rows = df.to_dicts()
    track_ids = [str(row["track_id"]) for row in rows]
    docs = [track_doc(row) for row in rows]
    metadata_text = {
        str(row["track_id"]): " ".join(
            part
            for part in [
                unwrap_text(row.get("track_name")),
                unwrap_text(row.get("artist_name")),
                unwrap_text(row.get("album_name")),
                unwrap_text(row.get("tag_list")),
            ]
            if part
        )
        for row in rows
    }
    release_dates = {str(row["track_id"]): parse_date(row.get("release_date")) for row in rows}
    return df, track_ids, docs, metadata_text, release_dates


def release_ordinals(track_ids: list[str], release_dates: dict[str, date | None]) -> np.ndarray:
    # Unknown release dates are allowed, so encode them as 0.
    return np.array(
        [release_dates[track_id].toordinal() if release_dates.get(track_id) is not None else 0 for track_id in track_ids],
        dtype=np.int32,
    )


def build_popularity(train_path: Path, track_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    train_df = pl.read_parquet(train_path, columns=["conversations"])
    counts = Counter(
        str(turn["content"])
        for conversation in train_df["conversations"]
        for turn in conversation
        if turn.get("role") == "music"
    )
    raw = np.array([math.log1p(counts[track_id]) for track_id in track_ids], dtype=np.float32)
    if raw.max() > raw.min():
        scores = (raw - raw.min()) / (raw.max() - raw.min())
    else:
        scores = np.zeros_like(raw)
    popular = [track_id for track_id, _ in counts.most_common()]
    seen = set(popular)
    popular.extend(track_id for track_id in track_ids if track_id not in seen)
    return scores, popular


def last_user_turn_number(conversation: list[dict[str, Any]]) -> int:
    return max(int(turn["turn_number"]) for turn in conversation if turn.get("role") == "user")


def prefixed_field_text(prefix: str, data: dict[str, Any] | None, fields: tuple[str, ...]) -> str:
    if not data:
        return ""
    parts = []
    for field in fields:
        value = unwrap_text(data.get(field))
        if value:
            parts.append(f"{prefix}_{field}_{value}")
            parts.append(value)
    return " ".join(parts)


def conversation_query(
    conversation: list[dict[str, Any]],
    target_turn: int,
    track_metadata_text: dict[str, str],
    user_profile: dict[str, Any] | None,
    conversation_goal: dict[str, Any] | None,
) -> tuple[str, list[str]]:
    parts: list[str] = []
    seen_tracks: list[str] = []
    profile_text = prefixed_field_text("profile", user_profile, PROFILE_FIELDS)
    goal_text = prefixed_field_text("goal", conversation_goal, GOAL_FIELDS)
    if profile_text:
        parts.append(profile_text)
    if goal_text:
        parts.extend([goal_text] * 3)
    for turn in conversation:
        turn_number = int(turn["turn_number"])
        role = turn.get("role")
        if turn_number > target_turn:
            break
        if role == "user" and turn_number == target_turn:
            text = unwrap_text(turn.get("content"))
            thought = unwrap_text(turn.get("thought"))
            parts.extend([text] * 8)
            if thought:
                parts.append(thought)
            break
        if turn_number < target_turn and role == "user":
            text = unwrap_text(turn.get("content"))
            thought = unwrap_text(turn.get("thought"))
            parts.extend([text] * 2)
            if thought:
                parts.append(thought)
        elif turn_number < target_turn and role == "assistant":
            parts.append(unwrap_text(turn.get("content")))
        elif turn_number < target_turn and role == "music":
            track_id = str(turn.get("content"))
            seen_tracks.append(track_id)
            track_text = track_metadata_text.get(track_id, "")
            if track_text:
                parts.extend([track_text] * 3)
            thought = unwrap_text(turn.get("thought"))
            if thought:
                parts.append(thought)
    return " ".join(part for part in parts if part), seen_tracks


def make_tasks(
    df: pl.DataFrame,
    track_metadata_text: dict[str, str],
    all_user_turns: bool,
) -> list[dict[str, Any]]:
    tasks = []
    for row in df.iter_rows(named=True):
        conversation = row["conversations"]
        target_turns = (
            [int(turn["turn_number"]) for turn in conversation if turn.get("role") == "user"]
            if all_user_turns
            else [last_user_turn_number(conversation)]
        )
        for target_turn in target_turns:
            query, seen_tracks = conversation_query(
                conversation=conversation,
                target_turn=target_turn,
                track_metadata_text=track_metadata_text,
                user_profile=row.get("user_profile"),
                conversation_goal=row.get("conversation_goal"),
            )
            tasks.append(
                {
                    "session_id": str(row["session_id"]),
                    "user_id": str(row["user_id"]),
                    "turn_number": int(target_turn),
                    "session_date": parse_date(row.get("session_date")),
                    "query": query,
                    "seen_tracks": seen_tracks,
                }
            )
    return tasks


def allowed_by_date(track_id: str, session_date: date | None, release_dates: dict[str, date | None]) -> bool:
    if session_date is None:
        return True
    release_date = release_dates.get(track_id)
    return release_date is None or release_date <= session_date


def top_tracks(
    scores: np.ndarray,
    track_ids: list[str],
    track_index: dict[str, int],
    release_ord: np.ndarray,
    popular_items: list[str],
    seen_tracks: list[str],
    session_date: date | None,
    release_dates: dict[str, date | None],
    top_k: int,
    filter_future_releases: bool,
) -> list[str]:
    blocked = set(seen_tracks)
    row_scores = scores.copy()
    if blocked:
        for track_id in blocked:
            idx = track_index.get(track_id)
            if idx is not None:
                row_scores[idx] = -np.inf
    if filter_future_releases and session_date is not None:
        row_scores[release_ord > session_date.toordinal()] = -np.inf

    finite_count = int(np.isfinite(row_scores).sum())
    recs: list[str] = []
    if finite_count > 0:
        k = min(max(top_k * 5, top_k), finite_count)
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
        if filter_future_releases and not allowed_by_date(track_id, session_date, release_dates):
            continue
        recs.append(track_id)
        if len(recs) >= top_k:
            break
    return recs[:top_k]


def generate_submission(args: argparse.Namespace) -> None:
    _, track_ids, docs, track_metadata_text, release_dates = load_track_metadata(args.track_metadata_path)
    track_index = {track_id: index for index, track_id in enumerate(track_ids)}
    release_ord = release_ordinals(track_ids, release_dates)
    popularity_scores, popular_items = build_popularity(args.train_path, track_ids)

    vocabulary, idf, track_matrix = fit_tfidf(docs, max_features=args.max_features)

    data_df = pl.read_parquet(args.eval_path)
    tasks = make_tasks(data_df, track_metadata_text, all_user_turns=args.all_user_turns)

    rows = []
    for start in range(0, len(tasks), args.batch_size):
        batch = tasks[start : start + args.batch_size]
        query_matrix = transform_tfidf([task["query"] for task in batch], vocabulary, idf)
        score_matrix = (query_matrix @ track_matrix.T).toarray().astype(np.float32)
        if args.popularity_weight:
            score_matrix += args.popularity_weight * popularity_scores[None, :]
        for local_idx, task in enumerate(batch):
            recs = top_tracks(
                scores=score_matrix[local_idx],
                track_ids=track_ids,
                track_index=track_index,
                release_ord=release_ord,
                popular_items=popular_items,
                seen_tracks=task["seen_tracks"],
                session_date=task["session_date"],
                release_dates=release_dates,
                top_k=args.top_k,
                filter_future_releases=args.filter_future_releases,
            )
            rows.append(
                {
                    "session_id": task["session_id"],
                    "user_id": task["user_id"],
                    "turn_number": task["turn_number"],
                    "predicted_track_ids": recs,
                    "predicted_response": "",
                }
            )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w") as f:
        json.dump(rows, f, indent=2)
    print(f"rows: {len(rows)}")
    print(f"saved submission to: {args.output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a text/metadata TF-IDF recommendation submission.")
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_BLIND_PATH)
    parser.add_argument("--track-metadata-path", type=Path, default=DEFAULT_TRACK_METADATA_PATH)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-features", type=int, default=200_000)
    parser.add_argument("--popularity-weight", type=float, default=0.03)
    parser.add_argument("--filter-future-releases", action="store_true")
    parser.add_argument("--all-user-turns", action="store_true", help="Use for devset evaluation; Blind A should omit this.")
    return parser.parse_args()


def main() -> None:
    generate_submission(parse_args())


if __name__ == "__main__":
    main()
