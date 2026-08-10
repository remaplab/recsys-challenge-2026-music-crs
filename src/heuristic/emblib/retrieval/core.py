from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import json
import numpy as np
import polars as pl
from scipy import sparse


QWEN_MODEL = "Qwen/Qwen3-Embedding-0.6B"
SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"

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


@dataclass(frozen=True)
class QueryWeights:
    last_user: int
    previous_user: int
    assistant: int
    music_metadata: int
    music_thought: int
    user_thought: int
    goal: int
    profile: int


@dataclass(frozen=True)
class DocWeights:
    name: str
    title: int
    artist: int
    album: int
    tags: int


@dataclass(frozen=True)
class SparseIndex:
    vocab: dict[str, int]
    idf_tfidf: np.ndarray
    idf_bm25: np.ndarray
    doc_counts: list[Counter[str]]
    doc_lengths: list[int]


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


def release_ordinals(track_ids: list[str], release_dates: dict[str, date | None]) -> np.ndarray:
    return np.array(
        [release_dates[track_id].toordinal() if release_dates.get(track_id) is not None else 0 for track_id in track_ids],
        dtype=np.int32,
    )


def metadata_text(row: dict[str, Any]) -> str:
    return " ".join(
        part
        for part in [
            unwrap_text(row.get("track_name")),
            unwrap_text(row.get("artist_name")),
            unwrap_text(row.get("album_name")),
            unwrap_text(row.get("tag_list")),
        ]
        if part
    )


def neural_metadata_doc(row: dict[str, Any]) -> str:
    title = unwrap_text(row.get("track_name"))
    artist = unwrap_text(row.get("artist_name"))
    album = unwrap_text(row.get("album_name"))
    tags = unwrap_text(row.get("tag_list"))
    release_date = parse_date(row.get("release_date"))
    year = str(release_date.year) if release_date is not None else ""
    return "\n".join(
        part
        for part in [
            f"Title: {title}" if title else "",
            f"Artist: {artist}" if artist else "",
            f"Album: {album}" if album else "",
            f"Tags: {tags}" if tags else "",
            f"Year: {year}" if year else "",
        ]
        if part
    )


def load_neural_track_metadata(
    path: Path,
) -> tuple[list[str], list[str], dict[str, str], dict[str, date | None], np.ndarray]:
    rows = pl.read_parquet(path).to_dicts()
    track_ids = [str(row["track_id"]) for row in rows]
    docs = [neural_metadata_doc(row) for row in rows]
    text_by_track = {str(row["track_id"]): doc for row, doc in zip(rows, docs)}
    release_dates = {str(row["track_id"]): parse_date(row.get("release_date")) for row in rows}
    popularity = np.array([float(row.get("popularity") or 0.0) for row in rows], dtype=np.float32)
    if popularity.max() > popularity.min():
        popularity = (popularity - popularity.min()) / (popularity.max() - popularity.min())
    return track_ids, docs, text_by_track, release_dates, popularity


def terms(text: str, ngram_max: int) -> list[str]:
    tokens = [token.lower() for token in TOKEN_RE.findall(text)]
    if not tokens:
        return []
    out = list(tokens)
    for n in range(2, max(ngram_max, 1) + 1):
        out.extend("_".join(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1))
    return out


def term_weight(count: int, mode: str) -> float:
    if mode == "binary":
        return 1.0
    if mode == "raw":
        return float(count)
    if mode == "sqrt":
        return math.sqrt(float(count))
    if mode == "log":
        return 1.0 + math.log(float(count))
    raise ValueError(f"unknown term frequency mode: {mode}")


def sparse_track_doc(row: dict[str, Any], weights: DocWeights) -> str:
    title = unwrap_text(row.get("track_name"))
    artist = unwrap_text(row.get("artist_name"))
    album = unwrap_text(row.get("album_name"))
    tags = unwrap_text(row.get("tag_list"))
    parts: list[str] = []
    parts.extend([title] * weights.title)
    parts.extend([artist] * weights.artist)
    parts.extend([album] * weights.album)
    parts.extend([tags] * weights.tags)
    return " ".join(part for part in parts if part)


def load_track_data(
    path: Path,
    doc_weights: DocWeights,
) -> tuple[list[str], list[str], dict[str, str], dict[str, date | None], np.ndarray]:
    rows = pl.read_parquet(path).to_dicts()
    track_ids = [str(row["track_id"]) for row in rows]
    docs = [sparse_track_doc(row, doc_weights) for row in rows]
    text_by_track = {str(row["track_id"]): metadata_text(row) for row in rows}
    release_dates = {str(row["track_id"]): parse_date(row.get("release_date")) for row in rows}
    popularity = np.array([float(row.get("popularity") or 0.0) for row in rows], dtype=np.float32)
    if popularity.max() > popularity.min():
        popularity = (popularity - popularity.min()) / (popularity.max() - popularity.min())
    return track_ids, docs, text_by_track, release_dates, popularity


def build_sparse_index(texts: list[str], max_features: int, ngram_max: int) -> SparseIndex:
    doc_counts: list[Counter[str]] = []
    df_counts: Counter[str] = Counter()
    tf_counts: Counter[str] = Counter()
    doc_lengths: list[int] = []
    for text in texts:
        counts = Counter(terms(text, ngram_max))
        doc_counts.append(counts)
        df_counts.update(counts.keys())
        tf_counts.update(counts)
        doc_lengths.append(sum(counts.values()))

    vocab_terms = [term for term, _ in tf_counts.most_common(max_features)]
    vocab = {term: idx for idx, term in enumerate(vocab_terms)}
    n_docs = len(texts)
    idf_tfidf = np.array(
        [math.log((1.0 + n_docs) / (1.0 + df_counts[term])) + 1.0 for term in vocab_terms],
        dtype=np.float32,
    )
    idf_bm25 = np.array(
        [math.log(1.0 + (n_docs - df_counts[term] + 0.5) / (df_counts[term] + 0.5)) for term in vocab_terms],
        dtype=np.float32,
    )
    return SparseIndex(
        vocab=vocab,
        idf_tfidf=idf_tfidf,
        idf_bm25=idf_bm25,
        doc_counts=doc_counts,
        doc_lengths=doc_lengths,
    )

# ---- caption-augmented track doc -------------------------------------------
_CAPTION_BY_TRACK: dict[str, str] | None = None

def load_caption_map(caption_path) -> dict[str, str]:
    """track_id -> caption string. Cached module-global so repeated calls are free."""
    global _CAPTION_BY_TRACK
    if _CAPTION_BY_TRACK is None:
        df = pl.read_parquet(caption_path)
        # second column is the caption regardless of its exact name
        cap_col = [c for c in df.columns if c != "track_id"][0]
        _CAPTION_BY_TRACK = {str(t): unwrap_text(c)
                             for t, c in zip(df["track_id"].to_list(), df[cap_col].to_list())}
    return _CAPTION_BY_TRACK

def neural_metadata_doc_with_caption(row, caption_by_track) -> str:
    """Base metadata doc + a Caption: field appended (Tags/Year stay before it so
    right-truncation eats caption last, after the structured fields)."""
    base = neural_metadata_doc(row)
    cap = caption_by_track.get(str(row.get("track_id")), "")
    if cap:
        return base + f"\nCaption: {cap}" if base else f"Caption: {cap}"
    return base

def load_neural_track_metadata_captioned(path, caption_path):
    """Same signature/returns as load_neural_track_metadata but docs include caption."""
    caption_by_track = load_caption_map(caption_path)
    rows = pl.read_parquet(path).to_dicts()
    track_ids = [str(r["track_id"]) for r in rows]
    docs = [neural_metadata_doc_with_caption(r, caption_by_track) for r in rows]
    text_by_track = {str(r["track_id"]): d for r, d in zip(rows, docs)}
    release_dates = {str(r["track_id"]): parse_date(r.get("release_date")) for r in rows}
    popularity = np.array([float(r.get("popularity") or 0.0) for r in rows], dtype=np.float32)
    if popularity.max() > popularity.min():
        popularity = (popularity - popularity.min()) / (popularity.max() - popularity.min())
    return track_ids, docs, text_by_track, release_dates, popularity

def transform_query_counts(
    texts: list[str],
    vocab: dict[str, int],
    query_tf: str,
    ngram_max: int,
) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for row_idx, text in enumerate(texts):
        for term, count in Counter(terms(text, ngram_max)).items():
            col_idx = vocab.get(term)
            if col_idx is None:
                continue
            rows.append(row_idx)
            cols.append(col_idx)
            values.append(term_weight(count, query_tf))
    return sparse.csr_matrix((values, (rows, cols)), shape=(len(texts), len(vocab)), dtype=np.float32)


def transform_bm25_docs(index: SparseIndex, k1: float, b: float) -> sparse.csr_matrix:
    avgdl = float(np.mean(index.doc_lengths)) if index.doc_lengths else 1.0
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for row_idx, counts in enumerate(index.doc_counts):
        dl = max(float(index.doc_lengths[row_idx]), 1.0)
        norm = k1 * (1.0 - b + b * dl / max(avgdl, 1e-6))
        for term, count in counts.items():
            col_idx = index.vocab.get(term)
            if col_idx is None:
                continue
            tf = float(count)
            value = float(index.idf_bm25[col_idx]) * (tf * (k1 + 1.0)) / (tf + norm)
            rows.append(row_idx)
            cols.append(col_idx)
            values.append(value)
    return sparse.csr_matrix((values, (rows, cols)), shape=(len(index.doc_counts), len(index.vocab)), dtype=np.float32)


def build_train_popularity(train_path: Path, track_ids: list[str]) -> tuple[np.ndarray, list[str]]:
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
    raw = np.array([math.log1p(counts[track_id]) for track_id in track_ids], dtype=np.float32)
    if raw.max() > raw.min():
        scores = (raw - raw.min()) / (raw.max() - raw.min())
    else:
        scores = np.zeros_like(raw)
    popular_items = [track_id for track_id, _ in counts.most_common()]
    seen = set(popular_items)
    popular_items.extend(track_id for track_id in track_ids if track_id not in seen)
    return scores, popular_items


def prefixed_field_text(prefix: str, data: dict | None, fields: tuple[str, ...]) -> str:
    if not data:
        return ""
    parts: list[str] = []
    for field in fields:
        value = unwrap_text(data.get(field))
        if value:
            parts.append(f"{prefix}_{field}_{value}")
            parts.append(value)
    return " ".join(parts)


def query_from_row_weighted(
    row: dict[str, Any],
    track_text_by_id: dict[str, str],
    weights: QueryWeights,
) -> tuple[str, list[str]]:
    conversation = row.get("conversation_prefix") or row["conversations"]
    parts: list[str] = []
    seen_tracks: list[str] = []
    profile_text = prefixed_field_text("profile", row.get("user_profile"), PROFILE_FIELDS)
    goal_text = prefixed_field_text("goal", row.get("conversation_goal"), GOAL_FIELDS)
    parts.extend([profile_text] * weights.profile)
    parts.extend([goal_text] * weights.goal)

    user_turns = [turn for turn in conversation if turn.get("role") == "user"]
    last_user_turn_number = int(user_turns[-1]["turn_number"]) if user_turns else -1
    for turn in conversation:
        role = turn.get("role")
        turn_number = int(turn.get("turn_number", -1))
        if role == "user":
            text = unwrap_text(turn.get("content"))
            thought = unwrap_text(turn.get("thought"))
            repeat = weights.last_user if turn_number == last_user_turn_number else weights.previous_user
            parts.extend([text] * repeat)
            parts.extend([thought] * weights.user_thought)
        elif role == "assistant":
            text = unwrap_text(turn.get("content"))
            parts.extend([text] * weights.assistant)
        elif role == "music":
            track_id = str(turn.get("content"))
            seen_tracks.append(track_id)
            track_text = track_text_by_id.get(track_id, "")
            thought = unwrap_text(turn.get("thought"))
            parts.extend([track_text] * weights.music_metadata)
            parts.extend([thought] * weights.music_thought)
    return " ".join(part for part in parts if part), seen_tracks


def profile_text(row: dict[str, Any]) -> str:
    profile = row.get("user_profile") or {}
    fields = [
        ("age group", profile.get("age_group")),
        ("country", profile.get("country_code")),
        ("language", profile.get("preferred_language")),
        ("musical culture", profile.get("preferred_musical_culture")),
    ]
    return "; ".join(f"{name}: {unwrap_text(value)}" for name, value in fields if unwrap_text(value))


CATEGORY_NAME = {"A": "audio-based discovery", "B": "lyrical discovery",
    "C": "visual-musical connections", "D": "contextual and situational",
    "E": "interactive refinement", "F": "metadata-rich exploration",
    "G": "mood and emotion based", "H": "artist and discography discovery",
    "I": "cultural and geographic", "J": "social and popularity context",
    "K": "temporal and era discovery"}
TARGET_DESC = {"H": "the listener seeks one specific track",
               "L": "many tracks could satisfy the request"}

def goal_text(row):
    goal = row.get("conversation_goal") or {}
    cat = (goal.get("category") or "").strip().upper()
    spec = (goal.get("specificity") or "").strip().upper()
    fields = [
        ("listener goal", goal.get("listener_goal")),
        ("goal type", CATEGORY_NAME.get(cat, cat)),
        ("target", TARGET_DESC.get(spec[1:] or "", spec)),
    ]
    return "; ".join(f"{name}: {unwrap_text(value)}" for name, value in fields if unwrap_text(value))


def build_query_text(
    row: dict[str, Any],
    track_text_by_id: dict[str, str],
    include_thoughts: bool,
    include_current_user_thought: bool = False,
) -> tuple[str, list[str]]:
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
            thought = unwrap_text(turn.get("thought"))
            if include_thoughts and thought:
                history_parts.append(f"Previous user thought: {thought}")
        elif role == "music":
            track_id = str(turn.get("content"))
            if track_id and track_id not in seen_tracks:
                seen_tracks.append(track_id)
            track_text = track_text_by_id.get(track_id, "")
            if track_text:
                history_parts.append(f"Previously played track:\n{track_text}")
            thought = unwrap_text(turn.get("thought"))
            if include_thoughts and thought:
                history_parts.append(f"Previous music thought: {thought}")
    if history_parts:
        parts.append("History:\n" + "\n".join(history_parts))

    current = unwrap_text(last_user.get("content"))
    if current:
        parts.append(f"Current user request: {current}")
    thought = unwrap_text(last_user.get("thought"))
    if (include_thoughts or include_current_user_thought) and thought:
        parts.append(f"Current user thought: {thought}")

    return "\n\n".join(parts), seen_tracks


def current_user_text(row: dict[str, Any]) -> str:
    conversation = row.get("conversation_prefix") or row.get("conversations") or []
    users = [turn for turn in conversation if turn.get("role") == "user"]
    return unwrap_text(users[-1].get("content")) if users else ""


def build_variant_query(
    row: dict[str, Any],
    variant: str,
    track_text_by_id: dict[str, str],
    include_thoughts: bool,
    include_current_user_thought: bool = False
) -> tuple[str, list[str]]:
    default_query, seen_tracks = build_query_text(row, track_text_by_id, include_thoughts,include_current_user_thought=include_current_user_thought)
    if variant == "default":
        return default_query, seen_tracks

    request = current_user_text(row)
    ptext = profile_text(row)
    gtext = goal_text(row)
    if variant == "request":
        parts = [f"Current user request: {request}"]
    elif variant == "goal_request":
        parts = [
            f"Conversation goal: {gtext}" if gtext else "",
            f"Current user request: {request}",
        ]
    elif variant == "profile_goal_request":
        parts = [
            f"User profile: {ptext}" if ptext else "",
            f"Conversation goal: {gtext}" if gtext else "",
            f"Current user request: {request}",
        ]
    elif variant == "profile_goal_date_request":
        parts = [
            f"User profile: {ptext}" if ptext else "",
            f"Conversation goal: {gtext}" if gtext else "",
            f"Session date: {row['session_date']}" if row.get("session_date") else "",
            f"Current user request: {request}",
        ]
    else:
        raise ValueError(f"unknown query variant: {variant}")
    return "\n\n".join(part for part in parts if part), seen_tracks


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


def visible_conversation_prefix(conversation: list[dict[str, Any]], target_turn: int) -> list[dict[str, Any]]:
    return [
        turn
        for turn in conversation
        if int(turn.get("turn_number", -1)) < target_turn
        or (int(turn.get("turn_number", -1)) == target_turn and turn.get("role") == "user")
    ]


def load_blind_task_rows(blind_path: Path) -> list[dict[str, Any]]:
    df = pl.read_parquet(blind_path)
    task_rows: list[dict[str, Any]] = []
    for row in df.iter_rows(named=True):
        row = dict(row)
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


def make_tasks_from_queries(
    task_rows: list[dict[str, Any]],
    queries_and_seen: list[tuple[str, list[str]]],
) -> tuple[list[dict[str, Any]], list[str]]:
    tasks: list[dict[str, Any]] = []
    query_texts: list[str] = []
    for task_row, (query, seen_tracks) in zip(task_rows, queries_and_seen):
        tasks.append(
            {
                "session_id": task_row["session_id"],
                "user_id": task_row["user_id"],
                "turn_number": task_row["turn_number"],
                "session_date": task_row["session_date"],
                "query": query,
                "seen_tracks": seen_tracks,
                "prefix_track_ids": task_row["prefix_track_ids"],
            }
        )
        query_texts.append(query)
    return tasks, query_texts


def rank_rows(
    *,
    score_rows: list[np.ndarray],
    tasks: list[dict[str, Any]],
    track_ids: list[str],
    track_index: dict[str, int],
    release_ord: np.ndarray,
    popular_items: list[str],
    release_dates: dict[str, date | None],
    top_k: int,
    filter_future_releases: bool,
) -> dict[tuple[str, str, int], list[str]]:
    recs_by_key: dict[tuple[str, str, int], list[str]] = {}
    for row_idx, task in enumerate(tasks):
        key = (task["session_id"], task["user_id"], int(task["turn_number"]))
        recs_by_key[key] = filter_and_rank(
            scores=np.asarray(score_rows[row_idx], dtype=np.float32),
            task=task,
            track_ids=track_ids,
            track_index=track_index,
            release_ord=release_ord,
            popular_items=popular_items,
            release_dates=release_dates,
            top_k=top_k,
            filter_future_releases=filter_future_releases,
        )
    return recs_by_key


def rows_from_base_with_recs(
    *,
    base_submission_path: Path,
    recs_by_key: dict[tuple[str, str, int], list[str]],
    top_k: int,
    require_all_base_rows: bool,
) -> list[dict[str, Any]]:
    rows = json.loads(base_submission_path.read_text())
    replaced = 0
    out_rows: list[dict[str, Any]] = []
    for base_row in rows:
        key = (str(base_row["session_id"]), str(base_row["user_id"]), int(base_row["turn_number"]))
        recs = recs_by_key.get(key)
        if recs is None:
            if require_all_base_rows:
                raise ValueError(f"missing recommendations for base row key={key}")
            out_rows.append(dict(base_row))
            continue
        if len(recs) != top_k:
            raise ValueError(f"wrong recommendation length for {key}: {len(recs)}")
        row = dict(base_row)
        row["predicted_track_ids"] = recs
        out_rows.append(row)
        replaced += 1

    if replaced != len(recs_by_key):
        raise ValueError(f"replaced {replaced} rows, expected {len(recs_by_key)}")

    lengths = sorted({len(row["predicted_track_ids"]) for row in out_rows})
    if lengths != [top_k]:
        raise ValueError(f"unexpected prediction lengths: {lengths}")
    return out_rows


def filter_and_rank(
    scores: np.ndarray,
    task: dict[str, Any],
    track_ids: list[str],
    track_index: dict[str, int],
    release_ord: np.ndarray,
    popular_items: list[str],
    release_dates: dict[str, date | None],
    top_k: int,
    filter_future_releases: bool,
) -> list[str]:
    row_scores = scores.copy()
    blocked = set(task["seen_tracks"])
    for track_id in blocked:
        idx = track_index.get(track_id)
        if idx is not None:
            row_scores[idx] = -np.inf
    if filter_future_releases and task["session_date"] is not None:
        row_scores[release_ord > task["session_date"].toordinal()] = -np.inf

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
        if filter_future_releases and task["session_date"] is not None:
            release_date = release_dates.get(track_id)
            if release_date is not None and release_date > task["session_date"]:
                continue
        recs.append(track_id)
        if len(recs) >= top_k:
            break
    return recs[:top_k]


def dense_score_rows(query_embeddings: np.ndarray, track_embeddings: np.ndarray, batch_size: int) -> list[np.ndarray]:
    rows: list[np.ndarray] = []
    tracks = track_embeddings.astype(np.float32, copy=False)
    for start in range(0, len(query_embeddings), batch_size):
        end = min(start + batch_size, len(query_embeddings))
        scores = query_embeddings[start:end].astype(np.float32, copy=False) @ tracks.T
        rows.extend([row.astype(np.float32, copy=False) for row in scores])
    return rows


def resolve_torch(device_arg: str, dtype_arg: str):
    import torch

    if device_arg == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_arg)

    if dtype_arg == "auto":
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    elif dtype_arg == "float16":
        dtype = torch.float16
    elif dtype_arg == "bfloat16":
        dtype = torch.bfloat16
    elif dtype_arg == "float32":
        dtype = torch.float32
    else:
        raise ValueError(f"unknown dtype: {dtype_arg}")
    return torch, device, dtype


def load_mlm_model(model_name: str, dtype: Any, trust_remote_code: bool, local_files_only: bool):
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    try:
        model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
    except TypeError:
        model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
    return tokenizer, model


class SpladeQueryEncoder:
    def __init__(
        self,
        model_name: str,
        device_arg: str,
        dtype_arg: str,
        trust_remote_code: bool,
        local_files_only: bool,
    ):
        torch, device, dtype = resolve_torch(device_arg, dtype_arg)
        tokenizer, model = load_mlm_model(model_name, dtype, trust_remote_code, local_files_only)
        self.torch = torch
        self.device = device
        self.tokenizer = tokenizer
        self.model = model.to(device).eval()
        self.vocab_size = len(tokenizer)

    def encode_many_top_terms(
        self,
        texts: list[str],
        max_length: int,
        top_terms_values: list[int],
        min_weight_values: list[float],
        batch_size: int,
    ) -> dict[tuple[int, float], sparse.csr_matrix]:
        max_top_terms = max(top_terms_values)
        if max_top_terms <= 0 or max_top_terms > self.vocab_size:
            max_top_terms = self.vocab_size

        buffers: dict[tuple[int, float], tuple[list[int], list[int], list[float]]] = {
            (top_terms, min_weight): ([], [], [])
            for top_terms in top_terms_values
            for min_weight in min_weight_values
        }

        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            print(f"  encode splade query len={max_length} {start}:{end}/{len(texts)}")
            tok = self.tokenizer(
                texts[start:end],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            with self.torch.no_grad():
                logits = self.model(**tok).logits.float()
                weights = self.torch.log1p(self.torch.relu(logits))
                weights = weights * tok["attention_mask"].unsqueeze(-1).float()
                vectors = weights.max(dim=1).values
                if max_top_terms < vectors.shape[1]:
                    top_values, top_indices = self.torch.topk(vectors, k=max_top_terms, dim=1)
                else:
                    top_values = vectors
                    top_indices = self.torch.arange(vectors.shape[1], device=self.device).expand(vectors.shape[0], -1)
            top_values_np = top_values.cpu().numpy()
            top_indices_np = top_indices.cpu().numpy()

            for local_row in range(top_values_np.shape[0]):
                row_number = start + local_row
                values = top_values_np[local_row]
                indices = top_indices_np[local_row].astype(int)
                for top_terms in top_terms_values:
                    limit = len(values) if top_terms <= 0 else min(top_terms, len(values))
                    for min_weight in min_weight_values:
                        keep = values[:limit] > min_weight
                        rows, cols, vals = buffers[(top_terms, min_weight)]
                        rows.extend([row_number] * int(keep.sum()))
                        cols.extend(indices[:limit][keep].tolist())
                        vals.extend(values[:limit][keep].astype(float).tolist())

        return {
            key: sparse.csr_matrix(
                (vals, (rows, cols)),
                shape=(len(texts), self.vocab_size),
                dtype=np.float32,
            )
            for key, (rows, cols, vals) in buffers.items()
        }


def encode_splade_sparse(
    model_name: str,
    texts: list[str],
    *,
    batch_size: int,
    max_length: int,
    top_terms: int,
    min_weight: float,
    device_arg: str,
    dtype_arg: str,
    trust_remote_code: bool,
    local_files_only: bool,
) -> sparse.csr_matrix:
    torch, device, dtype = resolve_torch(device_arg, dtype_arg)
    tokenizer, model = load_mlm_model(model_name, dtype, trust_remote_code, local_files_only)
    model = model.to(device).eval()
    vocab_size = len(tokenizer)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        print(f"  encode splade track {start}:{end}/{len(texts)}")
        tok = tokenizer(texts[start:end], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**tok).logits.float()
            weights = torch.log1p(torch.relu(logits))
            weights = weights * tok["attention_mask"].unsqueeze(-1).float()
            vectors = weights.max(dim=1).values
            if top_terms > 0 and top_terms < vectors.shape[1]:
                top_values, top_indices = torch.topk(vectors, k=top_terms, dim=1)
            else:
                top_values = vectors
                top_indices = torch.arange(vectors.shape[1], device=device).expand(vectors.shape[0], -1)
        top_values_np = top_values.cpu().numpy()
        top_indices_np = top_indices.cpu().numpy()
        for local_row in range(top_values_np.shape[0]):
            keep = top_values_np[local_row] > min_weight
            global_row = start + local_row
            rows.extend([global_row] * int(keep.sum()))
            cols.extend(top_indices_np[local_row][keep].astype(int).tolist())
            vals.extend(top_values_np[local_row][keep].astype(float).tolist())
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(texts), vocab_size), dtype=np.float32)


def load_or_generate_splade_track_matrix(
    *,
    cache_dir: Path,
    track_ids: list[str],
    track_docs: list[str],
    model_name: str,
    max_length: int,
    top_terms: int,
    min_weight: float,
    batch_size: int,
    device_arg: str,
    dtype_arg: str,
    local_files_only: bool,
    trust_remote_code: bool,
) -> sparse.csr_matrix:
    ids_path = cache_dir / "track_ids.npy"
    matrix_path = cache_dir / "doc_sparse.npz"
    if ids_path.exists() and matrix_path.exists():
        cached_ids = [str(track_id) for track_id in np.load(ids_path, allow_pickle=True).tolist()]
        if cached_ids == track_ids:
            print(f"loading cached SPLADE track matrix: {matrix_path}")
            return sparse.load_npz(matrix_path).tocsr()
        print(f"cached SPLADE track IDs mismatch, regenerating: {cache_dir}")

    print(f"generating SPLADE track matrix: {cache_dir}")
    doc_matrix = encode_splade_sparse(
        model_name,
        track_docs,
        batch_size=batch_size,
        max_length=max_length,
        top_terms=top_terms,
        min_weight=min_weight,
        device_arg=device_arg,
        dtype_arg=dtype_arg,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(ids_path, np.asarray(track_ids, dtype=object))
    sparse.save_npz(matrix_path, doc_matrix)
    return doc_matrix


def sparse_score_rows(query_matrix: sparse.csr_matrix, doc_matrix: sparse.csr_matrix, batch_size: int) -> list[np.ndarray]:
    doc_t = doc_matrix.T.tocsr()
    rows: list[np.ndarray] = []
    for start in range(0, query_matrix.shape[0], batch_size):
        end = min(start + batch_size, query_matrix.shape[0])
        scores = (query_matrix[start:end] @ doc_t).toarray().astype(np.float32)
        rows.extend([row for row in scores])
    return rows
