"""scripts/08b_encode_blinda_qwen_fallback.py

Encode the BLIND-A queries and write them where heuristic_200 reads them:

    models/query_emb_generic_cache/<key>/blind_a.npy
    models/query_emb_generic_cache/<key>/blind_a_meta.parquet

SELF-CONTAINED: the query builder (load_blind_task_rows / build_query_text and
helpers) is copied VERBATIM from scripts/launchers/qwen_fallback.py, and the
encoder (encode_qwen_texts / load_track_metadata and helpers) VERBATIM from
scripts/launchers/qwen_embeddings.py. Constants mirror config.py. So the query
vectors are produced by the same code qwen_fallback runs — no cross-package
imports, this file can live in scripts/.

USAGE (cluster, from repo root)
===============================
    uv run python scripts/08b_encode_blinda_qwen_fallback.py                  # all three
    uv run python scripts/08b_encode_blinda_qwen_fallback.py --models qwen3_0p6b
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


# ─── paths + constants (mirroring config.BestSubmissionConfig) ───────────────
REPO = Path(__file__).resolve().parents[1]                       # scripts/ -> repo root
DATA = REPO / "data/talkpl-ai"
TRACK_META_PATH = DATA / "TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet"
BLIND_PATH = DATA / "TalkPlayData-Challenge-Blind-A/data/test-00000-of-00001.parquet"
QUERY_CACHE = REPO / "models/query_emb_generic_cache"

QWEN_INSTRUCTION = "catalog"
QWEN_QUERY_MAX_LENGTH = 512
QWEN_QUERY_BATCH_SIZE = 16
DEVICE = "auto"
DTYPE = "auto"
LOCAL_FILES_ONLY = True
TRUST_REMOTE_CODE = False

MODEL_IDS: dict[str, str] = {
    "qwen3_0p6b": "Qwen/Qwen3-Embedding-0.6B",
    "qwen3_4b":   "Qwen/Qwen3-Embedding-4B",
    "qwen3_8b":   "Qwen/Qwen3-Embedding-8B",
}


# ─── VERBATIM from qwen_embeddings.py ────────────────────────────────────────
QWEN_INSTRUCTIONS = {
    "catalog": "Given a music recommendation request, retrieve the most relevant track from the catalog",
    "track": "Match this listener request to the track metadata that best satisfies it",
    "song": "Retrieve songs that match the listener intent, artist hints, mood, genre, era, and constraints",
    "none": "",
}


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


def track_metadata_text(row: dict[str, Any]) -> str:
    release_date = parse_date(row.get("release_date"))
    year = str(release_date.year) if release_date is not None else ""
    fields = [
        ("Title", unwrap_text(row.get("track_name"))),
        ("Artist", unwrap_text(row.get("artist_name"))),
        ("Album", unwrap_text(row.get("album_name"))),
        ("Tags", unwrap_text(row.get("tag_list"))),
        ("Year", year),
    ]
    return "\n".join(f"{name}: {value}" for name, value in fields if value)


def load_track_metadata(path: Path) -> tuple[list[str], list[str], dict[str, str], dict[str, date | None]]:
    rows = pl.read_parquet(path).to_dicts()
    track_ids = [str(row["track_id"]) for row in rows]
    docs = [track_metadata_text(row) for row in rows]
    text_by_track = {track_id: doc for track_id, doc in zip(track_ids, docs)}
    release_dates = {str(row["track_id"]): parse_date(row.get("release_date")) for row in rows}
    return track_ids, docs, text_by_track, release_dates


def resolve_torch(device_arg: str, dtype_arg: str):
    import torch

    if device_arg == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
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


def pool_last_token(hidden: Any, attention_mask: Any):
    import torch
    import torch.nn.functional as F

    is_left_padded = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if is_left_padded:
        pooled = hidden[:, -1]
    else:
        lengths = attention_mask.sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]
    return F.normalize(pooled.float(), p=2, dim=1)


def qwen_query_prefix(text: str, instruction_name: str) -> str:
    instruction = QWEN_INSTRUCTIONS.get(instruction_name, instruction_name)
    if not instruction:
        return text
    return f"Instruct: {instruction}\nQuery: {text}"


def encode_qwen_texts(
    *,
    model_name: str,
    texts: list[str],
    is_query: bool,
    instruction_name: str,
    max_length: int,
    batch_size: int,
    device_arg: str,
    dtype_arg: str,
    local_files_only: bool,
    trust_remote_code: bool,
) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer

    torch, device, dtype = resolve_torch(device_arg, dtype_arg)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left" if is_query else "right"
    try:
        model = AutoModel.from_pretrained(
            model_name,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
    except TypeError:
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
    model = model.to(device).eval()

    encoded_texts = [qwen_query_prefix(text, instruction_name) for text in texts] if is_query else texts
    arrays: list[np.ndarray] = []
    label = "query" if is_query else "track"
    for start in range(0, len(encoded_texts), batch_size):
        end = min(start + batch_size, len(encoded_texts))
        print(f"  encode qwen {label} {start}:{end}/{len(encoded_texts)}")
        tok = tokenizer(
            encoded_texts[start:end],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            hidden = model(**tok).last_hidden_state
            pooled = pool_last_token(hidden, tok["attention_mask"])
        arrays.append(pooled.cpu().numpy().astype(np.float32))
    return np.concatenate(arrays, axis=0)


# ─── VERBATIM from qwen_fallback.py (query side) ─────────────────────────────
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


# ─── cache writing ───────────────────────────────────────────────────────────
def write_blind_cache(key, task_rows, query_texts, seen_by_row, emb):
    out = QUERY_CACHE / key
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "blind_a.npy", emb.astype(np.float32, copy=False))
    pl.DataFrame(
        {
            "session_id":      [t["session_id"] for t in task_rows],
            "user_id":         [t["user_id"] for t in task_rows],
            "turn_number":     [int(t["turn_number"]) for t in task_rows],
            "prior_track_ids": [list(s) for s in seen_by_row],
            "query_text":      query_texts,
        },
        schema={
            "session_id": pl.String, "user_id": pl.String, "turn_number": pl.Int64,
            "prior_track_ids": pl.List(pl.String), "query_text": pl.String,
        },
    ).write_parquet(out / "blind_a_meta.parquet")
    print(f"  wrote {emb.shape} -> {out / 'blind_a.npy'}")
    print(f"  wrote {len(task_rows)} meta rows -> {out / 'blind_a_meta.parquet'}")
    return out / "blind_a.npy"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(MODEL_IDS), choices=list(MODEL_IDS))
    parser.add_argument("--batch-size", type=int, default=None,
                        help=f"Override query batch size (default mirrors config: {QWEN_QUERY_BATCH_SIZE}).")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--dtype", default=DTYPE)
    parser.add_argument("--allow-download", action="store_true",
                        help=f"Set local_files_only=False (default mirrors config: {LOCAL_FILES_ONLY}).")
    args = parser.parse_args()

    local_files_only = LOCAL_FILES_ONLY and not args.allow_download

    # Build the query texts ONCE (model-independent), exact qwen_fallback logic.
    print(f"loading blind task rows from {BLIND_PATH}")
    task_rows = load_blind_task_rows(BLIND_PATH)
    print(f"  {len(task_rows)} blind task rows (one per session, at the final user turn)")

    print(f"loading track metadata from {TRACK_META_PATH}")
    _ids, _docs, track_text_by_id, _release = load_track_metadata(TRACK_META_PATH)

    query_texts: list[str] = []
    seen_by_row: list[list[str]] = []
    for task in task_rows:
        text, seen = build_query_text(task["row"], track_text_by_id)
        query_texts.append(text)
        seen_by_row.append(seen)

    written: list[Path] = []
    for key in args.models:
        batch_size = args.batch_size or QWEN_QUERY_BATCH_SIZE
        print("\n" + "#" * 70)
        print(f"  {key}  ({MODEL_IDS[key]})  instruction={QWEN_INSTRUCTION!r}  "
              f"max_len={QWEN_QUERY_MAX_LENGTH}  batch={batch_size}")
        print("#" * 70)
        emb = encode_qwen_texts(
            model_name=MODEL_IDS[key],
            texts=query_texts,
            is_query=True,
            instruction_name=QWEN_INSTRUCTION,
            max_length=QWEN_QUERY_MAX_LENGTH,
            batch_size=batch_size,
            device_arg=args.device,
            dtype_arg=args.dtype,
            local_files_only=local_files_only,
            trust_remote_code=TRUST_REMOTE_CODE,
        )
        written.append(write_blind_cache(key, task_rows, query_texts, seen_by_row, emb))

    print("\nDone. Blind-A query caches (qwen_fallback-identical):")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()