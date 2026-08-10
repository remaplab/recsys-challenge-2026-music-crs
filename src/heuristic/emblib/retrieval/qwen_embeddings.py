from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


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
    """instruction_name may be a registry key (QWEN_INSTRUCTIONS) or a raw
    instruction string (per-bucket Level-2 prompts pass the full string)."""
    instruction = QWEN_INSTRUCTIONS.get(instruction_name, instruction_name)
    if not instruction:
        return text
    return f"Instruct: {instruction}\nQuery: {text}"


def encode_qwen_texts(
    *,
    model_name: str,
    texts: list[str],
    is_query: bool,
    instruction_name: str | list[str],
    max_length: int,
    batch_size: int,
    device_arg: str,
    dtype_arg: str,
    local_files_only: bool,
    trust_remote_code: bool,
) -> np.ndarray:
    """instruction_name: a single registry key / instruction string applied to all
    rows (legacy behaviour), OR a list with one instruction PER ROW (Level-2
    per-bucket prompts). A list must be row-aligned with `texts`."""
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

    if is_query:
        if isinstance(instruction_name, (list, tuple)):
            if len(instruction_name) != len(texts):
                raise ValueError(
                    f"per-row instructions: {len(instruction_name)} instructions "
                    f"for {len(texts)} texts"
                )
            encoded_texts = [qwen_query_prefix(t, ins) for t, ins in zip(texts, instruction_name)]
        else:
            encoded_texts = [qwen_query_prefix(text, instruction_name) for text in texts]
    else:
        encoded_texts = texts
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


def load_or_generate_track_embeddings(
    *,
    track_ids: list[str],
    track_docs: list[str],
    cache_dir: Path,
    model_name: str,
    max_length: int,
    batch_size: int,
    device_arg: str,
    dtype_arg: str,
    local_files_only: bool,
    trust_remote_code: bool,
) -> np.ndarray:
    ids_path = cache_dir / "track_ids.npy"
    embeddings_path = cache_dir / "embeddings.npy"
    if ids_path.exists() and embeddings_path.exists():
        cached_ids = [str(track_id) for track_id in np.load(ids_path, allow_pickle=True).tolist()]
        if cached_ids == track_ids:
            print(f"loading cached qwen track embeddings: {embeddings_path}")
            return np.load(embeddings_path, mmap_mode="r")
        print(f"cached qwen track IDs mismatch, regenerating: {cache_dir}")

    print(f"generating qwen track embeddings: {cache_dir}")
    embeddings = encode_qwen_texts(
        model_name=model_name,
        texts=track_docs,
        is_query=False,
        instruction_name="none",
        max_length=max_length,
        batch_size=batch_size,
        device_arg=device_arg,
        dtype_arg=dtype_arg,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(ids_path, np.asarray(track_ids, dtype=object))
    np.save(embeddings_path, embeddings)
    return embeddings