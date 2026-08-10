"""scripts/08c_encode_tracks_qwen_fallback.py

Build the TRACK tower for 0.6B / 4B / 8B and write it where heuristic_200 reads it.

SELF-CONTAINED: the encoder and metadata functions below are copied VERBATIM from
the submission scripts/launchers/qwen_embeddings.py, and the constants mirror
scripts/launchers/config.py. So the embedding-producing code is identical to what
qwen_fallback runs — no cross-package imports, this file can live in scripts/.

For 0.6B it calls load_or_generate_track_embeddings with config.qwen_track_cache_dir,
which simply LOADS the tower qwen_fallback already cached there
(models/retrieval_text_towers/Qwen__Qwen3-Embedding-0.6B/dense_tracks_len256_poollast),
so the 0.6B track vectors are byte-identical to qwen_fallback's (no GPU needed if
that cache exists). 4B/8B are generated with the same encoder.

OUTPUT (per model key) — what heuristic_200 reads:
    <track-cache-root>/<key>/track_ids.npy   (metadata-parquet order)
    <track-cache-root>/<key>/emb.npy
    <track-cache-root>/<key>/mask.npy         (all True; tooling parity)
default <track-cache-root> = models/track_tower_qwenfb_cache

USAGE (cluster, from repo root)
===============================
    uv run python scripts/08c_encode_tracks_qwen_fallback.py                 # all three
    uv run python scripts/08c_encode_tracks_qwen_fallback.py --models qwen3_0p6b
    uv run python scripts/08c_encode_tracks_qwen_fallback.py --models qwen3_8b --batch-size 2
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
QWEN_0P6B_TRACK_CACHE = REPO / "models/retrieval_text_towers/Qwen__Qwen3-Embedding-0.6B/dense_tracks_len256_poollast"
OUT_ROOT = REPO / "models/track_tower_qwenfb_cache"

QWEN_TRACK_MAX_LENGTH = 256
QWEN_TRACK_BATCH_SIZE = 16
DEVICE = "auto"
DTYPE = "auto"
LOCAL_FILES_ONLY = True
TRUST_REMOTE_CODE = False

MODEL_IDS: dict[str, str] = {
    "qwen3_0p6b": "Qwen/Qwen3-Embedding-0.6B",
    "qwen3_4b":   "Qwen/Qwen3-Embedding-4B",
    "qwen3_8b":   "Qwen/Qwen3-Embedding-8B",
}
DEFAULT_BATCH: dict[str, int] = {"qwen3_0p6b": QWEN_TRACK_BATCH_SIZE, "qwen3_4b": 8, "qwen3_8b": 4}


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


# ─── builder ─────────────────────────────────────────────────────────────────
def qwen_cache_dir_for(key: str) -> Path:
    """qwen_fallback-style track cache dir per model (0.6B reuses config's)."""
    if key == "qwen3_0p6b":
        return QWEN_0P6B_TRACK_CACHE
    folder = MODEL_IDS[key].replace("/", "__")
    return REPO / "models/retrieval_text_towers" / folder / "dense_tracks_len256_poollast"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(MODEL_IDS), choices=list(MODEL_IDS))
    parser.add_argument("--track-cache-root", type=Path, default=OUT_ROOT,
                        help="Where heuristic_200 will read the tower from.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override generation batch size (lower if you OOM on 4B/8B).")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--dtype", default=DTYPE)
    parser.add_argument("--allow-download", action="store_true",
                        help=f"Set local_files_only=False (default mirrors config: {LOCAL_FILES_ONLY}).")
    args = parser.parse_args()

    local_files_only = LOCAL_FILES_ONLY and not args.allow_download

    print(f"loading track metadata from {TRACK_META_PATH}")
    track_ids, track_docs, _text_by_id, _release = load_track_metadata(TRACK_META_PATH)
    print(f"  {len(track_ids)} tracks (metadata-parquet order)")

    written: list[Path] = []
    for key in args.models:
        batch_size = args.batch_size or DEFAULT_BATCH[key]
        cache_dir = qwen_cache_dir_for(key)
        print("\n" + "#" * 70)
        print(f"  {key}  ({MODEL_IDS[key]})  max_len={QWEN_TRACK_MAX_LENGTH}  batch={batch_size}")
        print(f"  qwen_fallback track cache: {cache_dir}")
        print("#" * 70)

        emb = load_or_generate_track_embeddings(
            track_ids=track_ids,
            track_docs=track_docs,
            cache_dir=cache_dir,
            model_name=MODEL_IDS[key],
            max_length=QWEN_TRACK_MAX_LENGTH,
            batch_size=batch_size,
            device_arg=args.device,
            dtype_arg=args.dtype,
            local_files_only=local_files_only,
            trust_remote_code=TRUST_REMOTE_CODE,
        )
        emb = np.asarray(emb, dtype=np.float32)  # materialize (may be mmap)

        out = args.track_cache_root / key
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "track_ids.npy", np.asarray(track_ids, dtype=object))
        np.save(out / "emb.npy", emb)
        np.save(out / "mask.npy", np.ones(len(track_ids), dtype=bool))
        print(f"  wrote {emb.shape} -> {out / 'emb.npy'}")
        written.append(out)

    print("\nDone. qwen_fallback-identical track towers:")
    for p in written:
        print(f"  {p}")
    print("\nNEXT:")
    print(f"  uv run python scripts/launchers/heuristic_200.py --model all "
          f"--track-cache-root {args.track_cache_root}")


if __name__ == "__main__":
    main()