"""Generate (and cache) a Qwen3-native track tower from track-metadata text.

The organizer ships a precomputed `metadata-qwen3_embedding_0.6b` tower, but it
was built with the organizer's own text format + pooling. To make the
query/track sides identical, this module re-encodes the track-metadata text
with OUR pipeline (`track_metadata_text` + last-token pooling, see
`src/qwen/qwen_embeddings.py`) — the exact mirror of how
`Qwen3NativeFrozenEncoder` encodes queries.

Guarantees:
  - track_ids are in the SAME canonical order as the organizer embedding shards,
    so previously-mined hard-negative indices (integer offsets into that order)
    stay valid;
  - output is a `TrackTower` with one modality ("metadata-qwen3-native", 1024-d),
    so all downstream code (mining / training / evaluation) is generic;
  - cached to disk so this is a one-shot expense.

Cache layout:
    cache_dir/track_ids.npy
    cache_dir/metadata-qwen3-native.npy
    cache_dir/metadata-qwen3-native__mask.npy
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from emblib.tracks.organizer_track_loader import TrackTower
from emblib.qwen.qwen_embeddings import track_metadata_text, encode_qwen_texts


QWEN_MODALITY = "metadata-qwen3-native"
QWEN_DIM = 1024


def _canonical_track_ids(embedding_shards_dir: Path) -> list[str]:
    """Canonical row order from the organizer's `all_tracks-*.parquet` shards."""
    shards = sorted(Path(embedding_shards_dir).glob("all_tracks-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No all_tracks-*.parquet files in {embedding_shards_dir}")
    ids: list[str] = []
    for shard in shards:
        df = pl.read_parquet(shard, columns=["track_id"])
        ids.extend(str(t) for t in df["track_id"].to_list())
    return ids


def build_qwen_track_tower(
    *,
    track_meta_path: Path,
    embedding_shards_dir: Path,
    cache_dir: Path,
    model_name: str,
    max_length: int,
    batch_size: int,
    device_arg: str,
    dtype_arg: str,
    local_files_only: bool,
    trust_remote_code: bool,
    instruction_name: str = "none",
    modality: str = QWEN_MODALITY,
) -> TrackTower:
    cache_dir = Path(cache_dir)
    ids_path = cache_dir / "track_ids.npy"
    emb_path = cache_dir / f"{modality}.npy"
    mask_path = cache_dir / f"{modality}__mask.npy"

    if ids_path.exists() and emb_path.exists() and mask_path.exists():
        print(f"Loading cached Qwen3 track tower from {cache_dir}")
        return load_qwen_track_tower(cache_dir, modality)

    canonical_ids = _canonical_track_ids(embedding_shards_dir)
    n = len(canonical_ids)
    print(f"  canonical ordering: {n} tracks (from organizer embedding shards)")

    print(f"Loading track metadata from {track_meta_path}")
    md = pl.read_parquet(track_meta_path)
    md_by_id: dict[str, dict] = {str(row["track_id"]): row for row in md.to_dicts()}
    print(f"  metadata rows: {len(md_by_id)}")

    texts: list[str] = []
    mask = np.zeros(n, dtype=bool)
    n_missing = 0
    for i, tid in enumerate(canonical_ids):
        row = md_by_id.get(tid)
        if row is None:
            texts.append("Unknown track")   # fed through so batches align; masked out below
            n_missing += 1
        else:
            doc = track_metadata_text(row)
            texts.append(doc if doc else "Unknown track")
            mask[i] = bool(doc)
    print(f"  text-coverage: {mask.mean():.3f} ({int(mask.sum())}/{n}); "
          f"{n_missing} tracks missing metadata")

    emb = encode_qwen_texts(
        model_name=model_name,
        texts=texts,
        is_query=False,
        instruction_name=instruction_name,
        max_length=max_length,
        batch_size=batch_size,
        device_arg=device_arg,
        dtype_arg=dtype_arg,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    ).astype(np.float32)
    assert emb.shape == (n, QWEN_DIM), f"shape {emb.shape} != ({n}, {QWEN_DIM})"

    # Zero out rows we couldn't build text for; eval/train -inf-mask these anyway.
    emb[~mask] = 0.0
    valid_norms = np.linalg.norm(emb[mask], axis=1)
    assert np.allclose(valid_norms, 1.0, atol=1e-3), \
        f"valid-row norms not unit; range {valid_norms.min():.4f}..{valid_norms.max():.4f}"

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(ids_path, np.asarray(canonical_ids, dtype=object))
    np.save(emb_path, emb)
    np.save(mask_path, mask)
    print(f"Cached Qwen3 track tower to {cache_dir}")

    tower = TrackTower(
        track_ids=canonical_ids,
        id_to_idx={tid: i for i, tid in enumerate(canonical_ids)},
        embeddings={modality: emb},
        masks={modality: mask},
    )
    return tower


def load_qwen_track_tower(cache_dir: Path, modality: str = QWEN_MODALITY) -> TrackTower:
    """Load a previously-built Qwen3 text track tower. Errors clearly if absent."""
    cache_dir = Path(cache_dir)
    ids_path = cache_dir / "track_ids.npy"
    emb_path = cache_dir / f"{modality}.npy"
    mask_path = cache_dir / f"{modality}__mask.npy"
    if not (ids_path.exists() and emb_path.exists() and mask_path.exists()):
        raise FileNotFoundError(
            f"No cached Qwen3 track tower at {cache_dir}. "
            f"Run `python scripts/04b_encode_qwen_tracks.py` first."
        )
    track_ids = [str(t) for t in np.load(ids_path, allow_pickle=True).tolist()]
    tower = TrackTower(
        track_ids=track_ids,
        id_to_idx={tid: i for i, tid in enumerate(track_ids)},
        embeddings={modality: np.load(emb_path)},
        masks={modality: np.load(mask_path)},
    )
    print(tower)
    return tower