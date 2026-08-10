"""Load all 6 organizer track-embedding modalities into a unified object.

The Track-Embeddings dataset has 4 shards covering ~47k tracks, each with
6 modality columns. Some tracks have empty embeddings for some modalities
(e.g., instrumentals have no lyrics). This loader keeps them in the index
and tracks per-modality validity via boolean masks.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import polars as pl


# All 6 organizer-provided track modalities and their dimensions.
TRACK_MODALITIES: Dict[str, int] = {
    "metadata-qwen3_embedding_0.6b": 1024,
    "attributes-qwen3_embedding_0.6b": 1024,
    "lyrics-qwen3_embedding_0.6b": 1024,
    "audio-laion_clap": 512,
    "image-siglip2": 768,
    "cf-bpr": 128,
}


@dataclass
class TrackTower:
    """Unified container for organizer track embeddings.

    Attributes:
        track_ids: list of track ids (length N), in row order.
        id_to_idx: dict mapping track_id -> row index.
        embeddings: dict mapping modality_name -> (N, D_m) float32 array, L2-normalized.
                    Rows where the embedding was empty are zero-filled.
        masks: dict mapping modality_name -> (N,) bool array; True if the track has
               a valid (non-empty) embedding for that modality.
    """
    track_ids: List[str]
    id_to_idx: Dict[str, int]
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    masks: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_tracks(self) -> int:
        return len(self.track_ids)

    def coverage(self) -> Dict[str, float]:
        return {m: float(self.masks[m].mean()) for m in self.embeddings}

    def __repr__(self) -> str:
        cov = self.coverage()
        lines = [f"TrackTower(n_tracks={self.n_tracks}, modalities={len(self.embeddings)})"]
        for m, dim in TRACK_MODALITIES.items():
            if m in self.embeddings:
                lines.append(f"  {m:40s} dim={dim}  coverage={cov[m]:.3f}")
        return "\n".join(lines)


def load_track_tower(
    shards_dir: Path,
    modalities: List[str] | None = None,
    cache_dir: Path | None = None,
) -> TrackTower:
    """Load specified modalities (default: all 6) into a TrackTower.

    The track-id index is built from the union of all rows across shards (which
    is the same for every modality — the parquet has one row per track and all
    columns are aligned). Empty embeddings get zero-filled and masked out.
    """
    if modalities is None:
        modalities = list(TRACK_MODALITIES.keys())
    for m in modalities:
        if m not in TRACK_MODALITIES:
            raise ValueError(f"Unknown modality {m!r}. Known: {list(TRACK_MODALITIES)}")

    shards_dir = Path(shards_dir)

    # Try cache
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        ids_path = cache_dir / "track_ids.npy"
        if ids_path.exists() and all((cache_dir / f"{m}.npy").exists() and (cache_dir / f"{m}__mask.npy").exists() for m in modalities):
            print(f"Loading cached TrackTower from {cache_dir}")
            track_ids = np.load(ids_path, allow_pickle=True).tolist()
            tower = TrackTower(
                track_ids=track_ids,
                id_to_idx={tid: i for i, tid in enumerate(track_ids)},
            )
            for m in modalities:
                tower.embeddings[m] = np.load(cache_dir / f"{m}.npy")
                tower.masks[m] = np.load(cache_dir / f"{m}__mask.npy")
            print(tower)
            return tower

    shard_files = sorted(shards_dir.glob("all_tracks-*.parquet"))
    if not shard_files:
        raise FileNotFoundError(f"No all_tracks-*.parquet files in {shards_dir}")

    # First pass: build the canonical track-id list from shard ordering
    print("Pass 1: collecting track ids...")
    all_ids: List[str] = []
    for shard in shard_files:
        df = pl.read_parquet(shard, columns=["track_id"])
        all_ids.extend(df["track_id"].to_list())
    n = len(all_ids)
    id_to_idx = {tid: i for i, tid in enumerate(all_ids)}
    print(f"  total tracks: {n}")

    # Second pass: load each modality
    embeddings: Dict[str, np.ndarray] = {}
    masks: Dict[str, np.ndarray] = {}
    for m in modalities:
        dim = TRACK_MODALITIES[m]
        emb = np.zeros((n, dim), dtype=np.float32)
        mask = np.zeros(n, dtype=bool)
        cursor = 0
        print(f"Loading {m} (dim={dim})...")
        for shard in shard_files:
            df = pl.read_parquet(shard, columns=["track_id", m])
            shard_vecs = df[m].to_list()
            for v in shard_vecs:
                if v is not None and len(v) == dim:
                    emb[cursor] = np.asarray(v, dtype=np.float32)
                    mask[cursor] = True
                cursor += 1
        # L2-normalize valid rows
        valid_norms = np.linalg.norm(emb[mask], axis=1, keepdims=True) + 1e-12
        emb[mask] = emb[mask] / valid_norms
        embeddings[m] = emb
        masks[m] = mask
        print(f"  coverage: {mask.mean():.3f} ({mask.sum()}/{n})")

    tower = TrackTower(track_ids=all_ids, id_to_idx=id_to_idx, embeddings=embeddings, masks=masks)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_dir / "track_ids.npy", np.asarray(all_ids, dtype=object))
        for m in modalities:
            np.save(cache_dir / f"{m}.npy", tower.embeddings[m])
            np.save(cache_dir / f"{m}__mask.npy", tower.masks[m])
        print(f"Cached to {cache_dir}")

    print(tower)
    return tower