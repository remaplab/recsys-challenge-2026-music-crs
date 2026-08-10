"""Encode raw track-metadata text with a HF model to build a per-backbone track tower.

The organizer ships one pre-computed track tower at
`metadata-qwen3_embedding_0.6b` (Qwen3-Embedding-0.6B over the track-metadata
text). Using that as the retrieval target for a BERT-based query encoder
forces BERT to learn a projection INTO Qwen3-1024 space — an asymmetric
handicap that confounds any fair encoder-comparison study.

This module reads the raw track-metadata parquet, builds a short text
description per track via `build_track_text`, and encodes it with the
backbone of choice (mean-pool + L2-norm). The resulting tower:

  - has the SAME track_ids in the SAME ORDER as the organizer's tower,
    so previously-mined hard-negative indices (which are integer offsets
    into that order) remain valid across all towers;
  - is cached to disk so this is a one-shot expense per backbone;
  - returns a `TrackTower` with the same shape as the organizer tower,
    so all downstream code (training, evaluation) is generic over which
    tower it gets.

Output cache layout:

    cache_dir/<backbone>/track_ids.npy
    cache_dir/<backbone>/<modality>.npy
    cache_dir/<backbone>/<modality>__mask.npy
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from emblib.data.parsing import build_track_text
from emblib.tracks.organizer_track_loader import TrackTower


# Friendly name → (HF model id, modality column name, native hidden size, max_length)
BACKBONE_MODELS: dict[str, tuple[str, str, int, int]] = {
    "bert":       ("bert-base-uncased",                       "metadata-bert-text",       768,  256),
    "sbert":      ("sentence-transformers/all-mpnet-base-v2", "metadata-sbert-text",      768,  384),
    "modernbert": ("answerdotai/ModernBERT-base",             "metadata-modernbert-text", 768, 1024),
}


def get_modality_name(backbone: str) -> str:
    """Lookup the modality column name for a backbone."""
    if backbone not in BACKBONE_MODELS:
        raise ValueError(f"unknown backbone {backbone!r}; choose from {list(BACKBONE_MODELS)}")
    return BACKBONE_MODELS[backbone][1]


def get_canonical_track_ids(embedding_shards_dir: Path) -> list[str]:
    """Read the canonical track-id ordering from the organizer's embedding shards.

    The organizer ships ~47k tracks split across `all_tracks-*.parquet`
    files. Their alphabetical concatenation defines the canonical row order
    used everywhere else in this codebase (organizer track tower,
    hard-negative mining indices, evaluation row order).
    """
    shards = sorted(Path(embedding_shards_dir).glob("all_tracks-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No all_tracks-*.parquet files in {embedding_shards_dir}")
    ids: list[str] = []
    for shard in shards:
        df = pl.read_parquet(shard, columns=["track_id"])
        ids.extend(df["track_id"].to_list())
    return ids


@torch.no_grad()
def _encode_texts(
    model_name: str,
    texts: list[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> np.ndarray:
    """Mean-pool the last hidden state, then L2-normalize. Returns (N, H) float32.

    Mean-pooling (as opposed to CLS) is the standard for sentence-level
    similarity tasks — it's what `sentence-transformers` defaults to and what
    matches Qwen3's last-token-pool spirit (a global summary vector).
    """
    print(f"Loading {model_name} on {device} (dtype={dtype})")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Some HF models (e.g. ModernBERT) want eager attn for output_attentions; we
    # don't ask for attentions here, so the default fast path is fine.
    model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()

    out: list[np.ndarray] = []
    n = len(texts)
    for s in tqdm(range(0, n, batch_size), desc=f"encode {model_name.split('/')[-1]}"):
        e = min(s + batch_size, n)
        tok = tokenizer(
            texts[s:e],
            padding=True, truncation=True, max_length=max_length,
            return_tensors="pt",
        ).to(device)
        last = model(**tok).last_hidden_state.float()        # (B, L, H)
        mask = tok["attention_mask"].float().unsqueeze(-1)   # (B, L, 1)
        pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        pooled = F.normalize(pooled, p=2, dim=-1)
        out.append(pooled.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


def encode_text_track_tower(
    backbone: str,
    track_meta_path: Path,
    embedding_shards_dir: Path,
    cache_dir: Path,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> TrackTower:
    """Build (and cache) a text-based track tower for one backbone.

    If the cache is already populated, just loads it and returns. Otherwise
    encodes all 47k tracks once and writes the cache.
    """
    if backbone not in BACKBONE_MODELS:
        raise ValueError(f"unknown backbone {backbone!r}; choose from {list(BACKBONE_MODELS)}")
    model_name, modality, dim, max_length = BACKBONE_MODELS[backbone]

    cache_dir = Path(cache_dir) / backbone
    ids_path = cache_dir / "track_ids.npy"
    emb_path = cache_dir / f"{modality}.npy"
    mask_path = cache_dir / f"{modality}__mask.npy"

    # ── cache hit ─────────────────────────────────────────────────────────
    if ids_path.exists() and emb_path.exists() and mask_path.exists():
        print(f"Loading cached text track tower from {cache_dir}")
        track_ids = np.load(ids_path, allow_pickle=True).tolist()
        emb = np.load(emb_path)
        mask = np.load(mask_path)
        tower = TrackTower(
            track_ids=track_ids,
            id_to_idx={tid: i for i, tid in enumerate(track_ids)},
            embeddings={modality: emb},
            masks={modality: mask},
        )
        print(tower)
        return tower

    # ── 1. canonical ordering (must match organizer tower exactly) ───────
    canonical_ids = get_canonical_track_ids(embedding_shards_dir)
    n = len(canonical_ids)
    print(f"  canonical ordering: {n} tracks (from organizer embedding shards)")

    # ── 2. metadata lookup keyed by track_id ─────────────────────────────
    print(f"Loading track metadata from {track_meta_path}")
    md = pl.read_parquet(track_meta_path)
    md_by_id: dict[str, dict] = {row["track_id"]: row for row in md.to_dicts()}
    print(f"  metadata rows: {len(md_by_id)}")

    # ── 3. assemble texts in canonical order; record which rows are valid ─
    texts: list[str] = []
    mask = np.zeros(n, dtype=bool)
    n_missing = 0
    for i, tid in enumerate(canonical_ids):
        row = md_by_id.get(tid)
        if row is None:
            # Track is in the embedding shards but not the metadata parquet.
            # Should be very rare; we still feed *something* through the encoder
            # so batch alignment isn't broken, then mask the row out below.
            texts.append("Unknown track")
            n_missing += 1
        else:
            texts.append(build_track_text(row))
            mask[i] = True
    print(f"  text-coverage: {mask.mean():.3f} ({mask.sum()}/{n}); "
          f"{n_missing} tracks missing metadata")

    # ── 4. encode ─────────────────────────────────────────────────────────
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb = _encode_texts(model_name, texts, max_length, batch_size, device)
    assert emb.shape == (n, dim), f"shape {emb.shape} != ({n}, {dim})"

    # Zero out rows we don't have metadata for, so that any code that forgets
    # to apply `mask` doesn't accidentally retrieve a "Unknown track" duplicate.
    # Eval/train code DOES apply the mask (-inf those rows), so this is belt-and-suspenders.
    emb[~mask] = 0.0

    # Sanity: valid rows are unit-norm
    valid_norms = np.linalg.norm(emb[mask], axis=1)
    assert np.allclose(valid_norms, 1.0, atol=1e-3), \
        f"valid-row norms not unit; range {valid_norms.min():.4f}..{valid_norms.max():.4f}"

    # ── 5. cache + return ─────────────────────────────────────────────────
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(ids_path, np.asarray(canonical_ids, dtype=object))
    np.save(emb_path, emb)
    np.save(mask_path, mask)
    print(f"Cached to {cache_dir}")

    tower = TrackTower(
        track_ids=canonical_ids,
        id_to_idx={tid: i for i, tid in enumerate(canonical_ids)},
        embeddings={modality: emb},
        masks={modality: mask},
    )
    print(tower)
    return tower


def load_text_track_tower(backbone: str, cache_dir: Path) -> TrackTower:
    """Load a previously-cached text track tower. Errors clearly if not cached."""
    if backbone not in BACKBONE_MODELS:
        raise ValueError(f"unknown backbone {backbone!r}; choose from {list(BACKBONE_MODELS)}")
    _, modality, _, _ = BACKBONE_MODELS[backbone]
    sub = Path(cache_dir) / backbone
    ids_path = sub / "track_ids.npy"
    emb_path = sub / f"{modality}.npy"
    mask_path = sub / f"{modality}__mask.npy"
    if not (ids_path.exists() and emb_path.exists() and mask_path.exists()):
        raise FileNotFoundError(
            f"No cached {backbone!r} text track tower at {sub}. "
            f"Run scripts/04_encode_tracks.py --backbone {backbone} first."
        )
    track_ids = np.load(ids_path, allow_pickle=True).tolist()
    return TrackTower(
        track_ids=track_ids,
        id_to_idx={tid: i for i, tid in enumerate(track_ids)},
        embeddings={modality: np.load(emb_path)},
        masks={modality: np.load(mask_path)},
    )