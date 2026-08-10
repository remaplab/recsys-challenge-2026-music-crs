"""High-level "build a feature dataset for one split/fold" pipeline.

Strategy
--------
We chunk the work BEFORE building the candidate pool, not after. The reason
is memory: with 18 CGs × 200 candidates × tens of thousands of (session,
turn) groups, the full long-format pool reaches ~150M rows. Outer-joining
18 such frames simultaneously blows RAM/swap on a 32 GB box.

Per-chunk pool assembly (new in this version)
---------------------------------------------
1. Pick ``n_chunks`` from the per-split number of ranking groups
   ``(session_id, turn)`` so each chunk has ``target_groups_per_chunk``
   groups (a more interpretable budget than a row count, since the
   post-union row count is unpredictable).
2. For each chunk ``c``:
   - Loop over the CGs. For each CG: read its candidate parquet, FILTER
     the rows where ``session_id.hash() % n_chunks == c``, explode to long
     and normalise scores. Each CG's contribution to this chunk is only
     ``~1/n_chunks`` of its full size.
   - Outer-join (``pool_union``) the small per-CG slices.
   - Add fusion features and run :class:`FeatureBuilder` on the small
     chunk.
   - Tag the binary ``label`` (``track_id == gt_track_id``) when GT is
     present, sort by ``(session_id, turn_number, track_id)``, and write
     to ``<save_dir>/chunk_<c:03d>.parquet``.

No final concatenation
----------------------
Every chunk is a stand-alone parquet under ``<save_dir>/``. Downstream
launchers iterate the chunks via the existing streaming primitives
(``xgb.DataIter`` for XGBoost, polars chunked reads for the others) so the
reranker never has to load a full-fold-sized DataFrame into RAM.

Side tables (track / user metadata, URM, session history) are loaded
ONCE and shared by every chunk's :class:`FeatureBuilder` — they're tiny
relative to the pool.
"""
from __future__ import annotations

import gc
import math
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

from .cg_calibration import GLOBAL_KEY, CalibrationArtifacts, apply_to_pool
from .emb_features import load_embedding_resources
from .feature_builder import FeatureBuilder
from .fusion_select import add_fused_and_truncate
from .pool import add_fusion_features, explode_and_normalize_scores, pool_union
from .resources import (
    build_session_history,
    build_urm_for_split,
    cg_candidate_path,
    load_track_metadata,
    load_user_metadata,
    load_warm_user_ids,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_groups(parquet_path: Path) -> int:
    """Return the number of unique ``(session_id, turn)`` ranking groups in
    a wide CG candidate parquet — used to size ``n_chunks``."""
    df = pl.scan_parquet(parquet_path).select("session_id", "turn").unique().collect()
    return df.height


def _split_save_dir(datasets_dir: Path, split: str, fold_idx: int) -> Path:
    """Directory holding the chunk parquets for one (split, fold)."""
    if split == "holdout":
        return datasets_dir / "holdout"
    return datasets_dir / f"fold_{fold_idx}_{split}"


def fold_chunk_paths(datasets_dir: Path, split: str, fold_idx: int) -> list[Path]:
    """Return the chunk parquets for one (split, fold), or the legacy
    single-file path if the chunked dir does not exist.

    Backwards-compat: if an old ``fold_{k}_train.parquet`` lingers next to
    the new ``fold_{k}_train/chunk_*.parquet`` directory, we prefer the
    chunked version.
    """
    d = _split_save_dir(datasets_dir, split, fold_idx)
    if d.exists():
        chunks = sorted(d.glob("chunk_*.parquet"))
        if chunks:
            return chunks
    legacy = (
        datasets_dir / "holdout.parquet" if split == "holdout"
        else datasets_dir / f"fold_{fold_idx}_{split}.parquet"
    )
    return [legacy] if legacy.exists() else []


# ---------------------------------------------------------------------------
# Pool helpers (per chunk)
# ---------------------------------------------------------------------------

def _preshard_cgs(
    split: str,
    fold_idx: int,
    cg_keep: list[str],
    n_chunks: int,
    shard_root: Path,
) -> None:
    """Read each CG candidate parquet ONCE and partition its rows by
    ``hash(session_id) % n_chunks`` into raw per-(cg, chunk) shards under
    ``shard_root/<cg>/chunk_<c>.parquet``.

    Replaces the previous design where ``_build_chunk_pool`` re-read every
    full CG parquet once per chunk (``n_chunks`` full decodes per CG). With
    pre-sharding the CG data is decoded once here and each chunk reads only
    its ~1/n_chunks shard, so total CG read I/O drops from ``n_chunks`` full
    passes to ~2 (this sharding pass + the distributed chunk-loop reads).

    Shards stay RAW (wide, un-exploded): the explode/normalize still happens
    per chunk in :func:`_build_chunk_pool`, preserving the per-chunk memory
    bound. Empty hash buckets are simply not written (the reader treats a
    missing shard as a 0-row slice).
    """

    def _shard_one(cg: str) -> tuple[str, int]:
        path = cg_candidate_path(cg, split, fold_idx)
        if not path.exists():
            raise FileNotFoundError(f"missing CG file: {path}")
        raw = pl.read_parquet(path).with_columns(
            (pl.col("session_id").hash() % n_chunks).cast(pl.Int64).alias("_cid")
        )
        parts = raw.partition_by("_cid", as_dict=True)
        cg_dir = shard_root / cg
        cg_dir.mkdir(parents=True, exist_ok=True)
        for c in range(n_chunks):
            sub = parts.get((c,))
            if sub is not None:
                sub.drop("_cid").write_parquet(cg_dir / f"chunk_{c:03d}.parquet")
        return cg, raw.height

    n_workers = min(8, len(cg_keep))
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for cg, h in ex.map(_shard_one, cg_keep):
            print(f"   [preshard/{split}/fold{fold_idx}] {cg}: {h:,} wide rows "
                  f"→ {n_chunks} shards")


def _build_chunk_pool(
    split: str,
    fold_idx: int,
    cg_keep: list[str],
    chunk_id: int,
    n_chunks: int,
    k_per_cg: int,
    shard_root: Path | None = None,
) -> pl.DataFrame:
    """Read each CG's slice for this chunk and outer-join them into a small
    pool. ``add_fusion_features`` is called here too so the returned
    DataFrame is ready for :class:`FeatureBuilder.build`.

    Memory footprint is bounded by ``~1/n_chunks`` of the full pool size.

    When ``shard_root`` is given the chunk shards were written once by
    :func:`_preshard_cgs`, so we just read the small
    ``shard_root/<cg>/chunk_<chunk_id>.parquet`` (no full-file re-decode).
    When ``shard_root`` is ``None`` we fall back to reading each full CG
    parquet and hash-filtering it in place — fine for tiny pools (e.g. the
    blind builder, n_chunks≈1) where pre-sharding would not pay off. Parquet
    decode releases the GIL, so the per-CG reads run in parallel via a thread
    pool.
    """

    def _read_one(args: tuple[int, str]) -> tuple[int, str, pl.DataFrame]:
        i, cg = args
        if shard_root is not None:
            shard = shard_root / cg / f"chunk_{chunk_id:03d}.parquet"
            if shard.exists():
                raw = pl.read_parquet(shard)
            else:
                # Empty hash bucket for this CG: read the source schema only
                # to build a 0-row slice so ``explode_and_normalize_scores``
                # keeps a consistent column set for ``pool_union``.
                raw = pl.read_parquet(cg_candidate_path(cg, split, fold_idx), n_rows=0)
        else:
            path = cg_candidate_path(cg, split, fold_idx)
            if not path.exists():
                raise FileNotFoundError(f"missing CG file: {path}")
            raw = pl.read_parquet(path).filter(
                (pl.col("session_id").hash() % n_chunks).cast(pl.Int64) == chunk_id
            )
        # Returns ``i`` so we can put the results back in the input order
        # for deterministic ``pool_union`` output (column order matters).
        return i, cg, explode_and_normalize_scores(raw, k=k_per_cg)

    # ``min(8, ...)`` caps the parallelism — beyond ~8 the disk becomes the
    # bottleneck on consumer SSDs and contention starts hurting. Adjust if
    # you have NVMe RAID or many CGs.
    n_workers = min(8, len(cg_keep))
    cg_long_indexed: list[tuple[int, str, pl.DataFrame]] = []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for i, cg, long_df in ex.map(
            _read_one, [(i, cg) for i, cg in enumerate(cg_keep)],
        ):
            cg_long_indexed.append((i, cg, long_df))
            # Print as results arrive (already-ordered prints would defeat
            # the parallelism). User sees progress not strictly sorted by
            # CG index, which is fine.
            print(
                f"   [pool/{split}/fold{fold_idx}/chunk{chunk_id}] "
                f"({i + 1:>2d}/{len(cg_keep)}) {cg}: {long_df.height:,} long rows"
            )
    cg_long_indexed.sort(key=lambda t: t[0])
    cg_long: list[tuple[str, pl.DataFrame]] = [(cg, df) for _, cg, df in cg_long_indexed]
    del cg_long_indexed
    gc.collect()

    pool = pool_union(cg_long)
    del cg_long
    gc.collect()

    pool = add_fusion_features(
        pool, cg_keep, k_rrf_grid=(5, 60), rp_penalty_grid=(200, 1000),
    )
    return pool


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def build_enriched_split_chunked(
    split: str,
    fold_idx: int,
    cg_keep: list[str],
    target_groups_per_chunk: int,
    save_dir: Path,
    k_per_cg: int = 200,
    n_chunks_override: int | None = None,
    force: bool = False,
    calibration_artifacts: CalibrationArtifacts | None = None,
    train_pool_top_k: int | None = None,
    fusion_selection: dict | None = None,
    embeddings_cfg: dict | None = None,
    emb_cache_dir: "str | Path | None" = None,
) -> Path:
    """Build the enriched feature parquets for one (split, fold) under
    ``save_dir`` — one parquet per chunk, no final concatenation.

    Parameters
    ----------
    split
        ``"train"`` | ``"val"`` | ``"holdout"`` (also ``"blind_a"`` when
        invoked from the submission launcher).
    fold_idx
        Fold index (used by ``train`` / ``val``; ignored for ``holdout``).
    cg_keep
        Logical CG names (looked up via :func:`resources.cg_candidate_path`).
    target_groups_per_chunk
        Aim for ~this many unique ``(session_id, turn)`` groups per chunk.
        ``2000`` is a safe default on 32 GB RAM. Reduce if you still OOM.
    save_dir
        Output directory. Each chunk lands at ``save_dir/chunk_<NNN>.parquet``.
    k_per_cg
        Top-K candidates kept per CG.
    n_chunks_override
        If provided, skip the auto-size and use this value.
    force
        If True, regenerate every chunk; else skip ones already on disk.
    train_pool_top_k
        Hard-negative truncation (A4). When set AND ``split == "train"``, keep
        per ``(session_id, turn_number)`` group only the GT row plus the top-K
        candidates by ``fusion_combsum``. ``None`` (default) or any non-train
        split → full pool, unchanged. Shrinks the train chunks (faster fits)
        without touching val / holdout / blind evaluation pools.
    fusion_selection
        Tuned weighted-fusion candidate selection. When set, after fusion
        features + calibration
        the pool is reduced to the top-``top_k`` candidates per group by the
        tuned fused score (GT row always kept), on EVERY split — replacing the
        union pool with the fused selection. ``None`` (default) → full union,
        unchanged. Mutually exclusive with ``train_pool_top_k``.

    Returns
    -------
    Path
        ``save_dir`` (now populated with chunk parquets).
    """
    if fusion_selection is not None and train_pool_top_k is not None:
        raise ValueError("fusion_selection and train_pool_top_k are mutually exclusive")
    print(f"\n========= {split.upper()} fold_{fold_idx} (CHUNKED, per-chunk pool) =========")
    t_total = time.time()
    save_dir.mkdir(parents=True, exist_ok=True)

    # Estimate the number of ranking groups in this split from the first CG
    # (all CGs share the same (session, turn) keys).
    proxy_path = cg_candidate_path(cg_keep[0], split, fold_idx)
    if n_chunks_override is not None:
        n_chunks = max(1, int(n_chunks_override))
        n_groups_total = _count_groups(proxy_path)
    else:
        n_groups_total = _count_groups(proxy_path)
        n_chunks = max(1, math.ceil(n_groups_total / max(1, target_groups_per_chunk)))
    print(
        f"[chunks] {n_groups_total:,} groups → n_chunks={n_chunks} "
        f"(~{n_groups_total // n_chunks:,} groups/chunk)"
    )

    # Side tables: loaded ONCE; shared across chunks. They are tiny relative
    # to the pool (track meta ~100k rows, user meta ~50k, URM a few M).
    # Embedding towers (family M). Loaded once per call; module-level cached so
    # repeated (split, fold) builds reuse the same in-RAM matrices. ``None`` →
    # family M is a no-op.
    emb_resources = load_embedding_resources(embeddings_cfg)
    # Family-M cosine cache keyed by (session_id, turn_number, track_id), one
    # parquet per (split, fold). ``None`` → recompute every chunk.
    emb_cache_path = (
        Path(emb_cache_dir) / f"emb_{split}_f{fold_idx}.parquet"
        if emb_cache_dir is not None else None
    )

    fb = FeatureBuilder(
        track_meta=load_track_metadata(),
        user_meta=load_user_metadata(),
        warm_user_ids=load_warm_user_ids(include_cold=False),
        urm_df=build_urm_for_split(split, fold_idx),
        session_history_df=build_session_history(split, fold_idx),
        emb_resources=emb_resources,
        emb_cache_path=emb_cache_path,
    )

    # ── pre-shard the CG candidate parquets ONCE ─────────────────────────
    # Read each CG file a single time and split it into per-chunk hash
    # buckets on disk, so the chunk loop reads only its small shard instead
    # of re-decoding every full CG parquet n_chunks times. The temp dir sits
    # next to ``save_dir`` (same filesystem, room for the transient shards)
    # and is removed in the ``finally`` below.
    chunks_to_build = [
        c for c in range(n_chunks)
        if force or not (save_dir / f"chunk_{c:03d}.parquet").exists()
    ]
    shard_root: Path | None = None
    if chunks_to_build:
        shard_root = Path(tempfile.mkdtemp(
            prefix=f"cgshard_{split}_f{fold_idx}_", dir=save_dir.parent,
        ))
        print(f"[preshard] CG shards → {shard_root}")
        t_ps = time.time()
        _preshard_cgs(split, fold_idx, cg_keep, n_chunks, shard_root)
        print(f"[preshard] done ({time.time() - t_ps:.1f}s)")

    try:
      for c in range(n_chunks):
        chunk_path = save_dir / f"chunk_{c:03d}.parquet"
        if chunk_path.exists() and not force:
            sz_mb = chunk_path.stat().st_size / 1e6
            print(f"[skip] {chunk_path.name} exists ({sz_mb:.1f} MB)")
            continue
        t_c = time.time()
        print(f"\n--- chunk {c + 1}/{n_chunks} ---")

        pool = _build_chunk_pool(
            split=split, fold_idx=fold_idx, cg_keep=cg_keep,
            chunk_id=c, n_chunks=n_chunks, k_per_cg=k_per_cg,
            shard_root=shard_root,
        )
        if pool.height == 0:
            print(f"  chunk {c} empty (no sessions hashed into bucket) — skipping")
            del pool
            gc.collect()
            continue

        # ── Hard-negative truncation for TRAIN only (A4) ──────────────────
        # Keep GT + top-K by fusion_combsum per (session, turn). Trivial
        # negatives dilute the rank:ndcg gradient and bloat fit time; eval
        # pools (val / holdout / blind) are never truncated.
        if train_pool_top_k is not None and split == "train":
            rows_before = pool.height
            is_gt = (
                (pl.col("track_id") == pl.col("gt_track_id"))
                if "gt_track_id" in pool.columns
                else pl.lit(False)
            )
            pool = pool.with_columns(
                pl.col("fusion_combsum").rank("ordinal", descending=True)
                  .over("session_id", "turn_number").alias("_cs_rank"),
                is_gt.fill_null(False).alias("_is_gt"),
            ).filter(
                (pl.col("_cs_rank") <= train_pool_top_k) | pl.col("_is_gt")
            ).drop("_cs_rank", "_is_gt")
            print(
                f"  [train_pool_top_k={train_pool_top_k}] pool "
                f"{rows_before:,} → {pool.height:,} rows"
            )

        print(f"  pool rows={pool.height:,}, cols={pool.width}  ({time.time() - t_c:.1f}s)")

        # ── per-CG calibration + conformal (LBO doc C + E) ────────────────
        # Honest fold split: for sessions in fold_idx use the calibrator fit
        # on folds ≠ fold_idx; for holdout / blind use the GLOBAL one.
        if calibration_artifacts is not None:
            fold_excluded = (
                fold_idx if split in ("train", "val") else GLOBAL_KEY
            )
            pool = apply_to_pool(
                pool, cg_keep=cg_keep, artifacts=calibration_artifacts,
                fold_excluded=fold_excluded,
            )
            cal_cols = [c for c in pool.columns
                         if c.startswith("calibrated_score_")
                         or c.startswith("set_size_")
                         or c == "mean_set_size_across_cgs"
                         or c == "min_set_size_across_cgs"]
            print(f"  + cg-calibration: +{len(cal_cols)} cols (excluded={fold_excluded})")

        # ── tuned weighted-fusion candidate selection (all splits) ────────
        if fusion_selection is not None:
            fs = fusion_selection
            buckets = [(b["name"], list(b["turns"])) for b in fs["turn_buckets"]]
            si = fs.get("score_input", "minmax")
            score_col = (
                (lambda cg: f"calibrated_score_{cg}") if si == "calibrated"
                else (lambda cg: f"score_minmax_{cg}")
            )
            rows_before = pool.height
            pool = add_fused_and_truncate(
                pool, cg_keep, fs["weights"], fs["method"],
                fs.get("method_params", {}), buckets,
                top_k=int(fs["top_k"]), score_col=score_col,
            )
            print(f"  [fusion_selection {fs['method']} top_k={fs['top_k']}] "
                  f"pool {rows_before:,} → {pool.height:,} rows")

        enriched = fb.build(pool, cg_names=cg_keep)
        del pool
        gc.collect()

        if "gt_track_id" in enriched.columns:
            # Use ``fill_null(False)`` BEFORE the cast so rows with no
            # gt_track_id (e.g. groups that no CG happens to provide labels
            # for) get label=0 (not nan). Downstream XGBoost rejects NaN
            # labels with "Label contains NaN, infinity or a value too large".
            enriched = enriched.with_columns(
                (pl.col("track_id") == pl.col("gt_track_id"))
                .fill_null(False)
                .cast(pl.Int8).alias("label")
            )
        enriched = enriched.sort("session_id", "turn_number", "track_id")
        enriched.write_parquet(chunk_path)
        print(
            f"  → {chunk_path.name} saved  shape={enriched.shape}  "
            f"size={chunk_path.stat().st_size / 1e6:.1f} MB  "
            f"({time.time() - t_c:.1f}s)"
        )
        del enriched
        gc.collect()
    finally:
        if shard_root is not None:
            shutil.rmtree(shard_root, ignore_errors=True)

    print(f"[{split}/fold{fold_idx}] TOTAL {time.time() - t_total:.1f}s")
    return save_dir
