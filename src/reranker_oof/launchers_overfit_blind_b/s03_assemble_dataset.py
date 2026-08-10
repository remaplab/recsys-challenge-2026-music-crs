"""Assemble the reranker feature dataset for the Blind-B overfit pipeline.

Given ``dataset.yaml`` (CGs, buckets, embeddings, test-tracks filter) and the
exported RRF config (``tune.rrf_out``), build per-chunk feature parquets for the
splitK TRAIN pool (folds train+val + holdout) and the Blind-B VAL, under
``models/reranker_oof/datasets/<name>/``.

Self-contained chunk loop — reuses only the generic pool helpers
(``_preshard_cgs`` / ``_build_chunk_pool``), the pure fusion math, and
``FeatureBuilder``. The test-tracks filter (Blind sessions only) is applied
BEFORE fusion so set-relative features are computed on the filtered set.

Splits/folds default to all. Usage:
    cd src/reranker_oof
    uv run python -m launchers_overfit_blind_b.s03_assemble_dataset \\
        --config configs/blind_v1/dataset.yaml
"""
from __future__ import annotations

import argparse
import gc
import math
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG_ROOT))
sys.path.insert(0, str(_PKG_ROOT / "src"))

import polars as pl                                                    # noqa: E402
import yaml                                                            # noqa: E402

from src.features.cg_calibration import (                             # noqa: E402
    GLOBAL_KEY, apply_to_pool, fit_artifacts,
)
from src.features.feature_builder import FeatureBuilder                # noqa: E402
from src.features.emb_features import load_embedding_resources         # noqa: E402
from src.features.fusion_select import add_fused_and_truncate          # noqa: E402
from src.features.pipeline import (                                    # noqa: E402
    _build_chunk_pool, _count_groups, _preshard_cgs, _split_save_dir,
)
from src.features.resources import (                                   # noqa: E402
    build_blind_a_session_history, build_full_urm, build_session_history,
    build_urm_for_split, cg_candidate_path, load_track_metadata,
    load_user_metadata, load_warm_user_ids,
)
from src.paths import (                                                # noqa: E402
    BLIND_A_RAW, BLIND_B_RAW, DATASETS_DIR, ensure_output_dirs,
)

from launchers_overfit_blind_b._common import (                        # noqa: E402
    BLIND_A_GT, BLIND_B_GT, assert_cgs_have, filter_candidates,
    load_config, register_cg_paths, test_tracks_spec,
)


def _apply_fusion(pool: pl.DataFrame, cg_keep: list[str], rrf: dict) -> pl.DataFrame:
    """Reduce the union pool to the tuned RRF top-K (GT row always kept). RRF is
    rank-based, so the minmax score column only defines the per-CG rank."""
    buckets = [(b["name"], list(b["turns"])) for b in rrf["turn_buckets"]]
    return add_fused_and_truncate(
        pool, cg_keep, rrf["weights"], rrf["method"],
        rrf.get("method_params", {}), buckets,
        top_k=int(rrf["top_k"]), score_col=lambda cg: f"score_minmax_{cg}",
    )


def _build_split(split: str, fold: int, cg_keep: list[str], k_per_cg: int,
                 groups_per_chunk: int, save_dir: Path, force: bool,
                 rrf: dict, embeddings_cfg: dict | None, tt_spec: dict | None,
                 emb_cache_dir: Path, calib=None, *, blind_raw: Path | None = None,
                 blind_gt: Path | None = None) -> None:
    blind = blind_raw is not None
    save_dir.mkdir(parents=True, exist_ok=True)
    proxy = cg_candidate_path(cg_keep[0], split, fold)
    n_groups = _count_groups(proxy)
    n_chunks = max(1, math.ceil(n_groups / max(1, groups_per_chunk)))
    print(f"\n========= {split.upper()} fold_{fold} =========")
    print(f"[chunks] {n_groups:,} groups → n_chunks={n_chunks}")

    # ── side tables (loaded once, shared by every chunk's FeatureBuilder) ─────
    gt_auth = None
    if blind:
        target_turns = (
            pl.read_parquet(proxy, columns=["session_id", "turn"])
            .rename({"turn": "turn_number"}).unique()
        )
        urm = build_full_urm()
        history = build_blind_a_session_history(blind_raw, target_turns)
        gt_auth = (
            pl.read_parquet(blind_gt, columns=["session_id", "turn_number", "track_id"])
            .rename({"track_id": "gt_track_id"})
            .unique(subset=["session_id", "turn_number"], keep="first")
        )
    else:
        urm = build_urm_for_split(split, fold)
        history = build_session_history(split, fold)

    fb = FeatureBuilder(
        track_meta=load_track_metadata(),
        user_meta=load_user_metadata(),
        warm_user_ids=load_warm_user_ids(include_cold=False),
        urm_df=urm,
        session_history_df=history,
        emb_resources=load_embedding_resources(embeddings_cfg),
        emb_cache_path=emb_cache_dir / f"emb_{split}_f{fold}.parquet",
    )

    # ── pre-shard CG parquets once (generic pool helper) ──────────────────────
    to_build = [c for c in range(n_chunks)
                if force or not (save_dir / f"chunk_{c:03d}.parquet").exists()]
    shard_root: Path | None = None
    if to_build:
        shard_root = Path(tempfile.mkdtemp(
            prefix=f"cgshard_{split}_f{fold}_", dir=save_dir.parent))
        _preshard_cgs(split, fold, cg_keep, n_chunks, shard_root)

    try:
        for c in range(n_chunks):
            cp = save_dir / f"chunk_{c:03d}.parquet"
            if cp.exists() and not force:
                print(f"[skip] {cp.name} exists")
                continue
            pool = _build_chunk_pool(split, fold, cg_keep, c, n_chunks,
                                     k_per_cg, shard_root)
            if pool.height == 0:
                del pool; gc.collect(); continue

            # Blind: override per-turn GT from the authoritative 280-turn file
            # (submission turns stay null → label 0).
            if blind:
                if "gt_track_id" in pool.columns:
                    pool = pool.drop("gt_track_id")
                pool = pool.join(gt_auth, on=["session_id", "turn_number"], how="left")

            # test-tracks filter BEFORE fusion (Blind sessions only).
            rows_before = pool.height
            pool = filter_candidates(pool, tt_spec)
            if tt_spec is not None:
                print(f"  [test_tracks] {rows_before:,} → {pool.height:,} rows")
            if pool.height == 0:
                del pool; gc.collect(); continue

            # Per-CG calibration + conformal set-size features (honest fold split:
            # train/val use the calibrator fit on folds≠fold; holdout/blind use
            # the global one). Added BEFORE fusion so the cross-CG aggregates see
            # the full candidate set.
            if calib is not None:
                excluded = fold if split in ("train", "val") else GLOBAL_KEY
                pool = apply_to_pool(pool, cg_keep=cg_keep, artifacts=calib,
                                     fold_excluded=excluded)

            pool = _apply_fusion(pool, cg_keep, rrf)
            enriched = fb.build(pool, cg_names=cg_keep)
            del pool; gc.collect()

            enriched = enriched.with_columns(
                (pl.col("track_id") == pl.col("gt_track_id"))
                .fill_null(False).cast(pl.Int8).alias("label")
            ).sort("session_id", "turn_number", "track_id")
            enriched.write_parquet(cp)
            print(f"  → {cp.name} shape={enriched.shape} positives={int(enriched['label'].sum())}")
            del enriched; gc.collect()
    finally:
        if shard_root is not None:
            shutil.rmtree(shard_root, ignore_errors=True)


@dataclass
class AssemblyContext:
    """Everything `_build_split` needs besides the (pre-fit or loaded)
    calibration artifacts — shared by the full assembler (`main()`) and any
    scoped single-split launcher (e.g. a blind-B-only, calibrator-free run)
    so the two can't drift on rrf/k_per_cg/emb_cache resolution."""
    cg_keep: list[str]
    rrf: dict
    k_per_cg: int
    groups_per_chunk: int
    embeddings_cfg: dict | None
    tt_spec: dict | None
    emb_cache_dir: Path
    out_root: Path


def load_assembly_context(
    cfg: dict, *, rrf_config: Path | None = None,
    groups_per_chunk: int | None = None, k_per_cg: int | None = None,
) -> AssemblyContext:
    cg_keep = list(cfg["cgs"])

    rrf_path = rrf_config or cfg.get("tune", {}).get("rrf_out")
    if rrf_path is None:
        raise SystemExit("no rrf config (set tune.rrf_out or --rrf_config)")
    rrf_path = Path(rrf_path)
    if not rrf_path.is_absolute():
        rrf_path = _PKG_ROOT / rrf_path
    with open(rrf_path) as f:
        rrf = yaml.safe_load(f)
    print(f"[blind_b/assemble] rrf={rrf_path.name} method={rrf['method']} "
          f"k={rrf.get('method_params', {}).get('k')} cgs={len(cg_keep)}")

    k_per_cg = int(k_per_cg if k_per_cg is not None
                   else cfg.get("max_candidates_per_cg", 200))
    groups_per_chunk = int(groups_per_chunk if groups_per_chunk is not None
                           else cfg.get("groups_per_chunk", 3000))
    embeddings_cfg = cfg.get("embeddings")
    tt_spec = test_tracks_spec(cfg)

    out_root = DATASETS_DIR / cfg["name"]
    # Embedding cosines are keyed by (split, fold, session, turn, track) and are
    # invariant to RRF weights / CG set / dataset, so the cache is SHARED across
    # datasets by default — no point rebuilding it per variant. Override with
    # embeddings.cache_dir (absolute, or relative to the package root).
    emb_cache_cfg = (embeddings_cfg or {}).get("cache_dir")
    if emb_cache_cfg:
        emb_cache_dir = Path(emb_cache_cfg)
        if not emb_cache_dir.is_absolute():
            emb_cache_dir = _PKG_ROOT / emb_cache_dir
    else:
        emb_cache_dir = DATASETS_DIR / "_emb_cache_shared"
    emb_cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[blind_b/assemble] emb_cache (shared) = {emb_cache_dir}")
    print(f"[blind_b/assemble] out={out_root} k_per_cg={k_per_cg} "
          f"groups_per_chunk={groups_per_chunk}")

    return AssemblyContext(
        cg_keep=cg_keep, rrf=rrf, k_per_cg=k_per_cg, groups_per_chunk=groups_per_chunk,
        embeddings_cfg=embeddings_cfg, tt_spec=tt_spec, emb_cache_dir=emb_cache_dir,
        out_root=out_root,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--rrf_config", type=Path, default=None)
    ap.add_argument("--splits", default="train,val,holdout,blind_b,blind_a")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--groups_per_chunk", type=int, default=None)
    ap.add_argument("--k_per_cg", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ensure_output_dirs()
    cfg = load_config(args.config)
    register_cg_paths(cfg)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    assert_cgs_have(cfg, splits)

    ctx = load_assembly_context(
        cfg, rrf_config=args.rrf_config,
        groups_per_chunk=args.groups_per_chunk, k_per_cg=args.k_per_cg,
    )
    cg_keep = ctx.cg_keep

    # Per-CG calibration + conformal artifacts (fit once on the OOF folds).
    calib = None
    cc = cfg.get("cg_calibration") or {}
    if cc.get("enabled", True):
        cg_oof = {k: {cg: cg_candidate_path(cg, "train", k) for cg in cg_keep}
                  for k in folds}
        print(f"[blind_b/assemble] fitting CG calibrators "
              f"({cc.get('method', 'isotonic')}, {cc.get('feature', 'reciprocal_rank')}) …")
        calib = fit_artifacts(
            cg_keep, cg_oof, method=cc.get("method", "isotonic"),
            feature_col=cc.get("feature", "reciprocal_rank"),
            alpha=float(cc.get("alpha", 0.1)), k_per_cg=ctx.k_per_cg, verbose=True,
        )

    print(f"[blind_b/assemble] splits={splits} folds={folds}")

    def build(split: str, fold: int, *, blind_raw: Path | None = None,
              blind_gt: Path | None = None) -> None:
        _build_split(split, fold, cg_keep, ctx.k_per_cg, ctx.groups_per_chunk,
                     _split_save_dir(ctx.out_root, split, fold), args.force,
                     ctx.rrf, ctx.embeddings_cfg, ctx.tt_spec, ctx.emb_cache_dir, calib,
                     blind_raw=blind_raw, blind_gt=blind_gt)
        gc.collect()

    if "holdout" in splits:
        build("holdout", 0)
    for fold in folds:
        for split in ("train", "val"):
            if split in splits:
                build(split, fold)
    if "blind_b" in splits:
        build("blind_b", 0, blind_raw=BLIND_B_RAW, blind_gt=BLIND_B_GT)
    if "blind_a" in splits:
        build("blind_a", 0, blind_raw=BLIND_A_RAW, blind_gt=BLIND_A_GT)
    print("\n[blind_b/assemble] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
