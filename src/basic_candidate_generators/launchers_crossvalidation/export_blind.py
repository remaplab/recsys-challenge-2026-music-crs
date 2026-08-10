"""Export ALL-turn candidates for an arbitrary blind dataset (e.g. blind-B).

Loads a CG's full-data checkpoint (``checkpoints/full.pkl``, saved by
``retrain_and_export``) and runs multiturn inference over EVERY turn of the
blind sessions — the history turns AND the withheld submission turn — via the
same ``run_inference_dispatch`` used for holdout/OOF. Writes one row per
(session, turn) to ``datasets/<out_name>`` in the CG candidate schema
(``session_id, user_id, turn, track_ids, scores, gt_track_id``).

This is the general path for new blinds; ``assemble_blind_a.py`` is the
Blind-A-only shortcut (Blind-A is already in the candidate store).

Usage:
    cd src/basic_candidate_generators
    uv run python -m launchers_crossvalidation.export_blind \\
        --model heuristic --urm_mode session \\
        --blind data/talkpl-ai/TalkPlayData-Challenge-Blind-A/data/test-00000-of-00001.parquet \\
        --out_name blind_a_all_turns_candidates.parquet
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _PKG_ROOT / "src"
_CV_ROOT = _PKG_ROOT / "launchers_crossvalidation"
sys.path.insert(0, str(_SRC_ROOT))
sys.path.insert(0, str(_CV_ROOT))

import polars as pl   # noqa: E402

from _cv_utils import (    # noqa: E402
    pkg_path, repo_path, resolve_param_paths, run_inference_dispatch,
)
from recommenders.interactions import explode_music_turns   # noqa: E402

# Reuse the sibling launcher's config conventions verbatim.
from launchers_crossvalidation.retrain_and_export import (   # noqa: E402
    _MODELS_NO_META, _OBJECTIVE_DEFAULT_K, _TRACK_META_PATH, _cfg_suffix,
)

_QUERY_TOWER_BASE = None  # lazily imported only for text-mode CGs


# ---------------------------------------------------------------------------
# Assemble a blind parquet into the all-turns inference frame
# ---------------------------------------------------------------------------

def _assemble_blind_full(blind_df: pl.DataFrame) -> pl.DataFrame:
    """One row per turn: every music turn (real track) + the submission turn
    (track null). Same shape ``run_inference_dispatch`` consumes for holdout.
    Mirrors ``retrain_and_export._predict_sessions`` assembly."""
    music_df = explode_music_turns(blind_df)
    last_turns = (
        blind_df.explode("conversations").unnest("conversations")
        .group_by("session_id").agg(pl.col("turn_number").max().alias("turn_number"))
    )
    all_meta = (
        blind_df.select(["session_id", "user_id", "session_date"]).unique(subset=["session_id"])
        .with_columns(
            pl.col("session_date").cast(pl.Utf8).str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
        )
    )
    target_rows = (
        all_meta.join(last_turns, on="session_id")
        .with_columns(pl.lit(None, dtype=pl.Utf8).alias("track_id"))
        .select(["session_id", "user_id", "session_date", "turn_number", "track_id"])
    )
    full_df = pl.concat([music_df, target_rows])
    if "conversation_goal" in blind_df.columns:
        goal = blind_df.select(["session_id", "conversation_goal"]).unique(subset=["session_id"])
        full_df = full_df.join(goal, on="session_id", how="left")
    return full_df


def _load_checkpoint(class_name: str, module_name: str, ckpt: Path):
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls.load(ckpt)


def run_export_blind(
    *, model: str, urm_mode: str, folder_key: str, cfg_path: Path, blind: Path,
    out_name: str, checkpoint: Path | None, top_k: int, storage_dir: str,
) -> Path:
    """Load ``folder_key``'s full checkpoint and export all-turn candidates for
    ``blind`` → ``<storage_dir>/<folder_key>/datasets/<out_name>``.

    Shared core so the dro / ndcg launchers (which only differ in folder_key +
    config path) are thin wrappers.
    """
    import yaml
    if not cfg_path.exists():
        sys.exit(f"[export_blind] config not found: {cfg_path}")
    cfg = yaml.safe_load(open(cfg_path))
    class_name, module_name = cfg["class"], cfg["module"]
    inference_mode = cfg.get("inference_mode", "standard")

    out_dir = repo_path(storage_dir) / folder_key
    ckpt = Path(checkpoint) if checkpoint else out_dir / "checkpoints" / "full.pkl"
    if not ckpt.exists():
        sys.exit(f"[export_blind] checkpoint missing: {ckpt}\nRun the retrain/export launcher first.")

    track_meta = None if model in _MODELS_NO_META else pl.read_parquet(repo_path(_TRACK_META_PATH))
    print(f"[export_blind] {folder_key} mode={inference_mode} ckpt={ckpt.name}")
    rec = _load_checkpoint(class_name, module_name, ckpt)

    blind_df = pl.read_parquet(blind if blind.is_absolute() else repo_path(str(blind)))
    if hasattr(rec, "encode_additional"):
        rec.encode_additional(blind_df)
    full_df = _assemble_blind_full(blind_df)

    query_bundle = None
    if inference_mode == "text":
        from embedding_based.query_tower import QUERY_TOWER_BASE, load_query_bundle
        query_bundle = load_query_bundle(repo_path(QUERY_TOWER_BASE), "blind")

    recs = run_inference_dispatch(rec, full_df, top_k, inference_mode, track_meta, query_bundle)
    recs = recs.with_columns(pl.lit(None, dtype=pl.Utf8).alias("gt_track_id"))
    out_path = out_dir / "datasets" / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    recs.write_parquet(out_path)
    print(f"[export_blind] {recs.shape[0]} rows ({recs['session_id'].n_unique()} sessions, "
          f"turns {recs['turn'].min()}-{recs['turn'].max()}) → {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--urm_mode", required=True, choices=["session", "user"])
    ap.add_argument("--blind", type=Path, required=True, help="Raw blind parquet path.")
    ap.add_argument("--out_name", default="blind_b_all_turns_candidates.parquet")
    ap.add_argument("--config", default=None, help="Explicit cv_best YAML.")
    ap.add_argument("--checkpoint", default=None, help="full.pkl (default: <out_dir>/checkpoints/full.pkl)")
    ap.add_argument("--top_k", type=int, default=200)
    ap.add_argument("--storage_dir", default="models/CG_crossvalidation")
    ap.add_argument("--objective", choices=("ndcg", "recall"), default="ndcg")
    ap.add_argument("--objective_k", type=int, default=None)
    args = ap.parse_args()

    folder_key = f"{args.model}_{args.urm_mode}"
    suffix = _cfg_suffix(args.objective, args.objective_k or _OBJECTIVE_DEFAULT_K[args.objective])
    if args.config:
        cfg_path = Path(args.config)
    else:
        tagged = pkg_path(f"configs/cv_best_{folder_key}{suffix}.yaml")
        bare = pkg_path(f"configs/cv_best_{folder_key}.yaml")
        cfg_path = tagged if tagged.exists() else bare
    run_export_blind(
        model=args.model, urm_mode=args.urm_mode, folder_key=folder_key,
        cfg_path=cfg_path, blind=args.blind, out_name=args.out_name,
        checkpoint=args.checkpoint, top_k=args.top_k, storage_dir=args.storage_dir,
    )


if __name__ == "__main__":
    main()
