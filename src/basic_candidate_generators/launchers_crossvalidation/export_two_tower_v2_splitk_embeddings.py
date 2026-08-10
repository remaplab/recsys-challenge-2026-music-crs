"""Export TwoTowerV2 embeddings for the whole splitK layout.

The script exports the final normalized vectors produced by two_tower_v2:

  - user_vectors.npy + user_vectors_meta.parquet for each splitK target row
  - item_vectors.npy + item_vectors_meta.parquet for the matching fitted model

By default it expects checkpoints created by
``launchers_crossvalidation.retrain_and_export``:

  models/CG_crossvalidation/two_tower_v2_session/checkpoints/

If a checkpoint is missing, pass ``--fit-missing`` and the script will train the
corresponding model from the best-params YAML before exporting embeddings.

Example:
    cd src/basic_candidate_generators
    uv run python -m launchers_crossvalidation.export_two_tower_v2_splitk_embeddings \
      --best-params ../../models/CG_crossvalidation/two_tower_v2_session/best_params_two_tower_v2_session_ndcg20.yaml \
      --fit-missing
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml


_PKG_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _PKG_ROOT / "src"
_REPO_ROOT = _PKG_ROOT.parent.parent
sys.path.insert(0, str(_SRC_ROOT))


_DEFAULT_BEST_PARAMS = (
    "models/CG_crossvalidation/two_tower_v2_session/"
    "best_params_two_tower_v2_session_ndcg20.yaml"
)
_DEFAULT_SPLITK_DIR = "data/splitK"
_DEFAULT_TRACK_META = (
    "data/talkpl-ai/TalkPlayData-Challenge-Track-Metadata/data/"
    "all_tracks-00000-of-00001.parquet"
)
_DEFAULT_OUTPUT_DIR = (
    "models/CG_crossvalidation/two_tower_v2_session/embeddings_splitK"
)
_DEFAULT_BLIND = (
    "data/talkpl-ai/TalkPlayData-Challenge-Blind-A/data/test-00000-of-00001.parquet"
)


def _repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _REPO_ROOT / p


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export final TwoTowerV2 user/item embeddings for splitK."
    )
    p.add_argument("--best-params", default=_DEFAULT_BEST_PARAMS)
    p.add_argument("--splitk-dir", default=_DEFAULT_SPLITK_DIR)
    p.add_argument("--track-metadata-path", default=_DEFAULT_TRACK_META)
    p.add_argument("--output-dir", default=_DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--checkpoint-dir",
        default=None,
        help=(
            "Directory containing fold_*.pkl/non_holdout.pkl/full.pkl. "
            "Defaults to <best_params parent>/checkpoints."
        ),
    )
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument(
        "--fold",
        action="append",
        type=int,
        default=None,
        help="Fold index to export. Can be passed multiple times. Defaults to all folds.",
    )
    p.add_argument(
        "--fit-missing",
        action="store_true",
        help="Fit and save a checkpoint when the expected checkpoint is missing.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing embedding files.",
    )
    p.add_argument(
        "--skip-holdout",
        action="store_true",
        help="Do not export non_holdout -> holdout_test embeddings.",
    )
    p.add_argument(
        "--include-blind",
        action="store_true",
        help="Also export full-model embeddings for Blind-A last-turn sessions.",
    )
    p.add_argument("--blind-path", default=_DEFAULT_BLIND)
    p.add_argument(
        "--include-inputs",
        action="store_true",
        help=(
            "Also export intermediate user/item tower inputs. These files can be "
            "large because user_context_item_vectors_input.npy is B x max_ctx x D."
        ),
    )
    p.add_argument(
        "--device",
        default=None,
        help="Override recommender device after loading/fitting, e.g. cpu or cuda.",
    )
    return p.parse_args()


def _read_best_params(path: Path) -> tuple[str, str, str, dict]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    class_name = cfg["class"]
    module_name = cfg["module"]
    model_name = cfg.get("model") or path.stem
    params = {**(cfg.get("fixed_params") or {}), **(cfg.get("best_params") or {})}
    return model_name, class_name, module_name, params


def _load_checkpoint(checkpoint: Path, device_override: str | None = None):
    import torch

    from recommenders.two_tower_v2 import TwoTowerV2Recommender

    with open(checkpoint, "rb") as f:
        state = pickle.load(f)
    rec = TwoTowerV2Recommender()
    if device_override:
        rec._device = torch.device(device_override)
    rec._set_model_state(state)
    if device_override:
        rec._device = torch.device(device_override)
        rec._model.to(rec._device)
    return rec


def _fit_checkpoint(
    checkpoint: Path,
    class_name: str,
    module_name: str,
    params: dict,
    train_df: pl.DataFrame,
    track_meta: pl.DataFrame,
    device_override: str | None,
):
    import importlib
    import torch

    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    rec = cls(urm_mode="session", **params)
    if device_override:
        rec._device = torch.device(device_override)
    t0 = time.time()
    rec.fit(train_df, track_metadata=track_meta)
    print(f"  [fit] {checkpoint.name}: {time.time() - t0:.1f}s")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    rec.save(checkpoint)
    print(f"  [save] {checkpoint}")
    return rec


def _get_or_fit_rec(
    checkpoint: Path,
    *,
    fit_missing: bool,
    class_name: str,
    module_name: str,
    params: dict,
    train_df: pl.DataFrame | None,
    track_meta: pl.DataFrame,
    device_override: str | None,
):
    if checkpoint.exists():
        print(f"  [load] {checkpoint}")
        return _load_checkpoint(checkpoint, device_override)
    if not fit_missing:
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint}. Pass --fit-missing to train it."
        )
    if train_df is None:
        raise ValueError(f"Cannot fit {checkpoint}: train_df is None.")
    return _fit_checkpoint(
        checkpoint,
        class_name,
        module_name,
        params,
        train_df,
        track_meta,
        device_override,
    )


def _make_sessions_for_eval(eval_df: pl.DataFrame) -> list[dict]:
    """Build one session dict per (session_id, target_turn) in a splitK eval df."""
    required = {"session_id", "user_id", "turn_number", "track_id"}
    missing = required - set(eval_df.columns)
    if missing:
        raise ValueError(f"eval_df missing columns: {sorted(missing)}")

    base = eval_df.sort(["session_id", "turn_number"])
    sessions: list[dict] = []
    for target_turn in sorted(base["turn_number"].unique().to_list()):
        target_rows = base.filter(pl.col("turn_number") == target_turn)
        target_sids = target_rows["session_id"].unique().to_list()
        target_set = set(target_sids)
        ctx_df = base.filter(
            pl.col("session_id").is_in(target_set)
            & (pl.col("turn_number") < target_turn)
        )
        ctx_by_sid: dict[str, list[str]] = {}
        for key, group in ctx_df.group_by("session_id", maintain_order=True):
            sid = key[0] if isinstance(key, tuple) else key
            ctx_by_sid[sid] = [
                row["track_id"]
                for row in group.sort("turn_number").iter_rows(named=True)
                if row.get("track_id") is not None
            ]
        for row in target_rows.iter_rows(named=True):
            sid = row["session_id"]
            sessions.append({
                "session_id": sid,
                "user_id": row["user_id"],
                "turn_number": int(target_turn),
                "context": ctx_by_sid.get(sid, []),
            })
    return sessions


def _make_sessions_for_context_df(context_df: pl.DataFrame) -> list[dict]:
    sessions: list[dict] = []
    sort_cols = ["session_id"]
    if "turn_number" in context_df.columns:
        sort_cols.append("turn_number")
    for key, group in context_df.sort(sort_cols).group_by("session_id", maintain_order=True):
        sid = key[0] if isinstance(key, tuple) else key
        first = group.row(0, named=True)
        sessions.append({
            "session_id": sid,
            "user_id": first["user_id"],
            "turn_number": int(first["target_turn"]),
            "context": [
                row["track_id"]
                for row in group.iter_rows(named=True)
                if row.get("track_id") is not None
            ],
        })
    return sessions


def _export_user_vectors(rec, sessions: list[dict], out_dir: Path, split_name: str) -> None:
    from recommenders.two_tower_v2 import _encode_users

    print(f"  [export] {split_name}: {len(sessions):,} user vectors")
    vectors = _encode_users(
        rec._model,
        sessions,
        rec.max_ctx,
        rec._track_features,
        rec._user_features,
        rec._text_embeds,
        rec._unk_track,
        rec._unk_user,
        rec._device,
    ).astype(np.float32, copy=False)
    meta = pl.DataFrame([
        {
            "split": split_name,
            "session_id": s["session_id"],
            "user_id": s["user_id"],
            "turn_number": s["turn_number"],
            "context_len": len(s["context"]),
        }
        for s in sessions
    ])
    meta.write_parquet(out_dir / "user_vectors_meta.parquet")
    np.save(out_dir / "user_vectors.npy", vectors)
    print(f"    wrote user_vectors.npy shape={vectors.shape}")


def _export_item_vectors(rec, out_dir: Path) -> None:
    import torch
    import torch.nn.functional as F

    from recommenders.two_tower_v2 import _item_tensors_batch

    print(f"  [export] {len(rec._tids):,} item vectors")
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        rec._model.eval()
        for start in range(0, len(rec._tids), 512):
            tids = rec._tids[start:start + 512]
            feats = {
                k: v.to(rec._device)
                for k, v in _item_tensors_batch(tids, rec._track_features, rec._unk_track).items()
            }
            chunks.append(F.normalize(rec._model.item_tower(**feats), dim=-1).cpu().numpy())
    vectors = np.vstack(chunks).astype(np.float32, copy=False)
    pl.DataFrame({"track_id": rec._tids}).write_parquet(out_dir / "item_vectors_meta.parquet")
    np.save(out_dir / "item_vectors.npy", vectors)
    print(f"    wrote item_vectors.npy shape={vectors.shape}")


def _assert_can_write(out_dir: Path, overwrite: bool) -> None:
    paths = [
        out_dir / "user_vectors.npy",
        out_dir / "user_vectors_meta.parquet",
        out_dir / "item_vectors.npy",
        out_dir / "item_vectors_meta.parquet",
    ]
    existing = [p for p in paths if p.exists()]
    if existing and not overwrite:
        names = ", ".join(str(p) for p in existing)
        raise SystemExit(f"Output already exists: {names}. Pass --overwrite.")


def _export_bundle(
    rec,
    sessions: list[dict],
    out_dir: Path,
    split_name: str,
    *,
    overwrite: bool,
    include_inputs: bool,
) -> None:
    from launchers_crossvalidation.export_two_tower_v2_text_embeddings import (
        export_item_tower_inputs,
        export_user_tower_inputs,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    _assert_can_write(out_dir, overwrite)
    _export_user_vectors(rec, sessions, out_dir, split_name)
    _export_item_vectors(rec, out_dir)
    if include_inputs:
        print("  [export] intermediate tower inputs")
        export_user_tower_inputs(rec, sessions, out_dir)
        export_item_tower_inputs(rec, out_dir)


def main() -> None:
    args = parse_args()
    best_params_path = _repo_path(args.best_params)
    splitk_dir = _repo_path(args.splitk_dir)
    output_dir = _repo_path(args.output_dir)
    checkpoint_dir = (
        _repo_path(args.checkpoint_dir)
        if args.checkpoint_dir
        else best_params_path.parent / "checkpoints"
    )

    model_name, class_name, module_name, params = _read_best_params(best_params_path)
    print(f"[config] model={model_name} class={class_name} module={module_name}")
    print(f"[config] checkpoints={checkpoint_dir}")
    print(f"[config] output={output_dir}")

    track_meta = pl.read_parquet(_repo_path(args.track_metadata_path))
    folds = args.fold if args.fold is not None else list(range(args.n_folds))

    for fold in folds:
        print(f"\n=== fold {fold}: cg_train -> cg_val ===")
        cg_val = pl.read_parquet(splitk_dir / f"fold_{fold}_cg_val.parquet")
        cg_train = None
        if args.fit_missing and not (checkpoint_dir / f"fold_{fold}_cg_train.pkl").exists():
            cg_train = pl.read_parquet(splitk_dir / f"fold_{fold}_cg_train.parquet")
        rec = _get_or_fit_rec(
            checkpoint_dir / f"fold_{fold}_cg_train.pkl",
            fit_missing=args.fit_missing,
            class_name=class_name,
            module_name=module_name,
            params=params,
            train_df=cg_train,
            track_meta=track_meta,
            device_override=args.device,
        )
        _export_bundle(
            rec,
            _make_sessions_for_eval(cg_val),
            output_dir / f"fold_{fold}_cg_val",
            f"fold_{fold}_cg_val",
            overwrite=args.overwrite,
            include_inputs=args.include_inputs,
        )

        print(f"\n=== fold {fold}: cg_train+cg_val -> reranker_val ===")
        reranker_val = pl.read_parquet(splitk_dir / f"fold_{fold}_reranker_val.parquet")
        cg_train_val = None
        if args.fit_missing and not (checkpoint_dir / f"fold_{fold}_cg_train_val.pkl").exists():
            cg_train_val = pl.concat([
                pl.read_parquet(splitk_dir / f"fold_{fold}_cg_train.parquet"),
                cg_val,
            ])
        rec = _get_or_fit_rec(
            checkpoint_dir / f"fold_{fold}_cg_train_val.pkl",
            fit_missing=args.fit_missing,
            class_name=class_name,
            module_name=module_name,
            params=params,
            train_df=cg_train_val,
            track_meta=track_meta,
            device_override=args.device,
        )
        _export_bundle(
            rec,
            _make_sessions_for_eval(reranker_val),
            output_dir / f"fold_{fold}_reranker_val",
            f"fold_{fold}_reranker_val",
            overwrite=args.overwrite,
            include_inputs=args.include_inputs,
        )

    if not args.skip_holdout:
        print("\n=== non_holdout -> holdout_test ===")
        holdout = pl.read_parquet(splitk_dir / "holdout_test.parquet")
        non_holdout = None
        if args.fit_missing and not (checkpoint_dir / "non_holdout.pkl").exists():
            non_holdout = pl.read_parquet(splitk_dir / "fold_0_cg_train.parquet")
            non_holdout = pl.concat([
                non_holdout,
                pl.read_parquet(splitk_dir / "fold_0_cg_val.parquet"),
                pl.read_parquet(splitk_dir / "fold_0_reranker_val.parquet"),
            ])
        rec = _get_or_fit_rec(
            checkpoint_dir / "non_holdout.pkl",
            fit_missing=args.fit_missing,
            class_name=class_name,
            module_name=module_name,
            params=params,
            train_df=non_holdout,
            track_meta=track_meta,
            device_override=args.device,
        )
        _export_bundle(
            rec,
            _make_sessions_for_eval(holdout),
            output_dir / "holdout",
            "holdout",
            overwrite=args.overwrite,
            include_inputs=args.include_inputs,
        )

    if args.include_blind:
        print("\n=== full -> blind_A ===")
        from launchers_crossvalidation.export_two_tower_v2_text_embeddings import (
            build_last_turn_context,
        )

        raw_train = _repo_path(params["raw_train_path"])
        raw_test = _repo_path(params["raw_test_path"])
        full_df = pl.concat([pl.read_parquet(raw_train), pl.read_parquet(raw_test)])
        full_train = full_df if args.fit_missing and not (checkpoint_dir / "full.pkl").exists() else None
        rec = _get_or_fit_rec(
            checkpoint_dir / "full.pkl",
            fit_missing=args.fit_missing,
            class_name=class_name,
            module_name=module_name,
            params=params,
            train_df=full_train,
            track_meta=track_meta,
            device_override=args.device,
        )
        blind_df = pl.read_parquet(_repo_path(args.blind_path))
        if hasattr(rec, "encode_additional"):
            rec.encode_additional(blind_df)
        _export_bundle(
            rec,
            _make_sessions_for_context_df(build_last_turn_context(blind_df)),
            output_dir / "blind_A",
            "blind_A",
            overwrite=args.overwrite,
            include_inputs=args.include_inputs,
        )

    print(f"\nDone. Embeddings written under: {output_dir}")


if __name__ == "__main__":
    main()
