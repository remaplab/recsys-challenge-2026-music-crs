"""Extract best hyperparameters from completed Optuna studies.

Reads SQLite DBs produced by tune_crossvalidation.py (5-fold CV on splitK) or
tune_fold.py (single-fold on splitK) and writes a YAML summary to two
locations per study:
  - {storage_dir}/{model}_{urm}/best_params_{model}_{urm}[_source][_objective].yaml
  - configs/cv_best_{model}_{urm}[_source][_objective].yaml

Objective handling
------------------
Every trial logs all metric@k values as user_attrs regardless of the study's
native objective. The best trial is always picked by
`user_attrs[{objective}@{K}]`, so any objective can be re-extracted from any
DB (e.g. ndcg@20 best from a recall@200-tuned study).

DB / study layout resolution (CV)
---------------------------------
The extractor probes three (file, study) candidates in priority order:
  1. optuna_{folder}.db                       + study {folder}_cv
     (standard layout written by tune_crossvalidation.py)
  2. optuna_{metric}{K}_{folder}.db           + study {folder}_cv_{metric}{K}
     (PR layout A: objective-tagged file and study)
  3. optuna_{folder}.db                       + study {folder}_cv_{metric}{K}
     (PR layout B: bare filename, objective-tagged study)
The first one that loads is used; fallback hits print a marker line.

Usage:
    cd src/basic_candidate_generators

    # Single model/mode (CV, ndcg@20)
    uv run python -m launchers_crossvalidation.extract_best_params \\
        --model item_knn --urm_mode session \\
        --source cv --objective ndcg --objective_k 20

    # Single model/mode (tune_fold, fold 0, recall@200)
    uv run python -m launchers_crossvalidation.extract_best_params \\
        --model item_knn --urm_mode session \\
        --source fold --fold 0 --objective recall --objective_k 200

    # All studies under storage_dir for a given (source, objective)
    uv run python -m launchers_crossvalidation.extract_best_params \\
        --all --source cv --objective ndcg --objective_k 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _PKG_ROOT / "src"
_CV_ROOT  = _PKG_ROOT / "launchers_crossvalidation"
sys.path.insert(0, str(_SRC_ROOT))
sys.path.insert(0, str(_CV_ROOT))

import optuna   # noqa: E402
import yaml     # noqa: E402
import tempfile
import shutil

from _cv_utils import _PKG_ROOT, _REPO_ROOT, pkg_path, repo_path  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_OBJECTIVE_DEFAULT_K = {"ndcg": 20, "recall": 200}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract best params from Optuna studies.")
    p.add_argument("--model",       default=None,
                   help="Model key. Omit together with --all to process all available DBs.")
    p.add_argument("--urm_mode",    default="session",
                   help="urm_mode (session|user). Ignored when --all is set.")
    p.add_argument("--all",         action="store_true",
                   help="Extract from every DB found in storage_dir matching source+objective.")
    p.add_argument("--storage_dir", default=None,
                   help="Default depends on --source: "
                        "cv -> models/CG_crossvalidation, fold -> models/CG_fold.")
    p.add_argument("--config",      default="configs/tune_crossvalidation.yaml")
    p.add_argument("--source",      choices=["cv", "fold"], default="cv",
                   help="cv = splitK 5-fold (tune_crossvalidation.py), "
                        "fold = single-fold (tune_fold.py)")
    p.add_argument("--fold",        type=int, default=0,
                   help="Fold index (only used when --source fold).")
    p.add_argument("--objective",   choices=("ndcg", "recall"), default="ndcg",
                   help="Optuna objective family. Must match the tuner run.")
    p.add_argument("--objective_k", type=int, default=None,
                   help="K for the objective. Defaults: ndcg=20, recall=200.")
    return p.parse_args()


# DB/study naming mirrors tune_crossvalidation.py and tune_fold.py:
#   cv    → optuna_{objective}{K}_{folder_key}.db
#           study={folder_key}_cv_{objective}{K}
#   fold  → optuna_fold{N}_{objective}{K}_{folder_key}.db
#           study={folder_key}_fold{N}_{objective}{K}

def _db_paths(
    storage_dir: Path, folder_key: str, source: str,
    objective_metric: str, objective_k: int, fold: int,
) -> tuple[Path, str]:
    """Return (db_path, study_name) for the *primary* (standard) layout.

    Standard CV layout — the one tune_crossvalidation.py writes — is
    `optuna_{folder_key}.db` with study `{folder_key}_cv`. The trial value is
    recall@200 by construction, but every metric@k is also recorded as a
    user_attr, so any objective can be re-extracted from this same DB.

    `--source fold` keeps its objective-tagged naming (tune_fold.py wrote
    these from day one).
    """
    if source == "fold":
        db = storage_dir / folder_key / f"optuna_fold{fold}_{objective_metric}{objective_k}_{folder_key}.db"
        study = f"{folder_key}_fold{fold}_{objective_metric}{objective_k}"
        return db, study
    # cv: standard layout
    db = storage_dir / folder_key / f"optuna_{folder_key}.db"
    study = f"{folder_key}_cv"
    return db, study


def _fallback_cv_db_paths(
    storage_dir: Path, folder_key: str, objective_metric: str, objective_k: int,
) -> tuple[Path, str]:
    """Fallback CV layout used by the sequential-CG PR for some models.

    DB: `optuna_{metric}{K}_{folder_key}.db`, study
    `{folder_key}_cv_{metric}{K}`. Trial value here is already the requested
    metric, so `study.best_trial` is meaningful directly.
    """
    db = storage_dir / folder_key / f"optuna_{objective_metric}{objective_k}_{folder_key}.db"
    study = f"{folder_key}_cv_{objective_metric}{objective_k}"
    return db, study


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_yaml_dict(model_name: str, urm_mode: str, mcfg: dict,
                     trial: optuna.trial.FrozenTrial, study: optuna.Study,
                     objective_key: str) -> dict:
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

    recall_keys = sorted(
        (k for k in trial.user_attrs if k.startswith("recall@")),
        key=lambda s: int(s.split("@")[1]),
    )
    ndcg_keys = sorted(
        (k for k in trial.user_attrs if k.startswith("ndcg@")),
        key=lambda s: int(s.split("@")[1]),
    )

    d: dict = {
        "model":    model_name,
        "class":    mcfg["class"],
        "module":   mcfg["module"],
        "urm_mode": urm_mode,
    }
    if "inference_mode" in mcfg:
        d["inference_mode"] = mcfg["inference_mode"]
    # `tune_crossvalidation.py` constructs the recommender with
    # `{**tuned_params, **fixed_params}`, so any non-tunable wiring
    # (feature_emb_paths, feature_modalities, icm_compressed_dim, ...) lives in
    # fixed_params. We persist it here so retrain_and_export.py can rebuild the
    # full kwargs from this YAML alone, without re-reading tune_crossvalidation.yaml.
    fixed_params = dict(mcfg.get("fixed_params") or {})
    d.update({
        "best_params":  dict(trial.params),
        "fixed_params": fixed_params,
        "cv_results": {
            "primary_metric": objective_key,
            "primary_value":  round(trial.value, 6),
            "recall": {int(k.split("@")[1]): round(trial.user_attrs[k], 6) for k in recall_keys},
            "ndcg":   {int(k.split("@")[1]): round(trial.user_attrs[k], 6) for k in ndcg_keys},
        },
        "optuna": {
            "study_name":         study.study_name,
            "n_trials_completed": n_complete,
            "best_trial_number":  trial.number,
        },
    })
    return d


def _write_yaml(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(payload, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
    print(f"  → {path}")


def _extract_one(model_name: str, urm_mode: str, mcfg: dict,
                 storage_dir: Path, source: str,
                 objective_metric: str, objective_k: int, fold: int) -> bool:
    """Extract best trial for one (model, urm_mode). Returns True on success."""
    folder_key = f"{model_name}_{urm_mode}"
    objective_key = f"{objective_metric}@{objective_k}"

    primary_db, primary_study = _db_paths(
        storage_dir, folder_key, source, objective_metric, objective_k, fold,
    )
    # CV: probe several known (file, study-name) layouts. The sequential-CG PR
    # mixed two non-standard layouts and one of them collides with the
    # standard filename (`optuna_{folder}.db` containing study
    # `{folder}_cv_recall200` instead of `{folder}_cv`), so filename alone
    # isn't enough — we try each candidate and load whichever exists.
    candidates: list[tuple[Path, str]] = [(primary_db, primary_study)]
    if source == "cv":
        fb_db, fb_study = _fallback_cv_db_paths(
            storage_dir, folder_key, objective_metric, objective_k,
        )
        candidates.append((fb_db, fb_study))            # PR layout A: metric-tagged file + study
        candidates.append((primary_db, fb_study))       # PR layout B: bare file, metric-tagged study

    db_path = study_name = None
    for cand_db, cand_study in candidates:
        if not cand_db.exists():
            continue
        try:
            optuna.load_study(study_name=cand_study, storage=f"sqlite:///{cand_db}")
        except KeyError:
            continue
        db_path, study_name = cand_db, cand_study
        if (db_path, study_name) != (primary_db, primary_study):
            print(f"[extract] {folder_key}: using fallback layout "
                  f"{db_path.name} (study={study_name})")
        break

    if db_path is None:
        print(f"[extract] SKIP {folder_key}: no matching DB/study found "
              f"(tried {[(c[0].name, c[1]) for c in candidates]})")
        return False

    # Snapshot .db + .db-wal + .db-shm so we see trials still in the WAL
    # (the tuner may not have checkpointed yet — without the sidecars we
    # silently read a stale committed-only view).
    snap_dir = Path(tempfile.mkdtemp(prefix="optuna_extract_"))
    try:
        for f in db_path.parent.glob(f"{db_path.name}*"):
            shutil.copy2(f, snap_dir / f.name)
        snap_db = snap_dir / db_path.name
        print(db_path, "→", snap_db)

        study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{snap_db}")

        complete_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        n_complete = len(complete_trials)
        if n_complete == 0:
            print(f"[extract] SKIP {folder_key}: no completed trials")
            return False

        # Unified ranking: always pick the best trial by
        # user_attrs[objective_key]. Every trial (standard or PR layout)
        # records every metric@k via trial.set_user_attr in the tuner, so
        # this works whether the study natively optimized the requested
        # metric (user_attr == trial.value) or a different one.
        scored = [
            (t.user_attrs.get(objective_key), t) for t in complete_trials
        ]
        scored = [(v, t) for v, t in scored if v is not None]
        if not scored:
            print(f"[extract] SKIP {folder_key}: no trial carries '{objective_key}'")
            return False
        best_value, best = max(scored, key=lambda x: x[0])

        print(f"[extract] {folder_key} [{source}]  trials={n_complete}  "
              f"best=#{best.number}  {objective_key}={best_value:.6f}")

        payload = _build_yaml_dict(model_name, urm_mode, mcfg, best, study, objective_key)
        # _build_yaml_dict stamps trial.value as primary_value; override with
        # the user-attr value of the actual chosen objective so the YAML's
        # primary_value matches its primary_metric.
        payload["cv_results"]["primary_value"] = round(float(best_value), 6)
    finally:
        shutil.rmtree(snap_dir, ignore_errors=True)

    suffix_parts: list[str] = []
    if source != "cv":
        suffix_parts.append(source)
        if source == "fold":
            suffix_parts.append(f"f{fold}")
    suffix_parts.append(f"{objective_metric}{objective_k}")
    suffix = "_" + "_".join(suffix_parts)

    _write_yaml(payload, storage_dir / folder_key / f"best_params_{folder_key}{suffix}.yaml")
    _write_yaml(payload, _PKG_ROOT / "configs" / f"cv_best_{folder_key}{suffix}.yaml")

    return True


def _discover_folders(storage_dir: Path, source: str,
                      objective_metric: str, objective_k: int, fold: int
                      ) -> list[tuple[str, str]]:
    """Scan storage_dir for folders containing an Optuna DB for this objective."""
    results = []
    for subdir in sorted(storage_dir.iterdir()):
        if not subdir.is_dir():
            continue
        name = subdir.name
        if name.endswith("_session"):
            urm_mode   = "session"
            model_name = name[: -len("_session")]
        elif name.endswith("_user"):
            urm_mode   = "user"
            model_name = name[: -len("_user")]
        else:
            continue
        db, _ = _db_paths(storage_dir, name, source, objective_metric, objective_k, fold)
        if db.exists():
            results.append((model_name, urm_mode))
            continue
        # CV fallback layout: PR sequential-CG studies may live under either
        # `optuna_{metric}{K}_{folder}.db` (with matching study name) or under
        # the standard filename but with a metric-tagged study. Either is
        # enough for _extract_one to find a usable trial.
        if source == "cv":
            fb_db, _ = _fallback_cv_db_paths(
                storage_dir, name, objective_metric, objective_k,
            )
            if fb_db.exists():
                results.append((model_name, urm_mode))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    cfg_path = pkg_path(args.config)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    storage_dir_default = "models/CG_crossvalidation" if args.source == "cv" else "models/CG_fold"
    storage_dir = repo_path(args.storage_dir or storage_dir_default)
    objective_metric = args.objective
    objective_k      = args.objective_k or _OBJECTIVE_DEFAULT_K[objective_metric]
    fold             = int(args.fold)

    if args.all:
        pairs = _discover_folders(
            storage_dir, args.source, objective_metric, objective_k, fold,
        )
        if not pairs:
            sys.exit(
                f"[extract] No DBs found in {storage_dir} "
                f"(source={args.source}, objective={objective_metric}@{objective_k}"
                f"{', fold=' + str(fold) if args.source == 'fold' else ''})"
            )
        print(
            f"[extract] Found {len(pairs)} studies (source={args.source}, "
            f"objective={objective_metric}@{objective_k}"
            f"{', fold=' + str(fold) if args.source == 'fold' else ''})\n"
        )
        ok, skip = 0, 0
        for model_name, urm_mode in pairs:
            if model_name not in cfg["models"]:
                print(f"[extract] SKIP {model_name}_{urm_mode}: not in config")
                skip += 1
                continue
            success = _extract_one(
                model_name, urm_mode, cfg["models"][model_name],
                storage_dir, args.source, objective_metric, objective_k, fold,
            )
            if success:
                ok += 1
            else:
                skip += 1
        print(f"\n[extract] done — {ok} extracted, {skip} skipped")

    else:
        if args.model is None:
            sys.exit("[extract] Provide --model MODEL or use --all")
        model_name = args.model
        urm_mode   = args.urm_mode
        if model_name not in cfg["models"]:
            sys.exit(f"[extract] Unknown model '{model_name}'. Available: {list(cfg['models'])}")
        success = _extract_one(
            model_name, urm_mode, cfg["models"][model_name],
            storage_dir, args.source, objective_metric, objective_k, fold,
        )
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
