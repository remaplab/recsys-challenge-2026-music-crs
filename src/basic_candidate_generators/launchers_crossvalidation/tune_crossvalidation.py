"""Cross-validation hyperparameter tuning for Candidate Generators.

Each Optuna trial trains and evaluates on all 5 splitK folds.
Train: cg_train split (80%). Eval: cg_val split (16%) — scored on every
(session, turn) pair via `run_inference_dispatch` (multiturn).
Per-fold scoring is macro-by-turn: per-session metric → mean within each
last-turn-position group → mean across non-empty groups. Final trial value
= arithmetic mean across the 5 fold scores. Native objective: recall@200.
All other metric@k values are stored as trial user_attrs so any of them
can be re-extracted by `extract_best_params.py --objective ...` post-hoc.

Path-like fixed_params (currently `feature_emb_paths`) are resolved
repo-relative via `_cv_utils.resolve_param_paths` before each trial, so
running from `src/basic_candidate_generators/` works without manual cwd
gymnastics.

Usage:
    cd src/basic_candidate_generators
    uv run python -m launchers_crossvalidation.tune_crossvalidation \\
        --model item_knn [--n_trials 300] [--n_jobs 6] \\
        [--storage_dir models/CG_crossvalidation] [--config configs/tune_crossvalidation.yaml]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _PKG_ROOT / "src"
_CV_ROOT  = _PKG_ROOT / "launchers_crossvalidation"
sys.path.insert(0, str(_SRC_ROOT))
sys.path.insert(0, str(_CV_ROOT))

import optuna                # noqa: E402
import polars as pl          # noqa: E402
import yaml                  # noqa: E402

from _cv_utils import (      # noqa: E402
    NDCG_KS,
    RECALL_KS,
    ResourceMonitor,
    _REPO_ROOT,
    aggregate_mean,
    build_params,
    instantiate_rec,
    load_eval,
    load_fold,
    make_storage,
    pkg_path,
    plot_study,
    repo_path,
    resolve_param_paths,
    run_inference_dispatch,
    score_fold,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CV hyperparameter tuning for a single CG model.")
    p.add_argument("--model",       required=True,  help="Model key in tune_crossvalidation.yaml")
    p.add_argument("--urm_mode",    default=None,
                   help="Run only this urm_mode (session|user). Default: all modes in config.")
    p.add_argument("--n_trials",    type=int,        default=None, help="Override n_trials from config")
    p.add_argument("--n_jobs",      type=int,        default=None, help="Override n_jobs from config")
    p.add_argument("--storage_dir", default="models/CG_crossvalidation",
                   help="Root dir for optuna DBs")
    p.add_argument("--config",      default="configs/tune_crossvalidation.yaml")
    p.add_argument("--monitor",     action="store_true",
                   help="Track and print peak RAM/CPU/GPU/VRAM per trial (local use only).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_one_mode(
    model_name: str,
    urm_mode: str,
    mcfg: dict,
    cfg: dict,
    args: argparse.Namespace,
) -> None:
    """Run (or resume) one Optuna study for a single (model, urm_mode) pair."""
    class_name     = mcfg["class"]
    module_name    = mcfg["module"]
    search_space   = mcfg["search_space"]
    fixed_params   = resolve_param_paths(mcfg.get("fixed_params") or {})
    inference_mode = mcfg.get("inference_mode", "standard")
    n_trials       = args.n_trials or mcfg.get("n_trials", 100)
    n_jobs         = args.n_jobs   or mcfg.get("n_jobs",   1)

    data_cfg   = cfg["data"]
    eval_cfg   = cfg["evaluation"]
    prune_cfg  = cfg.get("pruning", {})
    splitk_dir = repo_path(data_cfg["splitk_dir"])
    n_folds    = int(data_cfg.get("n_folds", 5))
    top_k      = int(eval_cfg.get("top_k", 700))
    recall_ks  = [int(k) for k in eval_cfg.get("recall_ks", RECALL_KS)]
    ndcg_ks    = [int(k) for k in eval_cfg.get("ndcg_ks",   NDCG_KS)]

    meta_path  = data_cfg.get("track_metadata_path")
    track_meta = pl.read_parquet(repo_path(meta_path)) if meta_path else None

    # Per-mode folder and DB
    folder_key  = f"{model_name}_{urm_mode}"
    storage_dir = repo_path(args.storage_dir) / folder_key
    storage_dir.mkdir(parents=True, exist_ok=True)
    db_path    = storage_dir / f"optuna_{folder_key}.db"
    study_name = f"{folder_key}_cv"

    print(f"\n{'='*60}")
    print(f"[cv] model:      {model_name} ({class_name})")
    print(f"[cv] urm_mode:   {urm_mode}")
    print(f"[cv] inference:  {inference_mode}")
    print(f"[cv] n_folds:    {n_folds}  top_k={top_k}")
    print(f"[cv] n_trials:   {n_trials}  n_jobs={n_jobs}")
    print(f"[cv] storage:    sqlite:///{db_path}")

    def objective(trial: optuna.Trial) -> float:
        tuned_params = build_params(trial, search_space)
        params = {**tuned_params, **fixed_params}

        fold_recall200: list[float] = []
        all_fold_metrics: list[dict] = []

        monitor = ResourceMonitor().start() if args.monitor else None
        try:
            for fold in range(n_folds):
                t0 = time.time()
                train_df = load_fold(splitk_dir, fold)
                eval_df  = load_eval(splitk_dir, fold)

                rec = instantiate_rec(class_name, module_name, params, urm_mode)
                rec.fit(train_df, track_metadata=track_meta)

                recs    = run_inference_dispatch(rec, eval_df, top_k, inference_mode, track_meta)
                metrics = score_fold(recs, recall_ks, ndcg_ks)
                fold_recall200.append(metrics["recall@200"])
                all_fold_metrics.append(metrics)

                mean_so_far = sum(fold_recall200) / len(fold_recall200)
                elapsed     = time.time() - t0
                print(f"  [trial {trial.number}] fold {fold}: recall@200={metrics['recall@200']:.4f}"
                      f"  mean={mean_so_far:.4f}  ({elapsed:.1f}s)")

                trial.report(mean_so_far, step=fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        finally:
            if monitor is not None:
                monitor.stop()

        if monitor is not None:
            monitor.print_summary(trial.number)

        mean_metrics = aggregate_mean(all_fold_metrics)
        for k, v in mean_metrics.items():
            trial.set_user_attr(k, v)
        return mean_metrics["recall@200"]

    n_startup = int(prune_cfg.get("n_startup_trials", 5))
    n_warmup  = int(prune_cfg.get("n_warmup_steps",   1))

    study = optuna.create_study(
        study_name=study_name,
        storage=make_storage(db_path),
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=n_startup,
            n_warmup_steps=n_warmup,
        ),
        load_if_exists=True,
    )

    done      = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - done)

    if remaining == 0:
        print(f"[cv] {done} complete trials already — nothing to run")
    else:
        print(f"[cv] {done} existing complete trials, running {remaining} more\n")

        def _callback(_study: optuna.Study, t: optuna.trial.FrozenTrial) -> None:
            if t.state != optuna.trial.TrialState.COMPLETE:
                return
            ps = ", ".join(f"{k}={v}" for k, v in t.params.items())
            print(f"[trial {t.number:>4d}] recall@200={t.value:.6f}  {ps}\n")

        study.optimize(
            objective,
            n_trials=remaining,
            n_jobs=n_jobs,
            gc_after_trial=True,
            callbacks=[_callback],
            show_progress_bar=False,
        )

    if study.best_trial is not None:
        print(f"\n[cv] best recall@200: {study.best_value:.6f}  (trial #{study.best_trial.number})")
        print("[cv] best params:")
        for k, v in study.best_params.items():
            print(f"  {k:22s} {v}")

    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"[cv] total complete trials: {n_complete}")
    plot_study(study, storage_dir / "plots")


def main() -> None:
    args = parse_args()

    cfg_path = pkg_path(args.config)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model_name = args.model
    if model_name not in cfg["models"]:
        sys.exit(f"[cv] Unknown model '{model_name}'. Available: {list(cfg['models'])}")

    mcfg = cfg["models"][model_name]

    # Determine which urm_modes to run
    config_modes = mcfg.get("urm_modes", ["session"])
    if args.urm_mode:
        if args.urm_mode not in config_modes:
            sys.exit(f"[cv] urm_mode '{args.urm_mode}' not in config urm_modes {config_modes}")
        modes_to_run = [args.urm_mode]
    else:
        modes_to_run = config_modes

    for urm_mode in modes_to_run:
        _run_one_mode(model_name, urm_mode, mcfg, cfg, args)


if __name__ == "__main__":
    main()
