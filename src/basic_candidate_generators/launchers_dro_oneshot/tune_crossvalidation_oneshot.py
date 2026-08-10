"""One-shot (turn-1-only) cross-validation HP tuning for Candidate Generators.

Sibling of `launchers_dro.tune_crossvalidation_dro`. Identical per-trial loop
(train on fold cg_train → infer on cg_val → BlindLikeEvaluator → robust scalar →
mean across folds), with two differences driven by the one-shot setting:

  * `data.turn_filter` (default 1): train_df, eval_df, and the evaluator GT are
    all restricted to that turn_number. Turn 1 = no prior session context, so
    every CG reduces to a query/profile retriever.
  * `evaluation.metric` defaults to recall@200 (CG = candidate coverage for the
    reranker; the NDCG@20 leaderboard metric is optimised downstream).

Output: models/CG_crossvalidation/<model>_oneshot/ (optuna db + plots; per-trial
NDCG/recall distributions persisted as user_attrs for extract_best_params).

Run:
    cd src/basic_candidate_generators
    uv run python -m launchers_dro_oneshot.tune_crossvalidation_oneshot --model dense_text_8b
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent   # src/basic_candidate_generators/
_SRC_ROOT = _PKG_ROOT / "src"
_CV_ROOT  = _PKG_ROOT / "launchers_crossvalidation"
_DRO_ROOT = _PKG_ROOT / "launchers_dro"
_REPO_ROOT = _PKG_ROOT.parent.parent
_LBO_SRC = _REPO_ROOT / "src" / "lower_bound_optimization" / "src"

for p in (_SRC_ROOT, _CV_ROOT, _DRO_ROOT, str(_LBO_SRC)):
    sys.path.insert(0, str(p))

import numpy as np      # noqa: E402
import optuna           # noqa: E402
import polars as pl     # noqa: E402
import yaml             # noqa: E402

from _cv_utils import (  # noqa: E402
    ResourceMonitor,
    build_fold_icm,
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
)
from _dro_objective import robust_score  # noqa: E402
from lbo.evaluator import BlindLikeEvaluator  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-shot (turn-1) CV HP tuning for a single CG.")
    p.add_argument("--model",       required=True)
    p.add_argument("--urm_mode",    default=None)
    p.add_argument("--n_trials",    type=int, default=None)
    p.add_argument("--n_jobs",      type=int, default=None)
    p.add_argument("--storage_dir", default="models/CG_crossvalidation",
                   help="Output root. One-shot folder appends '_oneshot' to the model key.")
    p.add_argument("--config",      default="launchers_dro_oneshot/configs/tune_oneshot.yaml")
    p.add_argument("--robust_mode", default=None, choices=["mean", "cvar", "group_dro"])
    p.add_argument("--robust_alpha", type=float, default=None)
    p.add_argument("--metric",      default=None)
    p.add_argument("--monitor",     action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# turn-1 GT parquets (filtered cg_val, written once → reused by the evaluator)
# ---------------------------------------------------------------------------

def _turn_filtered_gt(splitk_dir: Path, fold: int, turn: int, cache_root: Path) -> Path:
    """Write (once) a turn-filtered copy of fold_{fold}_cg_val and return its path."""
    src = splitk_dir / f"fold_{fold}_cg_val.parquet"
    dst = cache_root / f"fold_{fold}_cg_val_turn{turn}.parquet"
    if not dst.exists():
        cache_root.mkdir(parents=True, exist_ok=True)
        pl.read_parquet(src).filter(pl.col("turn_number") == turn).write_parquet(dst)
    return dst


def _build_fold_evaluators(cfg: dict, turn: int) -> list[BlindLikeEvaluator]:
    """One BlindLikeEvaluator per fold over the turn-filtered GT. Subsets cached."""
    data_cfg = cfg["data"]; eval_cfg = cfg["evaluation"]
    splitk_dir = repo_path(data_cfg["splitk_dir"])
    n_folds = int(data_cfg.get("n_folds", 5))
    cache_root = repo_path(eval_cfg["cache_dir"])
    cache_root.mkdir(parents=True, exist_ok=True)

    evaluators: list[BlindLikeEvaluator] = []
    for fold in range(n_folds):
        gt_path = _turn_filtered_gt(splitk_dir, fold, turn, cache_root)
        ev = BlindLikeEvaluator(
            gt_path=gt_path,
            blind_parquet=repo_path(data_cfg["blind_parquet"]),
            tracks_meta=repo_path(data_cfg["track_metadata_path"]),
            density_ratio=repo_path(data_cfg["density_ratio"]),
            n_subsets=int(eval_cfg.get("n_subsets", 2000)),
            subset_size=int(eval_cfg.get("subset_size", 80)),
            seed=int(eval_cfg.get("seed", 42)),
            strat_cols=tuple(eval_cfg.get(
                "strat_cols", ["specificity", "category", "pop_mean", "year_mean"])),
            strategies=tuple(eval_cfg.get(
                "strategies", ["calibrated", "density", "stratified"])),
            cache_path=cache_root / f"fold_{fold}_cg_val_turn{turn}.npz",
            cvar_alpha=float(cfg.get("robust", {}).get("alpha", 0.7)),
            verbose=False,
        )
        # The wrapper samples a per-session max_turn from the multiturn blind
        # PMF; recs/GT are then joined at target_turn = max_turn + 1. For a
        # one-shot (turn-`turn`) evaluation we force every session's target turn
        # to `turn` (max_turn = turn - 1). Subset SAMPLING is independent of
        # max_turn, so overriding it post-construction (incl. after a cache
        # load) is safe and gives ~100% coverage at the target turn.
        ev.evaluator._prep.max_turn_per_row[:] = turn - 1
        print(f"[oneshot] fold {fold} evaluator ready "
              f"({ev.eval_features.height} turn-{turn} eval sessions, "
              f"density coverage={ev.density_coverage})")
        evaluators.append(ev)
    return evaluators


# ---------------------------------------------------------------------------
# Optuna study
# ---------------------------------------------------------------------------

def _run_one_mode(model_name: str, urm_mode: str, mcfg: dict, cfg: dict,
                  args: argparse.Namespace) -> None:
    class_name     = mcfg["class"]
    module_name    = mcfg["module"]
    search_space   = mcfg["search_space"]
    fixed_params   = resolve_param_paths(mcfg.get("fixed_params") or {})
    inference_mode = mcfg.get("inference_mode", "standard")
    uses_colisten  = mcfg.get("uses_colisten", False)   # E3.6: needs unfiltered fold
    n_trials       = args.n_trials or mcfg.get("n_trials", 100)
    n_jobs         = args.n_jobs   or mcfg.get("n_jobs",   1)

    data_cfg = cfg["data"]; eval_cfg = cfg["evaluation"]
    rob_cfg  = cfg.get("robust", {}); prune_cfg = cfg.get("pruning", {})

    splitk_dir = repo_path(data_cfg["splitk_dir"])
    n_folds = int(data_cfg.get("n_folds", 5))
    turn = int(data_cfg.get("turn_filter", 1))
    top_k = int(eval_cfg.get("top_k_inference", 700))
    metric = args.metric or eval_cfg.get("metric", "recall@200")
    strategy = eval_cfg.get("strategy", "calibrated")
    robust_mode = args.robust_mode or rob_cfg.get("mode", "cvar")
    robust_alpha = (args.robust_alpha if args.robust_alpha is not None
                    else float(rob_cfg.get("alpha", 0.7)))

    meta_path = data_cfg.get("track_metadata_path")
    track_meta = pl.read_parquet(repo_path(meta_path)) if meta_path else None

    folder_key = f"{model_name}_oneshot"
    storage_dir = repo_path(args.storage_dir) / folder_key
    storage_dir.mkdir(parents=True, exist_ok=True)
    db_path = storage_dir / f"optuna_{folder_key}.db"
    suffix = (f"cvar{int(round(robust_alpha * 100))}" if robust_mode == "cvar"
              else robust_mode)
    study_name = f"{folder_key}_{suffix}"

    print(f"\n{'=' * 60}")
    print(f"[oneshot] model:    {model_name} ({class_name})")
    print(f"[oneshot] turn:     {turn}   inference={inference_mode}")
    print(f"[oneshot] metric:   {metric}  strategy={strategy}")
    print(f"[oneshot] robust:   {robust_mode} (alpha={robust_alpha})")
    print(f"[oneshot] folds:    {n_folds}  top_k_inference={top_k}")
    print(f"[oneshot] trials:   {n_trials}  n_jobs={n_jobs}")
    print(f"[oneshot] storage:  sqlite:///{db_path}")

    print("[oneshot] building per-fold BlindLikeEvaluators (turn-filtered GT)…")
    evaluators = _build_fold_evaluators(cfg, turn)

    # Per-fold ICMs (standard models that consume an ICM). HP-independent.
    fold_icms: list = [None] * n_folds
    if (track_meta is not None and inference_mode != "text"
            and mcfg.get("uses_icm", True)):
        print("[oneshot] prebuilding per-fold ICMs…")
        for fold in range(n_folds):
            fold_icms[fold] = build_fold_icm(
                load_fold(splitk_dir, fold).filter(pl.col("turn_number") == turn),
                track_meta, urm_mode)

    # Per-fold query bundles (text mode only).
    query_bundles: list = [None] * n_folds
    if inference_mode == "text":
        from embedding_based.query_tower import QUERY_TOWER_BASE, load_query_bundle
        print("[oneshot] loading per-fold cg_val query bundles…")
        for fold in range(n_folds):
            query_bundles[fold] = load_query_bundle(
                repo_path(QUERY_TOWER_BASE), "cg_val", fold)

    def objective(trial: optuna.Trial) -> float:
        tuned_params = build_params(trial, search_space)
        params = {**tuned_params, **fixed_params}

        fold_scalars: list[float] = []
        fold_means: list[float] = []
        all_per_subset: list[list[float]] = []

        monitor = ResourceMonitor().start() if args.monitor else None
        try:
            for fold in range(n_folds):
                t0 = time.time()
                train_df = load_fold(splitk_dir, fold).filter(pl.col("turn_number") == turn)
                eval_df  = load_eval(splitk_dir, fold).filter(pl.col("turn_number") == turn)

                rec = instantiate_rec(class_name, module_name, params, urm_mode)
                fit_kwargs = {"track_metadata": track_meta}
                if fold_icms[fold] is not None:
                    fit_kwargs["precomputed_icm"] = fold_icms[fold]
                if uses_colisten:
                    # co-listen needs full multi-turn sessions, not the turn-1 slice
                    fit_kwargs["colisten_df"] = load_fold(splitk_dir, fold)
                rec.fit(train_df, **fit_kwargs)
                recs = run_inference_dispatch(
                    rec, eval_df, top_k, inference_mode, track_meta,
                    query_bundle=query_bundles[fold])

                r = evaluators[fold].score(recs, metric=metric, strategy=strategy)
                scalar = robust_score(r.per_subset, mode=robust_mode, alpha=robust_alpha)
                fold_scalars.append(scalar)
                fold_means.append(float(np.nanmean(r.per_subset)))
                all_per_subset.append(r.per_subset.tolist())
                mean_so_far = sum(fold_scalars) / len(fold_scalars)
                print(f"  [trial {trial.number}] fold {fold}: "
                      f"{robust_mode}={scalar:.4f}  mean({metric})={fold_means[-1]:.4f}  "
                      f"running={mean_so_far:.4f}  ({time.time()-t0:.1f}s)")

                trial.report(mean_so_far, step=fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        finally:
            if monitor is not None:
                monitor.stop()
        if monitor is not None:
            monitor.print_summary(trial.number)

        trial.set_user_attr(f"{robust_mode}_per_fold", fold_scalars)
        trial.set_user_attr("mean_per_fold", fold_means)
        trial.set_user_attr("per_subset_per_fold", all_per_subset)
        trial.set_user_attr("metric", metric)
        trial.set_user_attr("robust_mode", robust_mode)
        trial.set_user_attr("robust_alpha", robust_alpha)
        return float(np.mean(fold_scalars))

    n_startup = int(prune_cfg.get("n_startup_trials", 5))
    n_warmup  = int(prune_cfg.get("n_warmup_steps", 1))
    study = optuna.create_study(
        study_name=study_name,
        storage=make_storage(db_path),
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=n_startup, n_warmup_steps=n_warmup),
        load_if_exists=True,
    )

    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - done)
    if remaining == 0:
        print(f"[oneshot] {done} complete trials already — nothing to run")
    else:
        print(f"[oneshot] {done} existing complete trials, running {remaining} more\n")

        def _callback(_study: optuna.Study, t: optuna.trial.FrozenTrial) -> None:
            if t.state != optuna.trial.TrialState.COMPLETE:
                return
            ps = ", ".join(f"{k}={v}" for k, v in t.params.items())
            print(f"[trial {t.number:>4d}] {robust_mode}={t.value:.6f}  {ps}\n")

        study.optimize(objective, n_trials=remaining, n_jobs=n_jobs,
                       gc_after_trial=True, callbacks=[_callback],
                       show_progress_bar=False)

    if study.best_trial is not None:
        print(f"\n[oneshot] best {robust_mode}: {study.best_value:.6f} "
              f"(trial #{study.best_trial.number})")
        for k, v in study.best_params.items():
            print(f"  {k:22s} {v}")
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"[oneshot] total complete trials: {n_complete}")
    plot_study(study, storage_dir / "plots")


def main() -> None:
    args = parse_args()
    with open(pkg_path(args.config)) as f:
        cfg = yaml.safe_load(f)

    model_name = args.model
    if model_name not in cfg["models"]:
        sys.exit(f"[oneshot] Unknown model '{model_name}'. Available: {list(cfg['models'])}")

    mcfg = cfg["models"][model_name]
    config_modes = mcfg.get("urm_modes", ["session"])
    if args.urm_mode:
        if args.urm_mode not in config_modes:
            sys.exit(f"[oneshot] urm_mode '{args.urm_mode}' not in {config_modes}")
        modes_to_run = [args.urm_mode]
    else:
        modes_to_run = config_modes

    for urm_mode in modes_to_run:
        _run_one_mode(model_name, urm_mode, mcfg, cfg, args)


if __name__ == "__main__":
    main()
