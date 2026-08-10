"""DRO-aware cross-validation HP tuning for Candidate Generators.

Per Optuna trial:
  1. Train the CG on each splitK fold's cg_train (5 folds total).
  2. Infer on cg_val → recs.
  3. Score recs via `BlindLikeEvaluator` (calibrated strategy by default)
     → full per-subset NDCG@20 distribution.
  4. Compute a robust scalar from that distribution
     (mean / CVaR_α / Group-DRO worst-group).
  5. Mean robust scalar across folds = Optuna objective.

Output layout under `{storage_dir}/{model}_{urm_mode}_dro/`:
    optuna_{model}_{urm_mode}_dro.db
    plots/...
    (every per-trial NDCG@20 distribution is persisted as a user_attr so
     extract_best_params_dro.py can refit with paired-EB + PoSI later).

Run:
    cd src/basic_candidate_generators
    uv run python -m launchers_dro.tune_crossvalidation_dro \\
        --model item_knn [--robust_mode cvar] [--robust_alpha 0.7]
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
from _dro_objective import (  # noqa: E402
    assign_subset_groups_by_session_mode,
    robust_score,
)
from lbo.evaluator import BlindLikeEvaluator  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DRO CV HP tuning for a single CG model.")
    p.add_argument("--model",       required=True)
    p.add_argument("--urm_mode",    default=None)
    p.add_argument("--n_trials",    type=int, default=None)
    p.add_argument("--n_jobs",      type=int, default=None)
    p.add_argument("--storage_dir", default="models/CG_crossvalidation",
                   help="Output root. DRO folder appends '_dro' to the model key.")
    p.add_argument("--config",      default="launchers_dro/configs/tune_crossvalidation_dro.yaml")
    p.add_argument("--robust_mode", default=None,
                   choices=["mean", "cvar", "group_dro"],
                   help="Override 'robust.mode' from config.")
    p.add_argument("--robust_alpha", type=float, default=None,
                   help="Override 'robust.alpha' from config.")
    p.add_argument("--metric",      default=None,
                   help="Override 'evaluation.metric' (e.g. ndcg@20).")
    p.add_argument("--monitor",     action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-fold scoring via BlindLikeEvaluator
# ---------------------------------------------------------------------------

def _build_fold_evaluators(
    cfg: dict, args: argparse.Namespace,
) -> list[BlindLikeEvaluator]:
    """One BlindLikeEvaluator per fold. Subsets cached → reused across trials."""
    data_cfg = cfg["data"]
    eval_cfg = cfg["evaluation"]
    splitk_dir = repo_path(data_cfg["splitk_dir"])
    n_folds = int(data_cfg.get("n_folds", 5))
    cache_root = repo_path(eval_cfg["cache_dir"])
    cache_root.mkdir(parents=True, exist_ok=True)

    evaluators: list[BlindLikeEvaluator] = []
    for fold in range(n_folds):
        gt_path = splitk_dir / f"fold_{fold}_cg_val.parquet"
        cache_path = cache_root / f"fold_{fold}_cg_val.npz"
        ev = BlindLikeEvaluator(
            gt_path=gt_path,
            blind_parquet=repo_path(data_cfg["blind_parquet"]),
            tracks_meta=repo_path(data_cfg["track_metadata_path"]),
            density_ratio=repo_path(data_cfg["density_ratio"]),
            n_subsets=int(eval_cfg.get("n_subsets", 2000)),
            subset_size=int(eval_cfg.get("subset_size", 80)),
            seed=int(eval_cfg.get("seed", 42)),
            strat_cols=tuple(eval_cfg.get(
                "strat_cols", ["specificity", "category", "pop_mean", "year_mean"],
            )),
            strategies=tuple(eval_cfg.get(
                "strategies", ["calibrated", "density", "stratified"],
            )),
            cache_path=cache_path,
            cvar_alpha=float(cfg.get("robust", {}).get("alpha", 0.7)),
            verbose=False,
        )
        print(
            f"[dro] fold {fold} evaluator ready "
            f"({ev.eval_features.height} eval sessions, "
            f"density coverage={ev.density_coverage})"
        )
        evaluators.append(ev)
    return evaluators


def _precompute_subset_groups(
    evaluators: list[BlindLikeEvaluator], cfg: dict, strategy: str,
) -> list[np.ndarray | None]:
    """For group_dro mode: build per-fold subset-group label arrays.

    Each subset → majority-vote `group_id_mode` from v2 density_ratio.parquet.
    Returns one np.ndarray per fold (same length as n_subsets) or None per
    fold if density ratio lacks `group_id_mode`.
    """
    dr_path = repo_path(cfg["data"]["density_ratio"])
    dr = pl.read_parquet(dr_path)
    if "group_id_mode" not in dr.columns:
        print("[dro] WARN: density_ratio.parquet has no group_id_mode column; "
              "group_dro mode unavailable")
        return [None] * len(evaluators)
    sid_to_group = dict(zip(
        dr["session_id"].to_list(), dr["group_id_mode"].to_list(),
    ))
    out: list[np.ndarray | None] = []
    for ev in evaluators:
        prep = ev.evaluator._prep
        subsets = prep.subsets[strategy]
        groups = assign_subset_groups_by_session_mode(
            subsets, prep.session_ids, sid_to_group,
        )
        out.append(groups)
    return out


def _score_fold(
    ev: BlindLikeEvaluator,
    recs: pl.DataFrame,
    *,
    metric: str,
    strategy: str,
    robust_mode: str,
    robust_alpha: float,
    subset_groups: np.ndarray | None,
) -> tuple[float, np.ndarray]:
    """Returns (robust_scalar, per_subset_array) for one fold."""
    r = ev.score(recs, metric=metric, strategy=strategy)
    scalar = robust_score(
        r.per_subset,
        mode=robust_mode,
        alpha=robust_alpha,
        subset_groups=subset_groups,
    )
    return scalar, r.per_subset


# ---------------------------------------------------------------------------
# Optuna study
# ---------------------------------------------------------------------------

def _run_one_mode(
    model_name: str, urm_mode: str, mcfg: dict, cfg: dict,
    args: argparse.Namespace,
) -> None:
    class_name     = mcfg["class"]
    module_name    = mcfg["module"]
    search_space   = mcfg["search_space"]
    fixed_params   = resolve_param_paths(mcfg.get("fixed_params") or {})
    inference_mode = mcfg.get("inference_mode", "standard")
    n_trials       = args.n_trials or mcfg.get("n_trials", 100)
    n_jobs         = args.n_jobs   or mcfg.get("n_jobs",   1)

    data_cfg = cfg["data"]
    eval_cfg = cfg["evaluation"]
    rob_cfg  = cfg.get("robust", {})
    prune_cfg = cfg.get("pruning", {})

    splitk_dir = repo_path(data_cfg["splitk_dir"])
    n_folds = int(data_cfg.get("n_folds", 5))
    top_k = int(eval_cfg.get("top_k_inference", 700))
    metric = args.metric or eval_cfg.get("metric", "ndcg@20")
    strategy = eval_cfg.get("strategy", "calibrated")
    robust_mode = args.robust_mode or rob_cfg.get("mode", "cvar")
    robust_alpha = (
        args.robust_alpha if args.robust_alpha is not None
        else float(rob_cfg.get("alpha", 0.7))
    )

    meta_path = data_cfg.get("track_metadata_path")
    track_meta = pl.read_parquet(repo_path(meta_path)) if meta_path else None

    # ── output paths ───────────────────────────────────────────────────────
    folder_key = f"{model_name}_{urm_mode}_dro"
    storage_dir = repo_path(args.storage_dir) / folder_key
    storage_dir.mkdir(parents=True, exist_ok=True)
    db_path = storage_dir / f"optuna_{folder_key}.db"
    suffix = (
        f"cvar{int(round(robust_alpha * 100))}" if robust_mode == "cvar"
        else robust_mode
    )
    study_name = f"{folder_key}_{suffix}"

    print(f"\n{'=' * 60}")
    print(f"[dro] model:      {model_name} ({class_name})")
    print(f"[dro] urm_mode:   {urm_mode}")
    print(f"[dro] inference:  {inference_mode}")
    print(f"[dro] metric:     {metric}  strategy={strategy}")
    print(f"[dro] robust:     {robust_mode} (alpha={robust_alpha})")
    print(f"[dro] folds:      {n_folds}  top_k_inference={top_k}")
    print(f"[dro] trials:     {n_trials}  n_jobs={n_jobs}")
    print(f"[dro] storage:    sqlite:///{db_path}")

    # ── one-time setup: per-fold evaluators (subsets cached) ──────────────
    print("[dro] building per-fold BlindLikeEvaluators…")
    evaluators = _build_fold_evaluators(cfg, args)
    subset_groups_per_fold: list[np.ndarray | None] = [None] * n_folds
    if robust_mode == "group_dro":
        print("[dro] precomputing subset-group labels (majority-vote group_id_mode)…")
        subset_groups_per_fold = _precompute_subset_groups(evaluators, cfg, strategy)

    # ── prebuild per-fold ICMs once (HP-independent) → reused by every trial ──
    # Standard (UserRecommender) models only; text CGs don't take precomputed_icm.
    # Skip for models that never consume the ICM (config `uses_icm: false`).
    fold_icms: list = [None] * n_folds
    if (track_meta is not None and inference_mode != "text"
            and mcfg.get("uses_icm", True)):
        print("[dro] prebuilding per-fold ICMs (reused across all trials)…")
        for fold in range(n_folds):
            fold_icms[fold] = build_fold_icm(
                load_fold(splitk_dir, fold), track_meta, urm_mode,
            )

    # ── per-fold query-tower bundles (text mode) ──────────────────────────
    # Precomputed query_text + query embeddings for each fold's cg_val, joined
    # per (session, turn) at inference. HP-independent → loaded once.
    query_bundles: list = [None] * n_folds
    if inference_mode == "text":
        from embedding_based.query_tower import QUERY_TOWER_BASE, load_query_bundle
        print("[dro] loading per-fold cg_val query bundles…")
        for fold in range(n_folds):
            query_bundles[fold] = load_query_bundle(
                repo_path(QUERY_TOWER_BASE), "cg_val", fold,
            )

    # ── per-fold tfidf rk cache (text models exposing tfidf_rk_table) ─────
    # tfidf vocab + query text are fixed across trials (only RRF weights / k1 /
    # b / decays vary), so the tfidf signal is identical every trial. Compute
    # its top-K rankings once per fold and inject → skips transform + scoring
    # in every trial. K = max searched top_k_per_signal.
    tfidf_cache: list = [None] * n_folds
    if inference_mode == "text" and "top_k_per_signal" in search_space:
        proto = instantiate_rec(class_name, module_name, fixed_params, urm_mode)
        if hasattr(proto, "tfidf_rk_table"):
            K = int(search_space["top_k_per_signal"]["high"])
            print(f"[dro] precomputing per-fold tfidf rk cache (K={K})…")
            proto.fit(None, track_metadata=track_meta)
            for fold in range(n_folds):
                qb = query_bundles[fold]
                keys = list(zip(qb["session_id"].to_list(),
                                qb["turn_number"].to_list()))
                mat = proto.tfidf_rk_table(qb["query_text"].to_list(), K).astype(np.int32)
                tfidf_cache[fold] = ({k: i for i, k in enumerate(keys)}, mat)

    # ── objective ─────────────────────────────────────────────────────────
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
                train_df = load_fold(splitk_dir, fold)
                eval_df  = load_eval(splitk_dir, fold)

                rec = instantiate_rec(class_name, module_name, params, urm_mode)
                fit_kwargs = {"track_metadata": track_meta}
                if fold_icms[fold] is not None:
                    fit_kwargs["precomputed_icm"] = fold_icms[fold]
                rec.fit(train_df, **fit_kwargs)
                if tfidf_cache[fold] is not None and hasattr(rec, "_tfidf_rk_inject"):
                    rec._tfidf_rk_inject = tfidf_cache[fold]
                recs = run_inference_dispatch(
                    rec, eval_df, top_k, inference_mode, track_meta,
                    query_bundle=query_bundles[fold],
                )

                scalar, per_subset = _score_fold(
                    evaluators[fold], recs,
                    metric=metric, strategy=strategy,
                    robust_mode=robust_mode,
                    robust_alpha=robust_alpha,
                    subset_groups=subset_groups_per_fold[fold],
                )
                fold_scalars.append(scalar)
                fold_means.append(float(np.nanmean(per_subset)))
                all_per_subset.append(per_subset.tolist())
                mean_so_far = sum(fold_scalars) / len(fold_scalars)
                elapsed = time.time() - t0
                print(
                    f"  [trial {trial.number}] fold {fold}: "
                    f"{robust_mode}={scalar:.4f}  mean(NDCG)={fold_means[-1]:.4f}  "
                    f"running={mean_so_far:.4f}  ({elapsed:.1f}s)"
                )

                trial.report(mean_so_far, step=fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        finally:
            if monitor is not None:
                monitor.stop()
        if monitor is not None:
            monitor.print_summary(trial.number)

        # Persist per-fold diagnostics so extract_best_params_dro.py can apply
        # paired-EB / PoSI without re-running every trial.
        trial.set_user_attr(f"{robust_mode}_per_fold", fold_scalars)
        trial.set_user_attr("mean_per_fold", fold_means)
        trial.set_user_attr("per_subset_per_fold", all_per_subset)
        trial.set_user_attr("metric", metric)
        trial.set_user_attr("robust_mode", robust_mode)
        trial.set_user_attr("robust_alpha", robust_alpha)

        return float(np.mean(fold_scalars))

    # ── study ─────────────────────────────────────────────────────────────
    n_startup = int(prune_cfg.get("n_startup_trials", 5))
    n_warmup  = int(prune_cfg.get("n_warmup_steps", 1))
    study = optuna.create_study(
        study_name=study_name,
        storage=make_storage(db_path),
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=n_startup, n_warmup_steps=n_warmup,
        ),
        load_if_exists=True,
    )

    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - done)
    if remaining == 0:
        print(f"[dro] {done} complete trials already — nothing to run")
    else:
        print(f"[dro] {done} existing complete trials, running {remaining} more\n")

        def _callback(_study: optuna.Study, t: optuna.trial.FrozenTrial) -> None:
            if t.state != optuna.trial.TrialState.COMPLETE:
                return
            ps = ", ".join(f"{k}={v}" for k, v in t.params.items())
            print(f"[trial {t.number:>4d}] {robust_mode}={t.value:.6f}  {ps}\n")

        study.optimize(
            objective,
            n_trials=remaining,
            n_jobs=n_jobs,
            gc_after_trial=True,
            callbacks=[_callback],
            show_progress_bar=False,
        )

    if study.best_trial is not None:
        print(f"\n[dro] best {robust_mode}: {study.best_value:.6f} "
              f"(trial #{study.best_trial.number})")
        for k, v in study.best_params.items():
            print(f"  {k:22s} {v}")

    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"[dro] total complete trials: {n_complete}")
    plot_study(study, storage_dir / "plots")


def main() -> None:
    args = parse_args()
    cfg_path = pkg_path(args.config)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model_name = args.model
    if model_name not in cfg["models"]:
        sys.exit(f"[dro] Unknown model '{model_name}'. "
                 f"Available: {list(cfg['models'])}")

    mcfg = cfg["models"][model_name]
    config_modes = mcfg.get("urm_modes", ["session"])
    if args.urm_mode:
        if args.urm_mode not in config_modes:
            sys.exit(f"[dro] urm_mode '{args.urm_mode}' not in {config_modes}")
        modes_to_run = [args.urm_mode]
    else:
        modes_to_run = config_modes

    for urm_mode in modes_to_run:
        _run_one_mode(model_name, urm_mode, mcfg, cfg, args)


if __name__ == "__main__":
    main()
