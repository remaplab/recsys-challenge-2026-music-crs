"""Pick the final DRO HP set from a finished Optuna DRO study.

Pipeline:
    1. Load all completed trials from
       `{storage_dir}/{model}_{urm}_dro/optuna_{model}_{urm}_dro.db`.
    2. Sort by Optuna objective (robust scalar). Take top-K.
    3. **Paired-EB dedup** on top-K via per-fold robust scalars
       (LBO doc C): drop trials CI-significantly dominated by a stronger one.
    4. Optional **PoSI on reranker_val OOF** (5×4% = ~20% of non-holdout):
       for each survivor and each fold k, refit CG on cg_train_k ∪ cg_val_k
       (deployment-like training set that will feed the reranker), infer on
       reranker_val_k, score via per-fold `BlindLikeEvaluator` → robust
       scalar_k. PoSI score = mean over folds.
       Holdout_test is NOT touched here — reserved for the reranker
       pipeline's own PoSI.
    5. Final pick = trial with highest PoSI score (or highest CV if --skip_posi).
    6. Write `configs/cv_best_{model}_{urm}_dro.yaml`.

Run:
    cd src/basic_candidate_generators
    uv run python -m launchers_dro.extract_best_params_dro \\
        --model item_knn --urm_mode session [--top_k 10] [--skip_posi]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
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
    instantiate_rec,
    load_fold,
    load_reranker_val,
    make_storage,
    pkg_path,
    repo_path,
    resolve_param_paths,
    run_inference_dispatch,
)
from _dro_objective import robust_score  # noqa: E402
from lbo.evaluator import BlindLikeEvaluator  # noqa: E402
from lbo.shift.multi_comp import paired_empirical_bernstein_ci  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract best DRO params + paired-EB dedup + optional PoSI."
    )
    p.add_argument("--model", required=True)
    p.add_argument("--urm_mode", default="session")
    p.add_argument("--storage_dir", default="models/CG_crossvalidation")
    p.add_argument("--config",
                   default="launchers_dro/configs/tune_crossvalidation_dro.yaml")
    p.add_argument("--top_k", type=int, default=10,
                   help="Take top-K trials by CV robust scalar before EB / PoSI.")
    p.add_argument("--eb_delta", type=float, default=0.05,
                   help="Paired-EB miscoverage level for dedup.")
    p.add_argument("--skip_posi", action="store_true",
                   help="Skip PoSI on holdout — pick top CV survivor directly.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eb_dedup(
    survivors: list[int], fold_scalars: dict[int, np.ndarray],
    *, delta: float,
) -> list[int]:
    """Drop trials whose per-fold robust scalar is CI-significantly dominated
    by a stronger survivor (paired-EB lower bound > 0).
    """
    keep = set(survivors)
    ordered = sorted(survivors, key=lambda t: -float(fold_scalars[t].mean()))
    for i, t1 in enumerate(ordered):
        if t1 not in keep:
            continue
        for t2 in ordered[i + 1:]:
            if t2 not in keep:
                continue
            d = fold_scalars[t1] - fold_scalars[t2]
            if float(d.mean()) <= 0:
                continue
            lo, _ = paired_empirical_bernstein_ci(d, delta=delta)
            if lo > 0:
                keep.discard(t2)
    return sorted(keep)


def _posi_score(
    trial_params: dict, fixed_params: dict, mcfg: dict, cfg: dict,
    *,
    robust_mode: str, robust_alpha: float, metric: str, strategy: str,
    top_k_inference: int,
) -> tuple[float, list[float]]:
    """OOF PoSI on reranker_val (5 × 4% = ~20% of non-holdout).

    For each fold k:
      train CG on cg_train_k ∪ cg_val_k (deployment-like — same training
      set that feeds the reranker's OOF candidates),
      infer on reranker_val_k,
      score via per-fold `BlindLikeEvaluator` → robust scalar_k.

    Returns (mean over folds, per-fold scalars). Holdout_test is not
    touched — reserved for the reranker pipeline.
    """
    class_name  = mcfg["class"]
    module_name = mcfg["module"]
    urm_mode    = mcfg["_urm_mode_internal"]  # injected by caller
    inference_mode = mcfg.get("inference_mode", "standard")

    data_cfg = cfg["data"]
    eval_cfg = cfg["evaluation"]
    splitk_dir = repo_path(data_cfg["splitk_dir"])
    n_folds = int(data_cfg.get("n_folds", 5))
    track_meta = pl.read_parquet(repo_path(data_cfg["track_metadata_path"]))

    posi_cache_root = repo_path(eval_cfg.get(
        "posi_cache_dir", "models/CG_crossvalidation/eval_cache_dro_posi",
    ))
    posi_cache_root.mkdir(parents=True, exist_ok=True)
    n_subsets = int(eval_cfg.get(
        "posi_n_subsets", eval_cfg.get("n_subsets", 2000),
    ))

    full_params = resolve_param_paths({**(fixed_params or {}), **trial_params})

    per_fold: list[float] = []
    for fold in range(n_folds):
        train_df = load_fold(
            splitk_dir, fold, parts=["cg_train", "cg_val"],
        )
        eval_df = load_reranker_val(splitk_dir, fold)

        rec = instantiate_rec(class_name, module_name, full_params, urm_mode)
        rec.fit(train_df, track_metadata=track_meta)
        recs = run_inference_dispatch(
            rec, eval_df, top_k_inference, inference_mode, track_meta,
        )

        ev = BlindLikeEvaluator(
            gt_path=splitk_dir / f"fold_{fold}_reranker_val.parquet",
            blind_parquet=repo_path(data_cfg["blind_parquet"]),
            tracks_meta=repo_path(data_cfg["track_metadata_path"]),
            density_ratio=repo_path(data_cfg["density_ratio"]),
            n_subsets=n_subsets,
            subset_size=int(eval_cfg.get("subset_size", 80)),
            seed=int(eval_cfg.get("seed", 42)),
            strat_cols=tuple(eval_cfg.get(
                "strat_cols", ["specificity", "category", "pop_mean", "year_mean"],
            )),
            strategies=tuple(eval_cfg.get(
                "strategies", ["calibrated", "density", "stratified"],
            )),
            cache_path=posi_cache_root / f"fold_{fold}_reranker_val.npz",
            cvar_alpha=robust_alpha,
            verbose=False,
        )
        r = ev.score(recs, metric=metric, strategy=strategy)
        scalar = float(robust_score(
            r.per_subset, mode=robust_mode, alpha=robust_alpha,
        ))
        per_fold.append(scalar)
        print(f"    fold {fold}: {robust_mode}={scalar:.4f}")

    return float(np.mean(per_fold)), per_fold


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    with open(pkg_path(args.config)) as f:
        cfg = yaml.safe_load(f)
    if args.model not in cfg["models"]:
        sys.exit(f"unknown model {args.model!r}")
    mcfg = dict(cfg["models"][args.model])
    mcfg["_urm_mode_internal"] = args.urm_mode

    rob_cfg = cfg.get("robust", {})
    robust_mode = rob_cfg.get("mode", "cvar")
    robust_alpha = float(rob_cfg.get("alpha", 0.7))
    eval_cfg = cfg["evaluation"]
    metric = eval_cfg.get("metric", "ndcg@20")
    strategy = eval_cfg.get("strategy", "calibrated")
    top_k_inference = int(eval_cfg.get("top_k_inference", 700))

    folder_key = f"{args.model}_{args.urm_mode}_dro"
    storage_dir = repo_path(args.storage_dir) / folder_key
    db_path = storage_dir / f"optuna_{folder_key}.db"
    if not db_path.exists():
        sys.exit(f"DB not found: {db_path}")

    suffix = (
        f"cvar{int(round(robust_alpha * 100))}" if robust_mode == "cvar"
        else robust_mode
    )
    study_name = f"{folder_key}_{suffix}"
    study = optuna.load_study(
        study_name=study_name, storage=make_storage(db_path),
    )

    complete = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    if not complete:
        sys.exit("no complete trials")
    print(f"[extract-dro] study={study_name}  complete trials={len(complete)}")

    # ── top-K by CV robust scalar ──────────────────────────────────────────
    top = sorted(complete, key=lambda t: -t.value)[: args.top_k]
    print(f"[extract-dro] top-{args.top_k} by CV {robust_mode}:")
    for t in top:
        print(f"  trial {t.number:>4d}  {t.value:.4f}")

    # ── paired-EB dedup over per-fold robust scalars ──────────────────────
    fold_arr = {
        t.number: np.asarray(
            t.user_attrs.get(f"{robust_mode}_per_fold", []), dtype=np.float64,
        )
        for t in top
    }
    survivors_after_eb = _eb_dedup(
        [t.number for t in top], fold_arr, delta=args.eb_delta,
    )
    print(f"[extract-dro] EB-dedup survivors: {survivors_after_eb}")

    # ── PoSI on reranker_val OOF (5 × 4%) ─────────────────────────────────
    posi: dict[int, float] = {}
    posi_per_fold: dict[int, list[float]] = {}
    if not args.skip_posi:
        print("[extract-dro] computing PoSI on reranker_val OOF (5 folds)…")
        for tn in survivors_after_eb:
            t = next(x for x in top if x.number == tn)
            print(f"  trial {tn:>4d}:")
            mean_score, per_fold = _posi_score(
                t.params, mcfg.get("fixed_params") or {}, mcfg, cfg,
                robust_mode=robust_mode, robust_alpha=robust_alpha,
                metric=metric, strategy=strategy,
                top_k_inference=top_k_inference,
            )
            posi[tn] = mean_score
            posi_per_fold[tn] = per_fold
            print(f"    → mean PoSI_{robust_mode}={mean_score:.4f}")
        best_tn = max(posi, key=posi.get)
    else:
        best_tn = survivors_after_eb[0]

    best_trial = next(t for t in top if t.number == best_tn)
    print(f"\n[extract-dro] picked trial {best_tn}  "
          f"CV {robust_mode}={best_trial.value:.4f}"
          f"  PoSI={posi.get(best_tn, float('nan')):.4f}")

    # ── write cv_best YAML ────────────────────────────────────────────────
    out_cfg = {
        "class":          mcfg["class"],
        "module":         mcfg["module"],
        "inference_mode": mcfg.get("inference_mode", "standard"),
        "urm_mode":       args.urm_mode,
        "robust": {
            "mode": robust_mode, "alpha": robust_alpha,
            "metric": metric, "strategy": strategy,
        },
        "study":          study_name,
        "best_trial":     best_tn,
        "cv_score":       float(best_trial.value),
        "posi_score":     posi.get(best_tn),
        "posi_per_fold":  posi_per_fold.get(best_tn),
        "eb_survivors":   list(survivors_after_eb),
        "best_params":    dict(best_trial.params),
        "fixed_params":   dict(mcfg.get("fixed_params") or {}),
    }
    suffix_y = (
        f"_cvar{int(round(robust_alpha * 100))}" if robust_mode == "cvar"
        else f"_{robust_mode}"
    )

    # Write to BOTH the package configs/ (so legacy tools that scan that dir
    # still find it) AND the model storage dir (mirrors the non-DRO pipeline:
    # best_params YAML lives next to the optuna DB + checkpoints).
    legacy_yaml = pkg_path(
        f"configs/cv_best_{args.model}_{args.urm_mode}_dro{suffix_y}.yaml"
    )
    storage_yaml = storage_dir / (
        f"best_params_{args.model}_{args.urm_mode}_dro{suffix_y}.yaml"
    )
    for out_yaml in (legacy_yaml, storage_yaml):
        out_yaml.parent.mkdir(parents=True, exist_ok=True)
        with open(out_yaml, "w") as f:
            yaml.safe_dump(out_cfg, f, sort_keys=False)
        print(f"[extract-dro] wrote → {out_yaml}")


if __name__ == "__main__":
    main()
