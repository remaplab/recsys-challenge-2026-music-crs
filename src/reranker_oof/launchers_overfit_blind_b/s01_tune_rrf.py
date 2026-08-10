"""Tune weighted RRF for the Blind-B overfit pipeline.

Per trial: sample a per-(cg, turn-bucket) weight matrix + the RRF ``k`` → fuse
the candidate rank pool → macro-by-turn recall@K (the config ``tune.metric``) =
objective, maximised over the ``tune.target`` GT (``blind_b`` = all 280 turns,
or ``holdout``). Optionally restricts candidates to the test-tracks catalogue
(Blind sessions only) when ``test_tracks.enabled``.

After every trial a 2-panel figure is refreshed: (left) the objective recall@K
across trials in green; (right) the current trial's recall@{1,5,10,20,50,100,200}.

Target / metric / Optuna settings are read from ``dataset.yaml``; CLI flags
override. Usage:
    cd src/reranker_oof
    uv run python -m launchers_overfit_blind_b.s01_tune_rrf \\
        --config configs/blind_v1/dataset.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG_ROOT))
sys.path.insert(0, str(_PKG_ROOT / "src"))

import matplotlib                                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
import optuna                                                          # noqa: E402

from src.paths import OPTUNA_DIR, PLOTS_DIR, ensure_output_dirs        # noqa: E402

from launchers_overfit_blind_b._common import (                        # noqa: E402
    PLOT_KS, assert_cgs_have, filter_candidates, fuse_rrf, last_turn_gt,
    load_config, load_gt, load_rank_pool, recall_at, register_cg_paths,
    test_tracks_spec, turn_buckets,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _metric_k(metric: str) -> int:
    name, k = metric.split("@")
    if name != "recall":
        raise SystemExit(f"only recall@K supported, got {metric!r}")
    return int(k)


def _sampler(name: str):
    return {"tpe": optuna.samplers.TPESampler,
            "random": optuna.samplers.RandomSampler,
            "cmaes": optuna.samplers.CmaEsSampler}[(name or "tpe").lower()](seed=42)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--target", choices=["blind_b", "holdout", "both"], default=None)
    ap.add_argument("--metric", default=None)
    ap.add_argument("--n_trials", type=int, default=None)
    args = ap.parse_args()

    ensure_output_dirs()
    cfg = load_config(args.config)
    register_cg_paths(cfg)

    tune = cfg.get("tune", {})
    target = args.target or tune.get("target", "blind_b")
    metric = args.metric or tune.get("metric", "recall@200")
    mk = _metric_k(metric)
    ks = sorted(set(PLOT_KS) | {mk})
    top_k = int(cfg.get("top_k", 200))
    buckets = turn_buckets(cfg)
    bucket_names = [b[0] for b in buckets]

    # target=both → optimise a weighted average of recall on holdout AND blind_b
    # with the SAME sampled weights. Holdout weight fixed at 1.0; blind_b weight
    # from the config. Single target → one eval split, weight 1.0 (== legacy).
    blind_w = float(tune.get("blind_b_weight", 1.0))
    if target == "both":
        eval_targets = ["holdout", "blind_b"]
        combo_w = {"holdout": 1.0, "blind_b": blind_w}
        study_tag = f"both_w{blind_w:g}"
    else:
        eval_targets = [target]
        combo_w = {target: 1.0}
        study_tag = target
    den = sum(combo_w.values())

    # blind_b candidates are asserted for every CG (downstream needs them too),
    # plus every tuning eval split.
    assert_cgs_have(cfg, sorted(set(["blind_b"] + eval_targets)))

    # Per eval split: load + test-tracks filter the rank pool, then semi-join to
    # the GT's (session, turn) groups so only scored groups remain.
    pools: dict[str, "object"] = {}
    gts: dict[str, "object"] = {}
    for t in eval_targets:
        p = filter_candidates(load_rank_pool(cfg, t), test_tracks_spec(cfg))
        g = load_gt(t)
        keys = g.select("session_id", "turn_number").unique()
        pools[t] = p.join(keys, on=["session_id", "turn_number"], how="semi")
        gts[t] = g
    pool_desc = " ".join(f"{t}(rows={pools[t].height:,},gt={gts[t].height:,})"
                         for t in eval_targets)
    print(f"[blind_b/tune] target={target} metric={metric} "
          f"{'blind_b_weight=' + format(blind_w, 'g') + ' ' if target == 'both' else ''}"
          f"cgs={len(cfg['cgs'])} buckets={len(buckets)} {pool_desc}")

    wcfg = cfg.get("weights", {"low": 1e-3, "high": 1.0, "log": True})
    kcfg = cfg.get("rrf_k", {"low": 1, "high": 200})

    history: list[float] = []
    metric_tag = metric.replace("@", "")
    png = PLOTS_DIR / f"blind_b_{cfg['name']}_rrf_{study_tag}_{metric_tag}.png"

    def _plot(per_target: dict[str, dict[int, float]]) -> None:
        fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 4))
        axl.plot(range(len(history)), history, "-o", color="green", ms=3)
        axl.set(title=f"objective {metric} (target={target}) per trial",
                xlabel="trial", ylabel=metric)
        axl.grid(alpha=0.3)
        for t, style in zip(eval_targets, ("-o", "-s", "-^")):
            axr.plot(PLOT_KS, [per_target[t][k] for k in PLOT_KS], style, label=t)
        axr.set(title=f"trial {len(history) - 1} recall@K", xlabel="K",
                ylabel="recall", xscale="log")
        axr.set_xticks(PLOT_KS)
        axr.set_xticklabels([str(k) for k in PLOT_KS])
        axr.grid(alpha=0.3)
        axr.legend()
        fig.suptitle(f"{cfg['name']} · RRF · target={target}")
        fig.tight_layout()
        fig.savefig(png, dpi=90)
        plt.close(fig)

    def objective(trial: optuna.Trial) -> float:
        weights = {
            cg: [trial.suggest_float(f"w_{cg}_{bn}", wcfg["low"], wcfg["high"],
                                     log=bool(wcfg.get("log", True)))
                 for bn in bucket_names]
            for cg in cfg["cgs"]
        }
        k = trial.suggest_int("rrf_k", int(kcfg["low"]), int(kcfg["high"]))
        per_target = {t: recall_at(fuse_rrf(pools[t], weights, k, top_k=top_k),
                                   gts[t], ks) for t in eval_targets}
        # Objective = normalized weighted average of recall@mk over eval splits
        # (single target ⇒ just that split's recall@mk).
        obj = sum(combo_w[t] * per_target[t][mk] for t in eval_targets) / den
        history.append(obj)
        trial.set_user_attr("recalls", {
            t: {str(kk): per_target[t][kk] for kk in ks} for t in eval_targets})
        _plot(per_target)
        detail = " ".join(f"{t}={per_target[t][mk]:.4f}" for t in eval_targets)
        print(f"[trial {trial.number}] obj={obj:.4f} ({detail}) rrf_k={k}")
        return obj

    db_dir = OPTUNA_DIR / "blind_b" / cfg["name"]
    db_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{db_dir / f'rrf_{study_tag}_{metric_tag}.db'}"
    study = optuna.create_study(
        study_name=f"rrf_{study_tag}_{metric_tag}", storage=storage,
        load_if_exists=True, direction="maximize",
        sampler=_sampler(cfg.get("optuna", {}).get("sampler")),
    )
    n_trials = int(args.n_trials if args.n_trials is not None
                   else cfg.get("optuna", {}).get("n_trials", 100))
    print(f"[blind_b/tune] storage={storage} n_trials={n_trials}")
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    print(f"\n[blind_b/tune] best {metric}={study.best_value:.4f} "
          f"trial #{study.best_trial.number}\n[blind_b/tune] plot → {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
