"""Tune the XGB reranker on the assembled Blind-B dataset.

Per trial: fit on the train pool (early-stop on ``val_target``) → score all
three eval sets (holdout, blind_b all, blind_b last) → ndcg@{1,5,10,20,50,100,
200}. Objective = ndcg@K(``metric``) on ``val_target``. After every trial a
2-panel figure refreshes: (left) objective per trial in green; (right) the three
eval sets' ndcg@K curves for the current trial.

Usage:
    cd src/reranker_oof
    uv run python -m launchers_overfit_blind_b.s05_tune_xgb \\
        --config configs/blind_v1/xgb_v1.yaml
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG_ROOT))
sys.path.insert(0, str(_PKG_ROOT / "src"))

import matplotlib                                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
import optuna                                                          # noqa: E402
import polars as pl                                                    # noqa: E402
import yaml                                                            # noqa: E402

from src.paths import (                                                # noqa: E402
    OPTUNA_DIR, PLOTS_DIR, active_subsamples_dir, ensure_output_dirs,
    set_active_dataset,
)
from src.rerankers.base import DatasetSpec                             # noqa: E402
from src.rerankers.xgb_ranker import XGBReranker                       # noqa: E402

from launchers_overfit_blind_b._common import PLOT_KS                  # noqa: E402
from launchers_overfit_blind_b._rerank import (                        # noqa: E402
    blind_chunks, build_infer_dmatrix, eval_gt, eval_keys, eval_scored,
    holdout_chunks, reshard, resolve_feats, subsample_train, train_pool_paths,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _sampler(name: str):
    return {"tpe": optuna.samplers.TPESampler, "random": optuna.samplers.RandomSampler,
            "cma": optuna.samplers.CmaEsSampler}[(name or "tpe").lower()](seed=42)


def _sample_params(trial, space: dict) -> dict:
    out = {}
    for name, spec in space.items():
        t = spec["type"]
        if t == "loguniform":
            out[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=True)
        elif t == "uniform":
            out[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]))
        elif t == "int":
            out[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
        elif t == "categorical":
            out[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"unknown search-space type {t!r}")
    return out


def run(cfg: dict, n_trials: int | None = None) -> int:
    ensure_output_dirs()
    set_active_dataset(cfg["dataset_name"])
    device = cfg.get("device", "cpu")
    static = dict(cfg.get("static", {}))
    early_stop = int(static.pop("early_stopping_rounds", 50))
    metric = cfg.get("metric", "ndcg@20")
    mk = int(metric.split("@")[1])
    ks = sorted(set(PLOT_KS) | {mk})
    target = cfg.get("val_target", "blind_b_all")
    # blind_only: no splitK in the dataset (e.g. the stacking meta dataset) →
    # train on blind history only, no holdout eval set.
    blind_only = bool(cfg.get("blind_only", False))
    # train_on_blind: fold blind_b history (minus each session's last turn) into
    # TRAIN and early-stop/objective on the held-out last turn (true overfit).
    train_on_blind = bool(cfg.get("train_on_blind", False)) or blind_only
    if train_on_blind:
        target = "blind_b_last"
    eval_sets = ("blind_b_all", "blind_b_last") if blind_only \
        else ("holdout", "blind_b_all", "blind_b_last")
    gpc = cfg.get("xgb_groups_per_chunk")          # null/0 → feed native chunks
    tag = cfg.get("run_tag") or cfg["dataset_name"]
    study_name = f"xgb_{tag}_{target}" + ("_onblind" if train_on_blind else "")

    # ── prepare train + eval sets ────────────────────────────────────────────
    # gpc set → re-shard to gpc groups/file (bounds DataIter ingest memory).
    # gpc null → feed s03 chunks directly; eval still gets filtered (no split).
    sub = active_subsamples_dir() / f"xgb{gpc or 'native'}"

    def _prep(paths, name, restrict=None):
        if restrict is None and not gpc:
            return list(paths)
        return reshard(paths, sub / name, gpc or 10**9, restrict=restrict)

    train_chunks = subsample_train(_prep(train_pool_paths(target), f"train_{target}"), cfg)
    if train_on_blind:
        # Add blind_b labelled non-last turns to TRAIN (never subsampled).
        nonlast = eval_keys("blind_b_all").join(
            eval_keys("blind_b_last"), on=["session_id", "turn_number"], how="anti")
        train_chunks = train_chunks + _prep(blind_chunks(), "train_blindnonlast",
                                            restrict=nonlast)
        print(f"[blind_b/tune-xgb] train_on_blind ON → +blind_b non-last turns; "
              f"objective/early-stop = blind_b_last")
    eval_chunk = {
        w: _prep(holdout_chunks() if w == "holdout" else blind_chunks(),
                 f"eval_{w}", restrict=eval_keys(w))
        for w in eval_sets
    }
    sample = pl.read_parquet(train_chunks[0], n_rows=10)
    blind_sample = pl.read_parquet(blind_chunks()[0], n_rows=1)
    feat_cols = resolve_feats(sample, cfg.get("feat_cols_keep"), blind_sample)
    print(f"[blind_b/tune-xgb] target={target} feats={len(feat_cols)} "
          f"train_chunks={len(train_chunks)} gpc={gpc}")

    # ── build dtrain once (no val), then eval DMatrices sharing its bins ──────
    # Per-config (tag-keyed) ext-mem cache: the quantised page cache depends on the
    # feature set AND is rewritten each run, so concurrent configs must NOT share it
    # (else torn pages / wrong-feature reuse). Reshard parquets stay shared (feature-
    # independent, written once) — warm them with one serial run before going parallel.
    use_ext = bool(cfg.get("external_memory", device == "cuda"))
    cache_dir = (active_subsamples_dir() / f"xgb_cache_{tag}_{target}") if use_ext else None
    ds = DatasetSpec(train_paths=train_chunks, val_paths=[], feat_cols=feat_cols)
    bundle = XGBReranker.build_dmatrix(ds, device=device, cache_dir=cache_dir,
                                       **dict(cfg.get("prepare_kwargs", {})))
    max_bin = cfg.get("prepare_kwargs", {}).get("max_bin")
    dmats: dict[str, tuple] = {}
    for w in eval_sets:
        dmats[w] = build_infer_dmatrix(eval_chunk[w], feat_cols, device,
                                       ref=bundle.dtrain, max_bin=max_bin)
    gts = {w: eval_gt(w) for w in eval_sets}
    target_dmat, _ = dmats[target]

    history: list[float] = []
    png = PLOTS_DIR / f"blind_b_{tag}_xgb_{target}.png"

    def _plot(curves: dict[str, dict[int, dict[str, float]]], params: dict) -> None:
        fig, (axl, axr, axt) = plt.subplots(
            1, 3, figsize=(17, 4.6), gridspec_kw={"width_ratios": [3, 3, 2]})
        axl.plot(range(len(history)), history, "-o", color="green", ms=3)
        axl.set(title=f"objective {metric} ({target}) per trial", xlabel="trial",
                ylabel=metric); axl.grid(alpha=0.3)
        for w, mk_ in (("holdout", "-o"), ("blind_b_all", "-s"), ("blind_b_last", "-^")):
            if w not in curves:
                continue
            axr.plot(PLOT_KS, [curves[w][k]["ndcg"] for k in PLOT_KS], mk_, label=w)
        axr.set(title=f"trial {len(history) - 1} ndcg@K", xlabel="K", ylabel="ndcg",
                xscale="log"); axr.set_xticks(PLOT_KS)
        axr.set_xticklabels([str(k) for k in PLOT_KS]); axr.grid(alpha=0.3); axr.legend()
        # Full xgb config for THIS trial (static + sampled) — EVERY param, so the
        # figure is a complete record of what produced the curves, not a subset.
        axt.axis("off")
        body = "\n".join(f"{k} = {params[k]}" for k in sorted(params))
        axt.text(0.0, 1.0, body, va="top", ha="left", family="monospace", fontsize=7,
                 transform=axt.transAxes)
        axt.set_title("full xgb config (this trial)", fontsize=9)
        fig.suptitle(f"{cfg['dataset_name']} · XGB · target={target}")
        fig.tight_layout(); fig.savefig(png, dpi=90); plt.close(fig)

    def objective(trial: optuna.Trial) -> float:
        params = {**static, **_sample_params(trial, cfg.get("search_space", {}))}
        model = XGBReranker()

        def prune_cb(step: int, value: float) -> bool:
            trial.report(value, step=step); return trial.should_prune()

        try:
            model.fit(dtrain=bundle.dtrain, dval=target_dmat, feat_cols=feat_cols,
                      params=params, device=device, early_stopping_rounds=early_stop,
                      pruning_callback=prune_cb)
        except optuna.TrialPruned:
            model.release(); raise

        curves: dict[str, dict[int, dict[str, float]]] = {}
        for w in eval_sets:
            dmat, meta = dmats[w]
            scored = model.predict_dval(dmat, meta)
            curves[w] = eval_scored(scored, gts[w], ks)
        obj = curves[target][mk]["ndcg"]          # macro-by-turn ndcg on target
        history.append(obj)
        _plot(curves, params)
        print(f"\033[32m[trial {trial.number}] macro {metric}({target})={obj:.4f}\033[0m")
        for w in ("blind_b_all", "blind_b_last"):
            c = curves[w]
            nd = " ".join(f"{k}:{c[k]['ndcg']:.3f}" for k in PLOT_KS)
            rc = " ".join(f"{k}:{c[k]['recall']:.3f}" for k in PLOT_KS)
            print(f"   {w:12s} ndcg[ {nd} ]")
            print(f"   {w:12s} rec [ {rc} ]")
        model.release(); gc.collect()
        return obj

    db_dir = OPTUNA_DIR / "blind_b" / tag
    db_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{db_dir / f'{study_name}.db'}"
    study = optuna.create_study(
        study_name=study_name, storage=storage, load_if_exists=True,
        direction="maximize", sampler=_sampler(cfg.get("optuna", {}).get("sampler")),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    n_trials = int(n_trials if n_trials is not None
                   else cfg.get("optuna", {}).get("n_trials", 100))
    print(f"[blind_b/tune-xgb] storage={storage} n_trials={n_trials}")
    study.optimize(objective, n_trials=n_trials,
                   timeout=cfg.get("optuna", {}).get("timeout_sec"), gc_after_trial=True)
    print(f"\n[blind_b/tune-xgb] best {metric}({target})={study.best_value:.4f} "
          f"trial #{study.best_trial.number}\n[blind_b/tune-xgb] plot → {png}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--n_trials", type=int, default=None)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    return run(cfg, n_trials=args.n_trials)


if __name__ == "__main__":
    raise SystemExit(main())
