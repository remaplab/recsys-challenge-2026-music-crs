"""Stage-2 RRF fusion for the one-shot CG ensemble.

TUNES on the turn-`turn_filter` slice of each component's cg_val lists (weighted
RRF, one tuned weight per component + a shared k_rrf) to maximise the fused
metric CVaR on the BlindLikeEvaluator subsets used in stage 1.

EXPORTS (`--export`) the fused candidates for ALL turns (per (session, turn)),
like a normal CG, into rrf_oneshot/datasets/ (fold_*_oof_cg_val,
fold_*_oof_reranker_val, holdout_candidates, blind_candidates).

Weighted RRF:  score(item) = Σ_c  w_c / (k_rrf + rank_c(item))

Prereq: tune each component (tune_crossvalidation_oneshot.py) then export it
(export_oneshot_candidates.py). Components missing their datasets are skipped.

Run:
    cd src/basic_candidate_generators
    uv run python -m launchers_dro_oneshot.tune_rrf_oneshot                # tune
    uv run python -m launchers_dro_oneshot.tune_rrf_oneshot --export       # + fuse holdout/blind
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _PKG_ROOT / "src"
_CV_ROOT  = _PKG_ROOT / "launchers_crossvalidation"
_DRO_ROOT = _PKG_ROOT / "launchers_dro"
_OS_ROOT  = _PKG_ROOT / "launchers_dro_oneshot"
_REPO_ROOT = _PKG_ROOT.parent.parent
_LBO_SRC = _REPO_ROOT / "src" / "lower_bound_optimization" / "src"
for p in (_SRC_ROOT, _CV_ROOT, _DRO_ROOT, _OS_ROOT, str(_LBO_SRC)):
    sys.path.insert(0, str(p))

import numpy as np     # noqa: E402
import optuna          # noqa: E402
import polars as pl    # noqa: E402
import yaml            # noqa: E402

from _cv_utils import make_storage, pkg_path, plot_study, repo_path  # noqa: E402
from _dro_objective import robust_score  # noqa: E402
from tune_crossvalidation_oneshot import _build_fold_evaluators  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Fusion index + weighted RRF
# ---------------------------------------------------------------------------

_Key = tuple[str, int]   # (session_id, turn)


class FusionIndex:
    """Flattened per-component (key_row, track_gidx, rank) for fast scatter.

    Keyed by (session_id, turn) so the fusion is per ranking group — the same
    granularity a normal CG exports. (Turn-1-only inputs simply have all keys at
    turn 1, which reproduces the original behaviour.)
    """

    def __init__(self, comp_lists: list[dict[_Key, list[str]]],
                 user_by_key: dict[_Key, str] | None = None):
        keys: list[_Key] = []
        seen: set[_Key] = set()
        for d in comp_lists:
            for k in d:
                if k not in seen:
                    seen.add(k); keys.append(k)
        self.keys = keys
        self.user_ids = ([user_by_key.get(k, "") for k in keys]
                         if user_by_key else None)
        self.key_to_row = {k: i for i, k in enumerate(keys)}

        vocab: dict[str, int] = {}
        self.flat: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for d in comp_lists:
            rows_l, cols_l, rank_l = [], [], []
            for k, tracks in d.items():
                r = self.key_to_row[k]
                for pos, t in enumerate(tracks):
                    g = vocab.get(t)
                    if g is None:
                        g = len(vocab); vocab[t] = g
                    rows_l.append(r); cols_l.append(g); rank_l.append(pos)
            self.flat.append((
                np.asarray(rows_l, dtype=np.int64),
                np.asarray(cols_l, dtype=np.int64),
                np.asarray(rank_l, dtype=np.float64)))
        self.vocab = np.empty(len(vocab), dtype=object)
        for t, g in vocab.items():
            self.vocab[g] = t
        self.n_tracks = len(vocab)

    def fuse(self, weights: list[float], k_rrf: float, top_k: int) -> pl.DataFrame:
        n = len(self.keys)
        M = np.zeros((n, self.n_tracks), dtype=np.float64)
        for (rows, cols, rank), w in zip(self.flat, weights):
            if w == 0.0 or rows.size == 0:
                continue
            np.add.at(M, (rows, cols), w / (k_rrf + rank))

        out_tracks: list[list[str]] = []
        out_scores: list[list[float]] = []
        for i in range(n):
            scores = M[i]
            nz = int((scores > 0).sum())
            if nz == 0:
                out_tracks.append([]); out_scores.append([]); continue
            k = min(top_k, nz)
            idx = np.argpartition(-scores, k - 1)[:k]
            idx = idx[np.argsort(-scores[idx])]
            out_tracks.append([self.vocab[j] for j in idx])
            out_scores.append([float(scores[j]) for j in idx])

        data = {"session_id": [k[0] for k in self.keys],
                "turn": [k[1] for k in self.keys],
                "track_ids": out_tracks, "scores": out_scores}
        if self.user_ids is not None:
            data["user_id"] = self.user_ids
        return pl.DataFrame(data)


def _load_comp(path: Path) -> tuple[dict[_Key, list[str]], dict[_Key, str]]:
    """Load a component parquet → {(session_id, turn): tracks}, {(…): user_id}."""
    df = pl.read_parquet(path)
    keys = list(zip(df["session_id"].to_list(),
                    [int(t) for t in df["turn"].to_list()]))
    lists = dict(zip(keys, df["track_ids"].to_list()))
    users = (dict(zip(keys, df["user_id"].to_list()))
             if "user_id" in df.columns else {})
    return lists, users


def _components_with_data(cfg: dict, storage_dir: Path, fname: str) -> list[str]:
    out = []
    for m in cfg["rrf"]["components"]:
        if (storage_dir / f"{m}_oneshot" / "datasets" / fname).exists():
            out.append(m)
        else:
            print(f"[rrf] skip {m}: missing datasets/{fname}")
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-2 RRF fusion tuning for one-shot CGs.")
    p.add_argument("--config", default="launchers_dro_oneshot/configs/tune_oneshot.yaml")
    p.add_argument("--storage_dir", default="models/CG_crossvalidation")
    p.add_argument("--n_trials", type=int, default=None)
    p.add_argument("--export", action="store_true",
                   help="After tuning, fuse holdout + blind with best weights → submission.")
    p.add_argument("--submit_k", type=int, default=20, help="tracks per blind submission record")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(pkg_path(args.config)) as f:
        cfg = yaml.safe_load(f)
    rrf_cfg = cfg["rrf"]
    eval_cfg = cfg["evaluation"]
    data_cfg = cfg["data"]
    turn = int(data_cfg.get("turn_filter", 1))
    metric = eval_cfg.get("metric", "recall@200")
    strategy = eval_cfg.get("strategy", "calibrated")
    alpha = float(cfg.get("robust", {}).get("alpha", 0.7))
    fuse_top_k = int(rrf_cfg.get("fuse_top_k", 200))
    n_trials = args.n_trials or int(rrf_cfg.get("n_trials", 400))
    storage_dir = repo_path(args.storage_dir)
    n_folds = int(data_cfg.get("n_folds", 5))

    components = _components_with_data(cfg, storage_dir, f"fold_0_oof_cg_val.parquet")
    if len(components) < 2:
        sys.exit(f"[rrf] need >=2 exported components, have {components}")
    print(f"[rrf] fusing {len(components)} components: {components}")

    # per-fold fusion indices over the OOF cg_val lists. Tuning stays on the
    # turn-`turn` slice (the components are tuned/evaluated on that turn); the
    # exporter fuses ALL turns. Filtering to (·, turn) keeps the tuning objective
    # identical even though the component parquets now carry every turn.
    print(f"[rrf] building per-fold fusion indices (tuning turn={turn})…")
    fold_index: list[FusionIndex] = []
    for fold in range(n_folds):
        comp_lists = []
        for m in components:
            lists, _ = _load_comp(storage_dir / f"{m}_oneshot" / "datasets"
                                  / f"fold_{fold}_oof_cg_val.parquet")
            comp_lists.append({k: v for k, v in lists.items() if k[1] == turn})
        fold_index.append(FusionIndex(comp_lists))

    evaluators = _build_fold_evaluators(cfg, turn)

    # RRF score is invariant to a global weight scale (only ratios matter), so
    # one component is pinned to weight 1.0 as the anchor; the rest are tuned
    # relative to it. Removes the scale degeneracy that wastes TPE budget.
    w_cfg, k_cfg = rrf_cfg["weight"], rrf_cfg["k_rrf"]
    w_log, k_log = bool(w_cfg.get("log", False)), bool(k_cfg.get("log", False))
    if w_log and w_cfg["low"] <= 0:
        sys.exit("[rrf] weight.log=true needs weight.low > 0 (log can't span 0)")
    anchor = rrf_cfg.get("anchor_component") or components[0]
    if anchor not in components:
        sys.exit(f"[rrf] anchor_component {anchor!r} not in components {components}")
    print(f"[rrf] anchor (weight≡1.0): {anchor}  | weight_log={w_log} k_log={k_log}")

    out_dir = storage_dir / "rrf_oneshot"
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "optuna_rrf_oneshot.db"
    # Namespace the study by the search-space variant: pinning the anchor and
    # switching k_rrf to log are incompatible distribution changes — resuming the
    # old study would raise an Optuna distribution-mismatch. A space tag gives the
    # new space a fresh study while the old one stays intact.
    space_tag = f"{'klog' if k_log else 'klin'}{'_wlog' if w_log else ''}_anc-{anchor}"
    study_name = f"rrf_oneshot_cvar{int(round(alpha*100))}_{space_tag}"

    def objective(trial: optuna.Trial) -> float:
        weights = [
            1.0 if m == anchor
            else trial.suggest_float(f"w_{m}", w_cfg["low"], w_cfg["high"], log=w_log)
            for m in components
        ]
        k_rrf = trial.suggest_int("k_rrf", k_cfg["low"], k_cfg["high"], log=k_log)
        fold_scalars = []
        means = []
        for fold in range(n_folds):
            t0 = time.time()
            recs = fold_index[fold].fuse(weights, float(k_rrf), fuse_top_k)
            r = evaluators[fold].score(recs, metric=metric, strategy=strategy)
            sc = robust_score(r.per_subset, mode="cvar", alpha=alpha)
            fold_scalars.append(sc)
            means.append(float(np.nanmean(r.per_subset)))
            print(f"  [trial {trial.number}] fold {fold}: cvar={sc:.4f} "
                  f"mean={float(np.nanmean(r.per_subset)):.4f} ({time.time()-t0:.1f}s)")
            trial.report(sum(fold_scalars)/len(fold_scalars), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        print(f"\033[32m[trial {trial.number}] FINAL CVAR={float(np.mean(fold_scalars)):.4f}, MEAN={float(np.mean(means)):.4f}\033[0m\n")
        return float(np.mean(fold_scalars))

    study = optuna.create_study(
        study_name=study_name, storage=make_storage(db_path), direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
        load_if_exists=True)
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - done)
    print(f"REMAINING TRIALS: {remaining}")
    if remaining:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True,
                       show_progress_bar=False, n_jobs=7)

    print(f"\n[rrf] best cvar: {study.best_value:.6f} (trial #{study.best_trial.number})")
    best = study.best_params
    for k, v in best.items():
        print(f"  {k:22s} {v}")
    plot_study(study, out_dir / "plots")

    best_yaml = out_dir / f"best_params_rrf_oneshot_cvar{int(round(alpha*100))}.yaml"
    with open(best_yaml, "w") as f:
        yaml.safe_dump({"components": components, "anchor_component": anchor,
                        "weights": best, "fuse_top_k": fuse_top_k,
                        "best_cvar": float(study.best_value)}, f)
    print(f"[rrf] wrote {best_yaml}")

    if args.export:
        _export(cfg, components, best, anchor, fuse_top_k, storage_dir, out_dir, args.submit_k)


def _weight_of(best: dict, m: str, anchor: str) -> float:
    """Anchor component is pinned to 1.0; the rest carry their tuned ``w_<m>``."""
    return 1.0 if m == anchor else float(best[f"w_{m}"])


def _export(cfg, components, best, anchor, fuse_top_k, storage_dir, out_dir, submit_k) -> None:
    """Fuse every split with the best weights → reranker-ready parquets.

    Writes the canonical CG dataset layout under ``rrf_oneshot/datasets/`` so the
    reranker feature-builder picks the fused CG up by name (``rrf_oneshot``):

        fold_{k}_oof_cg_val.parquet       (gt from splitK fold_k_cg_val)
        fold_{k}_oof_reranker_val.parquet (gt from splitK fold_k_reranker_val)
        holdout_candidates.parquet        (gt from splitK holdout_test)
        blind_candidates.parquet          (gt null)

    Schema = session_id, user_id, turn, track_ids, scores, gt_track_id. Fuses
    ALL turns (per (session, turn)); gt is joined per (session, turn).
    """
    k_rrf = float(best["k_rrf"])
    n_folds = int(cfg["data"].get("n_folds", 5))
    splitk_dir = repo_path(cfg["data"]["splitk_dir"])
    ds_dir = out_dir / "datasets"
    ds_dir.mkdir(parents=True, exist_ok=True)

    def _gt_map(gt_parquet: Path) -> pl.DataFrame:
        """(session_id, turn_number, gt_track_id) from a splitK parquet (all turns)."""
        return (pl.read_parquet(gt_parquet, columns=["session_id", "turn_number", "track_id"])
                .select("session_id", "turn_number", pl.col("track_id").alias("gt_track_id")))

    def _fuse_split(in_fname: str, out_fname: str, gt_parquet: Path | None) -> pl.DataFrame | None:
        present = [m for m in components
                   if (storage_dir / f"{m}_oneshot" / "datasets" / in_fname).exists()]
        if not present:
            print(f"[rrf] export {out_fname}: no component candidates — skip")
            return None
        ws = [_weight_of(best, m, anchor) for m in present]
        comp_lists, users = [], {}
        for m in present:
            lists, u = _load_comp(storage_dir / f"{m}_oneshot" / "datasets" / in_fname)
            comp_lists.append(lists); users.update(u)
        fused = FusionIndex(comp_lists, user_by_key=users).fuse(ws, k_rrf, fuse_top_k)
        if gt_parquet is not None and gt_parquet.exists():
            fused = fused.join(
                _gt_map(gt_parquet),
                left_on=["session_id", "turn"], right_on=["session_id", "turn_number"],
                how="left",
            )
        else:
            fused = fused.with_columns(pl.lit(None, dtype=pl.Utf8).alias("gt_track_id"))
        cols = ["session_id", "user_id", "turn", "track_ids", "scores", "gt_track_id"]
        fused = fused.select([c for c in cols if c in fused.columns])
        out_path = ds_dir / out_fname
        fused.write_parquet(out_path)
        print(f"[rrf] wrote {out_path} ({fused.height} rows)")
        return fused

    # per-fold OOF (reranker train = cg_val, reranker val = reranker_val)
    for fold in range(n_folds):
        _fuse_split(f"fold_{fold}_oof_cg_val.parquet",
                    f"fold_{fold}_oof_cg_val.parquet",
                    splitk_dir / f"fold_{fold}_cg_val.parquet")
        _fuse_split(f"fold_{fold}_oof_reranker_val.parquet",
                    f"fold_{fold}_oof_reranker_val.parquet",
                    splitk_dir / f"fold_{fold}_reranker_val.parquet")

    _fuse_split("holdout_candidates.parquet", "holdout_candidates.parquet",
                splitk_dir / "holdout_test.parquet")
    blind = _fuse_split("blind_candidates.parquet", "blind_candidates.parquet", None)

    # Blind-B all turns. No-op when components have no blind_b_candidates yet —
    # run export_oneshot_candidates.py --blind_b_only for each component first.
    # GT (shared per (session, turn) across components) is backfilled from the
    # first present component so the fused CG is internally validatable too.
    bb = _fuse_split("blind_b_candidates.parquet", "blind_b_candidates.parquet", None)
    if bb is not None:
        src = next((storage_dir / f"{m}_oneshot" / "datasets" / "blind_b_candidates.parquet"
                    for m in components
                    if (storage_dir / f"{m}_oneshot" / "datasets" / "blind_b_candidates.parquet").exists()),
                   None)
        if src is not None:
            gt = (pl.read_parquet(src, columns=["session_id", "turn", "gt_track_id"])
                  .unique(subset=["session_id", "turn"]))
            bb = bb.drop("gt_track_id").join(gt, on=["session_id", "turn"], how="left")
            bb.write_parquet(ds_dir / "blind_b_candidates.parquet")
            print(f"[rrf] blind-B GT backfilled "
                  f"({bb.filter(pl.col('gt_track_id').is_not_null()).height} turns)")

    # blind submission JSON (top-submit_k) — at each session's real target turn
    if blind is not None:
        recs = [{"session_id": r["session_id"], "user_id": r.get("user_id", ""),
                 "turn_number": int(r["turn"]), "predicted_response": "",
                 "predicted_track_ids": r["track_ids"][:submit_k]}
                for r in blind.to_dicts()]
        sub = out_dir / "blind_A_rrf_oneshot.json"
        sub.write_text(json.dumps(recs, indent=2))
        print(f"[rrf] wrote {sub} ({len(recs)} records, top-{submit_k})")


if __name__ == "__main__":
    main()
