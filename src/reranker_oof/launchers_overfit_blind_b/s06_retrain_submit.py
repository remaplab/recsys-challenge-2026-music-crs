"""Retrain the best XGB config twice and emit Blind-B submissions.

Variant 1 — train = OOF folds + holdout; val/early-stop = blind_b all 280 turns.
Variant 2 — train = OOF folds + holdout + blind_b labelled turns EXCEPT each
            session's last GT turn; val/early-stop = blind_b last GT turn.

Both use the best hyperparameters from the s05 study (val_target). For each:
write the competition submission JSON (top-20 for the 80 withheld turns), compute
ndcg + recall @ {1,5,10,20,50,100,200} over blind_b-all and blind_b-last (csv),
and save SHAP feature attribution + ndcg/recall curve plots.

Usage:
    cd src/reranker_oof
    uv run python -m launchers_overfit_blind_b.s06_retrain_submit \\
        --config configs/blind_v1/xgb_v1.yaml
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG_ROOT))
sys.path.insert(0, str(_PKG_ROOT / "src"))

import matplotlib                                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
import numpy as np                                                     # noqa: E402
import optuna                                                          # noqa: E402
import polars as pl                                                    # noqa: E402
import yaml                                                            # noqa: E402

from src.paths import (                                                # noqa: E402
    BLIND_A_RAW, OPTUNA_DIR, REPO_ROOT,
    active_subsamples_dir, ensure_output_dirs, set_active_dataset,
)
from src.features.cg_calibration import (                             # noqa: E402
    _conformal_quantile, _fit_one_calibrator,
)
from src.rerankers.base import DatasetSpec                             # noqa: E402
from src.rerankers.xgb_ranker import XGBReranker                       # noqa: E402

from launchers_overfit_blind_b._common import PLOT_KS                  # noqa: E402
from launchers_overfit_blind_b._rerank import (                        # noqa: E402
    blind_a_chunks, blind_chunks, build_infer_dmatrix, eval_gt, eval_keys,
    eval_scored, reshard, resolve_feats, submission_records, subsample_train,
    train_pool_paths,
)


def _load_study(cfg: dict) -> optuna.Study:
    target = cfg.get("val_target", "blind_b_all")
    train_on_blind = bool(cfg.get("train_on_blind", False))
    if train_on_blind:
        target = "blind_b_last"
    tag = cfg.get("run_tag") or cfg["dataset_name"]
    study_name = f"xgb_{tag}_{target}" + ("_onblind" if train_on_blind else "")
    storage = f"sqlite:///{OPTUNA_DIR / 'blind_b' / tag / f'{study_name}.db'}"
    return optuna.load_study(study_name=study_name, storage=storage)


def _topk_params(cfg: dict, k: int) -> list[dict]:
    """The param dicts of the top-``k`` COMPLETED trials by objective value,
    deduped by exact param set (TPE clusters near-identical configs, which would
    waste a bagging slot on a duplicate). k=1 ⇒ just the single best (legacy)."""
    study = _load_study(cfg)
    done = [t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    done.sort(key=lambda t: t.value, reverse=True)        # study direction = maximize
    seen: set = set()
    out: list[dict] = []
    for t in done:
        key = tuple(sorted(t.params.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append({"_trial": t.number, "_value": t.value, **t.params})
        if len(out) >= k:
            break
    head = ", ".join(f"#{m['_trial']}={m['_value']:.4f}" for m in out)
    print(f"[blind_b/retrain] bagging {len(out)} member(s) (top-{k}, deduped): {head}")
    return out


def _save_shap(model: XGBReranker, dmat, feat_cols: list[str], out_dir: Path,
               top: int = 20) -> None:
    import xgboost as xgb  # noqa: F401 (booster already holds device)
    contribs = model._booster.predict(dmat, pred_contribs=True)
    mean_abs = np.abs(contribs[:, :len(feat_cols)]).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    ranked = [(feat_cols[i], float(mean_abs[i])) for i in order]
    # Ordered list (most → least important), one feature per line.
    (out_dir / "shap.yaml").write_text(
        yaml.safe_dump([{f: v} for f, v in ranked], sort_keys=False, default_flow_style=False))
    sel = ranked[:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(sel) + 1))
    ax.barh([f for f, _ in sel], [v for _, v in sel], color="tab:purple")
    ax.set(title=f"SHAP mean|contrib| top-{top}", xlabel="mean |SHAP|")
    fig.tight_layout(); fig.savefig(out_dir / "shap.png", dpi=110); plt.close(fig)


def _save_curves(m_all: dict, m_last: dict, out_dir: Path, tag: str, fname: str) -> None:
    fig, (axn, axr) = plt.subplots(1, 2, figsize=(12, 4))
    for metric, ax in (("ndcg", axn), ("recall", axr)):
        ax.plot(PLOT_KS, [m_all[k][metric] for k in PLOT_KS], "-o", label="all turns")
        ax.plot(PLOT_KS, [m_last[k][metric] for k in PLOT_KS], "-s", label="last turn")
        ax.set(title=f"{metric}@K", xlabel="K", ylabel=metric, xscale="log")
        ax.set_xticks(PLOT_KS); ax.set_xticklabels([str(k) for k in PLOT_KS])
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle(tag)
    fig.tight_layout(); fig.savefig(out_dir / fname, dpi=100); plt.close(fig)


def _save_conformal_setsizes(scored: pl.DataFrame, out_dir: Path, kind: str,
                             alpha: float = 0.1, method: str = "isotonic",
                             suffix: str = "") -> None:
    """Conformal prediction-set sizes of the RERANKER: isotonic-calibrate its
    score → P(is_gt) on the labelled turns, set q̂ from the GT nonconformity,
    then per (session,turn) count candidates with 1−p_cal ≤ q̂. Plot the
    distribution over turns."""
    lab = scored.filter(pl.col("gt_track_id").is_not_null())
    if lab.height == 0:
        return
    y = (lab["track_id"] == lab["gt_track_id"]).cast(pl.Int8).to_numpy()
    cal = _fit_one_calibrator(np.nan_to_num(lab["score"].to_numpy()), y, method)
    gt = lab.filter(pl.col("track_id") == pl.col("gt_track_id"))
    q = _conformal_quantile(1.0 - np.asarray(cal(np.nan_to_num(gt["score"].to_numpy()))), alpha)
    pcal = np.asarray(cal(np.nan_to_num(scored["score"].to_numpy())))
    s = scored.with_columns(pl.Series("_in", ((1.0 - pcal) <= q).astype(np.int8)))
    sizes = (s.group_by("session_id", "turn_number")
             .agg(pl.col("_in").sum().alias("sz"))["sz"].to_numpy())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(sizes, bins=min(40, max(5, int(sizes.max()) + 1)), color="teal",
            edgecolor="k", alpha=0.8)
    ax.axvline(sizes.mean(), color="red", ls="--", label=f"mean={sizes.mean():.1f}")
    ax.set(title=f"{kind} reranker conformal set size (α={alpha}, q̂={q:.3f})",
           xlabel="set size", ylabel="# turns")
    ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / f"conformal_set_sizes_{kind}{suffix}.png", dpi=110); plt.close(fig)
    print(f"[blind_b/retrain] {kind} conformal set size: mean={sizes.mean():.2f} "
          f"median={np.median(sizes):.0f} max={int(sizes.max())} q_hat={q:.4f}")


def _score_chunks(model: XGBReranker, feat_cols: list[str],
                  chunks: list[Path]) -> pl.DataFrame:
    """Predict every chunk of one blind set and concat (one row per candidate,
    metadata cols — incl. gt_track_id — passed through by ``predict``)."""
    return pl.concat([model.predict(p, feat_cols) for p in chunks])


def _rank_average(frames: list[pl.DataFrame]) -> pl.DataFrame:
    """Bagging combine: per (session,turn) rank each member's candidates, average
    the ranks across members, and re-score by −avg_rank (so higher = better, as
    downstream expects). Rank-space is scale-free — raw scores from different
    trials live on different scales, so score-averaging would let one member
    dominate. Single member ⇒ return as-is (keeps raw scores for conformal)."""
    if len(frames) == 1:
        return frames[0]
    keys = ["session_id", "turn_number", "track_id"]
    merged = None
    rcols: list[str] = []
    for j, f in enumerate(frames):
        rc = f.select(*keys, pl.col("score").rank("ordinal", descending=True)
                      .over("session_id", "turn_number").alias(f"_rk{j}"))
        merged = rc if merged is None else merged.join(rc, on=keys, how="inner")
        rcols.append(f"_rk{j}")
    ens = (merged.with_columns((-pl.mean_horizontal(rcols)).alias("score"))
           .select(*keys, "score"))
    # Reattach member-0's metadata (gt_track_id, etc.); swap in the ensemble score.
    return frames[0].drop("score").join(ens, on=keys, how="left")


def _emit_scored(kind: str, scored: pl.DataFrame, blind_raw: Path, out_dir: Path,
                 ds_name: str, variant: str, n_members: int = 1,
                 conformal_alpha: float = 0.1, bag_label: str | None = None) -> None:
    """From an already-scored (possibly ensembled) frame: write the scored
    parquet, ndcg+recall metric table, submission JSON, curve + conformal plots.
    Artifacts are tagged ``bag{n_members}`` (default) so a bag-size sweep over one
    model folder doesn't clobber. Pass ``bag_label=""`` for a single fixed bag with
    clean, suffix-free names (one submission). Submissions land in
    ``out_dir/submissions/``."""
    bag = bag_label if bag_label is not None else f"bag{n_members}"
    sfx = f"_{bag}" if bag else ""
    scored.write_parquet(out_dir / f"scored_{kind}{sfx}.parquet")

    gt_all, gt_last = eval_gt(f"{kind}_all"), eval_gt(f"{kind}_last")
    m_all = eval_scored(scored, gt_all, PLOT_KS)
    m_last = eval_scored(scored, gt_last, PLOT_KS)
    table = pl.DataFrame({
        "k": PLOT_KS,
        "ndcg_all": [m_all[k]["ndcg"] for k in PLOT_KS],
        "recall_all": [m_all[k]["recall"] for k in PLOT_KS],
        "ndcg_last": [m_last[k]["ndcg"] for k in PLOT_KS],
        "recall_last": [m_last[k]["recall"] for k in PLOT_KS],
    })
    lbl = bag or "single"
    print(f"--- {kind} metrics ({variant} · {lbl}) ---")
    print(table)
    table.write_csv(out_dir / f"metrics_{kind}{sfx}.csv")

    records = submission_records(scored, blind_raw)
    sub_dir = out_dir / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)
    sub_path = sub_dir / f"{kind}_{ds_name}_{variant}{sfx}.json"
    sub_path.write_text(json.dumps(records, indent=2))
    print(f"[blind_b/retrain] {variant}/{kind}/{lbl}: {len(records)} submission turns → {sub_path}")
    _save_curves(m_all, m_last, out_dir, f"{variant} · {kind} · {lbl}", f"curves_{kind}{sfx}.png")
    _save_conformal_setsizes(scored, out_dir, kind, alpha=conformal_alpha, suffix=sfx)


def _export_candidates(kind: str, scored_members: list[pl.DataFrame],
                       ens_scored: pl.DataFrame, out_dir: Path, tag: str,
                       variant: str, top: int = 200, n_members: int = 1,
                       bag_label: str | None = None) -> None:
    """Top-K candidates per (session,turn) from the (ensembled) scores — folds in
    the old s07. When bagging, also attach each member's per-turn rank as
    ``rank_m{j}`` so disagreement between bag members is inspectable downstream.
    Reuses the members already fit in the retrain loop (no extra fit). Writes to
    ``out_dir/candidates/`` tagged ``bag{n_members}`` (default) or, with
    ``bag_label=""``, a single clean suffix-free file."""
    bag = bag_label if bag_label is not None else f"bag{n_members}"
    sfx = f"_{bag}" if bag else ""
    keys = ["session_id", "turn_number", "track_id"]
    grp = ["session_id", "turn_number"]
    base = ens_scored.with_columns(
        pl.col("score").rank("ordinal", descending=True).over(grp)
        .cast(pl.Int32).alias("rank"))
    rank_m: list[str] = []
    if len(scored_members) > 1:                      # per-member ranks only if bagged
        for j, f in enumerate(scored_members):
            rk = f.select(*keys, pl.col("score").rank("ordinal", descending=True)
                          .over(grp).cast(pl.Int32).alias(f"rank_m{j}"))
            base = base.join(rk, on=keys, how="left")
            rank_m.append(f"rank_m{j}")
    kept = base.filter(pl.col("rank") <= top).with_columns(pl.lit(kind).alias("kind"))
    m = eval_scored(ens_scored, eval_gt(f"{kind}_last"), [20, top])
    print(f"[blind_b/retrain] {kind}/{bag or 'single'} candidates: {kept.height:,} rows (top-{top}) · "
          f"last-turn ndcg@20={m[20]['ndcg']:.3f} recall@20={m[20]['recall']:.3f} "
          f"recall@{top}={m[top]['recall']:.3f}")
    cols = keys + ["score", "rank"] + rank_m + ["kind", "gt_track_id"]
    cand_dir = out_dir / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    out = cand_dir / f"cand_{tag}_{variant}_{kind}{sfx}.parquet"
    kept.select(cols).write_parquet(out)
    print(f"[blind_b/retrain] candidates → {out}")


def run(cfg: dict, variants_sel: list[str] | None = None,
        bag_top_k: int | None = None) -> int:
    ensure_output_dirs()
    set_active_dataset(cfg["dataset_name"])
    blind_only = bool(cfg.get("blind_only", False))
    device = cfg.get("device", "cpu")
    static = dict(cfg.get("static", {}))
    early_stop = int(static.pop("early_stopping_rounds", 50))
    prepare_kwargs = dict(cfg.get("prepare_kwargs", {}))
    gpc = cfg.get("xgb_groups_per_chunk")          # null/0 → feed native chunks
    blind_raw = Path(cfg["blind_raw"])
    if not blind_raw.is_absolute():
        blind_raw = REPO_ROOT / blind_raw
    # Bagging: retrain the top-K study trials and rank-average them into ONE
    # ensemble of size K (no sweep). CLI --bag-top-k overrides the config; default 10.
    k = bag_top_k if bag_top_k is not None else int(cfg.get("bag_top_k", 10))
    members = _topk_params(cfg, max(1, k))
    cand_top = int(cfg.get("cand_top", 200))       # top-K per turn in the candidate export
    tag = cfg.get("run_tag") or cfg["dataset_name"]
    sub_root = active_subsamples_dir() / f"xgb{gpc or 'native'}_retrain"

    def _prep(paths, name, restrict=None):
        # gpc set → re-shard to gpc groups/file; null → native chunks (filtered
        # subsets still get materialized once since they need the semi-join).
        if restrict is None and not gpc:
            return list(paths)
        return reshard(paths, sub_root / name, gpc or 10**9, restrict=restrict)

    # ── training pools per variant ───────────────────────────────────────────
    # v1: OOF folds + holdout; val = blind_b_all.
    # v2: + blind_b labelled non-last turns; val = blind_b_last.
    nonlast_keys = eval_keys("blind_b_all").join(eval_keys("blind_b_last"),
                                                 on=["session_id", "turn_number"], how="anti")
    # train_subsample thins the OOF/holdout base only (fast fitting); blind-
    # derived train rows (v2) are kept whole — they are the overfit signal.
    variants = {
        "v1_blind_all": {
            "train": lambda: subsample_train(_prep(train_pool_paths("blind_b_all"), "train_v1"), cfg),
            "val": "blind_b_all",
        },
        "v2_blind_last": {
            "train": lambda: (
                subsample_train(_prep(train_pool_paths("blind_b_last"), "train_v2_base"), cfg)
                + _prep(blind_chunks(), "train_v2_blindnonlast", restrict=nonlast_keys)
            ),
            "val": "blind_b_last",
        },
    }
    if blind_only:
        # No splitK in the dataset → v1 (needs folds+holdout) is impossible. Keep
        # the leave-last-out variant; its base train_pool_paths is empty, so
        # train = blind non-last only.
        variants = {"v2_blind_last": variants["v2_blind_last"]}

    # Variant selection — CLI --variants > config retrain_variants > default
    # (blind_last only; the train_on_blind regime the studies tune for).
    sel = variants_sel or cfg.get("retrain_variants") or ["v2_blind_last"]
    variants = {nm: v for nm, v in variants.items() if nm in sel}
    if not variants:
        raise SystemExit(
            f"[blind_b/retrain] no variants selected from {sel}; available: "
            f"{'v2_blind_last' if blind_only else 'v1_blind_all, v2_blind_last'}")

    for name, spec in variants.items():
        print(f"\n========== retrain {name} (val={spec['val']}) ==========")
        out_dir = REPO_ROOT / "models" / "reranker_oof" / "blind_b_retrain" / tag / name
        (out_dir / "boosters").mkdir(parents=True, exist_ok=True)

        train_chunks = spec["train"]()
        val_chunks = _prep(blind_chunks(), f"val_{name}", restrict=eval_keys(spec["val"]))
        feat_cols = resolve_feats(pl.read_parquet(train_chunks[0], n_rows=10),
                                  cfg.get("feat_cols_keep"),
                                  pl.read_parquet(blind_chunks()[0], n_rows=1))
        use_ext = bool(cfg.get("external_memory", device == "cuda"))
        cache_dir = (sub_root / f"cache_{name}") if use_ext else None

        ds = DatasetSpec(train_paths=train_chunks, val_paths=[], feat_cols=feat_cols)
        bundle = XGBReranker.build_dmatrix(ds, device=device, cache_dir=cache_dir, **prepare_kwargs)
        dval, _ = build_infer_dmatrix(val_chunks, feat_cols, device, ref=bundle.dtrain,
                                      max_bin=prepare_kwargs.get("max_bin"))

        # ── bagging: fit each member (shares the same dtrain/dval — only params
        # differ), score both blind sets, then rank-average across members. ─────
        c_alpha = float(cfg.get("conformal_alpha", 0.1))
        scored_b_members: list[pl.DataFrame] = []
        scored_a_members: list[pl.DataFrame] = []
        for j, m in enumerate(members):
            member_params = {kk: vv for kk, vv in m.items() if not kk.startswith("_")}
            full_params = {**static, **member_params}
            model = XGBReranker()
            model.fit(dtrain=bundle.dtrain, dval=dval, feat_cols=feat_cols,
                      params=full_params, device=device, early_stopping_rounds=early_stop)
            model._booster.save_model(str(out_dir / "boosters" / f"booster_{j}.json"))
            scored_b_members.append(_score_chunks(model, feat_cols, blind_chunks()))
            scored_a_members.append(_score_chunks(model, feat_cols, blind_a_chunks()))
            if j == 0:
                # SHAP of the best member as the representative attribution
                # (an ensemble has no single contrib matrix).
                _save_shap(model, dval, feat_cols, out_dir)
            model.release(); gc.collect()

        # Single fixed bag: rank-average ALL fit members (= --bag-top-k) into one
        # ensemble → one clean suffix-free submission + candidates (no bag sweep).
        n = len(members)
        ens_b = _rank_average(scored_b_members)
        ens_a = _rank_average(scored_a_members)
        _emit_scored("blind_b", ens_b, blind_raw, out_dir, tag, name,
                     n_members=n, conformal_alpha=c_alpha, bag_label="")
        _emit_scored("blind_a", ens_a, BLIND_A_RAW, out_dir, tag, name,
                     n_members=n, conformal_alpha=c_alpha, bag_label="")
        _export_candidates("blind_b", scored_b_members, ens_b, out_dir,
                           tag, name, cand_top, n, bag_label="")
        _export_candidates("blind_a", scored_a_members, ens_a, out_dir,
                           tag, name, cand_top, n, bag_label="")
        print(f"[blind_b/retrain] {name}: single bag of {n} member(s) → {out_dir}")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--variants", nargs="+", choices=["v1_blind_all", "v2_blind_last"],
                    default=None, help="retrain variant(s); default = config "
                    "retrain_variants or [v2_blind_last]")
    ap.add_argument("--bag-top-k", type=int, default=10,
                    help="fit the top-K study trials and rank-average them into a "
                    "single bag of size K → one submission + candidates. default 10.")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    return run(cfg, variants_sel=args.variants, bag_top_k=args.bag_top_k)


if __name__ == "__main__":
    raise SystemExit(main())
