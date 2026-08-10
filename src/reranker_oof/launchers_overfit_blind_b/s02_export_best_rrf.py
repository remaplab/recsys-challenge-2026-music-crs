"""Export the best RRF fusion + report Blind-B recall.

Loads the Optuna study (target/metric from ``dataset.yaml``), reconstructs the
best per-(cg, bucket) weight matrix + RRF ``k``, and writes the fusion config to
``tune.rrf_out``. Then evaluates that fusion on Blind-B and prints recall@
{1,5,10,20,50,100,200} over (a) all 280 GT turns and (b) the last GT turn of
each session. Tables are saved as csv next to a comparison plot.

Usage:
    cd src/reranker_oof
    uv run python -m launchers_overfit_blind_b.s02_export_best_rrf \\
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
import polars as pl                                                    # noqa: E402
import yaml                                                            # noqa: E402

from src.paths import OPTUNA_DIR, PLOTS_DIR, REPORTS_DIR, ensure_output_dirs  # noqa: E402

from launchers_overfit_blind_b._common import (                        # noqa: E402
    PLOT_KS, assert_cgs_have, filter_candidates, fuse_rrf, last_turn_gt,
    load_config, load_gt, load_rank_pool, recall_at, register_cg_paths,
    test_tracks_spec, turn_buckets,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--target", default=None)
    ap.add_argument("--metric", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ensure_output_dirs()
    cfg = load_config(args.config)
    register_cg_paths(cfg)

    tune = cfg.get("tune", {})
    target = args.target or tune.get("target", "blind_b")
    metric = args.metric or tune.get("metric", "recall@200")
    metric_tag = metric.replace("@", "")
    # Mirror s01's study-tag naming: target=both encodes blind_b_weight so studies
    # with different weights don't collide.
    blind_w = float(tune.get("blind_b_weight", 1.0))
    study_tag = f"both_w{blind_w:g}" if target == "both" else target
    buckets = turn_buckets(cfg)
    bucket_names = [b[0] for b in buckets]
    top_k = int(cfg.get("top_k", 200))

    # ── reconstruct best trial ───────────────────────────────────────────────
    storage = f"sqlite:///{OPTUNA_DIR / 'blind_b' / cfg['name'] / f'rrf_{study_tag}_{metric_tag}.db'}"
    study = optuna.load_study(study_name=f"rrf_{study_tag}_{metric_tag}", storage=storage)
    bt = study.best_trial
    weights = {
        cg: [float(bt.params[f"w_{cg}_{bn}"]) for bn in bucket_names]
        for cg in cfg["cgs"]
    }
    rrf_k = int(bt.params["rrf_k"])
    print(f"[blind_b/export] best trial #{bt.number} {metric}={bt.value:.4f} rrf_k={rrf_k}")

    rrf_cfg = {
        "method": "rrf",
        "top_k": top_k,
        "score_input": "minmax",
        "method_params": {"k": rrf_k},
        "turn_buckets": [{"name": n, "turns": t} for n, t in buckets],
        "weights": weights,
    }
    out = args.out or cfg.get("tune", {}).get("rrf_out", "configs/blind_v1/rrf_best.yaml")
    out = Path(out)
    if not out.is_absolute():
        out = _PKG_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.safe_dump(rrf_cfg, f, sort_keys=False)
    print(f"[blind_b/export] wrote RRF config → {out}")

    # ── re-evaluate on Blind-B ───────────────────────────────────────────────
    assert_cgs_have(cfg, ["blind_b"])
    pool = filter_candidates(load_rank_pool(cfg, "blind_b"), test_tracks_spec(cfg))
    recs = fuse_rrf(pool, weights, rrf_k, top_k=top_k)
    gt_all = load_gt("blind_b")
    gt_last = last_turn_gt(gt_all)
    r_all = recall_at(recs, gt_all, PLOT_KS)
    r_last = recall_at(recs, gt_last, PLOT_KS)

    table = pl.DataFrame({
        "k": PLOT_KS,
        "recall_all_280": [r_all[k] for k in PLOT_KS],
        "recall_last_turn": [r_last[k] for k in PLOT_KS],
    })
    print(f"\n[blind_b/export] Blind-B recall (all={gt_all.height} turns, "
          f"last_turn={gt_last.height} sessions):")
    print(table)
    csv = REPORTS_DIR / f"blind_b_{cfg['name']}_rrf_{study_tag}_{metric_tag}_recall.csv"
    table.write_csv(csv)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(PLOT_KS, [r_all[k] for k in PLOT_KS], "-o", label="all 280 turns")
    ax.plot(PLOT_KS, [r_last[k] for k in PLOT_KS], "-s", label="last GT turn")
    ax.set(xscale="log", xlabel="K", ylabel="recall",
           title=f"{cfg['name']} · best RRF · Blind-B")
    ax.set_xticks(PLOT_KS)
    ax.set_xticklabels([str(k) for k in PLOT_KS])
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    png = PLOTS_DIR / f"blind_b_{cfg['name']}_rrf_{study_tag}_{metric_tag}_export.png"
    fig.savefig(png, dpi=100)
    plt.close(fig)
    print(f"[blind_b/export] csv → {csv}\n[blind_b/export] plot → {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
