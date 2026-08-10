"""Distribution plots for per-subset evaluation results."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


_PALETTE = ["steelblue", "crimson", "seagreen", "darkorange", "mediumpurple"]


# ---------------------------------------------------------------------------
# Single-strategy histogram (original)
# ---------------------------------------------------------------------------

def plot_distribution(
    scores: np.ndarray,
    summary: dict,
    out_path: Path,
    title: str = "Score distribution",
    bins: int = 60,
) -> None:
    """Histogram of per-subset scores with mean ± std and 90/95/99 CIs."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(scores, bins=bins, alpha=0.65, color="steelblue", edgecolor="black", linewidth=0.3)
    m = summary["mean"]
    s = summary["std"]
    ax.axvline(m, color="black", linewidth=2, label=f"mean = {m:.4f}")
    ax.axvline(m - s, color="black", linewidth=1, linestyle=":", alpha=0.7)
    ax.axvline(m + s, color="black", linewidth=1, linestyle=":", alpha=0.7,
               label=f"±std ({s:.4f})")
    colors = {"ci90": "tab:green", "ci95": "tab:orange", "ci99": "tab:red"}
    for ci_key, color in colors.items():
        lo, hi = summary[ci_key]
        ax.axvline(lo, color=color, linewidth=1.5, linestyle="--",
                   label=f"{ci_key.upper()[2:]}% CI [{lo:.4f}, {hi:.4f}]")
        ax.axvline(hi, color=color, linewidth=1.5, linestyle="--")
    ax.set_xlabel("score")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Multi-strategy overlay histogram (original)
# ---------------------------------------------------------------------------

def plot_distribution_overlay(
    scores_per_strategy: dict[str, np.ndarray],
    summary_per_strategy: dict[str, dict],
    out_path: Path,
    title: str = "Score distribution — strategies overlay",
    bins: int = 60,
) -> None:
    """Overlay 2+ strategy distributions on the same axes."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (name, scores) in enumerate(scores_per_strategy.items()):
        color = _PALETTE[i % len(_PALETTE)]
        m = summary_per_strategy[name]["mean"]
        ax.hist(scores, bins=bins, alpha=0.45, color=color, edgecolor="black",
                linewidth=0.2, label=f"{name}  (mean={m:.4f})")
        ax.axvline(m, color=color, linewidth=2)
    ax.set_xlabel("score")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Violin + box — strategies side-by-side
# ---------------------------------------------------------------------------

def plot_violin_by_strategy(
    scores_per_strategy: dict[str, np.ndarray],
    summary_per_strategy: dict[str, dict],
    out_path: Path,
    title: str = "Per-subset score — strategies",
    metric_label: str = "score",
) -> None:
    """Violin + embedded box plot for each strategy, side by side."""
    names = list(scores_per_strategy.keys())
    data = [scores_per_strategy[n][~np.isnan(scores_per_strategy[n])] for n in names]
    fig, ax = plt.subplots(figsize=(max(6, 3 * len(names)), 6))
    parts = ax.violinplot(data, positions=range(len(names)), showmedians=False,
                          showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(_PALETTE[i % len(_PALETTE)])
        pc.set_alpha(0.55)
    # Box overlay
    bp = ax.boxplot(data, positions=range(len(names)), widths=0.12,
                    patch_artist=True, medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2),
                    flierprops=dict(marker=".", markersize=2, alpha=0.3))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(_PALETTE[i % len(_PALETTE)])
        patch.set_alpha(0.75)
    # Mean dot
    for i, (n, d) in enumerate(zip(names, data)):
        m = summary_per_strategy[n]["mean"]
        ax.scatter(i, m, zorder=5, color="white", s=40, edgecolors="black", linewidths=1.2,
                   label=f"{n}: mean={m:.4f}  CI95=[{summary_per_strategy[n]['ci95_lo']:.4f}, "
                         f"{summary_per_strategy[n]['ci95_hi']:.4f}]")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel(metric_label)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7.5)
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# CDF overlay — strategies
# ---------------------------------------------------------------------------

def plot_cdf(
    scores_per_strategy: dict[str, np.ndarray],
    out_path: Path,
    title: str = "CDF of per-subset scores",
    metric_label: str = "score",
) -> None:
    """Empirical CDF for each strategy overlaid on the same axes."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (name, scores) in enumerate(scores_per_strategy.items()):
        s = np.sort(scores[~np.isnan(scores)])
        if s.size == 0:
            continue
        cdf = np.arange(1, len(s) + 1) / len(s)
        color = _PALETTE[i % len(_PALETTE)]
        ax.step(s, cdf, where="post", color=color, linewidth=2,
                label=name, alpha=0.85)
        # mark median
        med_idx = np.searchsorted(cdf, 0.5)
        if med_idx < len(s):
            ax.axvline(s[med_idx], color=color, linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xlabel(metric_label)
    ax.set_ylabel("cumulative probability")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# CI summary (forest-plot style) — strategies
# ---------------------------------------------------------------------------

def plot_ci_summary(
    summary_per_strategy: dict[str, dict],
    out_path: Path,
    title: str = "Mean ± CI per strategy",
    metric_label: str = "score",
) -> None:
    """Horizontal forest-plot style: mean + 95% CI bar per strategy.

    Points are sorted by descending mean for readability.
    """
    items = sorted(summary_per_strategy.items(), key=lambda kv: -kv[1]["mean"])
    names = [k for k, _ in items]
    means = np.array([v["mean"] for _, v in items])
    lo = np.array([v["ci95_lo"] for _, v in items])
    hi = np.array([v["ci95_hi"] for _, v in items])
    std = np.array([v["std"] for _, v in items])

    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 1.4)))
    for i in y:
        color = _PALETTE[i % len(_PALETTE)]
        ax.errorbar(means[i], i, xerr=[[means[i] - lo[i]], [hi[i] - means[i]]],
                    fmt="o", color=color, markersize=8, linewidth=2,
                    capsize=5, capthick=2)
        ax.errorbar(means[i], i, xerr=std[i], fmt="", color=color,
                    linewidth=1, linestyle="--", alpha=0.45,
                    label="_nolegend_")
        ax.text(hi[i] + (hi[i] - lo[i]) * 0.05, i,
                f"{means[i]:.4f}  [{lo[i]:.4f}, {hi[i]:.4f}]",
                va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel(metric_label)
    ax.set_title(title)
    ax.axvline(means.mean(), color="gray", linestyle=":", linewidth=1, alpha=0.6,
               label="grand mean")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# s08 — Δ histogram with CI bounds
# ---------------------------------------------------------------------------

def plot_delta_distribution(
    delta: np.ndarray,
    lo: float,
    hi: float,
    mean_delta: float,
    out_path: Path,
    label_a: str = "A",
    label_b: str = "B",
    title: str = "Δ distribution",
    bins: int = 60,
) -> None:
    """Histogram of per-subset Δ = A − B with paired-EB CI bounds marked."""
    fig, ax = plt.subplots(figsize=(9, 5))
    d = delta[~np.isnan(delta)]
    color = "steelblue" if mean_delta >= 0 else "crimson"
    ax.hist(d, bins=bins, alpha=0.65, color=color, edgecolor="black", linewidth=0.3)
    ax.axvline(0, color="black", linewidth=1.5, linestyle="-", alpha=0.6, label="Δ = 0")
    ax.axvline(mean_delta, color="black", linewidth=2,
               label=f"mean Δ = {mean_delta:+.4f}")
    ax.axvspan(lo, hi, alpha=0.12, color="tab:orange",
               label=f"95% paired-EB CI [{lo:+.4f}, {hi:+.4f}]")
    ax.axvline(lo, color="tab:orange", linewidth=1.5, linestyle="--")
    ax.axvline(hi, color="tab:orange", linewidth=1.5, linestyle="--")
    verdict = (
        f"{label_a} wins (CI > 0)" if lo > 0 else
        f"{label_b} wins (CI < 0)" if hi < 0 else
        "tie — CI straddles 0"
    )
    ax.set_xlabel(f"Δ = {label_a} − {label_b}")
    ax.set_ylabel("count")
    ax.set_title(f"{title}  [{verdict}]")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# s08 — A vs B CDF overlay
# ---------------------------------------------------------------------------

def plot_ab_cdf_overlay(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    out_path: Path,
    label_a: str = "A",
    label_b: str = "B",
    title: str = "CDF — A vs B",
    metric_label: str = "score",
) -> None:
    """Empirical CDFs of A and B on the same axes, with fill between."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for scores, label, color in [
        (scores_a, label_a, "steelblue"),
        (scores_b, label_b, "crimson"),
    ]:
        s = np.sort(scores[~np.isnan(scores)])
        if s.size == 0:
            continue
        cdf = np.arange(1, len(s) + 1) / len(s)
        ax.step(s, cdf, where="post", color=color, linewidth=2,
                label=f"{label}  (mean={scores.mean():.4f})", alpha=0.9)
    ax.set_xlabel(metric_label)
    ax.set_ylabel("cumulative probability")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# s08 — forest plot across all (metric × strategy) comparisons
# ---------------------------------------------------------------------------

def plot_comparison_forest(
    comparisons_by_metric: dict[str, dict],
    out_path: Path,
    label_a: str = "A",
    label_b: str = "B",
) -> None:
    """Forest plot: Δ ± paired-EB CI for every (metric, strategy) pair.

    `comparisons_by_metric` is `report["comparisons"]` from s08.
    Rows with "error" keys are skipped.
    """
    rows: list[tuple[str, str, float, float, float, bool, bool]] = []
    for metric, by_strat in comparisons_by_metric.items():
        for strat, e in by_strat.items():
            if "error" in e or np.isnan(e.get("mean_delta", float("nan"))):
                continue
            rows.append((
                metric, strat,
                e["mean_delta"], e["ci_lo"], e["ci_hi"],
                bool(e["a_wins"]), bool(e["b_wins"]),
            ))
    if not rows:
        return

    labels = [f"{m}\n{s}" for m, s, *_ in rows]
    means = np.array([r[2] for r in rows])
    los   = np.array([r[3] for r in rows])
    his   = np.array([r[4] for r in rows])
    a_wins = [r[5] for r in rows]
    b_wins = [r[6] for r in rows]

    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, max(4, len(rows) * 0.8)))
    for i in y:
        color = "steelblue" if a_wins[i] else "crimson" if b_wins[i] else "gray"
        xerr = [[means[i] - los[i]], [his[i] - means[i]]]
        ax.errorbar(means[i], i, xerr=xerr, fmt="o", color=color,
                    markersize=7, linewidth=2, capsize=5, capthick=2)
        ax.text(his[i] + abs(his[i] - los[i]) * 0.04, i,
                f"{means[i]:+.4f}  [{los[i]:+.4f}, {his[i]:+.4f}]",
                va="center", fontsize=7.5)
    ax.axvline(0, color="black", linewidth=1.5, linestyle="--", alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(f"Δ = {label_a} − {label_b}")
    ax.set_title(f"Comparison: {label_a} vs {label_b} — all metrics × strategies\n"
                 f"(blue = {label_a} wins, red = {label_b} wins, gray = tie)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# internal
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, out_path: Path, dpi: int = 130) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
