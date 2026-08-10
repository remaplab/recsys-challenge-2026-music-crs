"""Recall@K + category x specificity heatmap evaluation.

Given:
  query_emb   (N, D), L2-normalized
  track_emb   (T, D), L2-normalized
  track_mask  (T,) bool
  meta_df     polars DataFrame with session_id, turn_number, gt_track_id,
              category, specificity, prior_track_ids
  id_to_idx   dict[str, int] mapping track_id -> row in track_emb

Computes recall@K for K in [20, 100, 150, 200, 250, 500], and per
(category, specificity) breakdown. Saves JSON + heatmap PNGs.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import torch
from tqdm import tqdm


DEFAULT_KS = (20, 100, 150, 200, 250, 500)


@torch.no_grad()
def compute_topk(
    query_emb: np.ndarray,
    track_emb: np.ndarray,
    track_mask: np.ndarray,
    meta_df: pl.DataFrame,
    id_to_idx: dict[str, int],
    max_k: int,
    batch_size: int = 256,
    mask_played: bool = True,
    device: str | None = None,
) -> np.ndarray:
    """Returns (N, max_k) int32 of tower indices."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    track_emb_t = torch.from_numpy(track_emb).to(device, dtype=torch.float32)
    track_mask_t = torch.from_numpy(track_mask).to(device)
    n = query_emb.shape[0]
    out = np.full((n, max_k), -1, dtype=np.int32)
    rows = meta_df.to_dicts()
    for s in tqdm(range(0, n, batch_size), desc="top-K"):
        e = min(s + batch_size, n)
        q_t = torch.from_numpy(query_emb[s:e]).to(device, dtype=torch.float32)
        scores = (q_t @ track_emb_t.T).masked_fill(
            ~track_mask_t.unsqueeze(0), float("-inf"))
        if mask_played:
            for bi in range(e - s):
                played = rows[s + bi].get("prior_track_ids") or []
                idxs = [id_to_idx[t] for t in played if t in id_to_idx]
                if idxs:
                    scores[bi, torch.tensor(idxs, device=device)] = float("-inf")
        out[s:e] = torch.topk(scores, k=max_k, dim=1).indices.cpu().numpy().astype(np.int32)
    return out


def compute_recall(
    topk: np.ndarray,
    meta_df: pl.DataFrame,
    id_to_idx: dict[str, int],
    ks: tuple = DEFAULT_KS,
) -> dict:
    """Overall + stratified recall by (category, specificity)."""
    rows = meta_df.to_dicts()
    overall = {k: [0, 0] for k in ks}                             # [hits, total]
    by_cell = defaultdict(lambda: {k: [0, 0] for k in ks})        # (cat, spec) -> {k: [h, t]}

    for i, r in enumerate(rows):
        tid = r.get("gt_track_id")
        if tid is None or tid not in id_to_idx:
            continue
        gt_idx = id_to_idx[tid]
        cat = r.get("category") or "?"
        spec = r.get("specificity") or "?"
        row_top = topk[i]
        for k in ks:
            overall[k][1] += 1
            by_cell[(cat, spec)][k][1] += 1
            if gt_idx in row_top[:k]:
                overall[k][0] += 1
                by_cell[(cat, spec)][k][0] += 1

    out_overall = {f"recall@{k}": h / max(t, 1) for k, (h, t) in overall.items()}
    out_overall["n_eval"] = overall[ks[0]][1]

    by_cell_out = {}
    for (cat, spec), kv in by_cell.items():
        by_cell_out[f"{cat}/{spec}"] = {
            f"recall@{k}": h / max(t, 1) for k, (h, t) in kv.items()
        } | {"n": kv[ks[0]][1]}
    return {"overall": out_overall, "by_cell": by_cell_out, "ks": list(ks)}


def save_heatmaps(report: dict, out_dir: Path, encoder_name: str) -> None:
    """One PNG per K showing recall on a (category x specificity) grid."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    cells = report["by_cell"]
    cats = sorted({k.split("/")[0] for k in cells})
    specs = sorted({k.split("/")[1] for k in cells})

    for k in report["ks"]:
        grid = np.full((len(cats), len(specs)), np.nan, dtype=np.float32)
        counts = np.zeros((len(cats), len(specs)), dtype=np.int64)
        for i, c in enumerate(cats):
            for j, s in enumerate(specs):
                cell = cells.get(f"{c}/{s}")
                if cell is None: continue
                grid[i, j] = cell[f"recall@{k}"]
                counts[i, j] = cell["n"]
        fig, ax = plt.subplots(figsize=(2 + len(specs), 2 + 0.4 * len(cats)))
        im = ax.imshow(grid, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(specs))); ax.set_xticklabels(specs)
        ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats)
        ax.set_title(f"{encoder_name} — recall@{k}")
        for i in range(len(cats)):
            for j in range(len(specs)):
                v = grid[i, j]
                if np.isnan(v): continue
                txt = f"{v:.2f}\n(n={counts[i, j]})"
                ax.text(j, i, txt, ha="center", va="center",
                        color="white" if v < 0.5 else "black", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        out_path = out_dir / f"heatmap_recall_at_{k}.png"
        plt.savefig(out_path, dpi=120); plt.close(fig)
    print(f"  saved {len(report['ks'])} heatmaps -> {out_dir}")


def save_report(report: dict, out_dir: Path, encoder_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"encoder": encoder_name, **report}
    (out_dir / "recall.json").write_text(json.dumps(payload, indent=2))
    print(f"  saved {out_dir / 'recall.json'}")
    for k, v in report["overall"].items():
        print(f"    {k} = {v:.4f}" if isinstance(v, float) else f"    {k} = {v}")