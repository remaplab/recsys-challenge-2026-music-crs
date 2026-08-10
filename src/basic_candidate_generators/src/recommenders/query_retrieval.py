"""Pure query→track retrieval CG (two-tower, intent-only).

Phase-1 of the query-retrieval line: a candidate generator that scores the WHOLE
catalogue by matching the TARGET-turn conversation query against each track —
deliberately ignoring the item history at scoring time, so it is complementary
to the artist-continuation engines (BERT4Rec / two_tower).  Failure analysis
([[project_failure_mode_artist_continuation]]) showed ~70% of misses are
new-artist targets the history-only models never retrieve; this tower goes after
exactly those.

Towers
------
* Query tower : frozen Qwen3 conversation embedding (from the per-turn cache)
                → PCA-initialised Linear → L2.  Trainable head, PCA init.
* Item tower  : per-modality L2 (qwen3-text / CF / CLAP) → PCA-initialised Linear
                → L2.  One catalogue index; full-catalogue scoring.
* Score       : cosine(u_query, item).

Training
--------
InfoNCE: positive = the track played at that turn; negatives = in-batch + random
+ HARD negatives mined from the artists already in the session history. The hard
negatives are the crux: they teach the query tower to rank the GT ABOVE the
continuation tracks when the query asks to pivot ("not southern rock", "a
different era") — which plain cosine on frozen embeddings cannot do.

Every turn (including turn 1) is a (query, track) training pair, so this CG also
covers the turn-1 cold-start the sequential models fall back on.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

from .interactions import explode_music_turns, parse_date
from .session_base import SessionRecommender
from .feature_bert4rec_identity_cosine_rope_mmh import _build_feature_matrix_per_modality
from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim_xattn_hardneg_query import (
    _load_query_emb_cache,
)


# ---------------------------------------------------------------------------
# PCA-initialised projection head
# ---------------------------------------------------------------------------

def _pca_linear(layer: nn.Linear, X: np.ndarray) -> None:
    """Initialise `layer` (in_dim→out_dim) so it projects onto the top
    principal components of X (mean-centred). Extra output rows (if out_dim >
    n_components) stay at the small default init."""
    out_dim, in_dim = layer.weight.shape
    n_comp = min(out_dim, in_dim, X.shape[0])
    pca = PCA(n_components=n_comp, random_state=0)
    pca.fit(X)
    W = layer.weight.data.clone().numpy()
    b = layer.bias.data.clone().numpy()
    W[:n_comp] = pca.components_.astype(np.float32)
    b[:n_comp] = -(pca.components_ @ pca.mean_).astype(np.float32)
    layer.weight.data.copy_(torch.from_numpy(W))
    layer.bias.data.copy_(torch.from_numpy(b))


class _QueryTower(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(q), dim=-1)


class _ItemTower(nn.Module):
    """Per-modality L2 → PCA-init Linear → L2."""

    def __init__(self, modality_dims: list[int], out_dim: int):
        super().__init__()
        self.modality_dims = list(modality_dims)
        bounds, s = [], 0
        for d in modality_dims:
            bounds.append((s, s + d)); s += d
        self.bounds = bounds
        self.proj = nn.Linear(s, out_dim)

    def _l2_per_modality(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([F.normalize(x[..., a:b], dim=-1) for a, b in self.bounds], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(self._l2_per_modality(x)), dim=-1)


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

class QueryRetrievalRecommender(SessionRecommender):
    RECOMMENDER_NAME = "QueryRetrieval"

    def __init__(
        self,
        *args: Any,
        feature_emb_paths: list[str],
        feature_modalities: list[str] | None = None,
        query_emb_dir: str = "models/query_emb_cache/qwen3_frozen",
        query_cache_splits: list[str] | None = None,
        out_dim: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 30,
        batch_size: int = 512,
        n_neg: int = 256,
        n_hardneg: int = 16,
        temperature: float = 0.07,
        max_future_years: float = 0.0,
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        super().__init__(urm_mode=kwargs.pop("urm_mode", "session"), **kwargs)
        self.feature_emb_paths = list(feature_emb_paths)
        self.feature_modalities = list(feature_modalities) if feature_modalities else None
        self.query_emb_dir = str(query_emb_dir)
        self.query_cache_splits = list(query_cache_splits) if query_cache_splits else ["train", "dev", "blind_a"]
        self.out_dim = int(out_dim)
        self.lr = float(lr); self.weight_decay = float(weight_decay)
        self.epochs = int(epochs); self.batch_size = int(batch_size)
        self.n_neg = int(n_neg); self.n_hardneg = int(n_hardneg)
        self.temperature = float(temperature)
        self.max_future_years = float(max_future_years)
        self.device_ = torch.device(
            "cuda" if (device == "auto" and torch.cuda.is_available()) else
            ("cpu" if device == "auto" else device)
        )
        # filled at fit
        self._modality_dims: list[int] = []
        self._query_table: np.ndarray | None = None
        self._query_lookup: dict[tuple[str, int], int] = {}
        self._feat_matrix: np.ndarray | None = None
        self._item_tower: _ItemTower | None = None
        self._query_tower: _QueryTower | None = None
        self._all_item_embs: torch.Tensor | None = None
        self._pop: np.ndarray | None = None

    # ------------------------------------------------------------------
    def _resolve(self, p: str) -> Path:
        path = Path(p)
        if path.is_absolute():
            return path
        from launchers._predict_fold_common import repo_path
        return repo_path(p)

    def _fit_model(self, urm: Any) -> None:
        # No URM model: training happens in fit() after super().fit() sets up
        # id_map / release_dates / fallback. This no-op satisfies the base hook.
        return None

    def fit(self, train_df: pl.DataFrame, track_metadata: pl.DataFrame | None = None, **kwargs: Any) -> None:
        super().fit(train_df, track_metadata=track_metadata, **kwargs)
        assert self.id_map is not None
        long = explode_music_turns(train_df)

        # --- item features (per-modality, aligned to id_map order) ---
        full_matrix, modality_dims = _build_feature_matrix_per_modality(
            self.feature_emb_paths, self.feature_modalities, self.id_map
        )
        self._feat_matrix = full_matrix.astype(np.float32)
        self._modality_dims = modality_dims
        n_tracks, D = full_matrix.shape
        print(f"[{self.RECOMMENDER_NAME}] item features {full_matrix.shape}, modality_dims={modality_dims}")

        # --- query cache ---
        cache_dir = self._resolve(self.query_emb_dir)
        self._query_table, self._query_lookup = _load_query_emb_cache(cache_dir, self.query_cache_splits)
        q_dim = self._query_table.shape[1]
        print(f"[{self.RECOMMENDER_NAME}] query table {self._query_table.shape}, "
              f"{len(self._query_lookup)} (sess,turn) keys")

        # --- popularity (cold fallback) ---
        pop = np.zeros(n_tracks, dtype=np.float32)
        for t in long["track_id"].to_list():
            j = self.id_map.track_to_idx.get(t)
            if j is not None:
                pop[j] += 1.0
        self._pop = np.log1p(pop)

        # --- primary artist per track + artist→tracks (for hard negs) ---
        track_artist = np.full(n_tracks, -1, dtype=np.int64)
        artist_to_tracks: dict[int, list[int]] = {}
        if track_metadata is not None and "artist_id" in track_metadata.columns:
            art_ids: dict[str, int] = {}
            for tid, arts in zip(track_metadata["track_id"].to_list(),
                                 track_metadata["artist_id"].to_list()):
                j = self.id_map.track_to_idx.get(tid)
                if j is None or not arts:
                    continue
                a0 = arts[0] if isinstance(arts, (list, tuple)) and len(arts) else arts
                a = art_ids.setdefault(str(a0), len(art_ids))
                track_artist[j] = a
                artist_to_tracks.setdefault(a, []).append(j)
        print(f"[{self.RECOMMENDER_NAME}] artists={len(artist_to_tracks)}")

        # --- training pairs (every turn with a query) + static hard negs ---
        q_rows: list[int] = []
        gt_idx: list[int] = []
        hard_np: list[np.ndarray] = []
        rng = np.random.default_rng(0)
        order_col = "turn_number" if "turn_number" in long.columns else None
        for _, grp in long.group_by("session_id", maintain_order=True):
            if order_col:
                grp = grp.sort(order_col)
            sid = grp["session_id"][0]
            turns = grp["turn_number"].to_list() if order_col else list(range(grp.height))
            tids = grp["track_id"].to_list()
            hist_artists: set[int] = set()
            for tn, tid in zip(turns, tids):
                j = self.id_map.track_to_idx.get(tid)
                qr = self._query_lookup.get((sid, int(tn)), 0) if tn is not None else 0
                if j is not None and qr != 0:
                    # hard negs from history artists (the continuation lure)
                    pool: list[int] = []
                    for a in hist_artists:
                        pool.extend(artist_to_tracks.get(a, ()))
                    pool = [p for p in pool if p != j]
                    if pool and self.n_hardneg > 0:
                        pick = rng.choice(np.asarray(pool), size=self.n_hardneg,
                                          replace=len(pool) < self.n_hardneg)
                    else:
                        pick = np.full(self.n_hardneg, -1, dtype=np.int64)
                    q_rows.append(qr); gt_idx.append(j); hard_np.append(pick.astype(np.int64))
                if j is not None and track_artist[j] >= 0:
                    hist_artists.add(int(track_artist[j]))
        if not q_rows:
            raise RuntimeError(f"[{self.RECOMMENDER_NAME}] no (query, track) training pairs")
        q_rows = np.asarray(q_rows, dtype=np.int64)
        gt_idx = np.asarray(gt_idx, dtype=np.int64)
        hard = np.stack(hard_np) if self.n_hardneg > 0 else np.zeros((len(q_rows), 0), np.int64)
        n_samples = len(q_rows)
        print(f"[{self.RECOMMENDER_NAME}] training pairs={n_samples} (n_hardneg={self.n_hardneg})")

        # --- towers (PCA-init) ---
        self._query_tower = _QueryTower(q_dim, self.out_dim).to(self.device_)
        _pca_linear(self._query_tower.proj.cpu(), self._query_table[1:])  # skip zero row
        self._query_tower.to(self.device_)
        self._item_tower = _ItemTower(modality_dims, self.out_dim).to(self.device_)
        # PCA on per-modality-L2 concat
        l2cat = np.concatenate(
            [full_matrix[:, a:b] / np.linalg.norm(full_matrix[:, a:b], axis=1, keepdims=True).clip(min=1e-9)
             for (a, b) in self._item_tower.bounds], axis=1).astype(np.float32)
        _pca_linear(self._item_tower.proj.cpu(), l2cat)
        self._item_tower.to(self.device_)

        feat_gpu = torch.from_numpy(self._feat_matrix).to(self.device_)
        qtab_gpu = torch.from_numpy(self._query_table).to(self.device_)
        q_rows_t = torch.from_numpy(q_rows); gt_t = torch.from_numpy(gt_idx)
        hard_t = torch.from_numpy(hard)

        opt = torch.optim.AdamW(
            list(self._query_tower.parameters()) + list(self._item_tower.parameters()),
            lr=self.lr, weight_decay=self.weight_decay)

        t0 = time.time()
        for ep in range(self.epochs):
            self._query_tower.train(); self._item_tower.train()
            perm = torch.randperm(n_samples)
            ep_loss, nb = 0.0, 0
            for s in range(0, n_samples, self.batch_size):
                b = perm[s:s + self.batch_size]
                B = b.numel()
                u = self._query_tower(qtab_gpu[q_rows_t[b].to(self.device_)])         # (B,out)
                pos_v = self._item_tower(feat_gpu[gt_t[b].to(self.device_)])          # (B,out)

                pos = (u * pos_v).sum(-1, keepdim=True)                                # (B,1)
                inb = u @ pos_v.T                                                      # (B,B)
                inb.fill_diagonal_(float("-inf"))
                neg_i = torch.randint(0, n_tracks, (self.n_neg,), device=self.device_)
                neg_v = self._item_tower(feat_gpu[neg_i])                              # (n_neg,out)
                rnd = u @ neg_v.T                                                      # (B,n_neg)
                parts = [pos, inb, rnd]
                if self.n_hardneg > 0:
                    h = hard_t[b].to(self.device_)                                    # (B,H)
                    hv = self._item_tower(feat_gpu[h.clamp(min=0)])                    # (B,H,out)
                    hl = (u.unsqueeze(1) * hv).sum(-1)                                 # (B,H)
                    hl = hl.masked_fill(h < 0, float("-inf"))
                    parts.append(hl)
                logits = torch.cat(parts, dim=1) / self.temperature
                loss = F.cross_entropy(logits, torch.zeros(B, dtype=torch.long, device=self.device_))

                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self._query_tower.parameters()) + list(self._item_tower.parameters()), 1.0)
                opt.step()
                ep_loss += float(loss.item()); nb += 1
            print(f"[{self.RECOMMENDER_NAME}] ep {ep+1}/{self.epochs} loss={ep_loss/max(nb,1):.4f}")

        # --- precompute all item embeddings ---
        self._item_tower.eval(); self._query_tower.eval()
        with torch.no_grad():
            chunks = []
            for s in range(0, n_tracks, 8192):
                chunks.append(self._item_tower(feat_gpu[s:s + 8192]))
            self._all_item_embs = torch.cat(chunks, dim=0)
        print(f"[{self.RECOMMENDER_NAME}] trained in {time.time()-t0:.1f}s; item index {tuple(self._all_item_embs.shape)}")

    # ------------------------------------------------------------------
    def recommend(self, context_df: pl.DataFrame, top_k: int = 20, remove_seen: bool = True,
                  max_future_years: float | None = None, turn: int = 8, **kwargs: Any) -> pl.DataFrame:
        if self._all_item_embs is None or self.id_map is None:
            raise RuntimeError("fit() before recommend()")
        if max_future_years is None:
            max_future_years = self.max_future_years
        if "track_id" not in context_df.columns:
            context_df = explode_music_turns(context_df)
        if "target_turn" not in context_df.columns:
            raise KeyError("context_df missing 'target_turn'")

        # seen tracks per session (for remove_seen)
        seen: dict[str, set[int]] = {}
        if "turn_number" in context_df.columns:
            for sid, grp in context_df.group_by("session_id", maintain_order=True):
                s = sid[0] if isinstance(sid, tuple) else sid
                seen[s] = {self.id_map.track_to_idx[t] for t in grp["track_id"].to_list()
                           if t in self.id_map.track_to_idx}

        meta = (context_df.select(["session_id", "user_id", "session_date", "target_turn"])
                .unique(subset=["session_id"]))
        out_s, out_u, out_t, out_tr, out_sc, out_fb = [], [], [], [], [], []
        qtab_gpu = torch.from_numpy(self._query_table).to(self.device_)
        for row in meta.iter_rows(named=True):
            sid = row["session_id"]; tt = int(row["target_turn"])
            sd = parse_date(row["session_date"])
            cand_mask = self._filter_candidate_mask(sd)  # uses self.max_future_years
            qr = self._query_lookup.get((sid, tt), 0)
            if qr == 0:  # no query → popularity fallback
                scores = self._pop.copy()
                fb = 1
            else:
                with torch.no_grad():
                    u = self._query_tower(qtab_gpu[qr:qr + 1])
                    scores = (u @ self._all_item_embs.T).squeeze(0).cpu().numpy()
                fb = 0
            recs, scs = self._topk_from_scores(scores, seen.get(sid, set()), top_k, cand_mask, remove_seen)
            out_s.append(sid); out_u.append(row["user_id"]); out_t.append(tt)
            out_tr.append(recs); out_sc.append(scs); out_fb.append([fb] * len(recs))

        return pl.DataFrame({"session_id": out_s, "user_id": out_u, "turn": out_t,
                             "track_ids": out_tr, "scores": out_sc, "fallback_used": out_fb})

    # ------------------------------------------------------------------
    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st.update({
            "feature_emb_paths": self.feature_emb_paths, "feature_modalities": self.feature_modalities,
            "query_emb_dir": self.query_emb_dir, "query_cache_splits": self.query_cache_splits,
            "out_dim": self.out_dim, "_modality_dims": self._modality_dims,
            "_feat_matrix": self._feat_matrix, "_pop": self._pop,
            "_query_table": self._query_table, "_query_lookup": self._query_lookup,
            "_item_tower": self._item_tower.state_dict() if self._item_tower else None,
            "_query_tower": self._query_tower.state_dict() if self._query_tower else None,
            "_all_item_embs": self._all_item_embs.cpu() if self._all_item_embs is not None else None,
        })
        return st

    def _set_model_state(self, state: dict) -> None:
        super()._set_model_state(state)
        for k in ("feature_emb_paths", "feature_modalities", "query_emb_dir", "query_cache_splits",
                  "out_dim", "_modality_dims", "_feat_matrix", "_pop", "_query_table", "_query_lookup"):
            setattr(self, k, state.get(k))
        if state.get("_item_tower") is not None:
            self._item_tower = _ItemTower(self._modality_dims, self.out_dim).to(self.device_)
            self._item_tower.load_state_dict(state["_item_tower"]); self._item_tower.eval()
        if state.get("_query_tower") is not None:
            self._query_tower = _QueryTower(self._query_table.shape[1], self.out_dim).to(self.device_)
            self._query_tower.load_state_dict(state["_query_tower"]); self._query_tower.eval()
        emb = state.get("_all_item_embs")
        self._all_item_embs = emb.to(self.device_) if emb is not None else None
