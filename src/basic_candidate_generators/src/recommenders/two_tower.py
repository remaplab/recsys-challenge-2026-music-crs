"""Embedding-native two-tower recommender.

Architecture
------------
* **Item tower**     — MLP over the *frozen* compressed track embedding plus an
                       optional projection of the ICM (truncated SVD). L2-normed.
* **Session tower**  — sequence encoder (GRU / LSTM / Transformer) over the same
                       compressed embeddings, followed by a projection head.
                       L2-normed.
* **Score**          — dot product of normalised embeddings (cosine).
* **Training**       — in-batch InfoNCE over a batch's negatives plus an extra
                       pool of ``n_neg`` randomly sampled negatives; the next
                       music turn in each session is the positive.

The previous BPR/SVD-only implementation is replaced — its config has been
ported (``configs/tune_two_tower_emb.yaml``).  Old class-name + module path are
preserved for downstream importers.
"""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize as sk_normalize

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
from BaseRecommender import BaseRecommender  # noqa: E402

from ._embedding_io import l2_normalize_rows, load_track_embeddings  # noqa: E402
from .interactions import (  # noqa: E402
    IdMap, build_icm, build_id_map, build_track_release_dates,
    explode_music_turns, parse_date,
)


_CORE_TYPES = {"gru", "lstm", "transformer"}


# ---------------------------------------------------------------------------
# Towers
# ---------------------------------------------------------------------------

class _ItemTower(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class _SessionTower(nn.Module):
    def __init__(self, core_type: str, emb_dim: int, hidden: int, out_dim: int,
                 num_layers: int, dropout: float, seq_len: int, nhead: int = 4):
        super().__init__()
        self.core_type = core_type
        if core_type == "gru":
            self.rnn = nn.GRU(emb_dim, hidden, num_layers=num_layers, batch_first=True,
                              dropout=dropout if num_layers > 1 else 0.0)
        elif core_type == "lstm":
            self.rnn = nn.LSTM(emb_dim, hidden, num_layers=num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        elif core_type == "transformer":
            nhead = max(h for h in [1, 2, 4, 8] if emb_dim % h == 0 and h <= nhead)
            self.pos = nn.Parameter(torch.zeros(1, seq_len, emb_dim))
            layer = nn.TransformerEncoderLayer(emb_dim, nhead, dim_feedforward=hidden,
                                               dropout=dropout, batch_first=True)
            self.tr = nn.TransformerEncoder(layer, num_layers=num_layers)
            hidden = emb_dim
        else:
            raise ValueError(f"unknown core_type {core_type!r}")
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x):
        if self.core_type == "gru":
            _, h = self.rnn(x)
            h = h[-1]
        elif self.core_type == "lstm":
            _, (h, _) = self.rnn(x)
            h = h[-1]
        else:
            x = x + self.pos[:, : x.size(1)]
            h = self.tr(x)[:, -1]
        return F.normalize(self.proj(h), dim=-1)


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

class TwoTowerRecommender(BaseRecommender):
    RECOMMENDER_NAME = "TwoTower"

    def __init__(
        self,
        core_type: str = "gru",
        out_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
        nhead: int = 4,
        seq_len: int = 6,
        embedding_cols: list[str] | None = None,
        icm_svd_components: int = 0,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 20,
        batch_size: int = 512,
        n_neg: int = 256,
        temperature: float = 0.07,
        max_future_years: float | None = 2.0,
        device: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if core_type not in _CORE_TYPES:
            raise ValueError(f"core_type must be one of {sorted(_CORE_TYPES)}, got {core_type!r}")
        self.core_type = core_type
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.nhead = int(nhead)
        self.seq_len = int(seq_len)
        self.embedding_cols = embedding_cols
        self.icm_svd_components = int(icm_svd_components)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.n_neg = int(n_neg)
        self.temperature = float(temperature)
        self.max_future_years = max_future_years
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.urm_mode = "session"

        self.id_map: IdMap | None = None
        self._track_ids: list[str] = []
        self._track_idx: dict[str, int] = {}
        self._emb: np.ndarray | None = None
        self._emb_gpu: torch.Tensor | None = None
        self._icm_proj: np.ndarray | None = None
        self._icm_proj_gpu: torch.Tensor | None = None
        self._pop: np.ndarray | None = None
        self._release_dates: np.ndarray | None = None
        self._svd: TruncatedSVD | None = None
        self._item_tower: _ItemTower | None = None
        self._session_tower: _SessionTower | None = None
        self._all_item_embs: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df: pl.DataFrame,
        track_embeddings: Any,
        track_metadata: pl.DataFrame | None = None,
        **kwargs: Any,
    ) -> None:
        long = explode_music_turns(train_df)

        # 1. Embeddings.
        track_ids, emb = load_track_embeddings(track_embeddings, self.embedding_cols)
        self._track_ids = track_ids
        self._track_idx = {t: i for i, t in enumerate(track_ids)}
        emb = l2_normalize_rows(emb.astype(np.float32, copy=False))
        self._emb = emb
        self._emb_gpu = torch.from_numpy(emb).to(self.device)
        d_emb = emb.shape[1]

        # 2. Optional ICM projection.
        extra_dim = 0
        if track_metadata is not None and self.icm_svd_components > 0:
            extra_track_ids = track_metadata["track_id"].to_list()
            self.id_map = build_id_map(long, extra_track_ids=extra_track_ids, mode="session")
            icm = build_icm(track_metadata, self.id_map, interactions=long)
            icm_norm = sk_normalize(icm.astype(np.float32), norm="l2", axis=1)
            n_comp = min(self.icm_svd_components, icm_norm.shape[1] - 1)
            self._svd = TruncatedSVD(n_components=n_comp, random_state=42)
            proj = self._svd.fit_transform(icm_norm).astype(np.float32)
            # Reindex proj from id_map order to embedding-catalogue order.
            aligned = np.zeros((len(track_ids), proj.shape[1]), dtype=np.float32)
            for tid, i in self._track_idx.items():
                j = self.id_map.track_to_idx.get(tid)
                if j is not None:
                    aligned[i] = proj[j]
            self._icm_proj = l2_normalize_rows(aligned)
            self._icm_proj_gpu = torch.from_numpy(self._icm_proj).to(self.device)
            extra_dim = self._icm_proj.shape[1]
            print(f"[{self.RECOMMENDER_NAME}] ICM SVD: {icm.shape} → {self._icm_proj.shape}")

        # 3. Popularity + release dates.
        pop = np.zeros(len(track_ids), dtype=np.float32)
        for tid in long["track_id"].to_list():
            j = self._track_idx.get(tid)
            if j is not None:
                pop[j] += 1.0
        self._pop = np.log1p(pop)

        if track_metadata is not None and "release_date" in track_metadata.columns:
            rd_map = {
                r["track_id"]: r["release_date"]
                for r in track_metadata.select(["track_id", "release_date"]).to_dicts()
            }
            dates = np.empty(len(track_ids), dtype="datetime64[D]")
            dates[:] = np.datetime64("NaT")
            for i, tid in enumerate(track_ids):
                d = rd_map.get(tid)
                if d:
                    try:
                        dates[i] = np.datetime64(str(d)[:10], "D")
                    except Exception:
                        pass
            self._release_dates = dates

        # 4. Build prefix→next-track training samples.
        order_col = "turn_number" if "turn_number" in long.columns else None
        contexts: list[np.ndarray] = []
        targets: list[int] = []
        for sid_val, group in long.group_by("session_id"):
            if order_col:
                group = group.sort(order_col)
            seq = [self._track_idx[t] for t in group["track_id"].to_list() if t in self._track_idx]
            if len(seq) < 2:
                continue
            for t in range(1, len(seq)):
                ctx = seq[max(0, t - self.seq_len):t]
                pad = self.seq_len - len(ctx)
                contexts.append(np.array(([-1] * pad) + ctx, dtype=np.int64))
                targets.append(seq[t])
        if not contexts:
            raise RuntimeError("No training sequences extracted")
        ctx_arr = np.stack(contexts)
        tgt_arr = np.asarray(targets, dtype=np.int64)
        n_samples = len(tgt_arr)
        print(f"[{self.RECOMMENDER_NAME}] training samples: {n_samples}")

        ctx_t = torch.from_numpy(ctx_arr)
        tgt_t = torch.from_numpy(tgt_arr)

        # 5. Towers.
        item_in_dim = d_emb + extra_dim
        self._item_tower = _ItemTower(item_in_dim, self.hidden_dim, self.out_dim, self.dropout).to(self.device)
        self._session_tower = _SessionTower(
            self.core_type, d_emb, self.hidden_dim, self.out_dim,
            self.num_layers, self.dropout, self.seq_len, nhead=self.nhead,
        ).to(self.device)

        opt = torch.optim.AdamW(
            list(self._item_tower.parameters()) + list(self._session_tower.parameters()),
            lr=self.lr, weight_decay=self.weight_decay,
        )

        n_items = len(track_ids)
        t0 = time.time()
        for ep in range(self.epochs):
            self._item_tower.train(); self._session_tower.train()
            order = torch.randperm(n_samples)
            ep_loss = 0.0
            n_b = 0
            for s in range(0, n_samples, self.batch_size):
                idx = order[s : s + self.batch_size]
                ctx_b = ctx_t[idx].to(self.device)            # (B, L)
                tgt_b = tgt_t[idx].to(self.device)            # (B,)
                B = ctx_b.size(0)

                flat = ctx_b.reshape(-1)
                valid = flat >= 0
                emb_in = torch.zeros(B * self.seq_len, d_emb, device=self.device)
                emb_in[valid] = self._emb_gpu[flat[valid]]
                emb_in = emb_in.view(B, self.seq_len, d_emb)

                u = self._session_tower(emb_in)               # (B, out)
                tgt_feats = self._item_feature_batch(tgt_b)   # (B, item_in_dim)
                pos_v = self._item_tower(tgt_feats)           # (B, out)

                neg_idx = torch.randint(0, n_items, (self.n_neg,), device=self.device)
                neg_feats = self._item_feature_batch(neg_idx)
                neg_v = self._item_tower(neg_feats)            # (N, out)

                pos_logits = (u * pos_v).sum(-1, keepdim=True) / self.temperature
                neg_logits = u @ neg_v.T / self.temperature
                logits = torch.cat([pos_logits, neg_logits], dim=1)
                labels = torch.zeros(B, dtype=torch.long, device=self.device)
                loss = F.cross_entropy(logits, labels)

                opt.zero_grad()
                loss.backward()
                params = list(self._item_tower.parameters()) + list(self._session_tower.parameters())
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                ep_loss += float(loss.item())
                n_b += 1
            print(f"[{self.RECOMMENDER_NAME}] ep {ep + 1}/{self.epochs} loss={ep_loss/max(n_b,1):.4f}")

        # 6. Pre-compute all item embeddings for fast inference.
        self._item_tower.eval(); self._session_tower.eval()
        with torch.no_grad():
            chunks = []
            chunk = 4096
            all_idx = torch.arange(n_items, device=self.device)
            for s in range(0, n_items, chunk):
                feats = self._item_feature_batch(all_idx[s : s + chunk])
                chunks.append(self._item_tower(feats))
            self._all_item_embs = torch.cat(chunks, dim=0)  # (n_items, out)
        print(f"[{self.RECOMMENDER_NAME}] trained in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # recommend
    # ------------------------------------------------------------------

    def recommend(
        self,
        context_df: pl.DataFrame,
        top_k: int = 20,
        remove_seen: bool = True,
        max_future_years: float | None = None,
        turn: int = 8,
        **kwargs: Any,
    ) -> pl.DataFrame:
        if self._all_item_embs is None:
            raise RuntimeError("Recommender not fitted")
        if max_future_years is None:
            max_future_years = self.max_future_years

        if "track_id" not in context_df.columns:
            context_df = explode_music_turns(context_df)

        order_col = "turn_number" if "turn_number" in context_df.columns else None
        ordered_ctx: dict[str, list[int]] = {}
        for sid_val, group in context_df.group_by("session_id"):
            sid = sid_val[0] if isinstance(sid_val, tuple) else sid_val
            if order_col:
                group = group.sort(order_col)
            ordered_ctx[sid] = [self._track_idx[t] for t in group["track_id"].to_list() if t in self._track_idx]

        session_meta = (
            context_df.select(["session_id", "user_id", "session_date"])
            .unique(subset=["session_id"])
        )
        rows = session_meta.to_dicts()
        n = len(rows)
        d_emb = self._emb.shape[1]

        ctx_indices = np.full((n, self.seq_len), -1, dtype=np.int64)
        warm = np.zeros(n, dtype=bool)
        for i, row in enumerate(rows):
            seq = ordered_ctx.get(row["session_id"], [])
            if not seq:
                continue
            warm[i] = True
            seq = seq[-self.seq_len:]
            ctx_indices[i, self.seq_len - len(seq):] = seq

        all_scores = np.empty((n, len(self._track_ids)), dtype=np.float32)
        warm_idx = np.where(warm)[0]
        if warm_idx.size:
            with torch.no_grad():
                ctx_t = torch.from_numpy(ctx_indices[warm_idx]).to(self.device)
                flat = ctx_t.reshape(-1)
                valid = flat >= 0
                emb_in = torch.zeros(len(warm_idx) * self.seq_len, d_emb, device=self.device)
                emb_in[valid] = self._emb_gpu[flat[valid]]
                emb_in = emb_in.view(len(warm_idx), self.seq_len, d_emb)
                u = self._session_tower(emb_in)
                scores = (u @ self._all_item_embs.T).cpu().numpy()
            all_scores[warm_idx] = scores
        cold_idx = np.where(~warm)[0]
        if cold_idx.size:
            all_scores[cold_idx] = self._pop[None, :]

        out_session, out_user, out_tracks, out_scores = [], [], [], []
        for i, row in enumerate(rows):
            sid = row["session_id"]
            sd = parse_date(row["session_date"])
            cand_mask = self._date_mask(sd, max_future_years)
            seen = set(ordered_ctx.get(sid, [])) if remove_seen else set()
            recs, scs = self._topk(all_scores[i], seen, top_k, cand_mask)
            out_session.append(sid)
            out_user.append(row["user_id"])
            out_tracks.append(recs)
            out_scores.append(scs)

        return pl.DataFrame({
            "session_id": out_session,
            "user_id": out_user,
            "turn": [turn] * len(out_session),
            "track_ids": out_tracks,
            "scores": out_scores,
        })

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _item_feature_batch(self, idx: torch.Tensor) -> torch.Tensor:
        e = self._emb_gpu[idx]
        if self._icm_proj_gpu is not None:
            return torch.cat([e, self._icm_proj_gpu[idx]], dim=-1)
        return e

    def _topk(self, scores, seen, top_k, mask):
        s = scores.astype(np.float64, copy=True)
        if seen:
            s[list(seen)] = -np.inf
        if mask is not None:
            s[~mask] = -np.inf
        finite = int(np.isfinite(s).sum())
        if finite == 0:
            return [], []
        k = min(top_k, finite)
        idx = np.argpartition(-s, k - 1)[:k]
        idx = idx[np.argsort(-s[idx])]
        return [self._track_ids[i] for i in idx], [float(s[i]) for i in idx]

    def _date_mask(self, sd: date | None, mfy: float | None) -> np.ndarray | None:
        if self._release_dates is None or sd is None or mfy is None:
            return None
        sd64 = np.datetime64(sd, "D")
        cutoff = sd64 + np.timedelta64(int(mfy * 365), "D")
        rd = self._release_dates
        return (rd <= cutoff) | np.isnat(rd)

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def _get_model_state(self) -> dict:
        st = {k: getattr(self, k) for k in (
            "core_type", "out_dim", "hidden_dim", "num_layers", "dropout", "nhead",
            "seq_len", "embedding_cols", "icm_svd_components", "lr", "weight_decay",
            "epochs", "batch_size", "n_neg", "temperature", "max_future_years",
            "device", "_track_ids", "_track_idx", "_emb", "_icm_proj", "_pop",
            "_release_dates",
        )}
        st["_svd"] = self._svd
        st["_item_tower_state"] = self._item_tower.state_dict() if self._item_tower else None
        st["_session_tower_state"] = self._session_tower.state_dict() if self._session_tower else None
        st["_all_item_embs"] = self._all_item_embs.cpu() if self._all_item_embs is not None else None
        return st

    def _set_model_state(self, state: dict) -> None:
        for k in (
            "core_type", "out_dim", "hidden_dim", "num_layers", "dropout", "nhead",
            "seq_len", "embedding_cols", "icm_svd_components", "lr", "weight_decay",
            "epochs", "batch_size", "n_neg", "temperature", "max_future_years",
            "device", "_track_ids", "_track_idx", "_emb", "_icm_proj", "_pop",
            "_release_dates",
        ):
            setattr(self, k, state[k])
        self.urm_mode = "session"
        self._svd = state.get("_svd")
        if self._emb is not None:
            self._emb_gpu = torch.from_numpy(self._emb).to(self.device)
        if self._icm_proj is not None:
            self._icm_proj_gpu = torch.from_numpy(self._icm_proj).to(self.device)
        d_emb = self._emb.shape[1]
        extra_dim = self._icm_proj.shape[1] if self._icm_proj is not None else 0
        item_in_dim = d_emb + extra_dim
        sd_item = state.get("_item_tower_state")
        sd_sess = state.get("_session_tower_state")
        if sd_item is not None:
            self._item_tower = _ItemTower(item_in_dim, self.hidden_dim, self.out_dim, self.dropout).to(self.device)
            self._item_tower.load_state_dict(sd_item)
            self._item_tower.eval()
        if sd_sess is not None:
            self._session_tower = _SessionTower(
                self.core_type, d_emb, self.hidden_dim, self.out_dim,
                self.num_layers, self.dropout, self.seq_len, nhead=self.nhead,
            ).to(self.device)
            self._session_tower.load_state_dict(sd_sess)
            self._session_tower.eval()
        all_emb = state.get("_all_item_embs")
        self._all_item_embs = all_emb.to(self.device) if all_emb is not None else None
