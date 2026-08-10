"""TowerEnsembleCG — RRF fusion of trained projection-head two-towers
(query8B×track8B, query8B×trackSigLIP2) + raw query8B·track8B dot + tfidf.

Subclasses HybridCG to reuse: tfidf caching, the 8B base tower (raw dot),
seen/future masking, the numba RRF kernel, and recs assembly.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import polars as pl
import torch

from .emb_matrix import load_image_tower
from .hybrid_cg import HybridCG, _maybe_torch, _query_text, _rrf_fuse
from .tower_ensemble_data import (
    align_tower_to_idx, build_training_pairs, load_query_store,
)
from .tower_ensemble_heads import (
    TrainConfig, build_member_towers, project_queries, train_member,
)


class TowerEnsembleCG(HybridCG):
    RECOMMENDER_NAME = "TowerEnsemble"

    def __init__(
        self,
        query_cache_root: str | None = None,     # tower base holding splitK query caches
        img_parquet_glob: str | None = None,     # SigLIP2 parquets (tower B)
        swag_k: int = 5,
        swag_max_rank: int = 5,
        swag_collect_every: int = 0,             # 0=per-epoch, N=every N steps
        d: int = 256,
        hidden: int = 512,
        epochs: int = 5,
        lr: float = 1e-3,
        tau: float = 0.05,
        # RRF group weights (the 4 tuned fusion knobs)
        w_towerA: float = 1.0,
        w_towerB: float = 1.0,
        w_dot: float = 1.0,
        w_tfidf: float = 1.0,
        **kwargs: Any,
    ):
        # This CG fuses its OWN four groups. We still inherit tfidf + the 8B
        # base tower (raw dot) from HybridCG.
        kwargs.setdefault("model_size", "qwen3_8b")
        super().__init__(**kwargs)
        self.query_cache_root = query_cache_root
        self.img_parquet_glob = img_parquet_glob
        self.swag_k = int(swag_k)
        self.swag_max_rank = int(swag_max_rank)
        self.swag_collect_every = int(swag_collect_every)
        self.d = int(d); self.hidden = int(hidden)
        self.epochs = int(epochs); self.lr = float(lr); self.tau = float(tau)
        self.w_towerA = float(w_towerA); self.w_towerB = float(w_towerB)
        self.w_dot = float(w_dot); self.w_tfidf = float(w_tfidf)
        self._members_A: list = []
        self._members_B: list = []

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, train_df, track_metadata: pl.DataFrame | None = None,
            **kwargs: Any) -> None:
        # Base fit: builds track_ids/track_to_idx, tfidf, the 8B tower
        # (self.track_emb) and uploads it (raw-dot base). track_emb_dir must be
        # the 8B dense_tracks dir.
        super().fit(train_df, track_metadata=track_metadata, **kwargs)
        # The DRO tuner calls fit(None, ...) once just to warm the tfidf cache
        # (tfidf_rk_table). With no training rows there are no tower members to
        # train — base artefacts from super().fit() are enough for that path.
        if train_df is None:
            return
        t0 = time.time()
        device = "cuda" if _maybe_torch(self.use_gpu) is not None else "cpu"

        store = load_query_store(self.query_cache_root)
        session_set = set(train_df["session_id"].to_list())

        # tower A catalogue = the 8B tower already aligned by HybridCG
        track8b = torch.from_numpy(np.ascontiguousarray(self.track_emb))
        q8b, pos8b = build_training_pairs(store, session_set, self.track_to_idx)
        if q8b.shape[0] == 0:
            raise RuntimeError(
                f"[{self.RECOMMENDER_NAME}] no training pairs — check that "
                f"query_cache_root ({self.query_cache_root}) covers the train "
                f"sessions and that gt tracks are in the catalogue."
            )

        # tower B catalogue = SigLIP2 aligned to the SAME track_to_idx
        img_ids, img_emb = load_image_tower(self.img_parquet_glob)
        img_aligned = align_tower_to_idx(img_ids, img_emb, self.track_to_idx,
                                         self.n_tracks)
        # tower-B training idx map drops zero (image-less) catalogue rows
        img_present = {self.track_ids[i]: i for i in range(self.n_tracks)
                       if np.any(img_aligned[i] != 0.0)}
        qB, posB = build_training_pairs(store, session_set, img_present)
        trackB = torch.from_numpy(np.ascontiguousarray(img_aligned))

        cfg = TrainConfig(
            d=self.d, hidden=self.hidden, epochs=self.epochs, lr=self.lr,
            tau=self.tau, swag_max_rank=self.swag_max_rank,
            swag_collect_every=self.swag_collect_every,
        )
        mA = train_member(q8b, pos8b, track8b, cfg, tower="A", device=device)
        self._members_A = build_member_towers(
            mA, track8b, swag_k=self.swag_k, device=device)
        self._members_B = []
        if qB.shape[0] > 0:
            mB = train_member(qB, posB, trackB, cfg, tower="B", device=device)
            self._members_B = build_member_towers(
                mB, trackB, swag_k=self.swag_k, device=device)
        print(f"[{self.RECOMMENDER_NAME}] trained "
              f"{len(self._members_A)} A-members, {len(self._members_B)} B-members "
              f"in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # recommend (text mode)
    # ------------------------------------------------------------------

    def _query_emb_matrix(self, rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """(ns, dq) frozen query embeddings + validity mask from sess_info rows."""
        ns = len(rows)
        dq = self.track_emb.shape[1]
        out = np.zeros((ns, dq), dtype=np.float32)
        valid = np.zeros(ns, dtype=bool)
        for i, r in enumerate(rows):
            qe = r.get("query_emb")
            if qe is not None:
                v = np.asarray(qe, dtype=np.float32)
                if v.shape[0] == dq:
                    out[i] = v; valid[i] = True
        return out, valid

    def _member_signals(self, q8b: np.ndarray, valid: np.ndarray,
                        m: int) -> list[tuple[np.ndarray, float]]:
        """One (ranked, weight) per member. Group weight split equally across
        members so a group's total RRF mass == its tuned weight regardless of
        member count."""
        out: list[tuple[np.ndarray, float]] = []
        torch_mod = self._torch
        for members, w_group in ((self._members_A, self.w_towerA),
                                  (self._members_B, self.w_towerB)):
            if not members or w_group == 0.0:
                continue
            w = w_group / len(members)
            for mem in members:
                qproj = project_queries(mem, q8b)                  # (ns, d)
                if torch_mod is not None:
                    # Score on GPU; cache the member's projected tower on CUDA
                    # so the 8 per-turn recommend calls reuse one upload.
                    if mem.gpu is None:
                        mem.gpu = torch_mod.from_numpy(
                            np.ascontiguousarray(mem.proj_tower)).to("cuda")
                    rk = self._emb_topk(qproj, valid, m, mem.gpu, mem.proj_tower)
                else:
                    rk = self._emb_topk(qproj, valid, m, None, mem.proj_tower)
                out.append((rk, w))
        return out

    def recommend_text(self, sess_info: pl.DataFrame, top_k: int = 100,
                       remove_seen: bool = True) -> pl.DataFrame:
        if not self._members_A and not self._members_B:
            raise RuntimeError("TowerEnsembleCG not fitted")
        rows = sess_info.to_dicts()
        ns = len(rows)
        m = self.top_k_per_signal
        t2i = self.track_to_idx
        ctx_idx = [np.asarray([t2i[t] for t in (r.get("ctx_tracks") or []) if t in t2i],
                              dtype=np.int64) for r in rows]

        q8b, valid = self._query_emb_matrix(rows)

        sig_rk: list[np.ndarray] = []
        sig_w: list[float] = []

        # group 3: raw dot (8B base tower, reused from HybridCG)
        if self.w_dot != 0.0:
            sig_rk.append(self._emb_topk(q8b, valid, m, self._tower_gpu, self.track_emb))
            sig_w.append(self.w_dot)

        # group 4: tfidf (cached / injected when available)
        if self.w_tfidf != 0.0:
            queries = [_query_text(r) for r in rows]
            if self._tfidf_rk_inject is not None:
                rk_tfidf = self._gather_injected_rk(rows, m)
            else:
                Qt = self._tfidf_vec.transform(queries).tocsr()
                rk_tfidf = self._text_topk(Qt, self._tfidf_gpu, self._tfidf_post, m)
            sig_rk.append(rk_tfidf); sig_w.append(self.w_tfidf)

        # groups 1+2: trained tower members
        for rk, w in self._member_signals(q8b, valid, m):
            sig_rk.append(rk); sig_w.append(w)

        ranked = np.stack(sig_rk, axis=1)
        weights = np.array(sig_w, np.float32)
        seen_flat, seen_ptr = self._seen_csr(ctx_idx, remove_seen, ns)
        cutoffs = self._future_cutoffs(rows)
        idx, scr = _rrf_fuse(ranked, weights, float(self.k_rrf), self.release_days,
                             cutoffs, seen_flat, seen_ptr, self.n_tracks, top_k)
        return self._to_recs_df(rows, idx, scr)

    # ------------------------------------------------------------------
    # persistence (params + projected members; refit not needed to restore)
    # ------------------------------------------------------------------

    def _get_model_state(self) -> dict:
        import dataclasses
        # Strip the lazy CUDA tower copies before pickling (not picklable /
        # huge); they re-upload on the next scoring call.
        members_a = [dataclasses.replace(m, gpu=None) for m in self._members_A]
        members_b = [dataclasses.replace(m, gpu=None) for m in self._members_B]
        st = super()._get_model_state()
        st.update({
            "query_cache_root": self.query_cache_root,
            "img_parquet_glob": self.img_parquet_glob,
            "swag_k": self.swag_k, "swag_max_rank": self.swag_max_rank,
            "swag_collect_every": self.swag_collect_every,
            "d": self.d, "hidden": self.hidden, "epochs": self.epochs,
            "lr": self.lr, "tau": self.tau,
            "w_towerA": self.w_towerA, "w_towerB": self.w_towerB,
            "w_dot": self.w_dot, "w_tfidf": self.w_tfidf,
            "_members_A": members_a, "_members_B": members_b,
        })
        return st

    def _set_model_state(self, state: dict) -> None:
        super()._set_model_state(state)
        for k in ("query_cache_root", "img_parquet_glob",
                  "swag_k", "swag_max_rank", "swag_collect_every", "d", "hidden",
                  "epochs", "lr", "tau",
                  "w_towerA", "w_towerB", "w_dot", "w_tfidf",
                  "_members_A", "_members_B"):
            if k in state:
                setattr(self, k, state[k])
