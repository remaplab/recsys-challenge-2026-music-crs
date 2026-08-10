"""TowerCFEnsembleCG — TowerEnsembleCG + one extra CF-BPR tower.

The parent fuses (RRF) two trained two-towers (query8B×track8B, query8B×SigLIP2)
plus raw 8B dot + tfidf. This subclass adds ONE more tower, the same
`head_q(query8B) × head_t(track_modality)` recipe as the parent's tower B:

    C : query8B × CF-BPR   (collaborative; item-side → usable on cold users)

CF-BPR lives in the SAME track-embedding parquet as SigLIP2 (different column),
so it reuses `img_parquet_glob`. The tower trains a head-pair (InfoNCE, in-batch
negs) + SWAG (reusing the parent's swag_k / swag_max_rank / swag_collect_every)
and contributes one RRF group with its own tuned weight (w_towerC); a useless
tower self-prunes to 0. The parent's `recommend_text` is reused verbatim — it
calls `_member_signals`, which we override to iterate all three tower groups.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import polars as pl
import torch

from .emb_matrix import load_modality_tower
from .hybrid_cg import _maybe_torch
from .tower_ensemble import TowerEnsembleCG
from .tower_ensemble_data import align_tower_to_idx, build_training_pairs, load_query_store
from .tower_ensemble_heads import TrainConfig, build_member_towers, project_queries, train_member


class TowerCFEnsembleCG(TowerEnsembleCG):
    RECOMMENDER_NAME = "TowerCFEnsemble"

    def __init__(
        self,
        *args: Any,
        w_towerC: float = 1.0,
        cf_column: str = "cf-bpr",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.w_towerC = float(w_towerC)
        self.cf_column = str(cf_column)
        self._members_C: list = []

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, train_df, track_metadata: pl.DataFrame | None = None,
            **kwargs: Any) -> None:
        # Parent fit: base artefacts + towers A (8B) and B (SigLIP2).
        super().fit(train_df, track_metadata=track_metadata, **kwargs)
        # tfidf-warm path (train_df is None) builds no towers — nothing to add.
        if train_df is None:
            return
        t0 = time.time()
        device = "cuda" if _maybe_torch(self.use_gpu) is not None else "cpu"
        cfg = TrainConfig(
            d=self.d, hidden=self.hidden, epochs=self.epochs, lr=self.lr,
            tau=self.tau, swag_max_rank=self.swag_max_rank,
            swag_collect_every=self.swag_collect_every,
        )

        # Reload the (session,turn)→(query_emb,gt) store + train sessions. The
        # parent doesn't stash these on self; re-reading keeps tower_ensemble
        # untouched (do-no-harm) at the cost of one extra cache read.
        store = load_query_store(self.query_cache_root)
        session_set = set(train_df["session_id"].to_list())

        # tower C — CF-BPR (from the track-embedding parquet)
        cf_ids, cf_emb = load_modality_tower(self.img_parquet_glob, self.cf_column)
        cf_aligned = align_tower_to_idx(cf_ids, cf_emb, self.track_to_idx, self.n_tracks)
        self._members_C = self._train_modality_tower(
            cf_aligned, store, session_set, cfg, "C", device)

        print(f"[{self.RECOMMENDER_NAME}] trained {len(self._members_C)} C members "
              f"in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    def _train_modality_tower(self, aligned: np.ndarray, store, session_set,
                              cfg: TrainConfig, name: str, device: str) -> list:
        """Train one cross-modal tower over an aligned (n_tracks, d) catalogue.
        Drops pairs whose gt track is missing this modality (zero row)."""
        present = {self.track_ids[i]: i for i in range(self.n_tracks)
                   if np.any(aligned[i] != 0.0)}
        q, pos = build_training_pairs(store, session_set, present)
        if q.shape[0] == 0:
            print(f"[{self.RECOMMENDER_NAME}] tower {name}: no training pairs — skipped")
            return []
        track_t = torch.from_numpy(np.ascontiguousarray(aligned))
        member = train_member(q, pos, track_t, cfg, tower=name, device=device)
        return build_member_towers(member, track_t, swag_k=self.swag_k, device=device)

    # ------------------------------------------------------------------
    # scoring: extend the parent's two-group iteration with tower C
    # ------------------------------------------------------------------
    def _member_signals(self, q8b: np.ndarray, valid: np.ndarray,
                        m: int) -> list[tuple[np.ndarray, float]]:
        out: list[tuple[np.ndarray, float]] = []
        torch_mod = self._torch
        groups = (
            (self._members_A, self.w_towerA), (self._members_B, self.w_towerB),
            (self._members_C, self.w_towerC),
        )
        for members, w_group in groups:
            if not members or w_group == 0.0:
                continue
            w = w_group / len(members)
            for mem in members:
                qproj = project_queries(mem, q8b)
                if torch_mod is not None:
                    if mem.gpu is None:
                        mem.gpu = torch_mod.from_numpy(
                            np.ascontiguousarray(mem.proj_tower)).to("cuda")
                    rk = self._emb_topk(qproj, valid, m, mem.gpu, mem.proj_tower)
                else:
                    rk = self._emb_topk(qproj, valid, m, None, mem.proj_tower)
                out.append((rk, w))
        return out

    # recommend_text inherited: its fitted-guard checks A/B only (A always
    # trains), then calls _member_signals (overridden) → C joins the RRF.

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def _get_model_state(self) -> dict:
        import dataclasses
        st = super()._get_model_state()
        st.update({
            "w_towerC": self.w_towerC, "cf_column": self.cf_column,
            # strip lazy CUDA copies (not picklable; re-upload on next scoring)
            "_members_C": [dataclasses.replace(x, gpu=None) for x in self._members_C],
        })
        return st

    def _set_model_state(self, state: dict) -> None:
        super()._set_model_state(state)
        for k in ("w_towerC", "cf_column", "_members_C"):
            if k in state:
                setattr(self, k, state[k])
