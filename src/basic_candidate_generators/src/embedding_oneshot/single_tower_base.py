"""SingleTowerCG — one trained query8B × modality two-tower retriever (turn-1).

A standalone version of one member group from tower_(cf_)ensemble: it trains a
single projection-head pair (frozen Qwen3-8B query encoder × a frozen track
modality) with InfoNCE + SWAG, then retrieves the catalogue by the SWAG-mean
projected cosine. No RRF, no tfidf/dot baggage — one tower, one signal, so each
modality can be tuned (and hacked) in isolation as its own CG.

Subclasses set the track modality only, via `_load_modality()`:
    tower_a        : 8B text track tower      (query8B × track8B)
    tower_b        : SigLIP2 image            (image-siglip2)
    tower_c        : CF-BPR                   (cf-bpr)
    tower_audioclap: LAION-CLAP audio         (audio-laion_clap)

Query embeddings (input, not label) for BOTH training pairs and inference come
from the splitK query caches under `query_cache_root` (always Qwen3-8B). Training
pairs use turns 1..`max_train_turn`: =1 specialises to one-shot but is data-
limited (~12k pairs); larger thresholds add nearer-turn pairs (more data, mild
distribution shift). The heads are always applied to turn-1 queries at inference.

Standard inference mode: subclasses BaseRecommender; query lookup keys on
(session_id, target_turn).
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import polars as pl
import torch

from BaseRecommender import BaseRecommender

from recommenders.interactions import parse_date

from embedding_based.dense_query_cg import _build_release_dates, _maybe_torch
from embedding_based.tower_ensemble_data import load_query_store
from embedding_based.tower_ensemble_heads import (
    TrainConfig, build_member_towers, project_queries, train_member,
)


class SingleTowerCG(BaseRecommender):
    RECOMMENDER_NAME = "SingleTowerCG"
    TOWER = "A"   # label passed to train_member; overridden per subclass

    def __init__(
        self,
        query_cache_root: str | None = None,
        # training
        d: int = 256, hidden: int = 512, epochs: int = 5, lr: float = 1e-3,
        tau: float = 0.05, swag_k: int = 5, swag_max_rank: int = 5,
        swag_collect_every: int = 0, max_train_turn: int = 1,
        # inference
        max_future_years: float = 2.0, block: int = 512,
        urm_mode: str = "session", use_gpu: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.query_cache_root = query_cache_root
        self.d = int(d); self.hidden = int(hidden); self.epochs = int(epochs)
        self.lr = float(lr); self.tau = float(tau)
        self.swag_k = int(swag_k); self.swag_max_rank = int(swag_max_rank)
        self.swag_collect_every = int(swag_collect_every)
        self.max_train_turn = int(max_train_turn)
        self.max_future_years = float(max_future_years)
        self.block = int(block); self.urm_mode = urm_mode; self.use_gpu = use_gpu

        self.track_ids: np.ndarray | None = None
        self.track_to_idx: dict[str, int] = {}
        self.release_dates: np.ndarray | None = None
        self.members: list = []
        self._store = None
        self._infer_emb: dict[tuple[str, int], np.ndarray] = {}

    # ------------------------------------------------------------------
    # modality source — subclasses override
    # ------------------------------------------------------------------
    def _load_modality(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (track_ids, L2-normed embeddings) for this tower's modality."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, train_df, track_metadata: pl.DataFrame | None = None, **kwargs: Any) -> None:
        if self.query_cache_root is None:
            raise ValueError(f"[{self.RECOMMENDER_NAME}] query_cache_root required")
        t0 = time.time()
        self.track_ids, mod_emb = self._load_modality()
        self.track_to_idx = {t: i for i, t in enumerate(self.track_ids)}
        if track_metadata is not None:
            self.release_dates = _build_release_dates(self.track_ids, track_metadata)

        self._store = load_query_store(self.query_cache_root)
        self._load_inference_queries()
        session_set = set(train_df["session_id"].to_list())
        q, pos = self._training_pairs(session_set)
        if q.shape[0] == 0:
            raise RuntimeError(
                f"[{self.RECOMMENDER_NAME}] no training pairs "
                f"(max_train_turn={self.max_train_turn}; check query_cache_root + "
                f"modality catalogue coverage)")

        device = "cuda" if _maybe_torch(self.use_gpu) is not None else "cpu"
        cfg = TrainConfig(
            d=self.d, hidden=self.hidden, epochs=self.epochs, lr=self.lr,
            tau=self.tau, swag_max_rank=self.swag_max_rank,
            swag_collect_every=self.swag_collect_every)
        track_t = torch.from_numpy(np.ascontiguousarray(mod_emb))
        member = train_member(q, pos, track_t, cfg, tower=self.TOWER, device=device)
        self.members = build_member_towers(member, track_t, swag_k=self.swag_k, device=device)
        print(f"[{self.RECOMMENDER_NAME}] fit in {time.time()-t0:.1f}s — "
              f"{len(self.track_ids)} tracks, {q.shape[0]} pairs, {len(self.members)} members")

    def _load_inference_queries(self) -> None:
        """(session,turn)→query_emb over ALL caches (incl blind/holdout), keyed
        for inference. Training pairs use the splitk-only `_store`; inference
        must also cover blind/holdout, whose dirs lack the 'splitk' token."""
        import glob as _glob
        from pathlib import Path as _Path
        out: dict[tuple[str, int], np.ndarray] = {}
        for d in sorted(_glob.glob(str(_Path(self.query_cache_root) / "dense_*query*"))):
            dp = _Path(d)
            if not (dp / "query_meta.parquet").exists() or not (dp / "query_embeddings.npy").exists():
                continue
            emb = np.asarray(np.load(dp / "query_embeddings.npy"), dtype=np.float32)
            meta = pl.read_parquet(dp / "query_meta.parquet",
                                   columns=["session_id", "turn_number"])
            for i, (sid, tn) in enumerate(zip(meta["session_id"].to_list(),
                                              meta["turn_number"].to_list())):
                out[(sid, int(tn))] = emb[i]
        self._infer_emb = out

    def _training_pairs(self, session_set: set[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """(query_emb, positive idx) pairs for train sessions, turns 1..max_train_turn.

        max_train_turn=1 → turn-1-only (~12k pairs, specialised but data-limited);
        higher thresholds add nearer-turn pairs (more data, mild distribution
        shift). Heads are always applied to turn-1 queries at inference.
        """
        qs: list[np.ndarray] = []
        pos: list[int] = []
        for (sid, turn), emb in self._store.emb_by_key.items():
            if sid not in session_set or turn < 1 or turn > self.max_train_turn:
                continue
            gt = self._store.gt_by_key.get((sid, turn))
            j = self.track_to_idx.get(gt) if gt is not None else None
            if j is None:
                continue
            qs.append(emb); pos.append(j)
        if not qs:
            return torch.empty(0, self._store.dim), torch.empty(0, dtype=torch.long)
        return torch.from_numpy(np.stack(qs)), torch.tensor(pos, dtype=torch.long)

    # ------------------------------------------------------------------
    # recommend
    # ------------------------------------------------------------------
    def recommend(
        self, context_df: pl.DataFrame, top_k: int = 200, remove_seen: bool = True,
        max_future_years: float | None = None, **kwargs: Any,
    ) -> pl.DataFrame:
        if not self.members:
            raise RuntimeError("Recommender not fitted")
        if "target_turn" not in context_df.columns:
            raise ValueError("context_df missing 'target_turn'. Use build_context_df().")
        if max_future_years is None:
            max_future_years = self.max_future_years

        meta = context_df.select(
            ["session_id", "user_id", "session_date", "target_turn"]
        ).unique(subset=["session_id"])

        ctx_map: dict[str, list[str]] = {}
        if "track_id" in context_df.columns and context_df.height > 0:
            grp = (context_df.filter(pl.col("track_id").is_not_null())
                   .group_by("session_id").agg(pl.col("track_id")))
            ctx_map = dict(zip(grp["session_id"].to_list(), grp["track_id"].to_list()))

        rows = meta.to_dicts()
        hit_rows: list[int] = []
        q_emb_list: list[np.ndarray] = []
        for ri, r in enumerate(rows):
            emb = self._infer_emb.get((r["session_id"], int(r["target_turn"])))
            if emb is not None:
                hit_rows.append(ri); q_emb_list.append(emb)

        out_tracks: list[list[str]] = [[] for _ in rows]
        out_scores: list[list[float]] = [[] for _ in rows]
        if hit_rows:
            Q = np.stack(q_emb_list).astype(np.float32)         # (H, dq)
            self._score_block(Q, rows, hit_rows, ctx_map, top_k, remove_seen,
                              float(max_future_years), out_tracks, out_scores)

        return pl.DataFrame(
            {"session_id": [r["session_id"] for r in rows],
             "user_id": [r["user_id"] for r in rows],
             "turn": [r["target_turn"] for r in rows],
             "track_ids": out_tracks, "scores": out_scores},
            schema={"session_id": pl.Utf8, "user_id": pl.Utf8, "turn": pl.Int64,
                    "track_ids": pl.List(pl.Utf8), "scores": pl.List(pl.Float64)},
        )

    def _score_block(self, Q, rows, hit_rows, ctx_map, top_k, remove_seen,
                     max_future_years, out_tracks, out_scores) -> None:
        """SWAG-mean projected-cosine scoring over the hit sessions, tiled."""
        n_tracks = len(self.track_ids)
        torch_mod = _maybe_torch(self.use_gpu)
        # per-member projected query matrices (H, d) + GPU track towers
        proj_q = [project_queries(mem, Q) for mem in self.members]   # list of (H, d)
        gpu_tow = None
        if torch_mod is not None:
            gpu_tow = [torch_mod.from_numpy(np.ascontiguousarray(m.proj_tower)).to("cuda")
                       for m in self.members]

        H = Q.shape[0]
        for s in range(0, H, self.block):
            e = min(s + self.block, H)
            if torch_mod is not None:
                acc = None
                for mi, tow in enumerate(gpu_tow):
                    qb = torch_mod.from_numpy(np.ascontiguousarray(proj_q[mi][s:e])).to("cuda")
                    sm = qb @ tow.T
                    acc = sm if acc is None else acc + sm
                S = (acc / len(self.members)).cpu().numpy()
            else:
                acc = np.zeros((e - s, n_tracks), dtype=np.float32)
                for mi, m in enumerate(self.members):
                    acc += proj_q[mi][s:e] @ m.proj_tower.T
                S = acc / len(self.members)

            for li, ri in enumerate(hit_rows[s:e]):
                r = rows[ri]
                scores = S[li].astype(np.float64)
                sd = parse_date(r["session_date"])
                if self.release_dates is not None and sd is not None:
                    cutoff = np.datetime64(sd, "D") + np.timedelta64(
                        int(max_future_years * 365), "D")
                    bad = (self.release_dates > cutoff) & ~np.isnat(self.release_dates)
                    scores[bad] = -np.inf
                if remove_seen:
                    for t in ctx_map.get(r["session_id"], []):
                        j = self.track_to_idx.get(t)
                        if j is not None:
                            scores[j] = -np.inf
                out_tracks[ri], out_scores[ri] = self._topk(scores, top_k, n_tracks)

        if gpu_tow is not None:
            torch_mod.cuda.empty_cache()

    def _topk(self, scores: np.ndarray, top_k: int, n_tracks: int):
        finite = int(np.isfinite(scores).sum())
        if finite == 0:
            return [], []
        k = min(top_k, finite)
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [self.track_ids[i] for i in idx], [float(scores[i]) for i in idx]

    # ------------------------------------------------------------------
    # persistence (params + projected members; query store reloaded on fit)
    # ------------------------------------------------------------------
    def _get_model_state(self) -> dict:
        import dataclasses
        return {
            "query_cache_root": self.query_cache_root,
            "d": self.d, "hidden": self.hidden, "epochs": self.epochs,
            "lr": self.lr, "tau": self.tau, "swag_k": self.swag_k,
            "swag_max_rank": self.swag_max_rank,
            "swag_collect_every": self.swag_collect_every,
            "max_train_turn": self.max_train_turn, "max_future_years": self.max_future_years,
            "block": self.block, "urm_mode": self.urm_mode, "use_gpu": self.use_gpu,
            "track_ids": self.track_ids, "track_to_idx": self.track_to_idx,
            "release_dates": self.release_dates,
            "members": [dataclasses.replace(m, gpu=None) for m in self.members],
        }

    def _set_model_state(self, state: dict) -> None:
        for k, v in state.items():
            setattr(self, k, v)
        # query store (inference query lookup) is reloaded lazily on next fit;
        # for export/load paths the caller refits or the store is rebuilt.
        # load() calls _set_model_state on a __new__'d instance (no __init__),
        # so _infer_emb may not exist yet — getattr instead of self._infer_emb.
        if self.query_cache_root is not None and not getattr(self, "_infer_emb", None):
            self._load_inference_queries()
