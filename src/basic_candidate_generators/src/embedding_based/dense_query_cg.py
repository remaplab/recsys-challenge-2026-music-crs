"""DenseQueryCG — query->track retrieval from frozen Qwen3 query/track caches.

Training-free intent retrieval: score the whole catalogue by
`cosine(query_emb, track_emb)`, ignoring item history. Complements the
continuation engines (CF / sequential), which miss new-artist targets.

Query embeddings are keyed by GLOBAL (session_id, turn_number) and live in the
per-bucket caches written by the emblib encode. We load *every* available query
cache for the model into one `{(sid, turn): row}` lookup at fit; this is safe —
the query text is model INPUT, not the label (GT lives separately) — and
sidesteps splitK bucket routing entirely.

Pipeline-native: subclasses `BaseRecommender`, exposes the standard
`recommend(context_df, ...)` signature so `run_inference` / `run_inference_dispatch`
(inference_mode="standard") drive it and attach gt afterwards. Sessions with no
cached query (cold / un-encoded) get empty candidate lists.
"""

from __future__ import annotations

import sys as _sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

_HERE = Path(__file__).resolve()
_sys.path.insert(0, str(_HERE.parent.parent))

from BaseRecommender import BaseRecommender  # noqa: E402

from recommenders.interactions import parse_date  # noqa: E402

from .emb_matrix import load_track_tower  # noqa: E402


def _build_release_dates(track_ids: np.ndarray, track_metadata: pl.DataFrame) -> np.ndarray:
    """datetime64[D] per track (tower order); NaT when missing/unparseable."""
    rd_map: dict[str, str] = {}
    if "release_date" in track_metadata.columns:
        for tid, rd in zip(
            track_metadata["track_id"].to_list(),
            track_metadata["release_date"].to_list(),
        ):
            rd_map[tid] = rd
    out = np.full(len(track_ids), np.datetime64("NaT", "D"), dtype="datetime64[D]")
    for i, tid in enumerate(track_ids):
        d = parse_date(rd_map.get(tid))
        if d is not None:
            out[i] = np.datetime64(d, "D")
    return out


class DenseQueryCG(BaseRecommender):
    RECOMMENDER_NAME = "DenseQueryCG"

    def __init__(
        self,
        track_emb_dir: str | None = None,     # dense_tracks_* dir
        query_cache_root: str | None = None,  # model folder holding dense_*query* dirs
        model_size: str = "qwen3_8b",
        urm_mode: str = "session",            # read by run_inference (no injection)
        max_future_years: float = 2.0,
        block: int = 512,                     # sessions per scoring tile
        use_gpu: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.track_emb_dir = track_emb_dir
        self.query_cache_root = query_cache_root
        self.model_size = model_size
        self.urm_mode = urm_mode
        self.max_future_years = max_future_years
        self.block = block
        self.use_gpu = use_gpu

        self.track_ids: np.ndarray | None = None
        self.track_emb: np.ndarray | None = None        # (n, d) L2-normed
        self.track_to_idx: dict[str, int] = {}
        self.release_dates: np.ndarray | None = None
        self.query_emb: np.ndarray | None = None        # (m, d) L2-normed
        self.query_key_to_row: dict[tuple[str, int], int] = {}

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, train_df, track_metadata: pl.DataFrame | None = None, **kwargs: Any) -> None:
        if self.track_emb_dir is None:
            raise ValueError(f"[{self.RECOMMENDER_NAME}] track_emb_dir required")
        t0 = time.time()
        self.track_ids, self.track_emb = load_track_tower(self.track_emb_dir)
        self.track_to_idx = {t: i for i, t in enumerate(self.track_ids)}
        if track_metadata is not None:
            self.release_dates = _build_release_dates(self.track_ids, track_metadata)
        self._load_query_caches()
        print(f"[{self.RECOMMENDER_NAME}] fit in {time.time()-t0:.1f}s — "
              f"{len(self.track_ids)} tracks, {len(self.query_key_to_row)} cached queries")

    def _load_query_caches(self) -> None:
        """Glob every query cache under the model folder → one global lookup."""
        if self.query_cache_root is None:
            raise ValueError(f"[{self.RECOMMENDER_NAME}] query_cache_root required")
        root = Path(self.query_cache_root)
        dirs = sorted(
            d for d in root.glob("dense_*query*")
            if (d / "query_meta.parquet").exists() and (d / "query_embeddings.npy").exists()
        )
        if not dirs:
            raise FileNotFoundError(
                f"[{self.RECOMMENDER_NAME}] no query caches under {root} "
                f"(expected dense_*query* dirs with query_embeddings.npy + query_meta.parquet). "
                f"Run the emblib encode --stages splitk/blind for {self.model_size}."
            )
        embs: list[np.ndarray] = []
        key_to_row: dict[tuple[str, int], int] = {}
        offset = 0
        for d in dirs:
            e = np.asarray(np.load(d / "query_embeddings.npy"), dtype=np.float32)
            meta = pl.read_parquet(d / "query_meta.parquet")
            sids = meta["session_id"].to_list()
            turns = meta["turn_number"].to_list()
            for i, (sid, tn) in enumerate(zip(sids, turns)):
                # global key; later buckets overwrite earlier on collision (none expected)
                key_to_row[(sid, int(tn))] = offset + i
            embs.append(e)
            offset += e.shape[0]
        self.query_emb = np.concatenate(embs, axis=0) if len(embs) > 1 else embs[0]
        self.query_key_to_row = key_to_row

    # ------------------------------------------------------------------
    # recommend
    # ------------------------------------------------------------------

    def recommend(
        self,
        context_df: pl.DataFrame,
        top_k: int = 200,
        remove_seen: bool = True,
        max_future_years: float | None = None,
        **kwargs: Any,
    ) -> pl.DataFrame:
        if self.track_emb is None:
            raise RuntimeError("Recommender not fitted")
        if "target_turn" not in context_df.columns:
            raise ValueError("context_df missing 'target_turn'. Use build_context_df().")
        if max_future_years is None:
            max_future_years = self.max_future_years

        session_meta = context_df.select(
            ["session_id", "user_id", "session_date", "target_turn"]
        ).unique(subset=["session_id"])

        # per-session context tracks (for seen removal)
        ctx_map: dict[str, list[str]] = {}
        if "track_id" in context_df.columns and context_df.height > 0:
            grouped = (
                context_df.filter(pl.col("track_id").is_not_null())
                .group_by("session_id").agg(pl.col("track_id"))
            )
            ctx_map = dict(zip(grouped["session_id"].to_list(),
                               grouped["track_id"].to_list()))

        rows = session_meta.to_dicts()
        # resolve which sessions have a cached query vector
        hit_rows: list[int] = []      # index into `rows`
        q_rows: list[int] = []        # row in self.query_emb
        for ri, r in enumerate(rows):
            qr = self.query_key_to_row.get((r["session_id"], int(r["target_turn"])))
            if qr is not None:
                hit_rows.append(ri)
                q_rows.append(qr)

        out_sid = [r["session_id"] for r in rows]
        out_uid = [r["user_id"] for r in rows]
        out_turn = [r["target_turn"] for r in rows]
        out_tracks: list[list[str]] = [[] for _ in rows]
        out_scores: list[list[float]] = [[] for _ in rows]

        if hit_rows:
            self._score_block(
                rows, hit_rows, q_rows, ctx_map, top_k, remove_seen,
                float(max_future_years), out_tracks, out_scores,
            )

        return pl.DataFrame(
            {"session_id": out_sid, "user_id": out_uid, "turn": out_turn,
             "track_ids": out_tracks, "scores": out_scores},
            schema={"session_id": pl.Utf8, "user_id": pl.Utf8, "turn": pl.Int64,
                    "track_ids": pl.List(pl.Utf8), "scores": pl.List(pl.Float64)},
        )

    def _score_block(self, rows, hit_rows, q_rows, ctx_map, top_k, remove_seen,
                     max_future_years, out_tracks, out_scores) -> None:
        """Tiled GEMM scoring over sessions that have a cached query vector."""
        track_emb = self.track_emb
        n_tracks = track_emb.shape[0]
        torch = _maybe_torch(self.use_gpu)
        ET_t = None
        if torch is not None:
            ET_t = torch.from_numpy(np.ascontiguousarray(track_emb.T)).to("cuda")

        for s in range(0, len(hit_rows), self.block):
            e = min(s + self.block, len(hit_rows))
            Q = self.query_emb[q_rows[s:e]]                      # (bb, d)
            if torch is not None:
                Qt = torch.from_numpy(np.ascontiguousarray(Q)).to("cuda")
                S = (Qt @ ET_t).cpu().numpy()                    # (bb, n)
            else:
                S = Q @ track_emb.T

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
                tracks, scs = self._topk(scores, top_k, n_tracks)
                out_tracks[ri] = tracks
                out_scores[ri] = scs

        if ET_t is not None:
            torch.cuda.empty_cache()

    def _topk(self, scores: np.ndarray, top_k: int, n_tracks: int):
        finite = int(np.isfinite(scores).sum())
        if finite == 0:
            return [], []
        k = min(top_k, finite)
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [self.track_ids[i] for i in idx], [float(scores[i]) for i in idx]

    # ------------------------------------------------------------------
    # persistence (store params/paths only; reload artefacts on load)
    # ------------------------------------------------------------------

    def _get_model_state(self) -> dict:
        return {
            "track_emb_dir": self.track_emb_dir,
            "query_cache_root": self.query_cache_root,
            "model_size": self.model_size, "urm_mode": self.urm_mode,
            "max_future_years": self.max_future_years, "block": self.block,
            "use_gpu": self.use_gpu,
        }

    def _set_model_state(self, state: dict) -> None:
        self.track_emb_dir = state.get("track_emb_dir")
        self.query_cache_root = state.get("query_cache_root")
        self.model_size = state.get("model_size", "qwen3_8b")
        self.urm_mode = state.get("urm_mode", "session")
        self.max_future_years = state.get("max_future_years", 2.0)
        self.block = state.get("block", 512)
        self.use_gpu = state.get("use_gpu", True)
        # lazy: artefacts reloaded on demand via fit() if needed


def _maybe_torch(use_gpu: bool):
    if not use_gpu:
        return None
    try:
        import torch
    except ImportError:
        return None
    return torch if torch.cuda.is_available() else None
