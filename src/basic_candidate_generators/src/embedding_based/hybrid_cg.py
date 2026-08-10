"""HybridCG — RRF fusion of five training-free content signals.

A single content-based candidate generator (no v1/v2). Five complementary
signals, each emitting a per-session ranked track list, fused by Reciprocal
Rank Fusion (`score(t) = Σ_s w_s / (k_rrf + rank_s(t))`):

  query text  ──► TFIDF        cosine(query, track-doc)         (sklearn TF-IDF)
              └─► BM25         BM25(query, track-doc)           (hand-rolled weights)
  context     ──► ICM          profile · W_cbf  (tag/artist/album/decade item-item)
  tracks      ├─► last_track   cos(track, last context track)   (Qwen-8B tower)
              └─► prev_n       Σ_{j≤n} cos(track, ctx_{-j})      (sum, not mean)

Self-contained: own TFIDF/BM25/ICM (mirrors recommenders.text_cg + interactions
but does not import them). Pure content → `train_df` is unused (no CF); this
matches the cold-user split where collaborative history is dead.

Pipeline-native text mode: driven by `run_inference_dispatch(inference_mode=
"text")` → `recommend_text(sess_info)`, where `sess_info` carries both the query
fields AND `ctx_tracks` per target session. Hot fusion + top-k run in Numba.

Default embeddings: frozen Qwen3-Embedding-8B track tower.
"""

from __future__ import annotations

import hashlib
import pickle
import sys as _sys
import time
from pathlib import Path
from typing import Any

import numba
import numpy as np
import polars as pl
import scipy.sparse as sp
from numba import njit, prange
from scipy.sparse import csr_matrix

_HERE = Path(__file__).resolve()
_sys.path.insert(0, str(_HERE.parent.parent))

from BaseRecommender import BaseRecommender  # noqa: E402

from recommenders.interactions import parse_date  # noqa: E402

from .emb_matrix import load_track_tower, sparsify_topk  # noqa: E402

_NAT = np.iinfo(np.int32).min   # release-day sentinel: "no date → never future"

import os as _os
_PROFILE = _os.environ.get("HYBRID_PROFILE", "") not in ("", "0")  # per-signal timers

from dataclasses import dataclass


@dataclass
class _EmbBackend:
    """One embedding tower contributing last / prev (/ query-emb) RRF signals.

    All towers share the same id_map index space (metadata order), so context
    track indices are valid against any backend. `emb_gpu` is the resident CUDA
    tensor (None on CPU → numpy path). `use_sess_qemb` toggles the query→track
    signal that reads `query_emb` from sess_info (only the base 8B tower, whose
    per-set query embeddings are attached by the inference dispatch).
    """
    size: str
    emb_cpu: np.ndarray
    emb_gpu: object
    w_last: float
    w_prev: float
    w_qemb: float
    use_sess_qemb: bool


# ---------------------------------------------------------------------------
# Text document builders
# ---------------------------------------------------------------------------

def _track_doc(row: dict) -> str:
    """Rich track document from metadata (name + artist + album + tags)."""
    parts: list[str] = []
    for col in ("track_name", "artist_name", "album_name", "tag_list"):
        v = row.get(col)
        if isinstance(v, list):
            parts.extend(str(x) for x in v if x)
        elif v:
            parts.append(str(v))
    return " ".join(parts)


def _query_text(row: dict) -> str:
    """Max query string: user_query + thought + listener_goal + profile prefs.

    Prefers a precomputed `query_text` (attached from the query-tower bundle,
    uniform across all inference sets); falls back to assembling from raw fields.
    """
    qt = row.get("query_text")
    if qt:
        return qt
    parts = [row.get("user_query") or "", row.get("user_thought") or ""]
    goal = row.get("conversation_goal")
    if isinstance(goal, dict):
        parts.append(goal.get("listener_goal") or "")
    prof = row.get("user_profile")
    if isinstance(prof, dict):
        parts.append(prof.get("preferred_musical_culture") or "")
        parts.append(prof.get("preferred_language") or "")
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------

@njit(cache=True)
def _rrf_fuse(ranked, weights, k_rrf, release_days, cutoffs,
              seen_flat, seen_ptr, n_tracks, top_k):
    """Reciprocal-Rank-Fusion of per-signal ranked lists, with seen/future masks.

    ranked      : (n_sess, n_sig, m) int64, track idx ranked per signal, -1 pad.
    weights     : (n_sig,) float32 RRF weights.
    release_days: (n_tracks,) int32 (days since epoch, _NAT = missing).
    cutoffs     : (n_sess,) int64 future cutoff in days (huge = no masking).
    seen_flat/seen_ptr: CSR of per-session seen track idx (removed from output).
    Returns (idx (n_sess, top_k) int64 [-1 pad], scores (n_sess, top_k) float32).
    """
    n_sess, n_sig, m = ranked.shape
    out_idx = np.full((n_sess, top_k), -1, np.int64)
    out_scr = np.zeros((n_sess, top_k), np.float32)

    acc = np.zeros(n_tracks, np.float32)
    inacc = np.zeros(n_tracks, np.uint8)
    seen = np.zeros(n_tracks, np.uint8)
    touched = np.empty(n_sig * m, np.int64)

    for s in range(n_sess):
        for p in range(seen_ptr[s], seen_ptr[s + 1]):
            seen[seen_flat[p]] = 1
        cutoff = cutoffs[s]
        cnt = 0
        for sig in range(n_sig):
            w = weights[sig]
            if w == 0.0:
                continue
            for rank in range(m):
                j = ranked[s, sig, rank]
                if j < 0:
                    break
                if seen[j]:
                    continue
                rd = release_days[j]
                if rd != _NAT and rd > cutoff:
                    continue
                if inacc[j] == 0:
                    inacc[j] = 1
                    touched[cnt] = j
                    cnt += 1
                acc[j] += w / (k_rrf + rank + 1.0)

        kk = top_k if top_k < cnt else cnt
        if cnt > 0:
            ts = np.empty(cnt, np.float32)
            ti = np.empty(cnt, np.int64)
            for i in range(cnt):
                ti[i] = touched[i]
                ts[i] = acc[touched[i]]
            order = np.argsort(-ts)
            for r in range(kk):
                o = order[r]
                out_idx[s, r] = ti[o]
                out_scr[s, r] = ts[o]

        for i in range(cnt):
            j = touched[i]
            acc[j] = 0.0
            inacc[j] = 0
        for p in range(seen_ptr[s], seen_ptr[s + 1]):
            seen[seen_flat[p]] = 0

    return out_idx, out_scr


@njit(parallel=True, cache=True)
def _text_inv_topk(q_data, q_idx, q_ptr, p_data, p_idx, p_ptr, n_tracks, top_k):
    """Inverted-index query→item top-k for sparse text signals (TFIDF / BM25).

    query CSR    : q_data/q_idx/q_ptr  (ns × vocab), per-query nonzero weights.
    postings CSR : p_data/p_idx/p_ptr  (vocab × n), row=token → (track, weight).
    Equivalent to top-k over (Q @ item_mat.T) but only touches tracks that
    share a token with the query (queries are very sparse → big win over the
    dense sparse-mm path). Returns (ns, top_k) int64 track idx, -1 pad.

    Parallel over thread chunks; each thread owns its accumulator scratch.
    """
    ns = q_ptr.shape[0] - 1
    out = np.full((ns, top_k), -1, np.int64)
    nthreads = numba.get_num_threads()
    chunk = (ns + nthreads - 1) // nthreads
    for c in prange(nthreads):
        acc = np.zeros(n_tracks, np.float32)
        inacc = np.zeros(n_tracks, np.uint8)
        touched = np.empty(n_tracks, np.int64)
        s0 = c * chunk
        s1 = min(s0 + chunk, ns)
        for s in range(s0, s1):
            cnt = 0
            for kk in range(q_ptr[s], q_ptr[s + 1]):
                qv = q_data[kk]
                col = q_idx[kk]
                for pp in range(p_ptr[col], p_ptr[col + 1]):
                    tr = p_idx[pp]
                    if inacc[tr] == 0:
                        inacc[tr] = 1
                        touched[cnt] = tr
                        cnt += 1
                    acc[tr] += qv * p_data[pp]

            kk_top = top_k if top_k < cnt else cnt
            if cnt > 0:
                ts = np.empty(cnt, np.float32)
                ti = np.empty(cnt, np.int64)
                for i in range(cnt):
                    ti[i] = touched[i]
                    ts[i] = acc[touched[i]]
                order = np.argsort(-ts)
                for r in range(kk_top):
                    o = order[r]
                    if ts[o] > 0.0:
                        out[s, r] = ti[o]

            for i in range(cnt):
                j = touched[i]
                acc[j] = 0.0
                inacc[j] = 0
    return out


# ---------------------------------------------------------------------------
# HybridCG
# ---------------------------------------------------------------------------

class HybridCG(BaseRecommender):
    RECOMMENDER_NAME = "HybridCG"

    def __init__(
        self,
        track_emb_dir: str | None = None,         # Qwen-8B dense_tracks_* dir
        model_size: str = "qwen3_8b",
        urm_mode: str = "session",                # read by the inference dispatch
        k_rrf: int = 60,
        top_k_per_signal: int = 600,
        prev_decay: float = 0.5,
        w_tfidf: float = 1.0,
        w_bm25: float = 1.0,
        w_icm: float = 1.0,
        w_last: float = 1.0,
        w_prev: float = 1.0,
        w_qemb: float = 1.0,
        tfidf_max_features: int = 50_000,
        tfidf_ngram_max: int = 2,
        bm25_max_features: int = 50_000,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        icm_k: int = 200,
        icm_cache_k: int = 400,                   # W_cbf cache neighbours (trimmed to icm_k)
        cache_dir: str | None = None,             # catalogue-global artifact cache
        max_future_years: float = 2.0,
        block: int = 512,
        use_gpu: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.track_emb_dir = track_emb_dir
        self.model_size = model_size
        self.urm_mode = urm_mode
        self.k_rrf = k_rrf
        self.top_k_per_signal = top_k_per_signal
        self.prev_decay = prev_decay
        self.w_tfidf = w_tfidf
        self.w_bm25 = w_bm25
        self.w_icm = w_icm
        self.w_last = w_last
        self.w_prev = w_prev
        self.w_qemb = w_qemb
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_max = tfidf_ngram_max
        self.bm25_max_features = bm25_max_features
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.icm_k = icm_k
        self.icm_cache_k = icm_cache_k
        self.cache_dir = cache_dir
        self.max_future_years = max_future_years
        self.block = block
        self.use_gpu = use_gpu

        # fitted artefacts
        self.track_ids: np.ndarray | None = None
        self.track_to_idx: dict[str, int] = {}
        self.n_tracks = 0
        self.release_days: np.ndarray | None = None       # (n,) int32
        self.track_emb: np.ndarray | None = None          # (n, d) L2-normed
        self._tfidf_vec = None
        self._tfidf_mat: csr_matrix | None = None         # (n, vocab)
        self._bm25_vec = None
        self._bm25_B: csr_matrix | None = None            # (n, vocab)
        self.W_cbf: csr_matrix | None = None              # (n, n) tag item-item
        # Text scoring backends, selected by device at fit:
        #   CUDA  → GPU sparse-mm (_tfidf_gpu/_bm25_gpu)
        #   CPU   → numba inverted index (_tfidf_post/_bm25_post)
        self._tfidf_post: tuple | None = None             # (data, idx, indptr)
        self._bm25_post: tuple | None = None              # (data, idx, indptr)
        # Optional cross-trial cache: {(session_id, turn): rk_row} injected by
        # the tuner so tfidf (signal invariant across trials) is computed once.
        self._tfidf_rk_inject: dict | None = None
        # GPU residents (uploaded once per fit; None when no CUDA)
        self._torch = None
        self._tower_gpu = None                            # (n, d) dense
        self._tfidf_gpu = None                            # (n, vocab) sparse csr
        self._bm25_gpu = None                             # (n, vocab) sparse csr
        # Embedding towers feeding last/prev(/qemb) RRF signals (built in _to_gpu).
        self._emb_backends: list[_EmbBackend] = []

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, train_df, track_metadata: pl.DataFrame | None = None, **kwargs: Any) -> None:
        if track_metadata is None:
            raise ValueError(f"[{self.RECOMMENDER_NAME}] track_metadata required")
        if self.track_emb_dir is None:
            raise ValueError(f"[{self.RECOMMENDER_NAME}] track_emb_dir required")
        t0 = time.time()

        # catalogue index (metadata order)
        self.track_ids = track_metadata["track_id"].to_numpy()
        self.track_to_idx = {t: i for i, t in enumerate(self.track_ids)}
        self.n_tracks = len(self.track_ids)
        sig = self._meta_sig()

        # Catalogue-global artefacts (HP- and fold-independent → cached once and
        # reused by every trial/fold). Only the searched bits are recomputed:
        # BM25 reweight (k1/b) from cached counts, W_cbf trim (icm_k) from cache.
        self.release_days = self._cached_release_days(track_metadata, sig)
        self._tfidf_vec, self._tfidf_mat = self._cached_tfidf(track_metadata, sig)
        self._bm25_vec, counts, idf, dl = self._cached_bm25(track_metadata, sig)
        self._bm25_B = self._bm25_reweight(counts, idf, dl)
        self.W_cbf = sparsify_topk(self._cached_wcbf(track_metadata, sig), self.icm_k)
        self.track_emb = self._cached_tower(sig)
        self._to_gpu()

        cache = "on" if self.cache_dir else "off"
        print(f"[{self.RECOMMENDER_NAME}] fit in {time.time()-t0:.1f}s (cache={cache}) — "
              f"{self.n_tracks} tracks, tfidf vocab={self._tfidf_mat.shape[1]}, "
              f"bm25 vocab={self._bm25_B.shape[1]}, W_cbf nnz={self.W_cbf.nnz}")

    # ------------------------------------------------------------------
    # caching: catalogue-global artefacts keyed by metadata signature
    # ------------------------------------------------------------------

    def _cdir(self) -> Path:
        p = Path(self.cache_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _meta_sig(self) -> str:
        """12-hex signature of the track-id ordering → invalidates stale caches."""
        h = hashlib.sha1()
        for t in self.track_ids:
            h.update(str(t).encode())
        return h.hexdigest()[:12]

    def _track_docs(self, meta: pl.DataFrame) -> list[str]:
        docs = [""] * self.n_tracks
        for row in meta.iter_rows(named=True):
            docs[self.track_to_idx[row["track_id"]]] = _track_doc(row)
        return docs

    def _cached_release_days(self, meta: pl.DataFrame, sig: str) -> np.ndarray:
        f = self._cdir() / f"release_{sig}.npy" if self.cache_dir else None
        if f is not None and f.exists():
            return np.load(f)
        days = np.full(self.n_tracks, _NAT, dtype=np.int32)
        for tid, rd in zip(meta["track_id"].to_list(), meta["release_date"].to_list()):
            d = parse_date(rd)
            if d is not None:
                days[self.track_to_idx[tid]] = np.datetime64(d, "D").astype("int64")
        if f is not None:
            np.save(f, days)
        return days

    def _cached_tfidf(self, meta: pl.DataFrame, sig: str):
        key = f"tfidf_{sig}_f{self.tfidf_max_features}_n{self.tfidf_ngram_max}"
        if self.cache_dir:
            vp, mp = self._cdir() / f"{key}.pkl", self._cdir() / f"{key}.npz"
            if vp.exists() and mp.exists():
                return pickle.loads(vp.read_bytes()), sp.load_npz(mp).astype(np.float32)
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=self.tfidf_max_features,
                              ngram_range=(1, self.tfidf_ngram_max), sublinear_tf=True)
        mat = vec.fit_transform(self._track_docs(meta)).astype(np.float32)
        if self.cache_dir:
            (self._cdir() / f"{key}.pkl").write_bytes(pickle.dumps(vec))
            sp.save_npz(self._cdir() / f"{key}.npz", mat)
        return vec, mat

    def _cached_bm25(self, meta: pl.DataFrame, sig: str):
        """Return (vectorizer, count matrix, idf, dl) — all k1/b-independent."""
        key = f"bm25_{sig}_f{self.bm25_max_features}"
        if self.cache_dir:
            vp = self._cdir() / f"{key}.pkl"
            cp = self._cdir() / f"{key}_counts.npz"
            ap = self._cdir() / f"{key}_idfdl.npz"
            if vp.exists() and cp.exists() and ap.exists():
                a = np.load(ap)
                return (pickle.loads(vp.read_bytes()),
                        sp.load_npz(cp).astype(np.float32), a["idf"], a["dl"])
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(max_features=self.bm25_max_features)
        tf = cv.fit_transform(self._track_docs(meta)).astype(np.float32)
        n = tf.shape[0]
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        idf = np.log((n - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
        dl = np.asarray(tf.sum(axis=1)).ravel().astype(np.float32)
        if self.cache_dir:
            (self._cdir() / f"{key}.pkl").write_bytes(pickle.dumps(cv))
            sp.save_npz(self._cdir() / f"{key}_counts.npz", tf)
            np.savez(self._cdir() / f"{key}_idfdl.npz", idf=idf, dl=dl)
        return cv, tf, idf, dl

    def _bm25_reweight(self, counts: csr_matrix, idf: np.ndarray, dl: np.ndarray) -> csr_matrix:
        """B[i,t] = tf*(k1+1)/(tf+k1*dl_norm_i) * idf[t] — cheap per-trial reweight."""
        dl_norm = (1.0 - self.bm25_b + self.bm25_b * dl / (dl.mean() or 1.0)).astype(np.float32)
        coo = counts.tocoo()
        vals = (coo.data * (self.bm25_k1 + 1.0)
                / (coo.data + self.bm25_k1 * dl_norm[coo.row]) * idf[coo.col])
        return sp.csr_matrix((vals, (coo.row, coo.col)), shape=counts.shape, dtype=np.float32)

    def _cached_wcbf(self, meta: pl.DataFrame, sig: str) -> csr_matrix:
        """W_cbf at icm_cache_k (trimmed to icm_k at fit). Build once, cache."""
        key = f"wcbf_{sig}_k{self.icm_cache_k}"
        if self.cache_dir:
            p = self._cdir() / f"{key}.npz"
            if p.exists():
                return sp.load_npz(p).astype(np.float32)
        W = self._sparse_item_knn(self._build_icm_matrix(meta), self.icm_cache_k)
        if self.cache_dir:
            sp.save_npz(self._cdir() / f"{key}.npz", W)
        return W

    def _cached_tower(self, sig: str) -> np.ndarray:
        return self._cached_tower_for(self.track_emb_dir, self.model_size, sig)

    def _cached_tower_for(self, track_emb_dir: str, model_size: str, sig: str) -> np.ndarray:
        """Load a Qwen track tower, remapped into id_map (metadata) order.

        Cache is keyed by (metadata signature, model_size), so several Qwen
        sizes coexist in the same cache_dir without collision.
        """
        key = f"tower_{sig}_{model_size}"
        if self.cache_dir:
            p = self._cdir() / f"{key}.npy"
            if p.exists():
                return np.load(p)
        ids, emb = load_track_tower(track_emb_dir)
        out = np.zeros((self.n_tracks, emb.shape[1]), dtype=np.float32)
        for row, tid in enumerate(ids):
            j = self.track_to_idx.get(tid)
            if j is not None:
                out[j] = emb[row]
        if self.cache_dir:
            np.save(self._cdir() / f"{key}.npy", out)
        return out

    def _build_icm_matrix(self, meta: pl.DataFrame) -> csr_matrix:
        """L2-normed tag/artist/album/decade ICM (deterministic sorted-unique cols)."""
        n = self.n_tracks
        tid_df = pl.DataFrame({"track_id": list(self.track_to_idx.keys()),
                               "ti": list(self.track_to_idx.values())})
        blocks: list[csr_matrix] = []

        def _onehot(col: str) -> None:
            df = (meta.select(["track_id", col]).explode(col).drop_nulls()
                  .join(tid_df, on="track_id"))
            if df.is_empty():
                return
            uniq = df.select(col).unique().sort(col).with_row_index("fi")  # deterministic
            df = df.join(uniq, on=col)
            r, c = df["ti"].to_numpy(), df["fi"].to_numpy().astype(np.int32)
            blocks.append(csr_matrix((np.ones(len(r), np.float32), (r, c)),
                                     shape=(n, uniq.height), dtype=np.float32))

        for col in ("tag_list", "artist_id", "album_id"):
            _onehot(col)

        rd = (meta.select(["track_id", "release_date"]).drop_nulls().join(tid_df, on="track_id")
              .with_columns(pl.col("release_date").str.slice(0, 4)
                            .cast(pl.Int32, strict=False).alias("y"))
              .filter(pl.col("y").is_not_null() & (pl.col("y") > 0))
              .with_columns(((pl.col("y") // 10) * 10).alias("dec")))
        if rd.height > 0:
            uniq = rd.select("dec").unique().sort("dec").with_row_index("fi")
            rd = rd.join(uniq, on="dec")
            r, c = rd["ti"].to_numpy(), rd["fi"].to_numpy().astype(np.int32)
            blocks.append(csr_matrix((np.ones(len(r), np.float32), (r, c)),
                                     shape=(n, uniq.height), dtype=np.float32))

        from sklearn.preprocessing import normalize
        return normalize(sp.hstack(blocks, format="csr").astype(np.float32),
                         norm="l2", axis=1).tocsr()

    def _sparse_item_knn(self, icm_norm: csr_matrix, k: int) -> csr_matrix:
        """Top-k cosine item-item over L2-normed sparse ICM (batched, self excl)."""
        n = icm_norm.shape[0]
        k = min(k, n - 1)
        rows, cols, vals = [], [], []
        icmT = icm_norm.T.tocsr()
        for a in range(0, n, self.block):
            b = min(a + self.block, n)
            S = (icm_norm[a:b] @ icmT).toarray()
            S[np.arange(b - a), np.arange(a, b)] = -1.0          # mask self
            top = np.argpartition(-S, k, axis=1)[:, :k]
            tv = np.take_along_axis(S, top, axis=1)
            for i in range(b - a):
                pos = tv[i] > 0
                if pos.any():
                    cc = top[i][pos]
                    rows.append(np.full(cc.size, a + i, np.int64))
                    cols.append(cc.astype(np.int64))
                    vals.append(tv[i][pos].astype(np.float32))
        if not rows:
            return csr_matrix((n, n), dtype=np.float32)
        return csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n, n), dtype=np.float32)

    # ------------------------------------------------------------------
    # GPU residents (uploaded once per fit, reused across all turn-calls)
    # ------------------------------------------------------------------

    def _to_gpu(self) -> None:
        """Pick text/emb backends by device.

        CUDA: upload tower + tfidf/bm25 sparse item matrices (GPU sparse-mm).
        CPU : build numba inverted-index postings (≫ scipy on CPU); no GPU.
        """
        torch = _maybe_torch(self.use_gpu)
        self._torch = torch
        if torch is None:
            self._tower_gpu = self._tfidf_gpu = self._bm25_gpu = None
            self._tfidf_post = self._text_postings(self._tfidf_mat)
            self._bm25_post = self._text_postings(self._bm25_B)
            self._emb_backends = self._build_emb_backends()
            return
        self._tower_gpu = torch.from_numpy(np.ascontiguousarray(self.track_emb)).to("cuda")
        self._tfidf_gpu = self._csr_to_gpu(self._tfidf_mat, torch)
        self._bm25_gpu = self._csr_to_gpu(self._bm25_B, torch)
        self._tfidf_post = self._bm25_post = None
        self._emb_backends = self._build_emb_backends()

    def _build_emb_backends(self) -> list[_EmbBackend]:
        """Embedding towers feeding last/prev/qemb signals. Base = 8B only;
        subclasses append more Qwen sizes (their towers must already be loaded +
        uploaded)."""
        return [_EmbBackend(
            self.model_size, self.track_emb, self._tower_gpu,
            self.w_last, self.w_prev, self.w_qemb, use_sess_qemb=True,
        )]

    def _csr_to_gpu(self, M: csr_matrix, torch):
        M = M.tocsr()
        return torch.sparse_csr_tensor(
            torch.from_numpy(M.indptr.astype(np.int64)),
            torch.from_numpy(M.indices.astype(np.int64)),
            torch.from_numpy(M.data.astype(np.float32)),
            size=M.shape, device="cuda")

    @staticmethod
    def _text_postings(item_mat: csr_matrix) -> tuple:
        """Transpose an item matrix (n × vocab) to inverted-index postings.

        Returns (data, idx, indptr) of the (vocab × n) CSR — row=token, cols=
        track, value=item weight — typed for the `_text_inv_topk` numba kernel.
        """
        post = item_mat.T.tocsr()
        return (post.data.astype(np.float32),
                post.indices.astype(np.int64),
                post.indptr.astype(np.int64))

    # ------------------------------------------------------------------
    # recommend (text mode)
    # ------------------------------------------------------------------

    def recommend_text(self, sess_info: pl.DataFrame, top_k: int = 100,
                       remove_seen: bool = True) -> pl.DataFrame:
        if self.track_emb is None:
            raise RuntimeError("Recommender not fitted")
        prof = _PROFILE
        tmr = {}

        def _lap(name, t):
            if prof:
                tmr[name] = tmr.get(name, 0.0) + (time.time() - t)

        rows = sess_info.to_dicts()
        ns = len(rows)
        m = self.top_k_per_signal
        t2i = self.track_to_idx

        # per-session context indices (ordered)
        ctx_idx: list[np.ndarray] = []
        for r in rows:
            cs = [t2i[t] for t in (r.get("ctx_tracks") or []) if t in t2i]
            ctx_idx.append(np.asarray(cs, dtype=np.int64))

        # ── signal 1: tfidf (query → track docs) ──
        # Cross-trial cache injected by the tuner short-circuits the whole
        # tfidf compute (signal is invariant across trials → computed once).
        queries = [_query_text(r) for r in rows]
        t = time.time()
        if self._tfidf_rk_inject is not None:
            rk_tfidf = self._gather_injected_rk(rows, m)
        else:
            Qt = self._tfidf_vec.transform(queries).tocsr()
            rk_tfidf = self._text_topk(Qt, self._tfidf_gpu, self._tfidf_post, m)
        _lap("tfidf", t)
        # ── signal 2: bm25 (binary query terms) ──
        t = time.time()
        Qb = (self._bm25_vec.transform(queries) > 0).tocsr()
        rk_bm25 = self._text_topk(Qb, self._bm25_gpu, self._bm25_post, m, binary=True)
        _lap("bm25", t)

        # ── signal 3: ICM (binary profile over ctx tracks · W_cbf) ──
        t = time.time()
        rk_icm = self._icm_topk(ctx_idx, m)
        _lap("icm", t)

        # ── signals 4..: embedding towers (last / prev / query-emb) ──
        # One {last, prev, qemb} set per backend tower (8B base; subclasses add
        # more Qwen sizes, all fused into the same RRF). See `_emb_signals`.
        t = time.time()
        sig_rk = [rk_tfidf, rk_bm25, rk_icm]
        sig_w = [self.w_tfidf, self.w_bm25, self.w_icm]
        for rk, w in self._emb_signals(rows, ctx_idx, ns):
            sig_rk.append(rk)
            sig_w.append(w)
        _lap("emb", t)

        # ── RRF fuse ──
        t = time.time()
        ranked = np.stack(sig_rk, axis=1)            # (ns, n_sig, m)
        weights = np.array(sig_w, np.float32)

        seen_flat, seen_ptr = self._seen_csr(ctx_idx, remove_seen, ns)
        cutoffs = self._future_cutoffs(rows)
        idx, scr = _rrf_fuse(ranked, weights, float(self.k_rrf), self.release_days,
                             cutoffs, seen_flat, seen_ptr, self.n_tracks, top_k)
        _lap("rrf", t)
        if prof:
            print(f"[HybridCG.profile] ns={ns} m={m} " +
                  " ".join(f"{k}={v:.1f}s" for k, v in tmr.items()), flush=True)

        return self._to_recs_df(rows, idx, scr)

    # ----- signal helpers (return (ns, m) int64 ranked idx, -1 pad) -----

    def _text_topk(self, Q: csr_matrix, item_gpu, postings, m: int,
                   binary: bool = False) -> np.ndarray:
        """Query→item top-k. GPU sparse-mm when CUDA, else numba inverted index."""
        if self._torch is not None and item_gpu is not None:
            return self._sparse_topk_gpu(Q.astype(np.float32), item_gpu, m)
        p_data, p_idx, p_ptr = postings
        q_data = np.ones(Q.nnz, np.float32) if binary else Q.data.astype(np.float32)
        return _text_inv_topk(q_data, Q.indices.astype(np.int64),
                              Q.indptr.astype(np.int64), p_data, p_idx, p_ptr,
                              self.n_tracks, m)

    def _sparse_topk_gpu(self, Q: csr_matrix, item_gpu, m: int) -> np.ndarray:
        """scores = item_gpu(n×vocab) @ Qᵀ(vocab×blk) on CUDA; top-k over n (dim=0)."""
        torch = self._torch
        ns = Q.shape[0]
        n = item_gpu.shape[0]
        k = min(m, n)
        out = np.full((ns, m), -1, np.int64)
        qn = np.asarray((Q != 0).sum(axis=1)).ravel()
        Q = Q.tocsr()
        for a in range(0, ns, self.block):
            b = min(a + self.block, ns)
            qd = torch.from_numpy(np.ascontiguousarray(
                Q[a:b].toarray().T.astype(np.float32))).to("cuda")     # (vocab, blk)
            S = torch.sparse.mm(item_gpu, qd)                           # (n, blk) dense
            tv, ti = torch.topk(S, k, dim=0)                           # (k, blk)
            tv = tv.cpu().numpy()
            ti = ti.cpu().numpy()
            del qd, S
            for i in range(b - a):
                if qn[a + i] == 0:
                    continue
                valid = tv[:, i] > 0
                cc = ti[:, i][valid]
                out[a + i, :cc.size] = cc
        return out

    def _gather_injected_rk(self, rows: list[dict], m: int) -> np.ndarray:
        """Assemble (ns, m) tfidf rk from the injected cache.

        `_tfidf_rk_inject` is `(key2row, mat)`: a `{(session_id, turn): row_idx}`
        map and an `(N, K)` int rk matrix (K ≥ any trial's top_k_per_signal).
        """
        ns = len(rows)
        out = np.full((ns, m), -1, np.int64)
        key2row, mat = self._tfidf_rk_inject
        kk = min(m, mat.shape[1])
        for i, r in enumerate(rows):
            ri = key2row.get((r["session_id"], r["turn_number"]))
            if ri is not None:
                out[i, :kk] = mat[ri, :kk]
        return out

    def tfidf_rk_table(self, queries: list[str], top_k: int) -> np.ndarray:
        """Top-k tfidf track rankings for a list of query strings (tuner cache).

        Returns (len(queries), top_k) int64 ranked track idx, -1 pad — the same
        rk_tfidf the recommend path produces, computed once for reuse.
        """
        Qt = self._tfidf_vec.transform(queries).tocsr()
        return self._text_topk(Qt, self._tfidf_gpu, self._tfidf_post, top_k)

    def _emb_signals(self, rows: list[dict], ctx_idx: list[np.ndarray],
                     ns: int) -> list[tuple[np.ndarray, float]]:
        """RRF (ranked, weight) tuples for every embedding backend tower.

        Per backend: last-track cos, decay-weighted prev-context cos, and (8B
        base only) query-emb→track cos. `prev` = decay-weighted sum over ALL
        prior ctx tracks (newest weight 1, older × prev_decay**age); decay→0
        recovers last-only, decay→1 a flat context sum. Context indices are
        shared id_map space, valid against every backend's tower.
        """
        m = self.top_k_per_signal
        out: list[tuple[np.ndarray, float]] = []
        for be in self._emb_backends:
            d = be.emb_cpu.shape[1]
            q_last = np.zeros((ns, d), np.float32)
            q_prev = np.zeros((ns, d), np.float32)
            v_ctx = np.zeros(ns, bool)
            for i, cs in enumerate(ctx_idx):
                if cs.size:
                    q_last[i] = be.emb_cpu[cs[-1]]
                    ages = np.arange(cs.size - 1, -1, -1, dtype=np.float32)  # newest age 0
                    w = (self.prev_decay ** ages).astype(np.float32)
                    q_prev[i] = (be.emb_cpu[cs] * w[:, None]).sum(axis=0)
                    v_ctx[i] = True
            out.append((self._emb_topk(q_last, v_ctx, m, be.emb_gpu, be.emb_cpu), be.w_last))
            out.append((self._emb_topk(q_prev, v_ctx, m, be.emb_gpu, be.emb_cpu), be.w_prev))

            if be.use_sess_qemb:
                q_qemb = np.zeros((ns, d), np.float32)
                v_qemb = np.zeros(ns, bool)
                for i, r in enumerate(rows):
                    qe = r.get("query_emb")
                    if qe is not None:
                        vec = np.asarray(qe, dtype=np.float32)
                        if vec.shape[0] == d:
                            q_qemb[i] = vec
                            v_qemb[i] = True
                out.append((self._emb_topk(q_qemb, v_qemb, m, be.emb_gpu, be.emb_cpu),
                            be.w_qemb))
        return out

    def _emb_topk(self, qvecs: np.ndarray, valid: np.ndarray, m: int,
                  tower_gpu: object = None, tower_cpu: np.ndarray | None = None) -> np.ndarray:
        """cos(query, tower) top-k. Defaults to the 8B base tower; pass a
        backend's (gpu tensor, cpu matrix) to score against another Qwen size.
        Reuses the persistent resident tensors (no re-upload)."""
        if tower_cpu is None:
            tower_cpu = self.track_emb
        if tower_gpu is None:
            tower_gpu = self._tower_gpu
        ns = qvecs.shape[0]
        n = tower_cpu.shape[0]
        k = min(m, n)
        out = np.full((ns, m), -1, np.int64)
        torch = self._torch
        for a in range(0, ns, self.block):
            b = min(a + self.block, ns)
            blk = qvecs[a:b]
            if torch is not None and tower_gpu is not None:
                qd = torch.from_numpy(np.ascontiguousarray(blk)).to("cuda")  # (blk, d)
                S = qd @ tower_gpu.T                                          # (blk, n)
                _, ti = torch.topk(S, k, dim=1)
                ti = ti.cpu().numpy()
                del qd, S
            else:
                S = blk @ tower_cpu.T
                ti = np.argpartition(-S, k - 1, axis=1)[:, :k]
                order = np.argsort(-np.take_along_axis(S, ti, axis=1), axis=1)
                ti = np.take_along_axis(ti, order, axis=1)
            for i in range(b - a):
                if valid[a + i]:
                    out[a + i] = ti[i][:m]
        return out

    def _icm_topk(self, ctx_idx: list[np.ndarray], m: int) -> np.ndarray:
        ns = len(ctx_idx)
        out = np.full((ns, m), -1, np.int64)
        if self.W_cbf is None or self.W_cbf.nnz == 0:
            return out
        rows_, cols_ = [], []
        for i, cs in enumerate(ctx_idx):
            if cs.size:
                rows_.append(np.full(cs.size, i, np.int64))
                cols_.append(cs)
        if not rows_:
            return out
        P = csr_matrix((np.ones(sum(c.size for c in cols_), np.float32),
                        (np.concatenate(rows_), np.concatenate(cols_))),
                       shape=(ns, self.n_tracks))
        R = (P @ self.W_cbf).tocsr()                       # (ns, n) sparse
        for i in range(ns):
            s, e = R.indptr[i], R.indptr[i + 1]
            if e == s:
                continue
            data, cols = R.data[s:e], R.indices[s:e]
            k = min(m, data.size)
            top = np.argpartition(-data, k - 1)[:k]
            top = top[np.argsort(-data[top])]
            out[i, :k] = cols[top]
        return out

    # ----- masking / output helpers -----

    def _seen_csr(self, ctx_idx, remove_seen, ns):
        if not remove_seen:
            return np.empty(0, np.int64), np.zeros(ns + 1, np.int64)
        ptr = np.zeros(ns + 1, np.int64)
        for i, cs in enumerate(ctx_idx):
            ptr[i + 1] = ptr[i] + cs.size
        flat = np.concatenate(ctx_idx) if ptr[-1] else np.empty(0, np.int64)
        return flat.astype(np.int64), ptr

    def _future_cutoffs(self, rows) -> np.ndarray:
        big = np.iinfo(np.int64).max
        out = np.full(len(rows), big, np.int64)
        span = int(self.max_future_years * 365)
        for i, r in enumerate(rows):
            d = parse_date(r.get("session_date"))
            if d is not None:
                out[i] = np.datetime64(d, "D").astype("int64") + span
        return out

    def _to_recs_df(self, rows, idx, scr) -> pl.DataFrame:
        out_t, out_s = [], []
        for i in range(len(rows)):
            valid = idx[i] >= 0
            out_t.append([self.track_ids[j] for j in idx[i][valid]])
            out_s.append([float(x) for x in scr[i][valid]])
        return pl.DataFrame(
            {"session_id": [r["session_id"] for r in rows],
             "user_id": [r["user_id"] for r in rows],
             "turn": [r["turn_number"] for r in rows],
             "track_ids": out_t, "scores": out_s,
             "gt_track_id": [r.get("track_id") for r in rows]},
            schema={"session_id": pl.Utf8, "user_id": pl.Utf8, "turn": pl.Int64,
                    "track_ids": pl.List(pl.Utf8), "scores": pl.List(pl.Float64),
                    "gt_track_id": pl.Utf8},
        )

    def recommend(self, *args, **kwargs):  # noqa: D401 - standard mode unsupported
        raise NotImplementedError(
            f"{self.RECOMMENDER_NAME} is text-mode; use recommend_text "
            f"(inference_mode='text')."
        )

    # ------------------------------------------------------------------
    # persistence (params only; refit to restore artefacts)
    # ------------------------------------------------------------------

    def _get_model_state(self) -> dict:
        return {
            "track_emb_dir": self.track_emb_dir, "model_size": self.model_size,
            "urm_mode": self.urm_mode, "k_rrf": self.k_rrf,
            "top_k_per_signal": self.top_k_per_signal, "prev_decay": self.prev_decay,
            "w_tfidf": self.w_tfidf, "w_bm25": self.w_bm25, "w_icm": self.w_icm,
            "w_last": self.w_last, "w_prev": self.w_prev, "w_qemb": self.w_qemb,
            "tfidf_max_features": self.tfidf_max_features,
            "tfidf_ngram_max": self.tfidf_ngram_max,
            "bm25_max_features": self.bm25_max_features,
            "bm25_k1": self.bm25_k1, "bm25_b": self.bm25_b, "icm_k": self.icm_k,
            "icm_cache_k": self.icm_cache_k, "cache_dir": self.cache_dir,
            "max_future_years": self.max_future_years,
            "block": self.block, "use_gpu": self.use_gpu,
        }

    def _set_model_state(self, state: dict) -> None:
        for k, v in state.items():
            if k != "recommender_name":
                setattr(self, k, v)
        # optional cross-trial tuner injection (hybrid_cg.py:316) — never
        # persisted, always unset after load (mirrors __init__).
        self._tfidf_rk_inject = None


def _maybe_torch(use_gpu: bool):
    if not use_gpu:
        return None
    try:
        import torch
    except ImportError:
        return None
    return torch if torch.cuda.is_available() else None
