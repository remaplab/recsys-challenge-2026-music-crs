"""LightFM ICM recommender — PyTorch BPR backend.

Same architecture as launchers_crossvalidation/test/lightfm_icm_cv.py,
wrapped as a UserRecommender-compatible class so it integrates with:
  - launchers_crossvalidation/tune_crossvalidation.py  (splitK 5-fold CV)
  - launchers/predict_blind.py                          (splitF/A blind submission)

Cold-start inference: user embedding = mean of item_repr over session context.
"""

from __future__ import annotations

import gc

import numpy as np
import polars as pl
import scipy.sparse as sps
from scipy.sparse import csr_matrix, hstack as sp_hstack
from sklearn.preprocessing import normalize as sk_normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from .user_base import UserRecommender


# ---------------------------------------------------------------------------
# ICM builder — identical feature set as lightfm_icm_cv._build_icm
# ---------------------------------------------------------------------------

def _build_icm_bpr(track_meta_df: pl.DataFrame, t2i: dict,
                   long_df: pl.DataFrame | None = None) -> csr_matrix:
    """Sparse ICM from track metadata (artist, album, tags, decade, pop, user_count, dur)."""
    n = len(t2i)
    blocks: list = []

    def _list_block(col: str) -> csr_matrix:
        fmap: dict = {}
        rs, cs = [], []
        for r in track_meta_df.iter_rows(named=True):
            ti = t2i.get(r["track_id"])
            if ti is None:
                continue
            for v in (r.get(col) or []):
                if v:
                    fmap.setdefault(v, len(fmap))
                    rs.append(ti)
                    cs.append(fmap[v])
        if not rs:
            return csr_matrix((n, 1), dtype=np.float32)
        return csr_matrix(
            (np.ones(len(rs), np.float32), (rs, cs)),
            shape=(n, len(fmap)), dtype=np.float32,
        )

    def _bin_block(tids: list, vals: np.ndarray, edges: np.ndarray) -> csr_matrix:
        bins = np.digitize(vals, edges)
        nb   = len(edges) + 1
        rs, cs = [], []
        for i, tid in enumerate(tids):
            ti = t2i.get(tid)
            if ti is not None:
                rs.append(ti)
                cs.append(int(bins[i]))
        if not rs:
            return csr_matrix((n, nb), dtype=np.float32)
        return csr_matrix(
            (np.ones(len(rs), np.float32), (rs, cs)),
            shape=(n, nb), dtype=np.float32,
        )

    blocks.append(_list_block("artist_id"))
    blocks.append(_list_block("album_id"))
    blocks.append(_list_block("tag_list"))

    # decade
    dm: dict = {}
    dr, dc = [], []
    for r in track_meta_df.iter_rows(named=True):
        ti = t2i.get(r["track_id"])
        rd = r.get("release_date")
        if ti is None or not rd or len(str(rd)) < 4:
            continue
        try:
            y = int(str(rd)[:4])
        except (ValueError, TypeError):
            continue
        if y <= 0:
            continue
        dec = (y // 10) * 10
        dm.setdefault(dec, len(dm))
        dr.append(ti)
        dc.append(dm[dec])
    if dr:
        blocks.append(csr_matrix(
            (np.ones(len(dr), np.float32), (dr, dc)),
            shape=(n, len(dm)), dtype=np.float32,
        ))

    # popularity bin
    pop_df = track_meta_df.filter(pl.col("popularity").is_not_null())
    if pop_df.height > 0:
        tids = pop_df["track_id"].to_list()
        pops = pop_df["popularity"].to_numpy().astype(np.float32)
        nz   = pops[pops > 0]
        edges = np.percentile(nz, [20, 40, 60, 80]) if len(nz) >= 5 else np.array([.25, .5, .75, 1.])
        blocks.append(_bin_block(tids, pops, edges))

    # user_count bin (from interactions)
    if long_df is not None:
        pc = (
            long_df.filter(pl.col("track_id").is_not_null())
            .group_by("track_id")
            .agg(pl.col("user_id").n_unique().alias("c"))
        )
        tids = pc["track_id"].to_list()
        cnts = pc["c"].to_numpy().astype(np.float32)
        nz   = cnts[cnts > 0]
        edges = (
            np.percentile(nz, [20, 40, 60, 80]) if len(nz) >= 5
            else np.array([1, 2, 5, 10], np.float32)
        )
        blocks.append(_bin_block(tids, cnts, edges))

    # duration bin
    dur_df = track_meta_df.filter(pl.col("duration").is_not_null())
    if dur_df.height > 0:
        tids = dur_df["track_id"].to_list()
        durs = dur_df["duration"].to_numpy().astype(np.float64)
        blocks.append(_bin_block(
            tids, durs,
            np.array([120_000, 180_000, 240_000, 360_000], np.float64),
        ))

    return sp_hstack(blocks, format="csr").astype(np.float32)


# ---------------------------------------------------------------------------
# BPR training — identical to _BPRRecommender.fit in lightfm_icm_cv.py
# ---------------------------------------------------------------------------

def _train_bpr(
    urm: csr_matrix,
    icm_dense: np.ndarray,
    n_components: int,
    learning_rate: float,
    item_alpha: float,
    user_alpha: float,
    epochs: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Train PyTorch BPR with ICM content features. Returns (item_repr, item_bias)."""
    n_users, n_items = urm.shape
    n_content = icm_dense.shape[1]
    d = n_components
    batch_size = 4096
    i_a, u_a = item_alpha, user_alpha

    class _Net(nn.Module):
        def __init__(self_) -> None:
            super().__init__()
            self_.ue = nn.Embedding(n_users, d)
            self_.ub = nn.Embedding(n_users, 1)
            self_.ie = nn.Embedding(n_items, d)
            self_.ib = nn.Embedding(n_items, 1)
            self_.ce = nn.Linear(n_content, d, bias=False)
            self_.cb = nn.Linear(n_content, 1, bias=False)
            nn.init.normal_(self_.ue.weight, std=0.01)
            nn.init.normal_(self_.ie.weight, std=0.01)
            nn.init.zeros_(self_.ub.weight)
            nn.init.zeros_(self_.ib.weight)
            nn.init.normal_(self_.ce.weight, std=0.01)
            nn.init.zeros_(self_.cb.weight)

        def _irep(self_, idx, c):
            return (
                self_.ie(idx) + self_.ce(c),
                self_.ib(idx).squeeze(-1) + self_.cb(c).squeeze(-1),
            )

        def forward(self_, u, pos, neg, pc, nc):
            ue = self_.ue(u)
            ub = self_.ub(u).squeeze(-1)
            pe, pb = self_._irep(pos, pc)
            ne, nb = self_._irep(neg, nc)
            ps = (ue * pe).sum(-1) + ub + pb
            ns = (ue * ne).sum(-1) + ub + nb
            loss = -F.logsigmoid(ps - ns).mean()
            reg  = u_a * self_.ue.weight.pow(2).sum() + i_a * self_.ie.weight.pow(2).sum()
            return loss + reg

    model = _Net().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=learning_rate)

    coo   = urm.tocoo()
    u_arr = coo.row.astype(np.int64)
    i_arr = coo.col.astype(np.int64)
    u_pos = [set() for _ in range(n_users)]
    for u, i in zip(u_arr, i_arr):
        u_pos[u].add(int(i))
    n_pairs = len(u_arr)

    icm_t = torch.from_numpy(icm_dense).to(device)

    for _ in tqdm(range(epochs), desc="BPR", leave=False):
        model.train()
        order = np.random.permutation(n_pairs)
        for s in range(0, n_pairs, batch_size):
            idx  = order[s:s + batch_size]
            u_np = u_arr[idx]
            p_np = i_arr[idx]
            n_np = np.random.randint(0, n_items, len(idx))
            for j in range(len(idx)):
                while n_np[j] in u_pos[u_np[j]]:
                    n_np[j] = np.random.randint(0, n_items)
            u_t = torch.from_numpy(u_np).to(device)
            p_t = torch.from_numpy(p_np).to(device)
            n_t = torch.from_numpy(n_np).to(device)
            opt.zero_grad()
            model(u_t, p_t, n_t, icm_t[p_np], icm_t[n_np]).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        id_emb  = model.ie.weight.cpu().numpy()
        ce_w    = model.ce.weight.cpu().numpy()
        id_bias = model.ib.weight.squeeze().cpu().numpy()
        cb_w    = model.cb.weight.squeeze().cpu().numpy()
        item_repr = (id_emb  + icm_dense @ ce_w.T).astype(np.float32)
        item_bias = (id_bias + icm_dense @ cb_w  ).astype(np.float32)

    del model, icm_t
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return item_repr, item_bias


# ---------------------------------------------------------------------------
# Recommender class
# ---------------------------------------------------------------------------

class LightFMICMRecommender(UserRecommender):
    """PyTorch BPR with ICM content features.

    Same architecture as launchers_crossvalidation/test/lightfm_icm_cv.py.
    Cold-start inference: user_emb = mean of item_repr[context_items].
    """

    RECOMMENDER_NAME = "LightFMICM"

    def __init__(
        self,
        n_components:  int   = 64,
        learning_rate: float = 0.05,
        item_alpha:    float = 1e-6,
        user_alpha:    float = 1e-6,
        epochs:        int   = 50,
        device:        str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.n_components  = n_components
        self.learning_rate = learning_rate
        self.item_alpha    = item_alpha
        self.user_alpha    = user_alpha
        self.epochs        = epochs
        self.device        = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._item_repr: np.ndarray | None = None
        self._item_bias: np.ndarray | None = None
        # stored in fit() before super() call so _fit_model() can access them
        self._long_df_bpr:   pl.DataFrame | None = None
        self._track_meta_bpr: pl.DataFrame | None = None

    # ------------------------------------------------------------------
    # fit — intercept to store long_df and track_meta before super()
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df: pl.DataFrame,
        track_metadata: pl.DataFrame | None = None,
        **kwargs,
    ) -> None:
        from .interactions import explode_music_turns
        self._long_df_bpr    = explode_music_turns(train_df)
        self._track_meta_bpr = track_metadata
        super().fit(train_df, track_metadata=track_metadata, **kwargs)

    # ------------------------------------------------------------------
    # _fit_model — called by UserRecommender.fit() after building URM
    # ------------------------------------------------------------------

    def _fit_model(self, urm: csr_matrix) -> None:
        t2i = self.id_map.track_to_idx   # track_id → column index in URM

        if self._track_meta_bpr is not None:
            icm = _build_icm_bpr(self._track_meta_bpr, t2i, self._long_df_bpr)
        else:
            icm = sps.eye(urm.shape[1], format="csr", dtype=np.float32)

        icm_dense = sk_normalize(icm.toarray(), norm="l2", axis=1).astype(np.float32)
        print(
            f"[{self.RECOMMENDER_NAME}] BPR ICM: {icm.shape}  nnz={icm.nnz}"
            f"  d={self.n_components}  epochs={self.epochs}  device={self.device}"
        )

        self._item_repr, self._item_bias = _train_bpr(
            urm,
            icm_dense,
            n_components=self.n_components,
            learning_rate=self.learning_rate,
            item_alpha=self.item_alpha,
            user_alpha=self.user_alpha,
            epochs=self.epochs,
            device=self.device,
        )
        print(f"[{self.RECOMMENDER_NAME}] item_repr={self._item_repr.shape}")

    # ------------------------------------------------------------------
    # _score_session_profile — called by UserRecommender.recommend()
    # ------------------------------------------------------------------

    def _score_session_profile(self, profile: csr_matrix) -> np.ndarray:
        n = self._item_repr.shape[0] if self._item_repr is not None else 0
        item_idxs = profile.nonzero()[1]
        if len(item_idxs) == 0 or self._item_repr is None:
            return np.zeros(n, dtype=np.float32)
        user_emb = self._item_repr[item_idxs].mean(axis=0)
        return (self._item_repr @ user_emb + self._item_bias).astype(np.float32)

    # ------------------------------------------------------------------
    # serialisation
    # ------------------------------------------------------------------

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st.update({
            "n_components":  self.n_components,
            "learning_rate": self.learning_rate,
            "item_alpha":    self.item_alpha,
            "user_alpha":    self.user_alpha,
            "epochs":        self.epochs,
            "device":        self.device,
            "_item_repr":    self._item_repr,
            "_item_bias":    self._item_bias,
        })
        return st

    def _set_model_state(self, state: dict) -> None:
        super()._set_model_state(state)
        for k in ("n_components", "learning_rate", "item_alpha", "user_alpha",
                  "epochs", "device"):
            setattr(self, k, state.get(k, getattr(self, k, None)))
        self._item_repr = state.get("_item_repr")
        self._item_bias = state.get("_item_bias")
