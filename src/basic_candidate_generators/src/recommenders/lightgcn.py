"""LightGCN with multimodal feature initialization.

Paper: "LightGCN: Simplifying and Powering Graph Convolution Network for
Recommendation" (He et al., SIGIR 2020).

Architecture overview
---------------------
1. Build bipartite session-item graph from URM (train interactions).
2. Init session embeddings randomly; init item embeddings from PCA of
   multimodal features (Qwen3 + CLAP + cf-bpr concat → hidden_size dims).
3. Propagate K times via normalized adjacency: E_k = D^-1/2 A D^-1/2 E_{k-1}.
   No transformations / non-linearities — that's what makes it "light".
4. Final embedding = mean of [E_0, E_1, ..., E_K].
5. Score = (session_emb · item_emb).

Inductive inference for cold sessions
-------------------------------------
Cold (unseen-at-train) sessions don't have a learned embedding. We represent
each test session as the MEAN of its prior tracks' final item embeddings.
This is consistent for both warm and cold sessions and handles cold items
naturally (their embedding still comes from PCA-init + graph propagation
through any neighbours they happen to have).

Training
--------
BPR loss with random negative sampling. For each (session, pos_item) pair in
URM, sample a random neg_item not in that session's history. Loss is the
classic Bayesian Personalized Ranking softplus.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.decomposition import PCA
from tqdm import tqdm

from .feature_bert4rec import _build_feature_matrix
from .interactions import explode_music_turns, parse_date
from .session_base import SessionRecommender


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class _LightGCNModel(nn.Module):
    """LightGCN: user/item embeddings + K-step graph propagation (no transforms)."""

    def __init__(self, n_users: int, n_items: int, hidden_size: int, n_layers: int) -> None:
        super().__init__()
        self.n_users     = n_users
        self.n_items     = n_items
        self.n_layers    = n_layers
        self.hidden_size = hidden_size

        self.user_embedding = nn.Embedding(n_users, hidden_size)
        self.item_embedding = nn.Embedding(n_items, hidden_size)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def init_item_emb_from_features(self, feature_matrix: np.ndarray) -> None:
        """PCA-init item embeddings from multimodal feature matrix.

        Scale to roughly unit norm per dim so propagation produces sensible
        magnitudes (~1 per row), matching what Xavier user embeddings give.
        """
        d = self.hidden_size
        pca = PCA(n_components=d, svd_solver="randomized", random_state=0)
        emb_init = pca.fit_transform(feature_matrix).astype(np.float32)
        # Standardize each PCA dim then scale to match Xavier range
        emb_init = (emb_init - emb_init.mean(axis=0)) / (emb_init.std(axis=0) + 1e-9)
        emb_init = emb_init * np.sqrt(1.0 / d)   # row norm ≈ 1
        with torch.no_grad():
            self.item_embedding.weight.copy_(torch.from_numpy(emb_init))
        explained = float(pca.explained_variance_ratio_.sum())
        print(f"  PCA item-init: explained variance {explained:.3f}")

    def propagate(self, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """K-step LightGCN propagation via scatter (autograd-safe, vs sparse.mm).

        edge_index: (2, n_edges) long — (src, dst) pairs
        edge_weight: (n_edges,) float — symmetric-normalized weights
        Returns (user_final, item_final).
        """
        all_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        N, D = all_emb.shape
        src, dst = edge_index[0], edge_index[1]
        ew = edge_weight.unsqueeze(-1)  # (n_edges, 1)

        embs = [all_emb]
        for _ in range(self.n_layers):
            msg = all_emb[src] * ew                                    # (n_edges, D)
            new_emb = torch.zeros(N, D, device=all_emb.device, dtype=all_emb.dtype)
            new_emb = new_emb.index_add(0, dst, msg)                   # functional, autograd-safe
            all_emb = new_emb
            embs.append(all_emb)
        combined = torch.stack(embs, dim=0).mean(dim=0)
        user_final, item_final = torch.split(combined, [self.n_users, self.n_items], dim=0)
        return user_final, item_final

    def bpr_forward(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        user_final, item_final = self.propagate(edge_index, edge_weight)
        u = user_final[users]
        p = item_final[pos_items]
        n = item_final[neg_items]
        pos_score = (u * p).sum(dim=-1)
        neg_score = (u * n).sum(dim=-1)
        # L2 reg uses RAW (pre-propagation) embeddings — LightGCN paper convention
        u_raw = self.user_embedding(users)
        p_raw = self.item_embedding(pos_items)
        n_raw = self.item_embedding(neg_items)
        return pos_score, neg_score, u_raw, p_raw, n_raw


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

class LightGCNRecommender(SessionRecommender):
    """LightGCN with PCA-initialized multimodal item embeddings.

    Inference is inductive (per-session embedding = mean of prior items' final
    embeddings), so both cold sessions and cold items are handled out of the box.
    """

    RECOMMENDER_NAME = "LightGCN"

    def __init__(
        self,
        feature_emb_paths: list[str],
        feature_modalities: list[str] | None = None,
        hidden_size: int = 64,
        n_layers: int = 3,
        epochs: int = 200,
        batch_size: int = 2048,
        lr: float = 1.0e-3,
        weight_decay: float = 1.0e-4,
        bpr_reg: float = 1.0e-4,
        val_ratio: float = 0.1,
        early_stop_patience: int = 20,
        max_future_years: float = 2.0,
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            urm_mode=kwargs.pop("urm_mode", "session"),
            max_future_years=max_future_years,
            **kwargs,
        )
        self.feature_emb_paths   = feature_emb_paths
        self.feature_modalities  = feature_modalities or ["metadata-qwen3_embedding_0.6b"]
        self.hidden_size         = hidden_size
        self.n_layers            = n_layers
        self.epochs              = epochs
        self.batch_size          = batch_size
        self.lr                  = lr
        self.weight_decay        = weight_decay
        self.bpr_reg             = bpr_reg
        self.val_ratio           = val_ratio
        self.early_stop_patience = early_stop_patience

        if device == "auto":
            self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device_ = torch.device(device)

        self.model:         _LightGCNModel | None  = None
        self._train_long:   pl.DataFrame  | None   = None
        self._edge_index:   torch.Tensor  | None   = None   # (2, n_edges) on device
        self._edge_weight:  torch.Tensor  | None   = None   # (n_edges,) on device
        self._item_final:   np.ndarray    | None   = None   # (n_items, hidden) — cached
        self._feature_dim:  int           | None   = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, train_df: pl.DataFrame, track_metadata: pl.DataFrame | None = None, **kwargs: Any) -> None:
        self._train_long = explode_music_turns(train_df)
        super().fit(train_df, track_metadata=track_metadata, **kwargs)

    def _fit_model(self, urm: csr_matrix) -> None:
        n_users, n_items = urm.shape
        print(f"[{self.RECOMMENDER_NAME}] users={n_users}, items={n_items}, interactions={urm.nnz}")

        # ---- Build normalized adjacency for bipartite graph ----
        # A = [[0, R], [R^T, 0]], normalized as D^-1/2 A D^-1/2
        urm_coo = coo_matrix(urm)
        rows = np.concatenate([urm_coo.row,            urm_coo.col + n_users])
        cols = np.concatenate([urm_coo.col + n_users,  urm_coo.row])
        data = np.ones(2 * urm.nnz, dtype=np.float32)
        N = n_users + n_items
        adj = coo_matrix((data, (rows, cols)), shape=(N, N))

        # Symmetric normalization
        degree = np.array(adj.sum(axis=1)).flatten()
        with np.errstate(divide="ignore"):
            d_inv_sqrt = np.power(degree, -0.5)
        d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0
        norm_data = adj.data * d_inv_sqrt[adj.row] * d_inv_sqrt[adj.col]

        # Store as edge_index/edge_weight (use with scatter — autograd-safe vs sparse.mm)
        self._edge_index  = torch.from_numpy(np.vstack([adj.row, adj.col])).long().to(self.device_)
        self._edge_weight = torch.from_numpy(norm_data).float().to(self.device_)
        print(f"  graph: {N} nodes, {len(norm_data)} edges (symmetric)")

        # ---- Load multimodal features for PCA init ----
        print(f"[{self.RECOMMENDER_NAME}] Loading feature embeddings: {self.feature_modalities}")
        full_matrix = _build_feature_matrix(
            self.feature_emb_paths, self.feature_modalities, self.id_map
        )
        self._feature_dim = full_matrix.shape[1]
        print(f"  features shape: {full_matrix.shape}")

        # ---- Build model ----
        self.model = _LightGCNModel(n_users, n_items, self.hidden_size, self.n_layers)
        self.model.init_item_emb_from_features(full_matrix)
        self.model.to(self.device_)

        # ---- Build (user, pos_item) training pairs and per-user positives ----
        urm_csr = csr_matrix(urm)
        pairs   = np.array(list(zip(*urm_csr.nonzero())), dtype=np.int64)  # (n_pairs, 2)
        n_pairs = len(pairs)
        print(f"  training pairs: {n_pairs}")

        # Per-user positives (for negative sampling rejection)
        user_positives: dict[int, set] = {}
        for u, i in pairs:
            user_positives.setdefault(int(u), set()).add(int(i))

        # ---- Train/val split on pairs ----
        rng = np.random.default_rng(0)
        rng.shuffle(pairs)
        n_val = max(1, int(n_pairs * self.val_ratio))
        val_pairs   = pairs[:n_val]
        train_pairs = pairs[n_val:]
        print(f"  train_pairs={len(train_pairs)}, val_pairs={len(val_pairs)}")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_val      = -float("inf")
        best_epoch    = 0
        best_state    = None
        patience_left = self.early_stop_patience

        epoch_bar = tqdm(range(1, self.epochs + 1), desc=f"[{self.RECOMMENDER_NAME}]",
                         unit="ep", dynamic_ncols=True, file=sys.stdout)
        for epoch in epoch_bar:
            self.model.train()
            rng.shuffle(train_pairs)
            total_loss = 0.0
            n_batches  = 0
            for start in range(0, len(train_pairs), self.batch_size):
                batch = train_pairs[start:start + self.batch_size]
                users_b = batch[:, 0]
                pos_b   = batch[:, 1]

                # Negative sampling with simple rejection (1-2 retries max in practice)
                neg_b = rng.integers(0, n_items, size=len(batch))
                for idx in range(len(batch)):
                    user_pos = user_positives.get(int(users_b[idx]), set())
                    tries = 0
                    while int(neg_b[idx]) in user_pos and tries < 5:
                        neg_b[idx] = rng.integers(0, n_items)
                        tries += 1

                u = torch.from_numpy(users_b).long().to(self.device_)
                p = torch.from_numpy(pos_b).long().to(self.device_)
                n = torch.from_numpy(neg_b).long().to(self.device_)

                pos_score, neg_score, u_raw, p_raw, n_raw = self.model.bpr_forward(
                    u, p, n, self._edge_index, self._edge_weight
                )
                bpr_loss = F.softplus(neg_score - pos_score).mean()
                reg = self.bpr_reg * 0.5 * (u_raw.pow(2).sum() + p_raw.pow(2).sum() + n_raw.pow(2).sum()) / len(batch)
                loss = bpr_loss + reg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches  += 1

            train_loss = total_loss / max(1, n_batches)

            # ---- Val: recall@20 on held-out pairs (subset for speed) ----
            self.model.eval()
            val_recall = self._eval_val_recall(val_pairs, urm_csr, k=20, sample=400)

            improved = val_recall > best_val
            if improved:
                best_val      = val_recall
                best_epoch    = epoch
                best_state    = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience_left = self.early_stop_patience
            else:
                patience_left -= 1

            epoch_bar.set_postfix(
                loss=f"{train_loss:.4f}",
                val_rec=f"{val_recall:.4f}",
                best=f"{best_val:.4f}@{best_epoch}",
                patience=patience_left,
            )

            if patience_left <= 0:
                print(f"\n[{self.RECOMMENDER_NAME}] early stopping at epoch {epoch}, "
                      f"best val_recall@20={best_val:.4f} at epoch {best_epoch}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"[{self.RECOMMENDER_NAME}] restored best checkpoint "
                  f"(val_recall@20={best_val:.4f}, epoch={best_epoch})")

        # Cache final item embeddings on CPU for inference
        self.model.eval()
        with torch.no_grad():
            _, item_final = self.model.propagate(self._edge_index, self._edge_weight)
        self._item_final = item_final.cpu().numpy()
        print(f"[{self.RECOMMENDER_NAME}] cached item_final: {self._item_final.shape}")

    def _eval_val_recall(self, val_pairs: np.ndarray, urm_csr: csr_matrix, k: int, sample: int) -> float:
        """Quick recall@k on val pairs (subset for speed). Uses learned user_emb."""
        with torch.no_grad():
            user_final, item_final = self.model.propagate(self._edge_index, self._edge_weight)
        item_final_np = item_final.cpu().numpy()
        idxs = np.random.default_rng(epoch_seed := 0).choice(len(val_pairs), size=min(sample, len(val_pairs)), replace=False)
        hits = 0
        n    = 0
        for u, i in val_pairs[idxs]:
            u_emb = user_final[int(u)].cpu().numpy()
            scores = u_emb @ item_final_np.T
            # Mask out user's other positives so we evaluate proper recall
            user_pos = urm_csr[int(u)].indices
            scores[user_pos] = -np.inf  # mask seen
            scores[int(i)] = u_emb @ item_final_np[int(i)]  # un-mask the val target
            topk = np.argpartition(-scores, k)[:k]
            if int(i) in topk:
                hits += 1
            n += 1
        return hits / n if n > 0 else 0.0

    # ------------------------------------------------------------------
    # recommend — inductive: session_emb = mean of prior item embeddings
    # ------------------------------------------------------------------

    def _score_session_sequence(self, prior: list[str]) -> np.ndarray:
        """Score all tracks given an ordered list of prior track IDs.

        session_emb = mean of prior items' final embeddings.
        scores = session_emb · item_final.T.
        """
        assert self.id_map is not None and self._item_final is not None
        prior_idxs = [self.id_map.track_to_idx[t] for t in prior if t in self.id_map.track_to_idx]
        if not prior_idxs:
            return np.full(self.id_map.n_tracks, -np.inf, dtype=np.float32)
        prior_embs  = self._item_final[prior_idxs]
        session_emb = prior_embs.mean(axis=0)
        scores      = self._item_final @ session_emb
        return scores.astype(np.float32)

    def _score_session_profile(self, profile: csr_matrix) -> np.ndarray:
        raise NotImplementedError

    def recommend(
        self,
        context_df: pl.DataFrame,
        top_k: int = 20,
        remove_seen: bool = True,
        max_future_years: float | None = None,
        turn: int = 8,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """One prediction per session — UserRecommender-compatible interface."""
        if self.id_map is None or self.model is None or self._item_final is None:
            raise RuntimeError("Call fit() before recommend()")

        if "track_id" not in context_df.columns:
            context_df = explode_music_turns(context_df)

        session_meta = (
            context_df
            .select(["session_id", "user_id", "session_date"])
            .unique(subset=["session_id"])
        )

        ctx_sorted = (
            context_df.sort(["session_id", "turn_number"])
            if "turn_number" in context_df.columns
            else context_df
        )
        ctx_map: dict[str, list[str]] = {}
        if ctx_sorted.height > 0:
            for sid, grp in ctx_sorted.group_by("session_id", maintain_order=True):
                sid_str = sid[0] if isinstance(sid, tuple) else sid
                ctx_map[sid_str] = [t for t in grp["track_id"].to_list() if t is not None]

        out_session  : list[str]            = []
        out_user     : list[str]            = []
        out_tracks   : list[list[str]]      = []
        out_scores   : list[list[float]]    = []
        out_fallback : list[list[int]]      = []

        for row in session_meta.iter_rows(named=True):
            sess_id = row["session_id"]
            user_id = row["user_id"]
            sd      = parse_date(row["session_date"])
            candidate_mask = self._filter_candidate_mask(sd)

            prior = ctx_map.get(sess_id, [])
            prior_idxs = {
                self.id_map.track_to_idx[t]
                for t in prior
                if t in self.id_map.track_to_idx
            }

            if not prior_idxs:
                if self.fallback is None:
                    recs, scs = [], []
                else:
                    recs, scs = self.fallback.recommend_one(sess_id, turn, sd, top_k)
                fb_flags = [1] * len(recs)
            else:
                scores  = self._score_session_sequence(prior)
                recs, scs = self._topk_from_scores(scores, prior_idxs, top_k, candidate_mask, remove_seen)
                fb_flags = [0] * len(recs)

            out_session.append(sess_id)
            out_user.append(user_id)
            out_tracks.append(recs)
            out_scores.append(scs)
            out_fallback.append(fb_flags)

        return pl.DataFrame({
            "session_id":    out_session,
            "user_id":       out_user,
            "turn":          [turn] * len(out_session),
            "track_ids":     out_tracks,
            "scores":        out_scores,
            "fallback_used": out_fallback,
        })

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st.update({
            "feature_emb_paths":   self.feature_emb_paths,
            "feature_modalities":  self.feature_modalities,
            "hidden_size":         self.hidden_size,
            "n_layers":            self.n_layers,
            "epochs":              self.epochs,
            "batch_size":          self.batch_size,
            "lr":                  self.lr,
            "weight_decay":        self.weight_decay,
            "bpr_reg":             self.bpr_reg,
            "val_ratio":           self.val_ratio,
            "early_stop_patience": self.early_stop_patience,
            "device":              str(self.device_),
            "feature_dim":         self._feature_dim,
            "item_final":          self._item_final,
            "model_state_dict":    self.model.state_dict() if self.model is not None else None,
        })
        return st

    def _set_model_state(self, state: dict) -> None:
        super()._set_model_state(state)
        self.feature_emb_paths   = state["feature_emb_paths"]
        self.feature_modalities  = state.get("feature_modalities", ["metadata-qwen3_embedding_0.6b"])
        self.hidden_size         = state["hidden_size"]
        self.n_layers            = state["n_layers"]
        self.epochs              = state["epochs"]
        self.batch_size          = state["batch_size"]
        self.lr                  = state["lr"]
        self.weight_decay        = state["weight_decay"]
        self.bpr_reg             = state.get("bpr_reg", 1e-4)
        self.val_ratio           = state.get("val_ratio", 0.1)
        self.early_stop_patience = state.get("early_stop_patience", 20)
        self.device_             = torch.device(state.get("device", "cpu"))
        self._feature_dim        = state.get("feature_dim")
        self._item_final         = state.get("item_final")

        sd = state.get("model_state_dict")
        if sd is not None and self.id_map is not None:
            n_users = self.id_map.n_users
            n_items = self.id_map.n_tracks
            self.model = _LightGCNModel(n_users, n_items, self.hidden_size, self.n_layers)
            self.model.load_state_dict(sd)
            self.model.to(self.device_)
