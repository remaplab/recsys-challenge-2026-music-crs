"""Multimodal Two-Tower with InfoNCE loss.

Architecture
------------
  Item tower : MLP(item_features) → L2-normalized 256d embedding
  User tower : MLP(mean of session item_features) → L2-normalized 256d embedding
  Score      : cosine similarity / temperature (learnable)
  Loss       : InfoNCE with in-batch negatives (sampled softmax)

Differences vs feature_bert4rec
-------------------------------
- No transformer / no sequence modeling — just mean pooling of session items
- InfoNCE (in-batch sampled softmax) instead of full softmax over 30K warm items
  → 10-100x faster training
- Symmetric towers (user and item have separate MLPs)
- Both warm and cold items get embeddings from the same item tower (no special
  handling)

Training
--------
For each (session, pos_item) pair in the URM, compute:
  user_feat = mean(features of OTHER items in session)   # leave-one-out
  user_emb  = normalize(user_tower(user_feat))
  item_emb  = normalize(item_tower(pos_item_features))
Within a batch of B pairs, compute B x B cosine similarity matrix; label is the
diagonal (the matching positive). Negatives = the other (B-1) positives in batch.
This is sampled softmax with in-batch negatives — standard for two-tower retrieval.

Inference
---------
- Precompute all_item_embs (one forward through item_tower)
- For each session: user_emb = normalize(user_tower(mean(prior features)))
- scores = user_emb @ all_item_embs.T
- Top-K
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from tqdm import tqdm

from .feature_bert4rec import _build_feature_matrix
from .interactions import explode_music_turns, parse_date
from .session_base import SessionRecommender


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class _TwoTowerModel(nn.Module):
    """Two separate MLPs (user & item towers) + L2-normalized cosine scoring."""

    def __init__(self, feature_dim: int, hidden_size: int, dropout: float,
                 init_tau: float = 0.1) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size

        self.item_tower = nn.Sequential(
            nn.Linear(feature_dim, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.user_tower = nn.Sequential(
            nn.Linear(feature_dim, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        # Learnable temperature (log-space for positivity)
        self.log_tau = nn.Parameter(torch.tensor(float(np.log(init_tau))))

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    def encode_items(self, item_features: torch.Tensor) -> torch.Tensor:
        """item_features: (B, feature_dim) → (B, hidden) L2-normalized."""
        return F.normalize(self.item_tower(item_features), dim=-1)

    def encode_users(self, user_features: torch.Tensor) -> torch.Tensor:
        """user_features: (B, feature_dim) → (B, hidden) L2-normalized."""
        return F.normalize(self.user_tower(user_features), dim=-1)

    def forward(
        self,
        user_features: torch.Tensor,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        """InfoNCE: returns logits matrix (B, B) where diagonal is the positive."""
        u = self.encode_users(user_features)  # (B, hidden)
        i = self.encode_items(item_features)  # (B, hidden)
        # Cosine similarity / temperature
        return (u @ i.T) / self.tau


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

class TwoTowerMultimodalRecommender(SessionRecommender):
    """Multimodal Two-Tower with InfoNCE training and inductive inference."""

    RECOMMENDER_NAME = "TwoTowerMultimodal"

    def __init__(
        self,
        feature_emb_paths:   list[str],
        feature_modalities:  list[str] | None = None,
        hidden_size:         int   = 256,
        dropout:             float = 0.2,
        init_tau:            float = 0.1,
        epochs:              int   = 100,
        batch_size:          int   = 512,
        lr:                  float = 1.0e-3,
        weight_decay:        float = 1.0e-4,
        val_ratio:           float = 0.1,
        early_stop_patience: int   = 15,
        max_future_years:    float = 2.0,
        device:              str   = "auto",
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
        self.dropout             = dropout
        self.init_tau            = init_tau
        self.epochs              = epochs
        self.batch_size          = batch_size
        self.lr                  = lr
        self.weight_decay        = weight_decay
        self.val_ratio           = val_ratio
        self.early_stop_patience = early_stop_patience

        if device == "auto":
            self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device_ = torch.device(device)

        self.model:          _TwoTowerModel | None = None
        self._train_long:    pl.DataFrame   | None = None
        self._feature_matrix: torch.Tensor  | None = None    # (n_items, feature_dim) on device
        self._all_item_emb:   np.ndarray    | None = None    # (n_items, hidden) — cached for inference
        # Precomputed per-session: sum of feature vectors + count, for leave-one-out user_features
        self._session_sum:    torch.Tensor  | None = None    # (n_sessions, feature_dim)
        self._session_count:  torch.Tensor  | None = None    # (n_sessions,)
        self._feature_dim:    int           | None = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, train_df: pl.DataFrame, track_metadata: pl.DataFrame | None = None, **kwargs: Any) -> None:
        self._train_long = explode_music_turns(train_df)
        super().fit(train_df, track_metadata=track_metadata, **kwargs)

    def _fit_model(self, urm: csr_matrix) -> None:
        n_users, n_items = urm.shape
        print(f"[{self.RECOMMENDER_NAME}] users={n_users}, items={n_items}, interactions={urm.nnz}")

        # ---- Load multimodal features ----
        print(f"[{self.RECOMMENDER_NAME}] Loading feature embeddings: {self.feature_modalities}")
        full_matrix = _build_feature_matrix(
            self.feature_emb_paths, self.feature_modalities, self.id_map
        )
        self._feature_dim = full_matrix.shape[1]
        print(f"  features shape: {full_matrix.shape}")

        # Keep features on device for fast lookup
        self._feature_matrix = torch.from_numpy(full_matrix).float().to(self.device_)

        # ---- Precompute per-session: sum and count of feature vectors ----
        # session_sum[s] = sum of feature vectors of items in session s
        # session_count[s] = number of items in session s
        # → user_features(s, exclude=p) = (session_sum[s] - feature[p]) / (session_count[s] - 1)
        urm_csr = csr_matrix(urm)
        sess_sum    = np.zeros((n_users, self._feature_dim), dtype=np.float32)
        sess_count  = np.zeros(n_users, dtype=np.int64)
        for u in range(n_users):
            items = urm_csr[u].indices
            if len(items) > 0:
                sess_sum[u]   = full_matrix[items].sum(axis=0)
                sess_count[u] = len(items)
        self._session_sum   = torch.from_numpy(sess_sum).to(self.device_)
        self._session_count = torch.from_numpy(sess_count.astype(np.float32)).to(self.device_)
        print(f"  session_sum: {self._session_sum.shape}, avg items/session = {sess_count.mean():.2f}")

        # ---- Build model ----
        self.model = _TwoTowerModel(self._feature_dim, self.hidden_size, self.dropout, self.init_tau)
        self.model.to(self.device_)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  model params: {n_params:,}")

        # ---- Training pairs (session, pos_item) ----
        pairs = np.array(list(zip(*urm_csr.nonzero())), dtype=np.int64)
        n_pairs = len(pairs)
        print(f"  training pairs: {n_pairs}")

        # Train/val split
        rng = np.random.default_rng(0)
        rng.shuffle(pairs)
        n_val = max(1, int(n_pairs * self.val_ratio))
        val_pairs   = pairs[:n_val]
        train_pairs = pairs[n_val:]
        print(f"  train_pairs={len(train_pairs)}, val_pairs={len(val_pairs)}")

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

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
                users_b = torch.from_numpy(batch[:, 0]).long().to(self.device_)
                pos_b   = torch.from_numpy(batch[:, 1]).long().to(self.device_)
                B = len(batch)
                if B < 2:  # InfoNCE needs ≥2 for negatives
                    continue

                # Leave-one-out: user_features = (session_sum - pos_features) / (count - 1)
                pos_features = self._feature_matrix[pos_b]                  # (B, F)
                sess_sum     = self._session_sum[users_b]                   # (B, F)
                sess_count   = self._session_count[users_b].clamp(min=2)    # (B,)  ≥2 for divisor
                user_features = (sess_sum - pos_features) / (sess_count - 1).unsqueeze(-1)

                # Forward
                logits = self.model(user_features, pos_features)            # (B, B)
                labels = torch.arange(B, device=self.device_)               # diagonal = positive
                loss = F.cross_entropy(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches  += 1

            train_loss = total_loss / max(1, n_batches)

            # ---- Val: in-batch recall@20 on held-out pairs (consistent w/ training proxy) ----
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
                tau=f"{self.model.tau.item():.3f}",
                patience=patience_left,
            )

            if patience_left <= 0:
                print(f"\n[{self.RECOMMENDER_NAME}] early stopping at epoch {epoch}, "
                      f"best val_recall@20={best_val:.4f} at epoch {best_epoch}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"[{self.RECOMMENDER_NAME}] restored best checkpoint "
                  f"(val_recall@20={best_val:.4f}, epoch={best_epoch}, tau={self.model.tau.item():.4f})")

        # ---- Cache all item embeddings for inference ----
        self.model.eval()
        with torch.no_grad():
            all_item_emb = self.model.encode_items(self._feature_matrix)  # (n_items, hidden)
        self._all_item_emb = all_item_emb.cpu().numpy()
        print(f"[{self.RECOMMENDER_NAME}] cached all_item_emb: {self._all_item_emb.shape}")

    def _eval_val_recall(self, val_pairs: np.ndarray, urm_csr: csr_matrix, k: int, sample: int) -> float:
        """Recall@k: for each (u, target), encode user from (session - target) features, score all items."""
        with torch.no_grad():
            all_item_emb = self.model.encode_items(self._feature_matrix).cpu().numpy()  # (n_items, hidden)

        idxs = np.random.default_rng(0).choice(
            len(val_pairs), size=min(sample, len(val_pairs)), replace=False
        )
        hits = 0
        n    = 0
        for u, i in val_pairs[idxs]:
            u_int = int(u); i_int = int(i)
            cnt = self._session_count[u_int].item()
            if cnt < 2:
                continue
            user_feat = (self._session_sum[u_int] - self._feature_matrix[i_int]) / (cnt - 1)
            with torch.no_grad():
                u_emb = self.model.encode_users(user_feat.unsqueeze(0)).cpu().numpy()[0]
            scores = u_emb @ all_item_emb.T
            user_pos = urm_csr[u_int].indices
            scores[user_pos] = -np.inf
            scores[i_int] = u_emb @ all_item_emb[i_int]
            topk = np.argpartition(-scores, k)[:k]
            if i_int in topk:
                hits += 1
            n += 1
        return hits / n if n > 0 else 0.0

    # ------------------------------------------------------------------
    # recommend — inductive: user_emb = user_tower(mean(prior features))
    # ------------------------------------------------------------------

    def _score_session_sequence(self, prior: list[str]) -> np.ndarray:
        assert self.id_map is not None and self._all_item_emb is not None
        prior_idxs = [self.id_map.track_to_idx[t] for t in prior if t in self.id_map.track_to_idx]
        if not prior_idxs:
            return np.full(self.id_map.n_tracks, -np.inf, dtype=np.float32)

        with torch.no_grad():
            prior_feat = self._feature_matrix[prior_idxs].mean(dim=0, keepdim=True)  # (1, F)
            user_emb   = self.model.encode_users(prior_feat).cpu().numpy()[0]        # (hidden,)

        scores = self._all_item_emb @ user_emb
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
        if self.id_map is None or self.model is None or self._all_item_emb is None:
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
            "dropout":             self.dropout,
            "init_tau":            self.init_tau,
            "epochs":              self.epochs,
            "batch_size":          self.batch_size,
            "lr":                  self.lr,
            "weight_decay":        self.weight_decay,
            "val_ratio":           self.val_ratio,
            "early_stop_patience": self.early_stop_patience,
            "device":              str(self.device_),
            "feature_dim":         self._feature_dim,
            "all_item_emb":        self._all_item_emb,
            "model_state_dict":    self.model.state_dict() if self.model is not None else None,
        })
        return st

    def _set_model_state(self, state: dict) -> None:
        super()._set_model_state(state)
        self.feature_emb_paths   = state["feature_emb_paths"]
        self.feature_modalities  = state.get("feature_modalities", ["metadata-qwen3_embedding_0.6b"])
        self.hidden_size         = state["hidden_size"]
        self.dropout             = state["dropout"]
        self.init_tau            = state.get("init_tau", 0.1)
        self.epochs              = state["epochs"]
        self.batch_size          = state["batch_size"]
        self.lr                  = state["lr"]
        self.weight_decay        = state["weight_decay"]
        self.val_ratio           = state.get("val_ratio", 0.1)
        self.early_stop_patience = state.get("early_stop_patience", 15)
        self.device_             = torch.device(state.get("device", "cpu"))
        self._feature_dim        = state.get("feature_dim")
        self._all_item_emb       = state.get("all_item_emb")

        sd = state.get("model_state_dict")
        if sd is not None and self._feature_dim is not None:
            self.model = _TwoTowerModel(self._feature_dim, self.hidden_size, self.dropout, self.init_tau)
            self.model.load_state_dict(sd)
            self.model.to(self.device_)
