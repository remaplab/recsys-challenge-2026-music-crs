"""Feature-based BERT4Rec — sequential recommendation without item ID lookup.

Instead of a per-item embedding table, each track is encoded from its
organizer-provided feature vectors (Qwen3, CLAP, CF, ...) through a shared
MLP encoder.

Warm/cold split — why it matters
---------------------------------
During training the softmax (cross-entropy loss) is computed over warm tracks
only (those seen in training interactions). If cold test tracks were included
in the softmax denominator they would act as distractors: the model would be
implicitly penalised for scoring them highly, biasing their ranking downward
at inference time — even when their features are similar to good warm tracks.

By restricting the training softmax to warm tracks and scoring cold tracks
separately at inference (via the same shared item encoder), we get:
  - Unbiased cold scoring: cold tracks are scored purely by feature similarity
  - Correct warm scoring: the training loss is over the right item space

Inference scoring
-----------------
1. Run the transformer on the input sequence → hidden state h at [MASK]
2. warm_scores = h @ encoder(warm_feature_matrix).T    (n_warm,)
3. cold_scores = h @ encoder(cold_feature_matrix).T    (n_cold,)
4. Combine into a (n_tracks,) array using stored index mappings

For cold-item coverage:
  - Pass track_metadata containing ALL tracks to fit() → adds cold tracks to id_map
  - Include test_tracks parquet in feature_emb_paths → cold tracks get feature vectors

Special tokens (same as bert4rec.py):
  PAD_TOKEN   = 0
  MASK_TOKEN  = 1
  ITEM_OFFSET = 2
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .interactions import IdMap, explode_music_turns, parse_date
from .session_base import SessionRecommender

PAD_TOKEN = 0
MASK_TOKEN = 1
ITEM_OFFSET = 2


# ---------------------------------------------------------------------------
# Feature matrix loader
# ---------------------------------------------------------------------------

def _build_feature_matrix(
    parquet_paths: list[str | Path],
    modalities: list[str],
    id_map: IdMap,
) -> np.ndarray:
    """Load track features from parquets and align them to id_map order.

    Returns float32 array of shape (n_tracks, total_feature_dim).
    Tracks missing from all parquets get zero vectors.
    Multiple parquets are merged: later files override zero entries from
    earlier files, so put imputed/fallback embeddings last.
    """
    n_tracks = id_map.n_tracks
    chunks: list[np.ndarray] = []

    for mod in modalities:
        lookup: dict[str, np.ndarray] = {}
        emb_dim: int | None = None

        for path in parquet_paths:
            df = pl.read_parquet(path)
            if mod not in df.columns:
                continue
            for tid, vec in zip(df["track_id"].to_list(), df[mod].to_list()):
                if vec is None:
                    continue
                arr = np.asarray(vec, dtype=np.float32)
                if arr.any():
                    lookup[tid] = arr
                    if emb_dim is None:
                        emb_dim = arr.shape[0]

        if emb_dim is None:
            raise ValueError(f"No valid features found for modality '{mod}' in {parquet_paths}")

        matrix = np.zeros((n_tracks, emb_dim), dtype=np.float32)
        found = 0
        for track_id, idx in id_map.track_to_idx.items():
            if track_id in lookup:
                matrix[idx] = lookup[track_id]
                found += 1

        coverage = found / n_tracks * 100
        print(f"  [{mod}] dim={emb_dim}, coverage={coverage:.1f}% ({found}/{n_tracks})")
        chunks.append(matrix)

    return np.concatenate(chunks, axis=1)  # (n_tracks, total_feature_dim)


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

class _FeatureBert4RecModel(nn.Module):
    """Transformer encoder with feature-based item representation.

    feature_matrix holds warm-track features only. The training forward()
    returns logits (B, L, n_warm) so the softmax stays within the warm
    item space. At inference time, encode_hidden() exposes the transformer
    output so the recommender can score warm and cold items separately via
    the same shared item_encoder.
    """

    def __init__(
        self,
        warm_feature_matrix: np.ndarray,  # (n_warm, feature_dim)
        hidden_size: int,
        max_seq_len: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        n_warm, feature_dim = warm_feature_matrix.shape
        self.hidden_size = hidden_size
        self.n_warm = n_warm

        self.register_buffer("feature_matrix", torch.from_numpy(warm_feature_matrix))

        self.item_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )

        # Separate learned embeddings for PAD (idx 0) and MASK (idx 1)
        self.special_embs = nn.Embedding(2, hidden_size)

        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)
        self.emb_norm = nn.LayerNorm(hidden_size)
        self.emb_dropout = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(hidden_size)

    def _build_seq_emb(self, x: torch.Tensor, items_table: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build sequence embeddings + pad mask from token IDs.

        items_table: (n_items, hidden) — item embeddings. Tokens index into
        [special_embs (2 rows) || items_table (n_items rows)] via direct lookup.

        At training, items_table = warm_embs only (tokens in [0, n_warm+1]).
        At inference, items_table can be the concat(warm_embs, cold_embs) so the
        sequence can contain cold items the model never saw at training time —
        their embeddings still come from the same shared item_encoder.
        """
        # (2 + n_items, hidden). Concat is a tracked op → gradients flow.
        full_table = torch.cat([self.special_embs.weight, items_table], dim=0)
        emb = full_table[x]  # (B, L, hidden), clean autograd

        pad_mask = x == PAD_TOKEN
        positions = torch.arange(emb.shape[1], device=x.device).unsqueeze(0)
        emb = emb + self.pos_emb(positions)
        return self.emb_dropout(self.emb_norm(emb)), pad_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Training forward — returns logits (B, L, n_warm).

        Softmax is over warm tracks only so cold tracks never appear as
        distractors in the cross-entropy denominator during training.
        """
        warm_embs = self.item_encoder(self.feature_matrix)  # (n_warm, hidden)
        emb, pad_mask = self._build_seq_emb(x, warm_embs)
        out = self.encoder(emb, src_key_padding_mask=pad_mask)
        out = self.output_norm(out)
        return out @ warm_embs.T  # (B, L, n_warm)

    def encode_hidden(self, x: torch.Tensor, items_table: torch.Tensor | None = None) -> torch.Tensor:
        """Inference — returns transformer hidden states (B, L, hidden_size).

        items_table: optional pre-computed item embeddings table; defaults to
        warm_embs. Pass concat(warm_embs, cold_embs) to include cold items
        in the input sequence (fixes the cold-in-prior limitation).
        """
        if items_table is None:
            items_table = self.item_encoder(self.feature_matrix)
        emb, pad_mask = self._build_seq_emb(x, items_table)
        out = self.encoder(emb, src_key_padding_mask=pad_mask)
        return self.output_norm(out)  # (B, L, hidden_size)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class _FeatureBert4RecDataset(Dataset):
    """Cloze-task dataset for the feature-based model.

    Sequences contain warm-local token IDs (warm_local_idx + ITEM_OFFSET), so
    labels are simply token - ITEM_OFFSET (already warm-local, no remapping needed).
    The model's training logits have shape (B, L, n_warm), consistent with this.
    At least one position per sequence is always masked to prevent zero-gradient
    batches on very short sequences (common when avg conversation length is 3-4).
    """

    def __init__(
        self,
        sequences: list[list[int]],
        n_warm: int,
        max_seq_len: int,
        mask_prob: float,
    ) -> None:
        self.sequences = sequences
        self.n_warm = n_warm
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx][-self.max_seq_len:]
        masked = list(seq)
        labels = [-100] * len(seq)

        to_mask = [i for i in range(len(seq)) if random.random() < self.mask_prob]
        if not to_mask:
            to_mask = [random.randrange(len(seq))]

        for i in to_mask:
            labels[i] = seq[i] - ITEM_OFFSET  # already warm-local (tokens use warm-local indices)
            r = random.random()
            if r < 0.8:
                masked[i] = MASK_TOKEN
            elif r < 0.9:
                masked[i] = random.randint(0, self.n_warm - 1) + ITEM_OFFSET
            # else: 10% keep original unchanged

        pad_len = self.max_seq_len - len(seq)
        masked = [PAD_TOKEN] * pad_len + masked
        labels = [-100] * pad_len + labels

        return (
            torch.tensor(masked, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

class FeatureBert4RecRecommender(SessionRecommender):
    """Feature-based BERT4Rec with warm/cold split scoring.

    Training: softmax restricted to warm tracks (seen in training interactions)
              so cold tracks do not appear as distractors in the loss.
    Inference: warm and cold tracks scored separately via the shared item_encoder,
               then combined into a single (n_tracks,) score vector.

    Cold-item setup:
      - Pass track_metadata with all tracks (including cold test tracks) to fit()
        so they are added to id_map via extra_track_ids.
      - Include test_tracks parquet in feature_emb_paths so cold tracks get
        non-zero feature vectors.
    """

    RECOMMENDER_NAME = "FeatureBert4Rec"

    def __init__(
        self,
        feature_emb_paths: list[str],
        feature_modalities: list[str] | None = None,
        max_seq_len: int = 50,
        hidden_size: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        mask_prob: float = 0.4,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        warmup_ratio: float = 0.1,
        val_ratio: float = 0.1,
        early_stop_patience: int = 10,
        lr_final_factor: float = 0.05,
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        # Sequential models default to urm_mode="session": each session is its own
        # sequence, no multi-session injection at inference (would break turn order).
        super().__init__(urm_mode=kwargs.pop("urm_mode", "session"), **kwargs)
        self.feature_emb_paths = feature_emb_paths
        self.feature_modalities = feature_modalities or ["metadata-qwen3_embedding_0.6b"]
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.mask_prob = mask_prob
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.val_ratio = val_ratio
        self.early_stop_patience = early_stop_patience
        # lr_final_factor: cosine annealing terminates at lr_final_factor * peak_lr
        # rather than at 0, so late epochs still update weights. The historic
        # default was 0 (max(0.0, ...) in the schedule), which combined with
        # high `epochs` could effectively freeze training near the end.
        self.lr_final_factor = float(lr_final_factor)

        if device == "auto":
            self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device_ = torch.device(device)

        self.model: _FeatureBert4RecModel | None = None
        self._train_long: pl.DataFrame | None = None
        self._feature_dim: int | None = None

        # Set during _fit_model — needed for warm/cold scoring at inference
        self._warm_global_indices: list[int] = []
        self._cold_global_indices: list[int] = []
        self._cold_feature_matrix: np.ndarray | None = None
        # Maps global id_map index → warm-local index (0..n_warm-1).
        # Stored as instance attribute so inference can convert prior track tokens.
        self._global_to_warm_local: dict[int, int] = {}
        # Maps global id_map index → cold-local index (0..n_cold-1).
        # Used at inference to encode cold prior tracks into the sequence.
        self._global_to_cold_local: dict[int, int] = {}

    # ------------------------------------------------------------------
    # LR schedule helper (used by every _fit_model in the family)
    # ------------------------------------------------------------------

    def _make_cosine_lr_lambda(self, total_steps: int, warmup_steps: int):
        """Linear warmup → cosine annealing schedule.

        Anneals from peak_lr down to `self.lr_final_factor * peak_lr`, NOT to
        zero. Historic default was 0, which meant late epochs (when early
        stopping hadn't yet triggered) ran with effectively no updates.
        """
        floor = float(self.lr_final_factor)

        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cos_factor = 0.5 * (1.0 + np.cos(np.pi * progress))   # 1.0 → 0.0 over the run
            return floor + (1.0 - floor) * cos_factor

        return _lr_lambda

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df: pl.DataFrame,
        track_metadata: pl.DataFrame | None = None,
        **kwargs: Any,
    ) -> None:
        self._train_long = explode_music_turns(train_df)
        super().fit(train_df, track_metadata=track_metadata, **kwargs)

    def _build_sequences(self) -> list[list[int]]:
        """Build per-session ordered token sequences using warm-local indices.

        Tokens are warm_local_idx + ITEM_OFFSET so that _build_seq_emb can
        index directly into warm_embs (shape n_warm × hidden) without any
        global→local remapping inside the model.
        Cold tracks in training data (edge case) are silently skipped.
        """
        assert self.id_map is not None and self._train_long is not None
        seqs: list[list[int]] = []
        for _, grp in (
            self._train_long
            .sort(["session_id", "turn_number"])
            .group_by("session_id", maintain_order=True)
        ):
            tokens = [
                self._global_to_warm_local[self.id_map.track_to_idx[t]] + ITEM_OFFSET
                for t in grp["track_id"].to_list()
                if t in self.id_map.track_to_idx
                and self.id_map.track_to_idx[t] in self._global_to_warm_local
            ]
            if len(tokens) >= 2:
                seqs.append(tokens)
        return seqs

    def _fit_model(self, urm: csr_matrix) -> None:
        assert self.id_map is not None and self._train_long is not None

        # Identify warm tracks (those that appear in at least one training interaction)
        warm_track_ids: set[str] = set(self._train_long["track_id"].to_list())
        warm_track_ids &= set(self.id_map.track_to_idx.keys())

        # Sorted lists of global id_map indices for warm and cold tracks
        self._warm_global_indices = sorted(
            self.id_map.track_to_idx[t] for t in warm_track_ids
        )
        self._cold_global_indices = sorted(
            idx for t, idx in self.id_map.track_to_idx.items()
            if t not in warm_track_ids
        )

        # Mappings global → warm/cold-local index.
        # warm: must be set before _build_sequences(), which uses it to filter tokens.
        # cold: only needed at inference; set here so save/load is consistent.
        self._global_to_warm_local = {g: l for l, g in enumerate(self._warm_global_indices)}
        self._global_to_cold_local = {g: l for l, g in enumerate(self._cold_global_indices)}

        sequences = self._build_sequences()

        n_warm = len(self._warm_global_indices)
        n_cold = len(self._cold_global_indices)
        print(
            f"[{self.RECOMMENDER_NAME}] warm={n_warm}, cold={n_cold}, "
            f"sequences={len(sequences)}, device={self.device_}"
        )

        # Build full feature matrix once, then slice into warm / cold
        print(f"[{self.RECOMMENDER_NAME}] Loading feature embeddings: {self.feature_modalities}")
        full_matrix = _build_feature_matrix(
            self.feature_emb_paths, self.feature_modalities, self.id_map
        )
        self._feature_dim = full_matrix.shape[1]

        warm_feature_matrix = full_matrix[self._warm_global_indices]   # (n_warm, feature_dim)
        self._cold_feature_matrix = full_matrix[self._cold_global_indices]  # (n_cold, feature_dim)
        print(f"  warm_features={warm_feature_matrix.shape}, cold_features={self._cold_feature_matrix.shape}")

        # 90/10 train/val split on sequences
        random.shuffle(sequences)
        n_val = max(1, int(len(sequences) * self.val_ratio))
        val_sequences   = sequences[:n_val]
        train_sequences = sequences[n_val:]
        print(f"  train_seqs={len(train_sequences)}, val_seqs={len(val_sequences)}")

        pin = self.device_.type == "cuda"
        train_loader = DataLoader(
            _FeatureBert4RecDataset(train_sequences, n_warm, self.max_seq_len, self.mask_prob),
            batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=pin,
        )
        val_loader = DataLoader(
            _FeatureBert4RecDataset(val_sequences, n_warm, self.max_seq_len, self.mask_prob),
            batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=pin,
        )

        self.model = _FeatureBert4RecModel(
            warm_feature_matrix, self.hidden_size, self.max_seq_len,
            self.n_layers, self.n_heads, self.dropout,
        )
        self.model.to(self.device_)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        total_steps = self.epochs * len(train_loader)
        warmup_steps = max(1, int(total_steps * self.warmup_ratio))

        _lr_lambda = self._make_cosine_lr_lambda(total_steps, warmup_steps)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

        epoch_bar = tqdm(
            range(1, self.epochs + 1),
            desc=f"[{self.RECOMMENDER_NAME}]",
            unit="ep",
            dynamic_ncols=True,
            file=sys.stdout,
        )

        best_val: float = float("inf")
        best_epoch: int = 0
        best_state: dict | None = None
        patience_left: int = self.early_stop_patience

        for epoch in epoch_bar:
            self.model.train()
            total_loss = 0.0
            batch_bar = tqdm(train_loader, desc=f"  ep {epoch:3d}", leave=False, unit="batch", dynamic_ncols=True, file=sys.stdout)
            for masked_seq, labels in batch_bar:
                masked_seq = masked_seq.to(self.device_)
                labels     = labels.to(self.device_)

                logits = self.model(masked_seq)  # (B, L, n_warm)
                loss = F.cross_entropy(
                    logits.view(-1, n_warm),
                    labels.view(-1),
                    ignore_index=-100,
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                batch_bar.set_postfix(loss=f"{loss.item():.4f}")

            train_avg = total_loss / len(train_loader)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for masked_seq, labels in val_loader:
                    masked_seq = masked_seq.to(self.device_)
                    labels     = labels.to(self.device_)
                    logits = self.model(masked_seq)
                    val_loss += F.cross_entropy(
                        logits.view(-1, n_warm),
                        labels.view(-1),
                        ignore_index=-100,
                    ).item()
            val_avg = val_loss / len(val_loader)

            # Early stopping + best-checkpoint tracking on val loss
            improved = val_avg < best_val
            if improved:
                best_val = val_avg
                best_epoch = epoch
                # Clone to CPU to avoid holding two model copies in GPU memory
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience_left = self.early_stop_patience
            else:
                patience_left -= 1

            epoch_bar.set_postfix(
                loss=f"{train_avg:.4f}",
                val=f"{val_avg:.4f}",
                best=f"{best_val:.4f}@{best_epoch}",
                patience=patience_left,
            )

            if patience_left <= 0:
                print(f"\n[{self.RECOMMENDER_NAME}] early stopping at epoch {epoch} "
                      f"(no val improvement for {self.early_stop_patience} epochs); "
                      f"best val={best_val:.4f} at epoch {best_epoch}")
                break

        # Restore best weights for inference
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"[{self.RECOMMENDER_NAME}] restored best checkpoint "
                  f"(val={best_val:.4f}, epoch={best_epoch})")

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    def _score_session_sequence(
        self,
        prior: list[str],
        warm_embs: torch.Tensor,
        cold_embs: torch.Tensor | None,
        all_embs: torch.Tensor,
    ) -> np.ndarray:
        """Score all tracks given an ordered list of prior track IDs.

        all_embs = concat(warm_embs, cold_embs) is used as items_table for the
        sequence so cold prior tracks contribute context instead of being dropped.
        warm_embs / cold_embs are separately used for the warm/cold scoring step.
        """
        assert self.model is not None and self.id_map is not None
        n_warm = warm_embs.shape[0]

        # Build tokens. Unified indexing:
        #   warm item → warm_local + ITEM_OFFSET     (in [2, n_warm+1])
        #   cold item → cold_local + n_warm + ITEM_OFFSET  (in [n_warm+2, n_warm+n_cold+1])
        tokens: list[int] = []
        for t in prior:
            if t not in self.id_map.track_to_idx:
                continue
            g = self.id_map.track_to_idx[t]
            if g in self._global_to_warm_local:
                tokens.append(self._global_to_warm_local[g] + ITEM_OFFSET)
            elif g in self._global_to_cold_local:
                tokens.append(self._global_to_cold_local[g] + n_warm + ITEM_OFFSET)

        tokens = tokens + [MASK_TOKEN]
        tokens = tokens[-self.max_seq_len:]
        pad_len = self.max_seq_len - len(tokens)
        tokens = [PAD_TOKEN] * pad_len + tokens

        x = torch.tensor([tokens], dtype=torch.long, device=self.device_)

        with torch.no_grad():
            hidden = self.model.encode_hidden(x, items_table=all_embs)
            h = hidden[0, -1, :]  # (hidden_size,) — MASK position
            warm_scores = (h @ warm_embs.T).cpu().numpy()
            cold_scores = (h @ cold_embs.T).cpu().numpy() if cold_embs is not None else None

        scores = np.full(self.id_map.n_tracks, -np.inf, dtype=np.float32)
        scores[self._warm_global_indices] = warm_scores
        if cold_scores is not None:
            scores[self._cold_global_indices] = cold_scores

        return scores

    def _score_session_profile(self, profile: csr_matrix) -> np.ndarray:
        raise NotImplementedError  # not reachable: recommend() bypasses this

    # ------------------------------------------------------------------
    # Per-session inference extras hook
    #
    # Subclasses whose _score_session_sequence accepts additional per-session
    # kwargs (e.g. user_idx for user-aware variants, cat_idx/spec_idx for
    # goal- and FiLM-conditioned variants) override this to look up the right
    # values for `sess_id` / `user_id` and return them as a dict. The default
    # is no extras, in which case _score_session_sequence is called with only
    # positional args.
    #
    # IMPORTANT: this is what restores correct behaviour for the
    # userbias/film/goal families after the predict_devset -> predict_fold
    # refactor — they used to be wired up by hand in their per-launcher
    # _predict_per_turn helpers, which no longer exist.
    # ------------------------------------------------------------------

    def _extra_score_kwargs_for_session(
        self, sess_id: str, user_id: str
    ) -> dict[str, Any]:
        return {}

    # ------------------------------------------------------------------
    # recommend — one prediction per session (UserRecommender-compatible)
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
        """Predict ONE next-track recommendation per session.

        context_df: one row per context track per session
            (session_id, user_id, session_date, track_id [, turn_number]).
        Returns: (session_id, user_id, turn, track_ids, scores) — one row per session.
        """
        if self.id_map is None or self.model is None:
            raise RuntimeError("Call fit() before recommend()")
        if max_future_years is None:
            max_future_years = self.max_future_years

        if "track_id" not in context_df.columns:
            context_df = explode_music_turns(context_df)

        # Precompute item encodings once (warm/cold features fixed after fit).
        # all_embs = concat(warm, cold) is used as items_table for the sequence
        # so cold prior tracks contribute context. warm/cold are kept separate
        # for the warm/cold scoring step.
        self.model.eval()
        with torch.no_grad():
            warm_embs = self.model.item_encoder(self.model.feature_matrix)  # (n_warm, hidden)
            if self._cold_feature_matrix is not None and len(self._cold_global_indices) > 0:
                cold_feat = torch.from_numpy(self._cold_feature_matrix).to(self.device_)
                cold_embs = self.model.item_encoder(cold_feat)  # (n_cold, hidden)
                all_embs = torch.cat([warm_embs, cold_embs], dim=0)
            else:
                cold_embs = None
                all_embs = warm_embs

        # One entry per session (some sessions may have zero context turns)
        session_meta = (
            context_df
            .select(["session_id", "user_id", "session_date"])
            .unique(subset=["session_id"])
        )

        # Per-session context tracks in turn order (turn_number is dropped by
        # build_context_df; if present, sort by it for deterministic ordering).
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

        out_session: list[str] = []
        out_user: list[str] = []
        out_tracks: list[list[str]] = []
        out_scores: list[list[float]] = []
        out_fallback: list[list[int]] = []

        for row in session_meta.iter_rows(named=True):
            sess_id = row["session_id"]
            user_id = row["user_id"]
            sd = parse_date(row["session_date"])
            candidate_mask = self._filter_candidate_mask(sd)

            prior = ctx_map.get(sess_id, [])
            prior_idxs = {
                self.id_map.track_to_idx[t]
                for t in prior
                if t in self.id_map.track_to_idx
            }

            if not prior:
                if self.fallback is None:
                    recs, scs = [], []
                else:
                    recs, scs = self.fallback.recommend_one(sess_id, turn, sd, top_k)
                fb_flags = [1] * len(recs)
            else:
                extra = self._extra_score_kwargs_for_session(sess_id, user_id)
                scores = self._score_session_sequence(
                    prior, warm_embs, cold_embs, all_embs, **extra
                )
                recs, scs = self._topk_from_scores(
                    scores, prior_idxs, top_k, candidate_mask, remove_seen
                )
                fb_flags = [0] * len(recs)

            out_session.append(sess_id)
            out_user.append(user_id)
            out_tracks.append(recs)
            out_scores.append(scs)
            out_fallback.append(fb_flags)

        return pl.DataFrame(
            {
                "session_id": out_session,
                "user_id": out_user,
                "turn": [turn] * len(out_session),
                "track_ids": out_tracks,
                "scores": out_scores,
                "fallback_used": out_fallback,
            }
        )

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st.update(
            {
                "feature_emb_paths": self.feature_emb_paths,
                "feature_modalities": self.feature_modalities,
                "max_seq_len": self.max_seq_len,
                "hidden_size": self.hidden_size,
                "n_layers": self.n_layers,
                "n_heads": self.n_heads,
                "dropout": self.dropout,
                "mask_prob": self.mask_prob,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "warmup_ratio": self.warmup_ratio,
                "val_ratio": self.val_ratio,
                "early_stop_patience": self.early_stop_patience,
                "device": str(self.device_),
                "feature_dim": self._feature_dim,
                "warm_global_indices": self._warm_global_indices,
                "cold_global_indices": self._cold_global_indices,
                "cold_feature_matrix": self._cold_feature_matrix,
                "global_to_warm_local": self._global_to_warm_local,
                "global_to_cold_local": self._global_to_cold_local,
                "model_state_dict": (
                    self.model.state_dict() if self.model is not None else None
                ),
            }
        )
        return st

    def _set_model_state(self, state: dict) -> None:
        super()._set_model_state(state)
        self._train_long = None
        self.feature_emb_paths = state["feature_emb_paths"]
        self.feature_modalities = state.get("feature_modalities", ["metadata-qwen3_embedding_0.6b"])
        self.max_seq_len = state["max_seq_len"]
        self.hidden_size = state["hidden_size"]
        self.n_layers = state["n_layers"]
        self.n_heads = state["n_heads"]
        self.dropout = state["dropout"]
        self.mask_prob = state.get("mask_prob", 0.4)
        self.epochs = state["epochs"]
        self.batch_size = state["batch_size"]
        self.lr = state["lr"]
        self.weight_decay = state["weight_decay"]
        self.warmup_ratio = state.get("warmup_ratio", 0.1)
        self.val_ratio = state.get("val_ratio", 0.1)
        self.early_stop_patience = state.get("early_stop_patience", 10)
        self.device_ = torch.device(state.get("device", "cpu"))
        self._feature_dim = state.get("feature_dim")
        self._warm_global_indices = state.get("warm_global_indices", [])
        self._cold_global_indices = state.get("cold_global_indices", [])
        self._cold_feature_matrix = state.get("cold_feature_matrix")
        self._global_to_warm_local = state.get("global_to_warm_local", {})
        # Backward compat: rebuild from _cold_global_indices if not in state
        self._global_to_cold_local = state.get(
            "global_to_cold_local",
            {g: l for l, g in enumerate(self._cold_global_indices)},
        )

        sd = state.get("model_state_dict")
        if sd is not None and self.id_map is not None and self._feature_dim is not None:
            n_warm = len(self._warm_global_indices)
            dummy = np.zeros((n_warm, self._feature_dim), dtype=np.float32)
            self.model = _FeatureBert4RecModel(
                dummy, self.hidden_size, self.max_seq_len,
                self.n_layers, self.n_heads, self.dropout,
            )
            self.model.load_state_dict(sd)
            self.model.to(self.device_)
