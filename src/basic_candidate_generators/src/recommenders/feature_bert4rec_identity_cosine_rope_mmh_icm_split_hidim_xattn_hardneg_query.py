"""hardneg + per-turn QUERY embedding injection at the masked position.

The dataset is TalkPlayData: conversations where the user explicitly describes
what music they want ("something more upbeat", "a chill version of X"). The
current best variant ignores this signal — it sees only the item sequence.

This variant injects, at each MASKED position, the precomputed Qwen3 embedding
of the user's query at that turn. Build_query_text_v2 already folds chat
history + user profile + prior tracks into each query embedding, so even
injecting at one position (the position being predicted) captures the full
"current intent given context" signal.

  seq_emb[t]      = item_encoder(token_t)                                 (normal)
  seq_emb[t_mask] = item_encoder(MASK_TOKEN) + query_proj(query_emb[t])   (only when masked)

At train: BERT-style masking picks K positions; each masked position gets its
own query injection. Non-masked positions: query contribution = 0.
At inference: the MASK position (last in the padded sequence) gets the
target-turn query. Prior positions: query contribution = 0.

Train/inference symmetric. No need to track per-prior-track turn numbers.

Query cache layout (produced by `embeddings-package/scripts/03_encode_queries.py`):
    models/query_emb_cache/<encoder>/<split>.npy           # (N, query_dim)
    models/query_emb_cache/<encoder>/<split>_meta.parquet  # session_id, turn_number, gt_track_id, ...
Splits are merged into one big lookup table (train.npy covers all splitK folds,
dev/blind_a cover the held-out test sessions).
"""

from __future__ import annotations

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

from .feature_bert4rec import (
    ITEM_OFFSET,
    MASK_TOKEN,
    PAD_TOKEN,
    _FeatureBert4RecDataset,  # only used for type/Constants ref; we subclass below
)
from .feature_bert4rec_identity_cosine_rope_mmh_icm_hardneg import (
    _build_hard_negatives,
)
from .feature_bert4rec_identity_cosine_rope_mmh_icm_xattn import (
    _MMHICMXAttnModel,
)
from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim_xattn_hardneg import (
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegRecommender,
)


# ---------------------------------------------------------------------------
# Query cache loader
# ---------------------------------------------------------------------------

def _load_query_emb_cache(
    cache_dir: Path,
    splits: list[str] = ("train", "dev", "blind_a"),
) -> tuple[np.ndarray, dict[tuple[str, int], int]]:
    """Load query embeddings from one or more split caches and concatenate.

    Returns:
      emb_matrix : (n_total, query_dim) float32. Row 0 is the "no-query"
                   zero vector (reserved index); rows 1..n_total-1 are the
                   loaded embeddings.
      lookup     : (session_id, turn_number) -> row index in emb_matrix
                   (always >= 1 for loaded rows; unknown keys mean "use 0").
    """
    cache_dir = Path(cache_dir)
    embs: list[np.ndarray] = []
    lookup: dict[tuple[str, int], int] = {}
    offset = 1  # reserve row 0 for the zero/no-query vector
    for sp in splits:
        npy = cache_dir / f"{sp}.npy"
        meta_p = cache_dir / f"{sp}_meta.parquet"
        if not npy.exists() or not meta_p.exists():
            print(f"  [query-cache:{sp}] missing (npy={npy.exists()}, meta={meta_p.exists()}) — skipping")
            continue
        emb = np.load(npy).astype(np.float32, copy=False)
        meta = pl.read_parquet(meta_p)
        sess_arr = meta["session_id"].to_list()
        turn_arr = meta["turn_number"].to_list()
        for i, (sid, tn) in enumerate(zip(sess_arr, turn_arr)):
            lookup[(sid, int(tn))] = offset + i
        embs.append(emb)
        offset += len(emb)
        print(f"  [query-cache:{sp}] {len(emb)} queries (offset now {offset})")
    if not embs:
        raise FileNotFoundError(f"No query cache found in {cache_dir} for splits {splits}")
    query_dim = embs[0].shape[1]
    zero_row = np.zeros((1, query_dim), dtype=np.float32)
    full = np.concatenate([zero_row] + embs, axis=0)
    return full, lookup


# ---------------------------------------------------------------------------
# Dataset — emits (masked_tokens, labels, query_idx_seq) tuples
# ---------------------------------------------------------------------------

class _FeatureBert4RecQueryDataset(Dataset):
    """Cloze-task dataset that also emits per-position query indices.

    Each sequence is a list of (token, query_idx) pairs aligned 1-to-1. After
    masking, query_idx_seq[i] is nonzero ONLY at positions where labels[i] !=
    -100 (i.e., positions to predict). Non-masked positions and padding get
    query_idx = 0 (which the model maps to the zero query vector → no
    contribution).
    """

    def __init__(
        self,
        sequences: list[list[tuple[int, int]]],
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        import random
        seq = self.sequences[idx][-self.max_seq_len:]
        tokens   = [tok for tok, _ in seq]
        q_idxs   = [qi  for _,  qi in seq]
        masked   = list(tokens)
        labels   = [-100] * len(tokens)
        q_at_mask = [0]   * len(tokens)

        to_mask = [i for i in range(len(tokens)) if random.random() < self.mask_prob]
        if not to_mask:
            to_mask = [random.randrange(len(tokens))]

        for i in to_mask:
            labels[i] = tokens[i] - ITEM_OFFSET
            r = random.random()
            if r < 0.8:
                masked[i] = MASK_TOKEN
            elif r < 0.9:
                masked[i] = random.randint(0, self.n_warm - 1) + ITEM_OFFSET
            q_at_mask[i] = q_idxs[i]

        pad_len = self.max_seq_len - len(tokens)
        masked    = [PAD_TOKEN] * pad_len + masked
        labels    = [-100]      * pad_len + labels
        q_at_mask = [0]         * pad_len + q_at_mask

        return (
            torch.tensor(masked,    dtype=torch.long),
            torch.tensor(labels,    dtype=torch.long),
            torch.tensor(q_at_mask, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Model — adds a query_table buffer and a query projector
# ---------------------------------------------------------------------------

class _MMHICMXAttnQueryModel(_MMHICMXAttnModel):
    """xattn model + frozen query embedding table + learned linear projector."""

    def __init__(
        self,
        *args: Any,
        query_emb_table: np.ndarray,   # (n_queries+1, query_dim); row 0 = zeros
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Non-persistent buffer: large derived artifact, rebuilt from the
        # on-disk cache on load (state_dict stays small).
        self.register_buffer(
            "query_table",
            torch.from_numpy(query_emb_table),
            persistent=False,
        )
        query_dim = query_emb_table.shape[1]
        self.query_proj = nn.Linear(query_dim, self.hidden_size, bias=True)
        # Zero-init the projection so training starts identical to the
        # parent (no query contribution at step 0) — lets the model learn
        # WHEN to use the query rather than being forced from epoch 1.
        nn.init.zeros_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)

    def _apply_query(self, emb: torch.Tensor, query_idx_seq: torch.Tensor) -> torch.Tensor:
        """Add query_proj(query_table[query_idx_seq]) to emb at each position.

        emb           : (B, L, hidden_size)
        query_idx_seq : (B, L) long — 0 means "no query", >=1 indexes table
        """
        if query_idx_seq is None:
            return emb
        q = self.query_table[query_idx_seq]            # (B, L, query_dim)
        return emb + self.query_proj(q)                 # (B, L, hidden)

    def forward(self, x: torch.Tensor, query_idx_seq: torch.Tensor | None = None) -> torch.Tensor:
        warm_embs = self.item_encoder(self.feature_matrix)
        emb, pad_mask = self._build_seq_emb(x, warm_embs)
        emb = self._apply_query(emb, query_idx_seq)
        out = self.encoder(emb, src_key_padding_mask=pad_mask)
        out = self.output_norm(out)
        out_n  = F.normalize(out,      dim=-1)
        warm_n = F.normalize(warm_embs, dim=-1)
        return (out_n @ warm_n.T) / self.tau

    def encode_hidden(
        self,
        x: torch.Tensor,
        items_table: torch.Tensor | None = None,
        query_idx_seq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if items_table is None:
            items_table = self.item_encoder(self.feature_matrix)
        emb, pad_mask = self._build_seq_emb(x, items_table)
        emb = self._apply_query(emb, query_idx_seq)
        out = self.encoder(emb, src_key_padding_mask=pad_mask)
        return self.output_norm(out)


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

class FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryRecommender(
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegRecommender,
):
    """split_hidim + xattn + hardneg + per-turn user-query injection."""

    RECOMMENDER_NAME = "FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQuery"

    def __init__(
        self,
        *args: Any,
        query_emb_dir: str = "models/query_emb_cache/qwen3_frozen",
        query_cache_splits: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.query_emb_dir = str(query_emb_dir)
        self.query_cache_splits = list(query_cache_splits) if query_cache_splits else ["train", "dev", "blind_a"]
        self._query_emb_table: np.ndarray | None = None
        self._query_lookup: dict[tuple[str, int], int] = {}
        self._session_target_turn: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_query_emb_dir(self) -> Path:
        """Resolve query_emb_dir relative to repo root if not absolute."""
        p = Path(self.query_emb_dir)
        if p.is_absolute():
            return p
        from launchers._predict_fold_common import repo_path
        return repo_path(self.query_emb_dir)

    def _load_query_cache(self) -> None:
        cache_dir = self._resolve_query_emb_dir()
        print(f"[{self.RECOMMENDER_NAME}] Loading query cache from {cache_dir}")
        emb, lookup = _load_query_emb_cache(cache_dir, self.query_cache_splits)
        self._query_emb_table = emb
        self._query_lookup    = lookup
        print(f"  query_emb_table: {emb.shape}  (row 0 is zero); {len(lookup)} (sess, turn) keys")

    def _build_sequences_with_queries(self) -> list[list[tuple[int, int]]]:
        """Per-session ordered (warm-token, query_idx) sequences.

        For each row in self._train_long (sorted by session_id, turn_number):
          - warm-local token = warm_idx + ITEM_OFFSET
          - query_idx        = lookup[(session_id, turn_number)] or 0 (no query)
        Sequences with <2 valid tokens are dropped (matches parent semantics).
        """
        assert self.id_map is not None and self._train_long is not None
        seqs: list[list[tuple[int, int]]] = []
        n_missing = 0
        n_total   = 0
        for _, grp in (
            self._train_long
            .sort(["session_id", "turn_number"])
            .group_by("session_id", maintain_order=True)
        ):
            sid_col = grp["session_id"].to_list()
            tn_col  = grp["turn_number"].to_list()
            tk_col  = grp["track_id"].to_list()
            pairs: list[tuple[int, int]] = []
            for sid, tn, tid in zip(sid_col, tn_col, tk_col):
                gidx = self.id_map.track_to_idx.get(tid)
                if gidx is None or gidx not in self._global_to_warm_local:
                    continue
                token = self._global_to_warm_local[gidx] + ITEM_OFFSET
                qidx  = self._query_lookup.get((sid, int(tn)), 0)
                n_total += 1
                if qidx == 0:
                    n_missing += 1
                pairs.append((token, qidx))
            if len(pairs) >= 2:
                seqs.append(pairs)
        cov = 1.0 - (n_missing / max(1, n_total))
        print(f"  built {len(seqs)} train sequences; query coverage: "
              f"{n_total - n_missing}/{n_total} = {cov:.1%}")
        return seqs

    def _build_train_val_sequences_with_queries(self) -> tuple[
        list[list[tuple[int, int]]],
        list[list[tuple[int, int]]],
    ]:
        """Same random-split logic as the parent but on (token, query) pairs."""
        import random
        sequences = self._build_sequences_with_queries()
        random.shuffle(sequences)
        n_val = max(1, int(len(sequences) * self.val_ratio))
        return sequences[n_val:], sequences[:n_val]

    def _make_query_dataset(
        self,
        sequences: list[list[tuple[int, int]]],
        n_warm: int,
        max_seq_len: int,
        mask_prob: float,
        is_train: bool = True,
    ) -> Dataset:
        """Factory hook so subclasses can swap the dataset implementation
        (e.g. the `query_full` variant injects queries at all positions, not
        only at masked ones, and the `cutaug` variants augment the TRAIN split
        with random-threshold prefix cuts while leaving val unaugmented)."""
        return _FeatureBert4RecQueryDataset(sequences, n_warm, max_seq_len, mask_prob)

    def _make_model(self, warm_feature_matrix: np.ndarray, modality_dims: list[int]):
        """Override: instantiate the xattn model + query head."""
        assert self._query_emb_table is not None, "_load_query_cache must run before _make_model"
        return _MMHICMXAttnQueryModel(
            warm_feature_matrix, self.hidden_size, self.max_seq_len,
            self.n_layers, self.n_heads, self.dropout,
            init_tau=self.init_tau,
            modality_dims=modality_dims,
            query_emb_table=self._query_emb_table,
        )

    # ------------------------------------------------------------------
    # Training (copies parent's _fit_model and swaps Dataset / forward call)
    # ------------------------------------------------------------------

    def _fit_model(self, urm: csr_matrix) -> None:
        assert self.id_map is not None and self._train_long is not None

        warm_track_ids: set[str] = set(self._train_long["track_id"].to_list())
        warm_track_ids &= set(self.id_map.track_to_idx.keys())
        self._warm_global_indices = sorted(self.id_map.track_to_idx[t] for t in warm_track_ids)
        self._cold_global_indices = sorted(
            idx for t, idx in self.id_map.track_to_idx.items() if t not in warm_track_ids
        )
        self._global_to_warm_local = {g: l for l, g in enumerate(self._warm_global_indices)}
        self._global_to_cold_local = {g: l for l, g in enumerate(self._cold_global_indices)}

        n_warm = len(self._warm_global_indices)
        n_cold = len(self._cold_global_indices)

        # 1. Query cache must be loaded BEFORE we build sequences (we look up
        # per-(sess, turn) query indices during sequence construction).
        self._load_query_cache()

        print(f"[{self.RECOMMENDER_NAME}] Loading per-modality features (with ICM)...")
        full_matrix, modality_dims = self._build_modality_feature_matrix()
        self._feature_dim = full_matrix.shape[1]
        self._modality_dims = modality_dims
        warm_feature_matrix = full_matrix[self._warm_global_indices]
        self._cold_feature_matrix = full_matrix[self._cold_global_indices]

        train_sequences, val_sequences = self._build_train_val_sequences_with_queries()
        print(
            f"[{self.RECOMMENDER_NAME}] warm={n_warm}, cold={n_cold}, "
            f"train_seqs={len(train_sequences)}, val_seqs={len(val_sequences)}, "
            f"device={self.device_}, init_tau={self.init_tau}, "
            f"hardneg_k={self.hardneg_k}, hardneg_weight={self.hardneg_weight}, "
            f"hardneg_tau={self.hardneg_tau}"
        )

        # 2. Hard negatives (same as parent — feature-cosine top-K).
        print(f"[{self.RECOMMENDER_NAME}] Pre-computing hard negatives (k={self.hardneg_k})...")
        feat_l2_chunks = []
        s = 0
        for d in modality_dims:
            chunk = warm_feature_matrix[:, s:s + d]
            norms = np.linalg.norm(chunk, axis=1, keepdims=True).clip(min=1e-10)
            feat_l2_chunks.append(chunk / norms)
            s += d
        feat_l2 = np.concatenate(feat_l2_chunks, axis=1).astype(np.float32)
        feat_l2 = feat_l2 / np.linalg.norm(feat_l2, axis=1, keepdims=True).clip(min=1e-10)
        hard_neg_idx = _build_hard_negatives(feat_l2, k=self.hardneg_k)
        hard_neg_tensor = torch.from_numpy(hard_neg_idx).to(self.device_)

        pin = self.device_.type == "cuda"
        train_loader = DataLoader(
            self._make_query_dataset(train_sequences, n_warm, self.max_seq_len, self.mask_prob, is_train=True),
            batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=pin,
        )
        val_loader = DataLoader(
            self._make_query_dataset(val_sequences, n_warm, self.max_seq_len, self.mask_prob, is_train=False),
            batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=pin,
        )

        self.model = self._make_model(warm_feature_matrix, modality_dims)
        self._pca_init_encoder_per_modality(warm_feature_matrix, modality_dims)
        self.model.to(self.device_)
        # Diagnostic: report query head parameter count. Subclasses may use a
        # different head name (e.g. `query_fusion` in qmod) — pick whichever
        # exists.
        for head_name in ("query_proj", "query_fusion"):
            head = getattr(self.model, head_name, None)
            if head is not None:
                n_qp = sum(p.numel() for p in head.parameters())
                print(f"  {head_name} params: {n_qp:,}")
                break

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        total_steps = self.epochs * len(train_loader)
        warmup_steps = max(1, int(total_steps * self.warmup_ratio))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, self._make_cosine_lr_lambda(total_steps, warmup_steps)
        )

        epoch_bar = tqdm(range(1, self.epochs + 1), desc=f"[{self.RECOMMENDER_NAME}]",
                          unit="ep", dynamic_ncols=True, file=sys.stdout)

        # Live-model best tracking
        best_val = float("inf")
        best_epoch = 0
        best_state: dict | None = None
        patience_left = self.early_stop_patience

        # EMA shadow: identical setup to the parent hardneg variant. Only
        # parameters (not buffers like query_table / RoPE caches) are EMA-ed.
        ema_enabled = self.ema_decay > 0.0
        ema_params: dict[str, torch.Tensor] = {}
        ema_best_val = float("inf")
        ema_best_epoch = 0
        ema_best_state: dict | None = None
        if ema_enabled:
            for n, p in self.model.named_parameters():
                ema_params[n] = p.detach().clone()
            print(f"[{self.RECOMMENDER_NAME}] EMA enabled: decay={self.ema_decay}, "
                  f"start_epoch={self.ema_start_epoch}, n_params={len(ema_params)}")

        for epoch in epoch_bar:
            self.model.train()
            total_loss = 0.0
            total_mlm  = 0.0
            total_hn   = 0.0
            for masked_seq, labels, q_idx in tqdm(
                train_loader, desc=f"  ep {epoch:3d}", leave=False,
                unit="batch", dynamic_ncols=True, file=sys.stdout,
            ):
                masked_seq = masked_seq.to(self.device_)
                labels     = labels.to(self.device_)
                q_idx      = q_idx.to(self.device_)

                logits = self.model(masked_seq, query_idx_seq=q_idx)
                loss_mlm = F.cross_entropy(
                    logits.view(-1, n_warm), labels.view(-1), ignore_index=-100
                )

                with torch.no_grad():
                    flat_logits = logits.view(-1, n_warm)
                    flat_labels = labels.view(-1)
                    valid = flat_labels != -100
                tgt_pos = valid.nonzero(as_tuple=False).squeeze(-1)
                if tgt_pos.numel() > 0:
                    tgt_idx = flat_labels[tgt_pos]
                    anchor = flat_logits[tgt_pos, tgt_idx].unsqueeze(1)
                    hn = hard_neg_tensor[tgt_idx]
                    neg_logits = flat_logits[tgt_pos.unsqueeze(1).expand_as(hn), hn]
                    hn_logits = torch.cat([anchor, neg_logits], dim=1) / self.hardneg_tau
                    hn_labels = torch.zeros(hn_logits.size(0), dtype=torch.long,
                                             device=hn_logits.device)
                    loss_hn = F.cross_entropy(hn_logits, hn_labels)
                else:
                    loss_hn = torch.zeros((), device=self.device_)

                loss = loss_mlm + self._get_hardneg_weight(epoch) * loss_hn

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                total_mlm  += loss_mlm.item()
                total_hn   += loss_hn.item()

                # EMA update — after warmup epochs to avoid pulling the shadow
                # toward early high-LR junk. Same recipe as the parent hardneg.
                if ema_enabled and epoch >= self.ema_start_epoch:
                    with torch.no_grad():
                        d = self.ema_decay
                        for n, p in self.model.named_parameters():
                            ema_params[n].mul_(d).add_(p.detach(), alpha=1.0 - d)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for masked_seq, labels, q_idx in val_loader:
                    masked_seq = masked_seq.to(self.device_)
                    labels     = labels.to(self.device_)
                    q_idx      = q_idx.to(self.device_)
                    logits = self.model(masked_seq, query_idx_seq=q_idx)
                    val_loss += F.cross_entropy(
                        logits.view(-1, n_warm), labels.view(-1), ignore_index=-100
                    ).item()
            val_avg = val_loss / len(val_loader)
            train_avg = total_loss / len(train_loader)

            # EMA val: temporarily swap params, eval, swap back. Only after
            # ema_start_epoch so the shadow has had a chance to fill.
            ema_val_avg: float | None = None
            ema_improved = False
            if ema_enabled and epoch >= self.ema_start_epoch:
                live_backup = {n: p.detach().clone() for n, p in self.model.named_parameters()}
                with torch.no_grad():
                    for n, p in self.model.named_parameters():
                        p.data.copy_(ema_params[n])
                    ema_val = 0.0
                    for masked_seq, labels, q_idx in val_loader:
                        masked_seq = masked_seq.to(self.device_)
                        labels     = labels.to(self.device_)
                        q_idx      = q_idx.to(self.device_)
                        logits = self.model(masked_seq, query_idx_seq=q_idx)
                        ema_val += F.cross_entropy(
                            logits.view(-1, n_warm), labels.view(-1), ignore_index=-100
                        ).item()
                    ema_val_avg = ema_val / len(val_loader)
                    for n, p in self.model.named_parameters():
                        p.data.copy_(live_backup[n])
                if ema_val_avg < ema_best_val:
                    ema_best_val = ema_val_avg
                    ema_best_epoch = epoch
                    # Snapshot: live state_dict overridden with EMA params.
                    snap = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    for n in ema_params:
                        snap[n] = ema_params[n].detach().cpu().clone()
                    ema_best_state = snap
                    ema_improved = True

            improved = val_avg < best_val
            if improved:
                best_val = val_avg
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            # Patience resets if either live or EMA improved this epoch.
            if improved or ema_improved:
                patience_left = self.early_stop_patience
            else:
                patience_left -= 1

            postfix = dict(
                loss=f"{train_avg:.4f}",
                mlm=f"{total_mlm/len(train_loader):.4f}",
                hn=f"{total_hn/len(train_loader):.4f}",
                val=f"{val_avg:.4f}",
                best=f"{best_val:.4f}@{best_epoch}",
                tau=f"{self.model.tau.item():.4f}",
                patience=patience_left,
            )
            if ema_val_avg is not None:
                postfix["ema_val"]  = f"{ema_val_avg:.4f}"
                postfix["ema_best"] = f"{ema_best_val:.4f}@{ema_best_epoch}"
            epoch_bar.set_postfix(**postfix)
            if patience_left <= 0:
                print(f"\n[{self.RECOMMENDER_NAME}] early stopping at epoch {epoch}; "
                      f"best live val={best_val:.4f}@{best_epoch}, "
                      f"best ema val={ema_best_val:.4f}@{ema_best_epoch}")
                break

        # Final restore: pick whichever of (live_best, ema_best) has lower val.
        # EMA-best is taken only if it's strictly better; ties go to live.
        use_ema = (
            ema_enabled
            and ema_best_state is not None
            and ema_best_val < best_val
        )
        if use_ema:
            self.model.load_state_dict(ema_best_state)
            print(f"[{self.RECOMMENDER_NAME}] restored EMA best "
                  f"(ema_val={ema_best_val:.4f}@{ema_best_epoch} < live_val={best_val:.4f}@{best_epoch})")
        elif best_state is not None:
            self.model.load_state_dict(best_state)
            tag = "live best" if not ema_enabled else f"live best (EMA worse: {ema_best_val:.4f})"
            print(f"[{self.RECOMMENDER_NAME}] restored {tag} "
                  f"(val={best_val:.4f}, epoch={best_epoch}, tau={self.model.tau.item():.4f})")

    # ------------------------------------------------------------------
    # Inference — populate the target_turn map, then thread query at MASK
    # ------------------------------------------------------------------

    def recommend(self, context_df: pl.DataFrame, *args: Any, **kwargs: Any) -> pl.DataFrame:
        """Cache the per-session target_turn so _extra_score_kwargs can look up
        the right query embedding at scoring time."""
        if "target_turn" in context_df.columns:
            unique_df = context_df.select(["session_id", "target_turn"]).unique(subset=["session_id"])
            self._session_target_turn = {
                row["session_id"]: int(row["target_turn"])
                for row in unique_df.iter_rows(named=True)
            }
        else:
            self._session_target_turn = {}
        return super().recommend(context_df, *args, **kwargs)

    def _extra_score_kwargs_for_session(self, sess_id: str, user_id: str) -> dict[str, Any]:
        target_turn = self._session_target_turn.get(sess_id)
        if target_turn is None:
            return {"target_query_idx": 0}
        qidx = self._query_lookup.get((sess_id, target_turn), 0)
        return {"target_query_idx": qidx}

    def _score_session_sequence(
        self,
        prior: list[str],
        warm_embs: torch.Tensor,
        cold_embs: torch.Tensor | None,
        all_embs: torch.Tensor,
        target_query_idx: int = 0,
    ) -> np.ndarray:
        """Same sequence-build as the parent, but ALSO feeds query_idx_seq
        with a single non-zero entry at the MASK (last) position."""
        assert self.model is not None and self.id_map is not None
        n_warm = warm_embs.shape[0]

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

        # query_idx_seq: 0 everywhere except the MASK position (last index).
        q_idx_seq = [0] * self.max_seq_len
        q_idx_seq[-1] = int(target_query_idx)

        x = torch.tensor([tokens],     dtype=torch.long, device=self.device_)
        q = torch.tensor([q_idx_seq],  dtype=torch.long, device=self.device_)

        with torch.no_grad():
            hidden = self.model.encode_hidden(x, items_table=all_embs, query_idx_seq=q)
            h = hidden[0, -1, :]
            h_n  = F.normalize(h, dim=-1)
            tau  = self.model.tau
            warm_n = F.normalize(warm_embs, dim=-1)
            warm_scores = ((h_n @ warm_n.T) / tau).cpu().numpy()
            if cold_embs is not None:
                cold_n = F.normalize(cold_embs, dim=-1)
                cold_scores = ((h_n @ cold_n.T) / tau).cpu().numpy()
            else:
                cold_scores = None

        scores = np.full(self.id_map.n_tracks, -np.inf, dtype=np.float32)
        scores[self._warm_global_indices] = warm_scores
        if cold_scores is not None:
            scores[self._cold_global_indices] = cold_scores
        return scores

    # ------------------------------------------------------------------
    # save / load — the query_table buffer is non-persistent (not in
    # state_dict), so on load we rebuild it from disk before load_state_dict.
    # ------------------------------------------------------------------

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st["query_emb_dir"]      = self.query_emb_dir
        st["query_cache_splits"] = self.query_cache_splits
        return st

    def _set_model_state(self, state: dict) -> None:
        self.query_emb_dir      = state.get("query_emb_dir", self.query_emb_dir)
        self.query_cache_splits = state.get("query_cache_splits", self.query_cache_splits)
        # Load query cache BEFORE super so that super()._set_model_state →
        # _make_model can pass _query_emb_table into the model constructor
        # (otherwise _make_model asserts and load fails). The buffer is
        # non-persistent, so it would not be in the state_dict anyway.
        self._load_query_cache()
        super()._set_model_state(state)
        # Defensive: ensure the buffer matches the freshly-loaded table and
        # lives on the right device (super may have moved the model).
        if self.model is not None:
            self.model.query_table = torch.from_numpy(self._query_emb_table).to(self.device_)
