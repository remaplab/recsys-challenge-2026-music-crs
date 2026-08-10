"""Two-Tower v2 recommender — SBERT text embeddings + InfoNCE training.

Same architecture as launchers_crossvalidation/test/two_tower_v2_cv.py,
wrapped as a UserRecommender-compatible class for integration with:
  - launchers_crossvalidation/tune_crossvalidation.py  (splitK 5-fold CV)
  - launchers/predict_blind.py                          (blind submission)

The class handles two data paths automatically:
  - tune_crossvalidation.py: train_df has no 'conversations' column (pre-exploded
    splitK files) → loads raw data from raw_train_path / raw_test_path fixed_params
  - predict_blind.py: train_df IS the full raw data (with 'conversations') →
    uses it directly without re-loading

Text embeddings are pre-computed once in fit() and cached as
{(session_id, turn_number): np.ndarray} for all sessions in the raw source.
encode_additional(raw_df) extends the cache for new sessions (e.g. blind_a).

At recommend() time, each session's text embedding is looked up by
(session_id, target_turn). For blind-style sessions, embeddings are also
created for the last unanswered user turn, including cold-start sessions with
no previous music turns. Unknown sessions fall back to a zero vector.
"""

from __future__ import annotations

import gc
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .user_base import UserRecommender

_HERE = Path(__file__).resolve()
_PKG_ROOT = _HERE.parent.parent.parent  # src/basic_candidate_generators/
_REPO_ROOT = _PKG_ROOT.parent.parent

PAD, UNK = 0, 1
TEXT_DIM    = 384
MAX_CTX_HIGH = 10   # samples stored at this depth; actual max_ctx used is ≤ this
MAX_TAGS     = 10
_MAX_EPOCHS  = 20
_PATIENCE    = 5

# Module-level cache keyed by (raw_train_path, raw_test_path).
# Shared across all TwoTowerV2Recommender instances in the same process, so
# tune_crossvalidation.py computes SBERT embeddings only once per tuning run.
_TT_PRECOMPUTE_CACHE: dict[tuple[str, str], dict] = {}


def _repo_path(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else _REPO_ROOT / path


# ---------------------------------------------------------------------------
# Vocabulary & feature helpers
# ---------------------------------------------------------------------------

def _build_vocab(values: list, max_size: int | None = None) -> dict:
    counter = Counter(v for v in values if v is not None)
    items = [v for v, _ in counter.most_common(max_size - 2)] if max_size else sorted(counter)
    return {v: i + 2 for i, v in enumerate(items)}


def _decade_idx(rd) -> int:
    if not rd or len(str(rd)) < 4:
        return 0
    try:
        y = int(str(rd)[:4])
    except (ValueError, TypeError):
        return 0
    return 0 if y <= 0 else max(1, min(8, (y // 10 - 195) + 1))


def _bin(val, edges: np.ndarray) -> int:
    return 0 if val is None else int(np.digitize(val, edges)) + 1


def _join_text_parts(*parts: str) -> str:
    return " [SEP] ".join(p.strip() for p in parts if p and p.strip())


def _format_user_profile(profile) -> str:
    if not isinstance(profile, dict):
        return ""
    fields = [
        ("age_group", "age group"),
        ("gender", "gender"),
        ("country_name", "country"),
        ("preferred_language", "preferred language"),
        ("preferred_musical_culture", "preferred musical culture"),
    ]
    parts = [
        f"{label}: {profile[key]}"
        for key, label in fields
        if profile.get(key) not in (None, "")
    ]
    return "User profile: " + "; ".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Towers
# ---------------------------------------------------------------------------

class _ItemTower(nn.Module):
    def __init__(self, D: int, dropout: float,
                 n_tracks: int, n_artists: int, n_tags: int):
        super().__init__()
        self.D = D
        self.track_emb  = nn.Embedding(n_tracks  + 2, D,  padding_idx=PAD)
        self.artist_emb = nn.Embedding(n_artists + 2, 64, padding_idx=PAD)
        self.tag_emb    = nn.Embedding(n_tags    + 2, 32, padding_idx=PAD)
        self.decade_emb = nn.Embedding(10, 16, padding_idx=PAD)
        self.pop_emb    = nn.Embedding( 7,  8, padding_idx=PAD)
        self.dur_emb    = nn.Embedding( 7,  8, padding_idx=PAD)
        self.mlp = nn.Sequential(
            nn.Linear(D + 64 + 32 + 16 + 8 + 8, 256),
            nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, D), nn.LayerNorm(D),
        )

    def forward(self, track_idx, artist_idx, tag_idxs, decade_idx, pop_bin, dur_bin):
        tag_v = self.tag_emb(tag_idxs)
        mask  = (tag_idxs != PAD).float().unsqueeze(-1)
        tag_p = (tag_v * mask).sum(-2) / mask.sum(-2).clamp(min=1)
        x = torch.cat([self.track_emb(track_idx), self.artist_emb(artist_idx),
                        tag_p, self.decade_emb(decade_idx),
                        self.pop_emb(pop_bin), self.dur_emb(dur_bin)], dim=-1)
        return self.mlp(x)


class _UserTower(nn.Module):
    def __init__(self, item_tower: _ItemTower, D: int, dropout: float,
                 n_ages: int, n_countries: int, n_genders: int,
                 recency_decay: float = 1.0):
        super().__init__()
        self.item_tower  = item_tower
        self.recency_decay = recency_decay
        self.text_proj   = nn.Sequential(nn.Linear(TEXT_DIM, D), nn.LayerNorm(D))
        self.age_emb     = nn.Embedding(n_ages     + 2, 16, padding_idx=PAD)
        self.country_emb = nn.Embedding(n_countries + 2, 32, padding_idx=PAD)
        self.gender_emb  = nn.Embedding(n_genders  + 2,  8, padding_idx=PAD)
        self.mlp = nn.Sequential(
            nn.Linear(D + D + 56, 256),
            nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, D), nn.LayerNorm(D),
        )

    def forward(self, txt_emb, ctx, ctx_mask, u_age, u_ctr, u_gen):
        txt   = self.text_proj(txt_emb.float())
        B, L  = ctx["track_idx"].shape
        D     = self.item_tower.D
        flat  = {k: v.reshape(B * L, *v.shape[2:]) for k, v in ctx.items()}
        ivecs = self.item_tower(**flat).view(B, L, D)
        msk   = ctx_mask.float().unsqueeze(-1)
        if self.recency_decay < 0.999:
            w = torch.tensor(
                [self.recency_decay ** (L - 1 - i) for i in range(L)],
                dtype=ivecs.dtype,
                device=ivecs.device,
            ).view(1, L, 1)
            msk = msk * w
        ctx_v = (ivecs * msk).sum(1) / msk.sum(1).clamp(min=1)
        demo  = torch.cat([self.age_emb(u_age), self.country_emb(u_ctr), self.gender_emb(u_gen)], dim=-1)
        return self.mlp(torch.cat([txt, ctx_v, demo], dim=-1))


class _TwoTowerModel(nn.Module):
    def __init__(self, D: int, dropout: float,
                 n_tracks: int, n_artists: int, n_tags: int,
                 n_ages: int, n_countries: int, n_genders: int,
                 recency_decay: float = 1.0,
                 hard_neg_weight: float = 1.0):
        super().__init__()
        self.item_tower = _ItemTower(D, dropout, n_tracks, n_artists, n_tags)
        self.user_tower = _UserTower(
            self.item_tower, D, dropout, n_ages, n_countries, n_genders,
            recency_decay=recency_decay,
        )
        self.log_temp   = nn.Parameter(torch.tensor(-2.66))
        self.hard_neg_weight = hard_neg_weight

    @property
    def D(self) -> int:
        return self.item_tower.D

    @property
    def temp(self) -> torch.Tensor:
        return self.log_temp.exp().clamp(0.01, 1.0)

    def forward(self, batch: dict) -> torch.Tensor:
        u = F.normalize(self.user_tower(
            batch["txt_emb"], batch["ctx"], batch["ctx_mask"],
            batch["u_age"], batch["u_ctr"], batch["u_gen"]), dim=-1)
        p = F.normalize(self.item_tower(**batch["pos"]), dim=-1)
        logits = u @ p.T / self.temp
        labels = torch.arange(len(u), device=u.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        if self.hard_neg_weight > 0 and "neg" in batch:
            B, H = batch["neg"]["track_idx"].shape
            flat_neg = {k: v.reshape(B * H, *v.shape[2:]) for k, v in batch["neg"].items()}
            neg = F.normalize(self.item_tower(**flat_neg), dim=-1).view(B, H, -1)
            pos_logits = (u * p).sum(-1, keepdim=True) / self.temp
            neg_logits = torch.einsum("bd,bhd->bh", u, neg) / self.temp
            hard_logits = torch.cat([pos_logits, neg_logits], dim=1)
            hard_labels = torch.zeros(B, dtype=torch.long, device=u.device)
            loss = loss + self.hard_neg_weight * F.cross_entropy(hard_logits, hard_labels)
        return loss


# ---------------------------------------------------------------------------
# Dataset / collate
# ---------------------------------------------------------------------------

class _TwoTowerDataset(Dataset):
    def __init__(self, samples: list) -> None: self._s = samples
    def __len__(self) -> int:                  return len(self._s)
    def __getitem__(self, i: int):             return self._s[i]


def _make_collate(max_ctx: int, track_features: dict, user_features: dict,
                  text_embeds: dict, unk_track: dict, unk_user: dict,
                  n_hard_neg: int = 0, hard_negatives: dict | None = None,
                  fallback_negatives: list | None = None):
    L, MT = max_ctx, MAX_TAGS
    hard_negatives = hard_negatives or {}
    fallback_negatives = fallback_negatives or list(track_features.keys())

    def collate(batch: list) -> dict:
        B = len(batch)
        ctx_ti = np.zeros((B, L), np.int64); ctx_ai = np.zeros((B, L), np.int64)
        ctx_tg = np.zeros((B, L, MT), np.int64)
        ctx_di = np.zeros((B, L), np.int64); ctx_pi = np.zeros((B, L), np.int64)
        ctx_ui = np.zeros((B, L), np.int64); ctx_mk = np.zeros((B, L), bool)
        u_age  = np.zeros(B, np.int64); u_ctr = np.zeros(B, np.int64); u_gen = np.zeros(B, np.int64)
        txt_emb = np.zeros((B, TEXT_DIM), np.float32)
        pos_tids: list = []
        neg_tids: list = []

        for b, s in enumerate(batch):
            ctx = s["context"][-L:]
            for j, tid in enumerate(ctx):
                f = track_features.get(tid, unk_track)
                ctx_ti[b, j] = f["track_idx"]; ctx_ai[b, j] = f["artist_idx"]
                t = f["tag_idxs"][:MT]; ctx_tg[b, j, :len(t)] = t
                ctx_di[b, j] = f["decade_idx"]; ctx_pi[b, j] = f["pop_bin"]
                ctx_ui[b, j] = f["dur_bin"];    ctx_mk[b, j] = True
            uf = user_features.get(s["user_id"], unk_user)
            u_age[b] = uf["age_idx"]; u_ctr[b] = uf["country_idx"]; u_gen[b] = uf["gender_idx"]
            txt_emb[b] = text_embeds.get((s["session_id"], s["turn_number"]),
                                          np.zeros(TEXT_DIM, np.float32))
            pos_tids.append(s["target"])
            if n_hard_neg > 0:
                pool = hard_negatives.get(s["target"]) or fallback_negatives
                banned = set(ctx)
                banned.add(s["target"])
                chosen = []
                seed_text = f"{s['session_id']}|{s['turn_number']}|{s['target']}"
                start = sum((i + 1) * ord(ch) for i, ch in enumerate(seed_text)) % max(1, len(pool))
                for off in range(len(pool)):
                    tid = pool[(start + off) % len(pool)]
                    if tid not in banned:
                        chosen.append(tid)
                    if len(chosen) >= n_hard_neg:
                        break
                if len(chosen) < n_hard_neg:
                    for tid in fallback_negatives:
                        if tid not in banned and tid not in chosen:
                            chosen.append(tid)
                        if len(chosen) >= n_hard_neg:
                            break
                neg_tids.extend(chosen[:n_hard_neg])

        T = lambda a: torch.tensor(a)
        pos_feats = _item_tensors_batch(pos_tids, track_features, unk_track)
        out = {
            "ctx": dict(track_idx=T(ctx_ti), artist_idx=T(ctx_ai), tag_idxs=T(ctx_tg),
                        decade_idx=T(ctx_di), pop_bin=T(ctx_pi), dur_bin=T(ctx_ui)),
            "ctx_mask": T(ctx_mk),
            "pos":      pos_feats,
            "u_age": T(u_age), "u_ctr": T(u_ctr), "u_gen": T(u_gen),
            "txt_emb": T(txt_emb),
        }
        if n_hard_neg > 0:
            neg_feats = _item_tensors_batch(neg_tids, track_features, unk_track)
            out["neg"] = {
                k: v.reshape(B, n_hard_neg, *v.shape[1:])
                for k, v in neg_feats.items()
            }
        return out
    return collate


def _item_tensors_batch(tids: list, track_features: dict, unk_track: dict) -> dict:
    B, MT = len(tids), MAX_TAGS
    ti = np.zeros(B, np.int64); ai = np.zeros(B, np.int64)
    tg = np.zeros((B, MT), np.int64)
    di = np.zeros(B, np.int64); pi = np.zeros(B, np.int64); ui = np.zeros(B, np.int64)
    for i, tid in enumerate(tids):
        f = track_features.get(tid, unk_track)
        ti[i] = f["track_idx"]; ai[i] = f["artist_idx"]
        t = f["tag_idxs"][:MT]; tg[i, :len(t)] = t
        di[i] = f["decade_idx"]; pi[i] = f["pop_bin"]; ui[i] = f["dur_bin"]
    T = lambda a: torch.tensor(a)
    return dict(track_idx=T(ti), artist_idx=T(ai), tag_idxs=T(tg),
                decade_idx=T(di), pop_bin=T(pi), dur_bin=T(ui))


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _to(batch: dict, device) -> dict:
    def _mv(v):
        if isinstance(v, torch.Tensor): return v.to(device)
        if isinstance(v, dict):         return {k: _mv(w) for k, w in v.items()}
        return v
    return {k: _mv(v) for k, v in batch.items()}


def _run_epoch(mdl: _TwoTowerModel, optimizer, loader: DataLoader,
               device, train: bool = True) -> float:
    mdl.train(train)
    total, n = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = _to(batch, device)
            loss  = mdl(batch)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                optimizer.step()
            total += loss.item() * len(batch["u_age"])
            n     += len(batch["u_age"])
    return total / n if n else 0.0


@torch.no_grad()
def _build_item_index(mdl: _TwoTowerModel, track_features: dict,
                      unk_track: dict, device):
    mdl.eval()
    tids, vecs = list(track_features.keys()), []
    for i in range(0, len(tids), 512):
        chunk = tids[i:i+512]
        feats = {k: v.to(device) for k, v in _item_tensors_batch(chunk, track_features, unk_track).items()}
        vecs.append(F.normalize(mdl.item_tower(**feats), dim=-1).cpu().numpy())
    import faiss
    mat = np.vstack(vecs).astype(np.float32)
    idx = faiss.IndexFlatIP(mdl.D)
    idx.add(mat)
    return idx, tids


@torch.no_grad()
def _ranking_metric(
    mdl: _TwoTowerModel,
    samples: list,
    max_ctx: int,
    track_features: dict,
    user_features: dict,
    text_embeds: dict,
    unk_track: dict,
    unk_user: dict,
    device,
    metric: str,
    k: int,
    batch_size: int = 256,
) -> float:
    if not samples:
        return 0.0
    idx, tids = _build_item_index(mdl, track_features, unk_track, device)
    vals: list[float] = []
    search_k = min(len(tids), max(k + max_ctx + 50, k))
    for i in range(0, len(samples), batch_size):
        chunk = samples[i:i + batch_size]
        uvecs = _encode_users(
            mdl, chunk, max_ctx, track_features, user_features, text_embeds,
            unk_track, unk_user, device,
        )
        _, nn_idx = idx.search(uvecs, search_k)
        for b, s in enumerate(chunk):
            seen = set(s["context"])
            recs = []
            for item_idx in nn_idx[b]:
                tid = tids[item_idx]
                if tid not in seen:
                    recs.append(tid)
                if len(recs) >= k:
                    break
            target = s["target"]
            if metric == "recall":
                vals.append(float(target in recs[:k]))
            elif metric == "ndcg":
                try:
                    rank = recs[:k].index(target) + 1
                    vals.append(1.0 / np.log2(rank + 1))
                except ValueError:
                    vals.append(0.0)
            else:
                raise ValueError("early_stop_metric must be 'loss', 'ndcg', or 'recall'")
    return float(np.mean(vals)) if vals else 0.0


@torch.no_grad()
def _encode_users(mdl: _TwoTowerModel, sessions: list, max_ctx: int,
                  track_features: dict, user_features: dict, text_embeds: dict,
                  unk_track: dict, unk_user: dict, device) -> np.ndarray:
    mdl.eval()
    B, L, MT = len(sessions), max_ctx, MAX_TAGS
    ctx_ti = np.zeros((B, L), np.int64); ctx_ai = np.zeros((B, L), np.int64)
    ctx_tg = np.zeros((B, L, MT), np.int64)
    ctx_di = np.zeros((B, L), np.int64); ctx_pi = np.zeros((B, L), np.int64)
    ctx_ui = np.zeros((B, L), np.int64); ctx_mk = np.zeros((B, L), bool)
    u_age  = np.zeros(B, np.int64); u_ctr = np.zeros(B, np.int64); u_gen = np.zeros(B, np.int64)
    txt_emb = np.zeros((B, TEXT_DIM), np.float32)

    for b, s in enumerate(sessions):
        ctx = s["context"][-L:]
        for j, tid in enumerate(ctx):
            f = track_features.get(tid, unk_track)
            ctx_ti[b, j] = f["track_idx"]; ctx_ai[b, j] = f["artist_idx"]
            t = f["tag_idxs"][:MT]; ctx_tg[b, j, :len(t)] = t
            ctx_di[b, j] = f["decade_idx"]; ctx_pi[b, j] = f["pop_bin"]
            ctx_ui[b, j] = f["dur_bin"];    ctx_mk[b, j] = True
        uf = user_features.get(s["user_id"], unk_user)
        u_age[b] = uf["age_idx"]; u_ctr[b] = uf["country_idx"]; u_gen[b] = uf["gender_idx"]
        txt_emb[b] = text_embeds.get((s["session_id"], s["turn_number"]),
                                      np.zeros(TEXT_DIM, np.float32))

    T = lambda a: torch.tensor(a).to(device)
    ctx_t = dict(track_idx=T(ctx_ti), artist_idx=T(ctx_ai), tag_idxs=T(ctx_tg),
                 decade_idx=T(ctx_di), pop_bin=T(ctx_pi), dur_bin=T(ctx_ui))
    u = mdl.user_tower(T(txt_emb), ctx_t, T(ctx_mk), T(u_age), T(u_ctr), T(u_gen))
    return F.normalize(u, dim=-1).cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Recommender class
# ---------------------------------------------------------------------------

class TwoTowerV2Recommender(UserRecommender):
    """Two-Tower v2 with SBERT text embeddings.

    Same architecture as launchers_crossvalidation/test/two_tower_v2_cv.py.
    """

    RECOMMENDER_NAME = "TwoTowerV2"

    def __init__(
        self,
        D:               int   = 128,
        lr:              float = 1e-3,
        wd:              float = 1e-4,
        batch_size:      int   = 512,
        max_ctx:         int   = 7,
        dropout:         float = 0.1,
        patience:        int   = _PATIENCE,
        text_model:      str   = "all-MiniLM-L6-v2",
        n_hard_neg:      int   = 0,
        hard_neg_weight: float = 1.0,
        recency_decay:   float = 1.0,
        early_stop_metric: str = "loss",
        early_stop_k:    int   = 20,
        raw_train_path:  str | None = None,
        raw_test_path:   str | None = None,
        user_meta_path:  str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.D              = D
        self.lr             = lr
        self.wd             = wd
        self.batch_size     = batch_size
        self.max_ctx        = max_ctx
        self.dropout        = dropout
        self.patience       = patience
        self.text_model     = text_model
        self.n_hard_neg     = n_hard_neg
        self.hard_neg_weight = hard_neg_weight
        self.recency_decay  = recency_decay
        self.early_stop_metric = early_stop_metric
        self.early_stop_k   = early_stop_k
        self.raw_train_path = raw_train_path
        self.raw_test_path  = raw_test_path
        self.user_meta_path = user_meta_path

        self._device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._text_embeds:  dict = {}
        self._track_features: dict = {}
        self._user_features:  dict = {}
        self._unk_track:    dict = {}
        self._unk_user:     dict = {}
        self._vocab_sizes:  dict = {}
        self._hard_negatives: dict = {}
        self._fallback_negatives: list = []
        self._item_idx      = None
        self._tids:         list = []
        self._model_state:  dict | None = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df:       pl.DataFrame,
        track_metadata: pl.DataFrame | None = None,
        **kwargs,
    ) -> None:
        print(f"[{self.RECOMMENDER_NAME}] device={self._device}")

        # 1. Raw conversation source
        if "conversations" in train_df.columns:
            raw_source = train_df
            print(f"[{self.RECOMMENDER_NAME}] using train_df with conversations ({train_df.shape[0]:,} sessions)")
        else:
            if not self.raw_train_path or not self.raw_test_path:
                raise RuntimeError(
                    "train_df has no 'conversations' column and raw_train_path/raw_test_path "
                    "are not set. Pass them as fixed_params in tune_crossvalidation.yaml."
                )
            print(f"[{self.RECOMMENDER_NAME}] loading raw data from paths...")
            raw_source = pl.concat([
                pl.read_parquet(_repo_path(self.raw_train_path)),
                pl.read_parquet(_repo_path(self.raw_test_path)),
            ])
            print(f"[{self.RECOMMENDER_NAME}] raw_source: {raw_source.shape[0]:,} sessions")

        # 2. Track & user feature encoding
        if track_metadata is None:
            raise RuntimeError(f"[{self.RECOMMENDER_NAME}] track_metadata is required")
        self._build_features(track_metadata, raw_source)

        # 3. SBERT text embeddings
        self._compute_text_embeds(raw_source)

        # 4. Extract train session IDs from train_df
        from .interactions import explode_music_turns
        long = explode_music_turns(train_df)
        train_sids = set(long["session_id"].unique().to_list())

        # 5. Extract samples
        all_samples = self._extract_samples(raw_source, allowed_sess=train_sids)
        print(f"[{self.RECOMMENDER_NAME}] {len(all_samples):,} training samples from {len(train_sids):,} sessions")

        # 6. Internal 80/20 split by session_id for early stopping
        sids = sorted(train_sids)
        rng  = np.random.RandomState(42)
        rng.shuffle(sids)
        split      = int(len(sids) * 0.8)
        train_set  = set(sids[:split])
        val_set    = set(sids[split:])
        train_samp = [s for s in all_samples if s["session_id"] in train_set]
        val_samp   = [s for s in all_samples if s["session_id"] in val_set]
        print(f"[{self.RECOMMENDER_NAME}] train={len(train_samp):,}  val={len(val_samp):,}")

        # 7. Train
        vs = self._vocab_sizes
        model = _TwoTowerModel(
            D=self.D, dropout=self.dropout,
            n_tracks=vs["n_tracks"], n_artists=vs["n_artists"], n_tags=vs["n_tags"],
            n_ages=vs["n_ages"], n_countries=vs["n_countries"], n_genders=vs["n_genders"],
            recency_decay=self.recency_decay,
            hard_neg_weight=self.hard_neg_weight,
        ).to(self._device)

        opt   = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=_MAX_EPOCHS)

        cfn   = _make_collate(self.max_ctx, self._track_features, self._user_features,
                               self._text_embeds, self._unk_track, self._unk_user,
                               n_hard_neg=self.n_hard_neg,
                               hard_negatives=self._hard_negatives,
                               fallback_negatives=self._fallback_negatives)
        t_ldr = DataLoader(_TwoTowerDataset(train_samp), self.batch_size, shuffle=True,
                           num_workers=2, collate_fn=cfn, pin_memory=True)
        v_ldr = DataLoader(_TwoTowerDataset(val_samp),   self.batch_size, shuffle=False,
                           num_workers=2, collate_fn=cfn, pin_memory=True)

        metric_mode = (self.early_stop_metric or "loss").lower()
        if metric_mode not in {"loss", "ndcg", "recall"}:
            raise ValueError("early_stop_metric must be one of: loss, ndcg, recall")
        best_score = float("inf") if metric_mode == "loss" else -float("inf")
        patience_cnt, best_state = 0, None
        for ep in range(1, _MAX_EPOCHS + 1):
            t0 = time.time()
            tl = _run_epoch(model, opt, t_ldr, self._device, train=True)
            vl = _run_epoch(model, None, v_ldr, self._device, train=False)
            sched.step()
            if metric_mode == "loss":
                score = vl
                improved = score < best_score
                score_msg = f"val={vl:.4f}"
            else:
                score = _ranking_metric(
                    model, val_samp, self.max_ctx,
                    self._track_features, self._user_features, self._text_embeds,
                    self._unk_track, self._unk_user, self._device,
                    metric=metric_mode, k=self.early_stop_k,
                    batch_size=min(512, self.batch_size),
                )
                improved = score > best_score
                score_msg = f"val_loss={vl:.4f}  {metric_mode}@{self.early_stop_k}={score:.4f}"
            flag = ""
            if improved:
                best_score, patience_cnt = score, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                flag = " ★"
            else:
                patience_cnt += 1
            print(f"  ep {ep:02d} | train={tl:.4f}  {score_msg}  temp={model.temp.item():.3f}"
                  f"  {time.time()-t0:.0f}s{flag}")
            if patience_cnt >= self.patience:
                print(f"  early stop")
                break

        if best_state:
            model.load_state_dict({k: v.to(self._device) for k, v in best_state.items()})
        self._model_state = best_state

        # 8. Build item index
        self._item_idx, self._tids = _build_item_index(
            model, self._track_features, self._unk_track, self._device
        )
        self._model = model
        print(f"[{self.RECOMMENDER_NAME}] fit done — {len(self._tids):,} items indexed")

    # ------------------------------------------------------------------
    # encode_additional — pre-compute embeds for extra sessions (e.g. blind_a)
    # ------------------------------------------------------------------

    def encode_additional(self, raw_df: pl.DataFrame) -> None:
        """Extend self._text_embeds with sessions not yet encoded."""
        self._compute_text_embeds(raw_df, incremental=True)

    # ------------------------------------------------------------------
    # recommend — override UserRecommender completely
    # ------------------------------------------------------------------

    def recommend(
        self,
        context_df: pl.DataFrame,
        top_k: int = 200,
        remove_seen: bool = True,
        **kwargs,
    ) -> pl.DataFrame:
        if self._item_idx is None:
            raise RuntimeError(f"[{self.RECOMMENDER_NAME}] call fit() first")

        if "turn_number" in context_df.columns:
            context_df = context_df.sort(["session_id", "turn_number"])

        # Parse context_df: one row per (session, context_track)
        # Columns: session_id, user_id, session_date, track_id, target_turn
        per_sess: dict = {}
        for row in context_df.iter_rows(named=True):
            sid = row["session_id"]
            if sid not in per_sess:
                per_sess[sid] = {
                    "session_id":  sid,
                    "user_id":     row["user_id"],
                    "turn_number": row.get("target_turn") or 0,
                    "context":     [],
                }
            tid = row.get("track_id")
            if tid:
                per_sess[sid]["context"].append(tid)

        sessions = list(per_sess.values())
        rows: list = []
        BS = 256

        for i in range(0, len(sessions), BS):
            chunk = sessions[i:i+BS]
            uvecs = _encode_users(
                self._model, chunk, self.max_ctx,
                self._track_features, self._user_features, self._text_embeds,
                self._unk_track, self._unk_user, self._device,
            )
            scores, I = self._item_idx.search(uvecs, top_k + 50)
            for b, s in enumerate(chunk):
                seen = set(s["context"]) if remove_seen else set()
                cands, cand_sc = [], []
                for rank, idx in enumerate(I[b]):
                    tid = self._tids[idx]
                    if tid not in seen:
                        cands.append(tid)
                        cand_sc.append(float(scores[b, rank]))
                    if len(cands) >= top_k:
                        break
                rows.append({
                    "session_id":   s["session_id"],
                    "turn":         s["turn_number"],
                    "track_ids":    cands,
                    "scores":       cand_sc,
                    "fallback_used": [0] * len(cands),
                })

        return pl.DataFrame(rows, schema={
            "session_id":    pl.Utf8,
            "turn":          pl.Int64,
            "track_ids":     pl.List(pl.Utf8),
            "scores":        pl.List(pl.Float64),
            "fallback_used": pl.List(pl.Int32),
        })

    # ------------------------------------------------------------------
    # Unused stubs required by UserRecommender
    # ------------------------------------------------------------------

    def _fit_model(self, urm) -> None:
        pass  # fit() is fully overridden

    def _score_session_profile(self, profile) -> np.ndarray:
        return np.zeros(0)  # recommend() is fully overridden

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st.update({
            "D": self.D, "lr": self.lr, "wd": self.wd,
            "batch_size": self.batch_size, "max_ctx": self.max_ctx,
            "dropout": self.dropout, "patience": self.patience,
            "text_model": self.text_model,
            "n_hard_neg": self.n_hard_neg,
            "hard_neg_weight": self.hard_neg_weight,
            "recency_decay": self.recency_decay,
            "early_stop_metric": self.early_stop_metric,
            "early_stop_k": self.early_stop_k,
            "raw_train_path": self.raw_train_path,
            "raw_test_path": self.raw_test_path,
            "user_meta_path": self.user_meta_path,
            "_model_state":    self._model_state,
            "_text_embeds":    self._text_embeds,
            "_track_features": self._track_features,
            "_user_features":  self._user_features,
            "_unk_track":      self._unk_track,
            "_unk_user":       self._unk_user,
            "_vocab_sizes":    self._vocab_sizes,
            "_hard_negatives": self._hard_negatives,
            "_fallback_negatives": self._fallback_negatives,
            "_tids":           self._tids,
        })
        return st

    def _set_model_state(self, state: dict) -> None:
        super()._set_model_state(state)
        if not hasattr(self, "_device"):
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for k in ("D", "lr", "wd", "batch_size", "max_ctx", "dropout",
                  "patience", "text_model", "n_hard_neg", "hard_neg_weight",
                  "recency_decay", "early_stop_metric", "early_stop_k",
                  "raw_train_path", "raw_test_path",
                  "user_meta_path"):
            setattr(self, k, state.get(k, getattr(self, k, None)))
        self._model_state    = state.get("_model_state")
        self._text_embeds    = state.get("_text_embeds", {})
        self._track_features = state.get("_track_features", {})
        self._user_features  = state.get("_user_features", {})
        self._unk_track      = state.get("_unk_track", {})
        self._unk_user       = state.get("_unk_user", {})
        self._vocab_sizes    = state.get("_vocab_sizes", {})
        self._hard_negatives = state.get("_hard_negatives", {})
        self._fallback_negatives = state.get("_fallback_negatives", list(self._track_features.keys()))
        self._tids           = state.get("_tids", [])
        if self._model_state and self._vocab_sizes:
            vs = self._vocab_sizes
            self._model = _TwoTowerModel(
                D=self.D, dropout=self.dropout,
                n_tracks=vs["n_tracks"], n_artists=vs["n_artists"], n_tags=vs["n_tags"],
                n_ages=vs["n_ages"], n_countries=vs["n_countries"], n_genders=vs["n_genders"],
                recency_decay=self.recency_decay,
                hard_neg_weight=self.hard_neg_weight,
            ).to(self._device)
            self._model.load_state_dict(
                {k: v.to(self._device) for k, v in self._model_state.items()}
            )
            self._item_idx, _ = _build_item_index(
                self._model, self._track_features, self._unk_track, self._device
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_features(self, track_meta: pl.DataFrame,
                         raw_source: pl.DataFrame) -> None:
        # Return cached features if available (avoids rebuilding vocabs each Optuna trial)
        if self.raw_train_path and self.raw_test_path:
            _ck = (str(self.raw_train_path), str(self.raw_test_path))
            if _ck in _TT_PRECOMPUTE_CACHE and "track_features" in _TT_PRECOMPUTE_CACHE[_ck]:
                _c = _TT_PRECOMPUTE_CACHE[_ck]
                self._track_features = _c["track_features"]
                self._user_features  = _c["user_features"]
                self._unk_track      = _c["unk_track"]
                self._unk_user       = _c["unk_user"]
                self._vocab_sizes    = _c["vocab_sizes"]
                self._hard_negatives = _c.get("hard_negatives", {})
                self._fallback_negatives = _c.get("fallback_negatives", list(self._track_features.keys()))
                print(f"[{self.RECOMMENDER_NAME}] features restored from cache")
                return

        track_vocab   = _build_vocab(track_meta["track_id"].to_list())
        artist_vocab  = _build_vocab(track_meta["artist_id"].explode().drop_nulls().to_list())
        all_tags      = [t for tags in track_meta["tag_list"].to_list() if tags for t in tags]
        tag_vocab     = _build_vocab(all_tags, max_size=2000)

        pop_nz    = track_meta["popularity"].drop_nulls().to_numpy()
        pop_nz    = pop_nz[pop_nz > 0]
        pop_edges = np.percentile(pop_nz, [20, 40, 60, 80]) if len(pop_nz) >= 5 \
                    else np.array([.25, .5, .75, 1.])
        dur_edges = np.array([120_000, 180_000, 240_000, 360_000], dtype=np.float64)

        tf: dict = {}
        for r in track_meta.iter_rows(named=True):
            tid     = r["track_id"]
            tags    = r.get("tag_list") or []
            artists = r.get("artist_id") or []
            top_tags = sorted(tags, key=lambda t: tag_vocab.get(t, 0), reverse=True)[:MAX_TAGS]
            tf[tid] = {
                "track_idx":  track_vocab.get(tid, UNK),
                "artist_idx": artist_vocab.get(artists[0] if artists else None, UNK),
                "tag_idxs":   [tag_vocab.get(t, UNK) for t in top_tags],
                "decade_idx": _decade_idx(r.get("release_date")),
                "pop_bin":    _bin(r.get("popularity"), pop_edges),
                "dur_bin":    _bin(r.get("duration"),   dur_edges),
            }
        self._track_features = tf
        self._unk_track = {
            "track_idx": UNK, "artist_idx": UNK, "tag_idxs": [],
            "decade_idx": 0, "pop_bin": 0, "dur_bin": 0,
        }

        # User features
        age_vocab = country_vocab = gender_vocab = {}
        uf: dict = {}
        if "user_id" in raw_source.columns and self.user_meta_path:
            user_meta  = pl.read_parquet(_repo_path(self.user_meta_path))
            age_vocab     = _build_vocab(user_meta["age_group"].to_list())
            country_vocab = _build_vocab(user_meta["country_code"].to_list())
            gender_vocab  = _build_vocab(user_meta["gender"].to_list())
            for r in user_meta.iter_rows(named=True):
                uf[r["user_id"]] = {
                    "age_idx":     age_vocab.get(r["age_group"],    UNK),
                    "country_idx": country_vocab.get(r["country_code"], UNK),
                    "gender_idx":  gender_vocab.get(r["gender"],    UNK),
                }
        self._user_features = uf
        self._unk_user = {"age_idx": UNK, "country_idx": UNK, "gender_idx": UNK}

        self._vocab_sizes = {
            "n_tracks":    len(track_vocab),
            "n_artists":   len(artist_vocab),
            "n_tags":      len(tag_vocab),
            "n_ages":      len(age_vocab),
            "n_countries": len(country_vocab),
            "n_genders":   len(gender_vocab),
        }
        print(f"[{self.RECOMMENDER_NAME}] tracks={len(track_vocab)}  "
              f"artists={len(artist_vocab)}  tags={len(tag_vocab)}")
        self._build_hard_negative_pools()

        # Populate cache
        if self.raw_train_path and self.raw_test_path:
            _ck = (str(self.raw_train_path), str(self.raw_test_path))
            _TT_PRECOMPUTE_CACHE.setdefault(_ck, {}).update({
                "track_features": self._track_features,
                "user_features":  self._user_features,
                "unk_track":      self._unk_track,
                "unk_user":       self._unk_user,
                "vocab_sizes":    self._vocab_sizes,
                "hard_negatives": self._hard_negatives,
                "fallback_negatives": self._fallback_negatives,
            })

    def _build_hard_negative_pools(self, max_pool: int = 512) -> None:
        by_artist: dict[int, list] = {}
        by_tag: dict[int, list] = {}
        for tid, f in self._track_features.items():
            by_artist.setdefault(f["artist_idx"], []).append(tid)
            for tag in f["tag_idxs"]:
                by_tag.setdefault(tag, []).append(tid)

        pools: dict = {}
        for tid, f in self._track_features.items():
            candidates: list = []
            candidates.extend(by_artist.get(f["artist_idx"], []))
            for tag in f["tag_idxs"][:3]:
                candidates.extend(by_tag.get(tag, []))
            seen = {tid}
            pool = []
            for cand in candidates:
                if cand not in seen:
                    seen.add(cand)
                    pool.append(cand)
                if len(pool) >= max_pool:
                    break
            pools[tid] = pool
        self._hard_negatives = pools
        self._fallback_negatives = list(self._track_features.keys())

    def _compute_text_embeds(self, raw_df: pl.DataFrame,
                              incremental: bool = False) -> None:
        from sentence_transformers import SentenceTransformer

        # Restore from cache if available (avoids re-encoding each Optuna trial)
        if not incremental and self.raw_train_path and self.raw_test_path:
            _ck = (str(self.raw_train_path), str(self.raw_test_path))
            if _ck in _TT_PRECOMPUTE_CACHE and "text_embeds" in _TT_PRECOMPUTE_CACHE[_ck]:
                self._text_embeds.update(_TT_PRECOMPUTE_CACHE[_ck]["text_embeds"])
                print(f"[{self.RECOMMENDER_NAME}] text_embeds restored from cache "
                      f"({len(self._text_embeds):,} entries)")
                return

        # Build (session_id, turn_number) → text lookup
        lookup: dict = {}
        for row in raw_df.iter_rows(named=True):
            sid   = row["session_id"]
            convs = row.get("conversations") or []
            profile_text = _format_user_profile(row.get("user_profile"))
            user_by_turn = {c["turn_number"]: c["content"]
                            for c in convs if c["role"] == "user"}
            for c in convs:
                if c["role"] == "music":
                    tn = c["turn_number"]
                    if incremental and (sid, tn) in self._text_embeds:
                        continue
                    lookup[(sid, tn)] = _join_text_parts(profile_text, user_by_turn.get(tn, ""))
            # Last unanswered user turn (for blind prediction). This includes
            # cold-start blind sessions where no music has been recommended yet.
            music_turns = [c["turn_number"] for c in convs if c["role"] == "music"]
            last_conv_turn = max((c["turn_number"] for c in convs), default=0)
            has_unanswered_music_context = music_turns and last_conv_turn > max(music_turns)
            has_cold_start_unanswered = not music_turns and last_conv_turn > 0
            if has_unanswered_music_context or has_cold_start_unanswered:
                tn = last_conv_turn
                if not incremental or (sid, tn) not in self._text_embeds:
                    last_user = next(
                        (c["content"] for c in reversed(sorted(convs, key=lambda c: c["turn_number"]))
                         if c["role"] == "user"), ""
                    )
                    lookup[(sid, tn)] = _join_text_parts(profile_text, last_user)

        if not lookup:
            return

        keys  = list(lookup.keys())
        texts = list(lookup.values())
        print(f"[{self.RECOMMENDER_NAME}] encoding {len(texts):,} texts with {self.text_model}...")
        t0 = time.time()
        encoder = SentenceTransformer(self.text_model, device=str(self._device))
        embs = encoder.encode(texts, batch_size=512, show_progress_bar=True,
                               convert_to_numpy=True, normalize_embeddings=True)
        print(f"[{self.RECOMMENDER_NAME}] done in {time.time()-t0:.1f}s")
        del encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for i, k in enumerate(keys):
            self._text_embeds[k] = embs[i]
        print(f"[{self.RECOMMENDER_NAME}] text_embeds total: {len(self._text_embeds):,}")

        # Populate cache
        if not incremental and self.raw_train_path and self.raw_test_path:
            _ck = (str(self.raw_train_path), str(self.raw_test_path))
            _TT_PRECOMPUTE_CACHE.setdefault(_ck, {})["text_embeds"] = dict(self._text_embeds)

    def _extract_samples(self, raw_df: pl.DataFrame,
                          allowed_sess: set | None = None) -> list:
        samples = []
        for row in raw_df.iter_rows(named=True):
            sid = row["session_id"]
            if allowed_sess is not None and sid not in allowed_sess:
                continue
            uid   = row["user_id"]
            convs = sorted(row.get("conversations") or [], key=lambda c: c["turn_number"])
            music_turns = [(c["turn_number"], c["content"])
                           for c in convs if c["role"] == "music"]
            for i, (tn, tid) in enumerate(music_turns):
                if tid not in self._track_features:
                    continue
                if (sid, tn) not in self._text_embeds:
                    continue
                context = [t for _, t in music_turns[max(0, i - MAX_CTX_HIGH):i]]
                samples.append({
                    "session_id":  sid,
                    "user_id":     uid,
                    "turn_number": tn,
                    "context":     context,
                    "target":      tid,
                })
        return samples
