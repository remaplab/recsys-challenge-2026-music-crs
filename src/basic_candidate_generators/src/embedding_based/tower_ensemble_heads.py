"""Trainable projection heads, losses, hard negatives, and SWAG for the
tower_ensemble CG. Encoders are frozen; only these light heads are trained.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Projection head
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """2-layer MLP into a shared space; L2-normed output."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


# ---------------------------------------------------------------------------
# Retrieval loss
# ---------------------------------------------------------------------------

def infonce_loss(q: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor,
                 *, tau: float = 0.05) -> torch.Tensor:
    """Sampled-softmax: positive vs (positive + per-row negatives), temperature τ.

    q, pos: (B, d). neg: (B, N, d). All L2-normed → dot == cosine.
    """
    pos_logit = (q * pos).sum(-1, keepdim=True) / tau          # (B, 1)
    neg_logit = torch.einsum("bd,bnd->bn", q, neg) / tau        # (B, N)
    logits = torch.cat([pos_logit, neg_logit], dim=1)          # (B, 1+N)
    target = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)
    return F.cross_entropy(logits, target)


# ---------------------------------------------------------------------------
# SWAG
# ---------------------------------------------------------------------------

class SWAG:
    """SWA-Gaussian over a head's flattened weights.

    Collects post-burn-in weight snapshots → running first moment (mean),
    second moment (for the diagonal), and a low-rank deviation buffer. At
    inference, `draw="mean"` loads the SWA mean; an int draw samples
    w = mean + (1/√2)·diag^½·z1 + (1/√(2(K-1)))·D·z2  (Maddox et al. 2019).
    """

    def __init__(self, head: nn.Module, max_rank: int = 5):
        self.shapes = [p.shape for p in head.parameters()]
        self.max_rank = max_rank
        self.n = 0
        self.mean = None          # (P,)
        self.sq_mean = None       # (P,)
        self.dev_cols: list[torch.Tensor] = []

    @staticmethod
    def _flat(head: nn.Module) -> torch.Tensor:
        # Keep on the param device: collect() runs on the training hot path
        # (per-step), so avoid a GPU→CPU sync every snapshot. finalize() moves
        # the accumulated stats to CPU once, where sampling happens.
        return torch.cat([p.detach().flatten() for p in head.parameters()])

    def collect(self, head: nn.Module) -> None:
        w = self._flat(head)
        if self.mean is None:
            self.mean = torch.zeros_like(w)
            self.sq_mean = torch.zeros_like(w)
        self.n += 1
        self.mean += (w - self.mean) / self.n
        self.sq_mean += (w * w - self.sq_mean) / self.n
        self.dev_cols.append(w - self.mean)
        if len(self.dev_cols) > self.max_rank:
            self.dev_cols.pop(0)

    def finalize(self) -> None:
        # Move accumulated stats to CPU once; sampling runs there.
        self.diag = torch.clamp(self.sq_mean - self.mean * self.mean,
                                min=1e-12).cpu()
        self.D = (torch.stack(self.dev_cols, dim=1) if self.dev_cols
                  else torch.zeros(self.mean.numel(), 1)).cpu()
        self.mean = self.mean.cpu()

    def _load_flat(self, head: nn.Module, w: torch.Tensor) -> nn.Module:
        off = 0
        with torch.no_grad():
            for p, shape in zip(head.parameters(), self.shapes):
                numel = int(np.prod(shape))
                p.copy_(w[off:off + numel].view(shape).to(p.device))
                off += numel
        return head

    def sample_into(self, head: nn.Module, draw="mean", seed: int = 0) -> nn.Module:
        if draw == "mean":
            return self._load_flat(head, self.mean)
        g = torch.Generator().manual_seed(seed + int(draw))
        z1 = torch.randn(self.mean.numel(), generator=g)
        k = self.D.shape[1]
        z2 = torch.randn(k, generator=g)
        w = (self.mean
             + (self.diag.sqrt() * z1) / (2 ** 0.5)
             + (self.D @ z2) / ((2 * max(1, k - 1)) ** 0.5))
        return self._load_flat(head, w)


# ---------------------------------------------------------------------------
# Single-member training loop
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    d: int = 256
    hidden: int = 512
    epochs: int = 5
    lr: float = 1e-3
    batch_size: int = 1024
    tau: float = 0.05
    swag_max_rank: int = 5
    swag_burnin_frac: float = 0.5         # start collecting after this epoch fraction
    swag_collect_every: int = 0           # 0 = once per epoch; N>0 = every N steps
    grad_clip: float = 1.0                # max grad norm; <=0 disables
    seed: int = 0


def train_member(q_emb: torch.Tensor, pos_idx: torch.Tensor,
                 track_emb: torch.Tensor, cfg: TrainConfig, *,
                 tower: str, device: str = "cuda") -> dict:
    """Train one tower member (infonce, in-batch negatives) + SWAG stats.

    q_emb:    (M, dq) frozen query embeddings for training pairs.
    pos_idx:  (M,) catalogue index of the positive track per pair.
    track_emb:(n, dt) frozen catalogue track tower (same modality as `tower`).
    `tower`:  "A" (query8B×track8B) or "B" (cross-modal query8B×SigLIP2).
    """
    torch.manual_seed(cfg.seed)
    # TF32 for the remaining fp32 matmuls (SWAG accumulation etc.); the hot
    # head/projection matmuls run under bf16 autocast below.
    torch.set_float32_matmul_precision("high")
    dq, dt = q_emb.shape[1], track_emb.shape[1]
    head_q = ProjectionHead(dq, cfg.d, cfg.hidden).to(device)
    head_t = ProjectionHead(dt, cfg.d, cfg.hidden).to(device)
    track_emb = track_emb.to(device)
    q_emb = q_emb.to(device)
    pos_idx = pos_idx.to(device)

    params = list(head_q.parameters()) + list(head_t.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr)
    swag_q, swag_t = SWAG(head_q, cfg.swag_max_rank), SWAG(head_t, cfg.swag_max_rank)
    burnin = int(cfg.epochs * cfg.swag_burnin_frac)
    dev_type = "cuda" if str(device).startswith("cuda") else "cpu"

    M = q_emb.shape[0]
    losses: list[float] = []
    step = 0
    # Progress bar over all training steps; leave=False so it self-erases on
    # completion (the printed per-trial summary stays).
    pbar = tqdm(total=cfg.epochs * math.ceil(M / cfg.batch_size),
                desc=f"train tower {tower}", leave=False)
    for ep in range(cfg.epochs):
        perm = torch.randperm(M, device=device)
        ep_loss = torch.zeros((), device=device)      # accumulate on GPU
        nb = 0
        for s in range(0, M, cfg.batch_size):
            b = perm[s:s + cfg.batch_size]
            with torch.autocast(device_type=dev_type, dtype=torch.bfloat16):
                qb = head_q(q_emb[b])                  # (B, d)
                tb = head_t(track_emb[pos_idx[b]])     # (B, d) positives
                bb = qb.shape[0]                       # in-batch negatives
                if bb < 2:
                    # degenerate batch: fall back to the positive as its own
                    # (trivial) negative so the loss stays finite.
                    neg = tb.unsqueeze(1)
                else:
                    neg = tb.unsqueeze(0).expand(bb, -1, -1)
                    neg = neg[~torch.eye(bb, dtype=torch.bool, device=device)] \
                        .view(bb, bb - 1, cfg.d)
                loss = infonce_loss(qb, tb, neg, tau=cfg.tau)
            opt.zero_grad(); loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            ep_loss += loss.detach(); nb += 1          # no per-step host sync
            step += 1
            pbar.update(1)
            # per-step SWAG snapshot (richer posterior than 1/epoch)
            if (cfg.swag_collect_every and ep >= burnin
                    and step % cfg.swag_collect_every == 0):
                swag_q.collect(head_q); swag_t.collect(head_t)
        losses.append(float(ep_loss) / max(1, nb))     # single host sync / epoch
        pbar.set_postfix(ep=ep + 1, loss=f"{losses[-1]:.4f}")
        if not cfg.swag_collect_every and ep >= burnin:
            swag_q.collect(head_q); swag_t.collect(head_t)
    pbar.close()
    # guard: ensure at least one snapshot exists before finalize
    if swag_q.mean is None:
        swag_q.collect(head_q); swag_t.collect(head_t)
    swag_q.finalize(); swag_t.finalize()
    return {"head_q": head_q.cpu(), "head_t": head_t.cpu(),
            "swag_q": swag_q, "swag_t": swag_t, "losses": losses,
            "d": cfg.d, "tower": tower}


# ---------------------------------------------------------------------------
# Projected-member builder
# ---------------------------------------------------------------------------

@dataclass
class ProjectedMember:
    tower: str                      # "A" | "B"
    proj_tower: np.ndarray          # (n, d) L2-normed projected track tower
    head_q_state: dict              # head_q weights for projecting queries
    in_dim_q: int
    d: int
    gpu: object = None              # lazy CUDA copy of proj_tower (set at scoring)


def _rebuild_head(state: dict, in_dim: int, d: int) -> ProjectionHead:
    # Recover the hidden width from the first linear's weight so the rebuilt
    # head matches whatever `hidden` the member was trained with.
    hidden = int(state["net.0.weight"].shape[0])
    h = ProjectionHead(in_dim, d, hidden=hidden)
    h.load_state_dict(state)
    h.eval()
    return h


def build_member_towers(member: dict, track_emb: torch.Tensor, *,
                        swag_k: int, device: str = "cuda") -> list[ProjectedMember]:
    """Expand a trained member into `max(1, swag_k)` ProjectedMembers.

    swag_k == 0 → one member using the SWA mean weights. swag_k > 0 → that many
    SWAG weight draws. Each member applies head_t to the catalogue tower to get
    a projected (n, d) tower; head_q is stored for query projection at scoring.
    """
    d = member["d"]
    in_dim_q = member["head_q"].net[0].in_features
    in_dim_t = member["head_t"].net[0].in_features
    track_emb = track_emb.to(device)
    dev_type = "cuda" if str(device).startswith("cuda") else "cpu"
    draws = ["mean"] if swag_k == 0 else list(range(swag_k))
    out: list[ProjectedMember] = []
    for draw in tqdm(draws, desc=f"swag-project {member.get('tower', 'A')}",
                     leave=False):
        hq = _rebuild_head(member["head_q"].state_dict(), in_dim_q, d)
        ht = _rebuild_head(member["head_t"].state_dict(), in_dim_t, d)
        member["swag_q"].sample_into(hq, draw=draw)
        member["swag_t"].sample_into(ht, draw=draw)
        ht = ht.to(device)
        # bf16 on the full-catalogue projection (swag_k× per member = the fit
        # bottleneck); cast back to fp32 for the stored tower.
        with torch.no_grad(), torch.autocast(device_type=dev_type,
                                              dtype=torch.bfloat16,
                                              enabled=dev_type == "cuda"):
            proj = ht(track_emb)
        proj = proj.float().cpu().numpy().astype(np.float32)
        out.append(ProjectedMember(
            tower=member.get("tower", "A"), proj_tower=proj,
            head_q_state={k: v.cpu() for k, v in hq.state_dict().items()},
            in_dim_q=in_dim_q, d=d,
        ))
    return out


def project_queries(member: ProjectedMember, q_emb: np.ndarray) -> np.ndarray:
    """Apply a member's head_q to query embeddings → (M, d) L2-normed."""
    hq = _rebuild_head(member.head_q_state, member.in_dim_q, member.d)
    with torch.no_grad():
        out = hq(torch.from_numpy(np.ascontiguousarray(q_emb, dtype=np.float32)))
    return out.numpy().astype(np.float32)
