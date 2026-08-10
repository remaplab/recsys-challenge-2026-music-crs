"""RecVAE — Revisiting Variational Autoencoders for Collaborative Filtering
(Shenbin et al. 2020).

Key differences vs MultVAE:
  - LayerNorm in encoder
  - Composite prior KL (simplified here: standard N(0,1) prior)
  - Alternating encoder / decoder updates (enc_steps : dec_steps per batch)
  - gamma controls KL weight (replaces anneal schedule)
  - Per-batch sparse densification — no full dense URM
"""

from __future__ import annotations

import time

import numpy as np
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

from .user_base import UserRecommender


def _make_recvae(n_items: int, hidden: int, latent: int, dropout: float, device: str):
    import torch
    import torch.nn as nn

    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.drop = nn.Dropout(dropout)
            self.l1 = nn.Linear(n_items, hidden)
            self.l2 = nn.Linear(hidden, hidden)
            self.mu = nn.Linear(hidden, latent)
            self.lv = nn.Linear(hidden, latent)
            self.ln1 = nn.LayerNorm(hidden)
            self.ln2 = nn.LayerNorm(hidden)

        def forward(self, x):
            h = torch.tanh(self.ln1(self.l1(self.drop(x))))
            h = torch.tanh(self.ln2(self.l2(h)))
            return self.mu(h), self.lv(h)

    class _RecVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = Enc()
            self.dec = nn.Linear(latent, n_items)
            self.latent = latent

        def forward(self, x):
            mu, lv = self.enc(x)
            z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu) if self.training else mu
            return self.dec(z), mu, lv

    return _RecVAE().to(device)


class RecVAERecommender(UserRecommender):
    RECOMMENDER_NAME = "RecVAE"

    def __init__(
        self,
        hidden: int = 600,
        latent: int = 200,
        dropout: float = 0.5,
        gamma: float = 0.005,
        lr: float = 5e-4,
        batch_size: int = 512,
        epochs: int = 50,
        enc_steps: int = 3,
        dec_steps: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden = hidden
        self.latent = latent
        self.dropout = dropout
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.enc_steps = enc_steps
        self.dec_steps = dec_steps
        self.device: str | None = None
        self.model = None

    def _fit_model(self, urm: csr_matrix) -> None:
        import torch
        import torch.nn.functional as F
        from torch import optim

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        m, n = urm.shape
        urm = urm.tocsr()
        self.model = _make_recvae(n, self.hidden, self.latent, self.dropout, self.device)
        opt_e = optim.Adam(self.model.enc.parameters(), lr=self.lr)
        opt_d = optim.Adam(self.model.dec.parameters(), lr=self.lr)

        t0 = time.time()
        for _ in tqdm(range(self.epochs), desc=f"  {self.RECOMMENDER_NAME}"):
            self.model.train()
            order = np.random.permutation(m)
            for s in range(0, m, self.batch_size):
                idx = order[s : s + self.batch_size]
                x = torch.from_numpy(
                    urm[idx].toarray().astype(np.float32)
                ).to(self.device)
                xn = F.normalize(x, p=1, dim=1)

                for _ in range(self.enc_steps):
                    opt_e.zero_grad()
                    logits, mu, lv = self.model(xn)
                    recon = -torch.sum(F.log_softmax(logits, 1) * x, 1).mean()
                    kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp(), 1).mean()
                    (recon + self.gamma * kl).backward()
                    opt_e.step()

                for _ in range(self.dec_steps):
                    opt_d.zero_grad()
                    logits, mu, lv = self.model(xn)
                    recon = -torch.sum(F.log_softmax(logits, 1) * x, 1).mean()
                    recon.backward()
                    opt_d.step()

        print(f"[{self.RECOMMENDER_NAME}] {self.epochs}ep in {time.time() - t0:.1f}s")

    def _score_session_profile(self, profile: csr_matrix) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(
                profile.toarray().astype(np.float32)
            ).to(self.device)
            xn = F.normalize(x, p=1, dim=1)
            logits, _, _ = self.model(xn)
            return logits.cpu().numpy().ravel()

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st.update({k: getattr(self, k) for k in (
            "hidden", "latent", "dropout", "gamma", "lr",
            "batch_size", "epochs", "enc_steps", "dec_steps", "device",
        )})
        st["model_state_dict"] = self.model.state_dict() if self.model else None
        st["n_items"] = self.urm.shape[1] if self.urm is not None else None
        return st

    def _set_model_state(self, state: dict) -> None:
        super()._set_model_state(state)
        for k in ("hidden", "latent", "dropout", "gamma", "lr",
                  "batch_size", "epochs", "enc_steps", "dec_steps", "device"):
            setattr(self, k, state[k])
        n_items = state.get("n_items")
        if state.get("model_state_dict") is not None and n_items:
            self.model = _make_recvae(
                n_items, self.hidden, self.latent, self.dropout, self.device
            )
            self.model.load_state_dict(state["model_state_dict"])
