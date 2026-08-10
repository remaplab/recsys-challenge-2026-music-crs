"""MultVAE — Variational Auto-Encoder for collaborative filtering
(Liang et al. 2018).

Per-batch sparse densification: never materializes full dense URM.
ICM kept sparse; session features computed per batch via sparse matmul.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

from .user_base import UserRecommender


def _make_multvae_model(n_items, n_input, hidden_dim, latent_dim, dropout, device):
    """Build a _MultVAEModel instance, importing torch on first call."""
    import torch
    import torch.nn as nn

    class _MultVAEModel(nn.Module):
        def __init__(self, n_items, n_input, hidden_dim, latent_dim, dropout):
            super().__init__()
            self.n_items = n_items
            self.dropout = nn.Dropout(dropout)
            self.encoder = nn.Sequential(
                nn.Linear(n_input, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            )
            self.fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, n_items),
            )
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

        def encode(self, x):
            h = self.encoder(self.dropout(x))
            return self.fc_mu(h), self.fc_logvar(h)

        def forward(self, x):
            mu, logvar = self.encode(x)
            if self.training:
                z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            else:
                z = mu
            return self.decoder(z), mu, logvar

    return _MultVAEModel(n_items, n_input, hidden_dim, latent_dim, dropout).to(device)


class MultVAERecommender(UserRecommender):
    RECOMMENDER_NAME = "MultVAE"

    def __init__(
        self,
        hidden_dim: int = 600,
        latent_dim: int = 200,
        dropout: float = 0.5,
        beta: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 512,
        epochs: int = 100,
        anneal_cap: float = 0.2,
        total_anneal_steps: int = 200_000,
        weight_decay: float = 1e-5,
        device: str | None = None,
        use_icm: bool = False,
        icm_weight: float = 0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.beta = beta
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.anneal_cap = anneal_cap
        self.total_anneal_steps = total_anneal_steps
        self.weight_decay = weight_decay
        self.device = device  # resolved to "cuda"/"cpu" in _fit_model if None
        self.use_icm = use_icm
        self.icm_weight = icm_weight
        self.model: object | None = None
        self._icm_sp: csr_matrix | None = None  # normalized sparse ICM for batch augmentation

    def _augment_batch(self, batch_dense: np.ndarray) -> np.ndarray:
        """Append per-session ICM content features to a batch of URM rows.

        batch_dense: (b, n_items) float32 — only this batch, not the full URM.
        """
        if not (self.use_icm and self._icm_sp is not None):
            return batch_dense
        sess_icm = (csr_matrix(batch_dense) @ self._icm_sp).toarray().astype(np.float32)
        rn = np.maximum(np.linalg.norm(sess_icm, axis=1, keepdims=True), 1e-10)
        return np.hstack([batch_dense, sess_icm / rn * self.icm_weight])

    def _fit_model(self, urm: csr_matrix) -> None:
        import torch
        import torch.nn.functional as F
        import torch.optim as optim
        from sklearn.preprocessing import normalize as sk_normalize

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        n_users, n_items = urm.shape
        urm = urm.tocsr()

        self._icm_sp = None
        n_input = n_items
        if self.use_icm and self.icm is not None:
            # Keep ICM sparse — never densify n_items × n_feat
            self._icm_sp = sk_normalize(self.icm.astype(np.float32), norm="l2", axis=1).tocsr()
            n_input = n_items + self._icm_sp.shape[1]
            print(f"[{self.RECOMMENDER_NAME}] ICM augmented input: {n_input} dims (sparse ICM)")

        self.model = _make_multvae_model(
            n_items, n_input, self.hidden_dim, self.latent_dim, self.dropout, self.device
        )
        opt = optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        update_count = 0
        t0 = time.time()

        for _ in tqdm(range(self.epochs), desc=f"  {self.RECOMMENDER_NAME}"):
            self.model.train()
            order = np.random.permutation(n_users)
            for s in range(0, n_users, self.batch_size):
                idx = order[s : s + self.batch_size]
                # Densify only this batch — never full URM dense
                batch_dense = urm[idx].toarray().astype(np.float32)
                x = torch.from_numpy(self._augment_batch(batch_dense)).to(self.device)
                xn = F.normalize(x, p=1, dim=1)
                opt.zero_grad()
                logits, mu, logvar = self.model(xn)
                x_items = x[:, :n_items]
                recon = -torch.sum(F.log_softmax(logits, dim=1) * x_items, dim=1).mean()
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
                ann = min(self.anneal_cap, update_count / max(self.total_anneal_steps, 1))
                loss = recon + ann * self.beta * kl
                loss.backward()
                opt.step()
                update_count += 1

        print(f"[{self.RECOMMENDER_NAME}] {self.epochs}ep in {time.time()-t0:.1f}s")

    def _score_session_profile(self, profile: csr_matrix) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        self.model.eval()
        with torch.no_grad():
            p_dense = profile.toarray().astype(np.float32)
            x = torch.from_numpy(self._augment_batch(p_dense)).to(self.device)
            xn = F.normalize(x, p=1, dim=1)
            logits, _, _ = self.model(xn)
            return logits.cpu().numpy().ravel()

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st.update({k: getattr(self, k) for k in (
            "hidden_dim", "latent_dim", "dropout", "beta", "learning_rate",
            "batch_size", "epochs", "anneal_cap", "total_anneal_steps",
            "weight_decay", "device", "use_icm", "icm_weight",
        )})
        st["model_state_dict"] = self.model.state_dict() if self.model else None
        st["n_items"] = self.urm.shape[1] if self.urm is not None else None
        st["n_input"] = self.model.encoder[0].in_features if self.model else None
        st["_icm_sp"] = self._icm_sp
        return st

    def _set_model_state(self, state: dict) -> None:
        super()._set_model_state(state)
        for k in ("hidden_dim", "latent_dim", "dropout", "beta", "learning_rate",
                  "batch_size", "epochs", "anneal_cap", "total_anneal_steps",
                  "weight_decay", "device"):
            setattr(self, k, state[k])
        self.use_icm = state.get("use_icm", False)
        self.icm_weight = state.get("icm_weight", 0.3)
        self._icm_sp = state.get("_icm_sp")
        n_items = state.get("n_items")
        n_input = state.get("n_input") or n_items
        if state.get("model_state_dict") is not None and n_items:
            self.model = _make_multvae_model(
                n_items, n_input, self.hidden_dim, self.latent_dim, self.dropout, self.device
            )
            self.model.load_state_dict(state["model_state_dict"])
