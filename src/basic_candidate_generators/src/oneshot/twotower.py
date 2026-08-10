"""Two-Tower Contrastive Recommender per query testuali e metadati tracce.

Elabora le query usando gli embedding pre-calcolati (Qwen) e mappa le tracce
in uno spazio latente ottimizzato tramite InfoNCE Loss.

Supporta sia `inference_mode: text` (chiamando recommend_text) che 
`inference_mode: standard` (estraendo i testi dal contesto).
"""

import os
# --- FIX CRASH SILENZIOSO MACOS ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"
# ----------------------------------


import copy
import os
import sys as _sys
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Risoluzione path per importare BaseRecommender come gli altri file
_HERE = Path(__file__).resolve()
_sys.path.insert(0, str(_HERE.parent.parent))
from BaseRecommender import BaseRecommender  # noqa: E402

_REPO_ROOT = _HERE.parents[4]

# ---------------------------------------------------------------------------
# Architettura PyTorch
# ---------------------------------------------------------------------------

class Tower(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=256, dropout_rate=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class TwoTowerModel(nn.Module):
    def __init__(self, hidden_dim, output_dim, dropout_rate):
        super().__init__()
        self.query_tower = Tower(hidden_dim=hidden_dim, output_dim=output_dim, dropout_rate=dropout_rate)
        self.track_tower = Tower(hidden_dim=hidden_dim, output_dim=output_dim, dropout_rate=dropout_rate)
        
    def forward(self, queries, tracks):
        q_emb = F.normalize(self.query_tower(queries), p=2, dim=1)
        t_emb = F.normalize(self.track_tower(tracks), p=2, dim=1)
        return q_emb, t_emb

def info_nce_loss(q_emb, t_emb, temperature):
    logits = torch.matmul(q_emb, t_emb.T) / temperature
    labels = torch.arange(logits.size(0)).to(q_emb.device)
    return F.cross_entropy(logits, labels)

# ---------------------------------------------------------------------------
# Classe Recommender Ufficiale
# ---------------------------------------------------------------------------

class TwoTowerRecommender(BaseRecommender):
    RECOMMENDER_NAME = "TwoTower"

    def __init__(
        self,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout_rate: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        temperature: float = 0.05,
        epochs: int = 40,
        batch_size: int = 512,
        urm_mode: str = "session",
        validation_split: float = 0.1,  # Percentuale del train set usata per early stopping
        patience: int = 5,              # Epoche di tolleranza senza miglioramenti
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.epochs = epochs
        self.batch_size = batch_size
        self.urm_mode = urm_mode
        self.validation_split = validation_split
        self.patience = patience

        self.model = None
        self.tracks_mapped = None
        self.track_ids = None
        self.track_id_to_idx = None
        self.faiss_index = None
        
        self._query_map = None

    def _load_all_query_embeddings(self):
        """Carica globalmente in RAM gli embedding Qwen delle query per fare lookup veloci."""
        if self._query_map is not None:
            return
            
        self._query_map = {}
        emb_dir = _REPO_ROOT / "data/embeddings_2805_qwen3_frozen"
        
        def _load_if_exists(npy_name, parquet_name):
            npy_path = emb_dir / npy_name
            pq_path = emb_dir / parquet_name
            if npy_path.exists() and pq_path.exists():
                q_arr = np.load(npy_path).astype(np.float32)
                m_df = pl.read_parquet(pq_path)
                for i, row in enumerate(m_df.iter_rows(named=True)):
                    self._query_map[(row["session_id"], row["turn_number"])] = q_arr[i]

        _load_if_exists("train.npy", "train_meta.parquet")
        _load_if_exists("dev.npy", "dev_meta.parquet")
        _load_if_exists("blind_a.npy", "blind_a_meta.parquet")

    # ------------------------------------------------------------------
    # Fit (Training con Early Stopping Nativo)
    # ------------------------------------------------------------------

    def fit(self, train_df: pl.DataFrame, track_metadata: pl.DataFrame | None = None, **kwargs) -> None:
        t0 = time.time()
        
        # 1. Filtro rigoroso per la Cross-Validation (Niente Data Leakage!)
        valid_sessions = set(train_df["session_id"].to_list())

        # 2. Caricamento base dati Tracce
        cache_dir = _REPO_ROOT / "data/trackemb/track_tower_cache"
        tracks_emb_raw = np.load(cache_dir / "metadata-qwen3_embedding_0.6b.npy")
        track_ids_raw = np.load(cache_dir / "track_ids.npy", allow_pickle=True)
        mask = np.load(cache_dir / "metadata-qwen3_embedding_0.6b__mask.npy")

        tracks_emb = tracks_emb_raw[mask].astype(np.float32)
        self.track_ids = track_ids_raw[mask]
        self.track_id_to_idx = {tid: i for i, tid in enumerate(self.track_ids)}

        # 3. Preparazione Set di Addestramento filtrato
        self._load_all_query_embeddings()
        train_meta_df = pl.read_parquet(_REPO_ROOT / "data/embeddings_2805_qwen3_frozen/train_meta.parquet")
        
        X_train_q, X_train_t = [], []
        for row in train_meta_df.iter_rows(named=True):
            if row["session_id"] in valid_sessions:
                target_track = row["gt_track_id"]
                if target_track and target_track in self.track_id_to_idx:
                    q_emb = self._query_map.get((row["session_id"], row["turn_number"]))
                    if q_emb is not None:
                        X_train_q.append(q_emb)
                        X_train_t.append(tracks_emb[self.track_id_to_idx[target_track]])

        if not X_train_q:
            raise ValueError(f"[{self.RECOMMENDER_NAME}] Zero coppie trovate per l'addestramento in questo fold.")

        X_train_q = torch.tensor(np.vstack(X_train_q))
        X_train_t = torch.tensor(np.vstack(X_train_t))
        
        # 4. Creazione dello Split di Validazione Interna per Early Stopping
        val_size = int(len(X_train_q) * self.validation_split)
        if val_size > 0:
            indices = torch.randperm(len(X_train_q))
            train_idx, val_idx = indices[val_size:], indices[:val_size]
            train_dataset = TensorDataset(X_train_q[train_idx], X_train_t[train_idx])
            val_dataset = TensorDataset(X_train_q[val_idx], X_train_t[val_idx])
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            print(f"[{self.RECOMMENDER_NAME}] Fit data: {len(train_idx)} train | {len(val_idx)} val (Early Stopping attivo)")
        else:
            train_dataset = TensorDataset(X_train_q, X_train_t)
            val_dataloader = None
            print(f"[{self.RECOMMENDER_NAME}] Fit data: {len(X_train_q)} coppie (Nessuna validazione interna)")

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # 5. Addestramento PyTorch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TwoTowerModel(self.hidden_dim, self.output_dim, self.dropout_rate).to(device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.epochs):
            self.model.train()
            total_train_loss = 0.0
            for batch_q, batch_t in train_dataloader:
                batch_q, batch_t = batch_q.to(device), batch_t.to(device)
                optimizer.zero_grad()
                q_emb, t_emb = self.model(batch_q, batch_t)
                loss = info_nce_loss(q_emb, t_emb, self.temperature)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_dataloader)

            # --- Lógica di Early Stopping ---
            if val_dataloader is not None:
                self.model.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    for batch_q, batch_t in val_dataloader:
                        batch_q, batch_t = batch_q.to(device), batch_t.to(device)
                        q_emb, t_emb = self.model(batch_q, batch_t)
                        loss = info_nce_loss(q_emb, t_emb, self.temperature)
                        total_val_loss += loss.item()
                
                avg_val_loss = total_val_loss / len(val_dataloader)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_state = copy.deepcopy(self.model.state_dict())
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    print(f"[{self.RECOMMENDER_NAME}] Early stopping innescato all'epoca {epoch+1} (Miglior Val Loss: {best_val_loss:.4f})")
                    break
            else:
                # Se non c'è validazione, salviamo semplicemente l'ultimo stato
                best_model_state = copy.deepcopy(self.model.state_dict())

        # Ricarica i pesi della migliore epoca
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        # 6. Pre-calcolo dello spazio vettoriale (FAISS)
        self.model.eval()
        with torch.no_grad():
            all_tracks_tensor = torch.tensor(tracks_emb).to(device)
            mapped = []
            for i in range(0, len(all_tracks_tensor), 4096):
                batch = all_tracks_tensor[i:i+4096]
                mapped.append(F.normalize(self.model.track_tower(batch), p=2, dim=1).cpu().numpy())
            self.tracks_mapped = np.vstack(mapped)

        self._build_faiss()
        print(f"[{self.RECOMMENDER_NAME}] Fit completato in {time.time()-t0:.1f}s")

    def _build_faiss(self):
        """Ricostruisce l'indice in RAM dopo il fit o dopo il caricamento dal disco."""
        if self.tracks_mapped is None:
            return
        self.faiss_index = faiss.IndexFlatIP(self.output_dim)
        self.faiss_index.add(self.tracks_mapped)

    # ------------------------------------------------------------------
    # Inference (Compatibilità Totale)
    # ------------------------------------------------------------------

    def recommend_text(self, sess_info: pl.DataFrame, top_k: int = 20, remove_seen: bool = True, **kwargs) -> pl.DataFrame:
        return self._predict_batch(sess_info, top_k, remove_seen, turn_col="turn_number", ctx_col="ctx_tracks")

    def recommend(self, context_df: pl.DataFrame, top_k: int = 20, remove_seen: bool = True, **kwargs) -> pl.DataFrame:
        if context_df.height > 0:
            ctx_map = context_df.group_by("session_id").agg(pl.col("track_id").drop_nulls().alias("ctx_tracks"))
            target_turns = context_df.select(["session_id", "user_id", "target_turn"]).unique(subset=["session_id"])
            df = target_turns.join(ctx_map, on="session_id", how="left").with_columns(pl.col("ctx_tracks").fill_null([]))
        else:
            df = pl.DataFrame({"session_id": [], "user_id": [], "target_turn": [], "ctx_tracks": []})

        return self._predict_batch(df, top_k, remove_seen, turn_col="target_turn", ctx_col="ctx_tracks")

    def _predict_batch(self, df: pl.DataFrame, top_k: int, remove_seen: bool, turn_col: str, ctx_col: str) -> pl.DataFrame:
        self._load_all_query_embeddings()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

        out_session, out_user, out_turn, out_tracks, out_scores = [], [], [], [], []

        # Estrazione massiva per sfruttare la GPU/vettorizzazione della rete neurale
        q_embs, valid_indices = [], []
        for i, row in enumerate(df.iter_rows(named=True)):
            q_emb = self._query_map.get((row["session_id"], row[turn_col]))
            if q_emb is not None:
                q_embs.append(q_emb)
                valid_indices.append(i)

        if len(q_embs) > 0:
            q_tensor = torch.tensor(np.vstack(q_embs)).to(device)
            with torch.no_grad():
                q_mapped = F.normalize(self.model.query_tower(q_tensor), p=2, dim=1).cpu().numpy()
            scores, indices = self.faiss_index.search(q_mapped, top_k + (50 if remove_seen else 0))
        else:
            scores, indices = [], []

        # Generazione Output Mappato
        q_idx = 0
        for i, row in enumerate(df.iter_rows(named=True)):
            sess_id = row["session_id"]
            user_id = row.get("user_id", None)
            turn = row[turn_col]
            seen = set(row[ctx_col]) if remove_seen else set()

            final_tracks, final_scores = [], []
            if i in valid_indices:
                raw_scores, raw_idx = scores[q_idx], indices[q_idx]
                q_idx += 1

                for rank, t_idx in enumerate(raw_idx):
                    tid = self.track_ids[t_idx]
                    if tid not in seen:
                        final_tracks.append(tid)
                        final_scores.append(float(raw_scores[rank]))
                        if len(final_tracks) == top_k:
                            break

            out_session.append(sess_id)
            out_user.append(user_id)
            out_turn.append(turn)
            out_tracks.append(final_tracks)
            out_scores.append(final_scores)

        return pl.DataFrame({
            "session_id": out_session,
            "user_id": out_user,
            "turn": out_turn,
            "track_ids": out_tracks,
            "scores": out_scores,
        })

    # ------------------------------------------------------------------
    # Salvataggio/Caricamento Modello per la Submission
    # ------------------------------------------------------------------

    def _get_model_state(self) -> dict:
        return {
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "dropout_rate": self.dropout_rate,
            "tracks_mapped": self.tracks_mapped,
            "track_ids": self.track_ids,
            "track_id_to_idx": self.track_id_to_idx,
            "model_state_dict": self.model.state_dict() if self.model else None
        }

    def _set_model_state(self, state: dict) -> None:
        self.hidden_dim = state.get("hidden_dim", 512)
        self.output_dim = state.get("output_dim", 256)
        self.dropout_rate = state.get("dropout_rate", 0.1)
        self.tracks_mapped = state["tracks_mapped"]
        self.track_ids = state["track_ids"]
        self.track_id_to_idx = state["track_id_to_idx"]

        self.model = TwoTowerModel(self.hidden_dim, self.output_dim, self.dropout_rate)
        if state["model_state_dict"]:
            self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()

        self._build_faiss()