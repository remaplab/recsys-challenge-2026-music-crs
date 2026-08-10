"""LoRA fine-tunable Qwen3-0.6B query encoder, with routing toggle.

use_routing=True  -> per-user soft prompt: warm = linear(cf_bpr), cold = learned token
use_routing=False -> a single shared learned token for everyone (no user signal)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel, AutoTokenizer


@dataclass
class LoRAQueryEncoderConfig:
    base_model: str = "Qwen/Qwen3-Embedding-0.6B"
    user_cf_dim: int = 128
    hidden_dim: int = 1024
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")
    max_length: int = 384
    instruction: str = (
        "Given a music recommendation conversation, retrieve the most relevant "
        "track from the catalog"
    )
    use_routing: bool = True       # True = per-user soft prompt; False = shared token


class Qwen3LoRAQueryEncoder(nn.Module):
    def __init__(self, cfg: LoRAQueryEncoderConfig):
        super().__init__()
        self.cfg = cfg

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.base_model, padding_side="right", truncation_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModel.from_pretrained(cfg.base_model, torch_dtype=torch.bfloat16)
        assert base.config.hidden_size == cfg.hidden_dim

        peft_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=list(cfg.target_modules),
        )
        self.model = get_peft_model(base, peft_cfg)
        for n, p in self.model.named_parameters():
            if "lora_" not in n:
                p.requires_grad = False

        # Routing components — only trained when use_routing=True
        self.user_proj = nn.Linear(cfg.user_cf_dim, cfg.hidden_dim, bias=True)
        nn.init.normal_(self.user_proj.weight, std=0.02)
        nn.init.zeros_(self.user_proj.bias)
        self.cold_user_token = nn.Parameter(torch.randn(cfg.hidden_dim) * 0.02)

        # Shared token for the no-routing variant
        self.shared_user_token = nn.Parameter(torch.randn(cfg.hidden_dim) * 0.02)

        self.type_emb = nn.Embedding(2, cfg.hidden_dim)
        nn.init.normal_(self.type_emb.weight, std=0.01)

        # When use_routing=False, freeze the routing-only weights so they don't
        # waste optimizer steps. Saves ~0.5M trainable params and is cleaner.
        if not cfg.use_routing:
            for p in self.user_proj.parameters():
                p.requires_grad = False
            self.cold_user_token.requires_grad = False

    @property
    def output_dim(self) -> int:
        # Required by 05_train_encoder.py's dim sanity check, and by anything
        # else that relies on the QueryEncoder-style API. Returns the Qwen3
        # backbone's native hidden size (1024 for Qwen3-Embedding-0.6B).
        return self.cfg.hidden_dim

    @property
    def trainable_param_names(self) -> List[str]:
        return [n for n, p in self.named_parameters() if p.requires_grad]

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (p for _, p in self.named_parameters() if p.requires_grad)

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def _build_user_token(self, user_cf, is_cold):
        if not self.cfg.use_routing:
            B = user_cf.size(0)
            return self.shared_user_token.unsqueeze(0).expand(B, -1)
        warm = self.user_proj(user_cf)
        cold = self.cold_user_token.unsqueeze(0).expand_as(warm)
        return torch.where(is_cold.unsqueeze(-1), cold, warm)

    def _embed_text(self, input_ids):
        return self.model.get_input_embeddings()(input_ids)

    def forward(self, texts, user_cf, is_cold, device=None):
        if device is None:
            device = next(self.parameters()).device

        prefixed = [f"Instruct: {self.cfg.instruction}\nQuery: {t}" for t in texts]
        tok = self.tokenizer(
            prefixed, padding=True, truncation=True,
            max_length=self.cfg.max_length, return_tensors="pt",
        ).to(device)
        ids, attn = tok["input_ids"], tok["attention_mask"]
        B, L = ids.shape

        tok_emb = self._embed_text(ids).to(torch.bfloat16)
        user_tok = self._build_user_token(
            user_cf.to(device), is_cold.to(device)
        ).to(torch.bfloat16).unsqueeze(1)

        type_ids = torch.cat([
            torch.zeros(B, 1, dtype=torch.long, device=device),
            torch.ones(B, L, dtype=torch.long, device=device),
        ], dim=1)
        type_e = self.type_emb(type_ids).to(torch.bfloat16)

        full_emb = torch.cat([user_tok, tok_emb], dim=1) + type_e
        full_attn = torch.cat([
            torch.ones(B, 1, dtype=attn.dtype, device=device), attn,
        ], dim=1)

        out = self.model(
            inputs_embeds=full_emb, attention_mask=full_attn,
            output_hidden_states=False, return_dict=True,
        )
        last = out.last_hidden_state
        seq_lens = full_attn.sum(dim=1) - 1
        pooled = last[torch.arange(B, device=device), seq_lens].float()
        return F.normalize(pooled, p=2, dim=-1)

    def save_adapter(self, path: Path) -> None:
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path / "lora")
        torch.save({
            "user_proj": self.user_proj.state_dict(),
            "cold_user_token": self.cold_user_token.detach().cpu(),
            "shared_user_token": self.shared_user_token.detach().cpu(),
            "type_emb": self.type_emb.state_dict(),
        }, path / "soft_prompt.pt")
        torch.save(self.cfg, path / "cfg.pt")

    @classmethod
    def load_adapter(cls, path: Path, device: torch.device) -> "Qwen3LoRAQueryEncoder":
        from peft import PeftModel
        path = Path(path)
        cfg = torch.load(path / "cfg.pt", map_location="cpu", weights_only=False)
        instance = cls(cfg)
        instance.model = PeftModel.from_pretrained(
            instance.model.get_base_model(), path / "lora",
        )
        for n, p in instance.model.named_parameters():
            if "lora_" not in n:
                p.requires_grad = False
        sp = torch.load(path / "soft_prompt.pt", map_location="cpu", weights_only=False)
        instance.user_proj.load_state_dict(sp["user_proj"])
        with torch.no_grad():
            instance.cold_user_token.copy_(sp["cold_user_token"])
            if "shared_user_token" in sp:
                instance.shared_user_token.copy_(sp["shared_user_token"])
        instance.type_emb.load_state_dict(sp["type_emb"])
        return instance.to(device)


@torch.no_grad()
def encode_corpus(encoder, texts, user_cf_arr, is_cold_arr,
                  batch_size=16, show_progress=True):
    encoder.eval()
    device = next(encoder.parameters()).device
    out = []
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="LoRA encode")
    for s in iterator:
        e = min(s + batch_size, len(texts))
        emb = encoder(
            texts[s:e],
            torch.from_numpy(user_cf_arr[s:e]).float(),
            torch.from_numpy(is_cold_arr[s:e]).bool(),
        )
        out.append(emb.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)