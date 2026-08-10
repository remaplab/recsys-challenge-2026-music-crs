"""HF-backbone query encoder with a learned projection head.

Family: BERT, SBERT (mpnet), ModernBERT.
Variants:
    - backbone {frozen, LoRA}
    - user routing {on, off}

Pooling: concat([user-token's contextual repr at pos 0, mean of text tokens]).
The first half carries user signal (warm: cf-bpr; cold: learned) when routing
is on; when routing is off it carries the contextualized shared learned token
(CLS-like). The second half is a content summary. The projector then maps
(2*H_backbone) -> output_dim.

Output is L2-normalized.

──────────────────────────────────────────────────────────────────────────────
WHY output_dim DEFAULTS TO 768 (NATIVE H)
──────────────────────────────────────────────────────────────────────────────
Originally output_dim was 1024 to match the organizer's Qwen3 track tower.
That made the comparison BERT-vs-Qwen3 unfair: BERT was forced to learn a
foreign-space projection, while Qwen3 stayed in its home space.

With per-backbone text track towers (see src/tracks/text_track_loader.py),
each backbone now matches against an L2-normed mean-pool of its OWN encoding
of the track-metadata text. Native dims are H=768 for BERT, MPNet, and
ModernBERT — so the projector target is also 768.

Pre-existing checkpoints saved with output_dim=1024 still load correctly
(cfg.output_dim is persisted in cfg.pt) — but they will only retrieve
against the legacy Qwen3 tower. New training runs after this change land in
native space.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from emblib.encoders.base import QueryEncoder


@dataclass
class HFProjectedConfig:
    base_model: str
    output_dim: int = 768               # native HF backbone size; matches per-backbone text tower
    user_cf_dim: int = 128
    max_length: int = 256
    use_routing: bool = True
    use_lora: bool = False              # False = fully frozen backbone
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("query", "key", "value", "dense")
    proj_hidden_mult: int = 2           # MLP hidden = mult * H_backbone


class HFProjectedEncoder(QueryEncoder):
    def __init__(self, cfg: HFProjectedConfig):
        super().__init__()
        self.cfg = cfg

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.base_model, truncation_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"

        backbone = AutoModel.from_pretrained(cfg.base_model, torch_dtype=torch.bfloat16)
        self._H = backbone.config.hidden_size

        if cfg.use_lora:
            from peft import LoraConfig, TaskType, get_peft_model
            peft_cfg = LoraConfig(
                r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout, bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=list(cfg.lora_target_modules),
            )
            self.backbone = get_peft_model(backbone, peft_cfg)
            for n, p in self.backbone.named_parameters():
                if "lora_" not in n:
                    p.requires_grad = False
        else:
            self.backbone = backbone
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        H = self._H
        # Projector: concat(user_pos, mean_text) (size 2H) -> output_dim
        self.proj = nn.Sequential(
            nn.Linear(2 * H, H * cfg.proj_hidden_mult),
            nn.GELU(),
            nn.Linear(H * cfg.proj_hidden_mult, cfg.output_dim),
        )

        # Routing components — always allocated for save/load symmetry
        self.user_proj = nn.Linear(cfg.user_cf_dim, H, bias=True)
        nn.init.normal_(self.user_proj.weight, std=0.02)
        nn.init.zeros_(self.user_proj.bias)
        self.cold_user_token = nn.Parameter(torch.randn(H) * 0.02)
        self.shared_user_token = nn.Parameter(torch.randn(H) * 0.02)

        # Type embedding: 0 = soft prompt, 1 = text token
        self.type_emb = nn.Embedding(2, H)
        nn.init.normal_(self.type_emb.weight, std=0.01)

        # Freeze unused soft-prompt params
        if cfg.use_routing:
            self.shared_user_token.requires_grad = False
        else:
            for p in self.user_proj.parameters():
                p.requires_grad = False
            self.cold_user_token.requires_grad = False

    @property
    def output_dim(self) -> int:
        return self.cfg.output_dim

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def train(self, mode: bool = True):
        """Override so the backbone stays in eval mode when frozen
        (keeps dropout / layernorm running stats stable)."""
        super().train(mode)
        if not self.cfg.use_lora:
            self.backbone.eval()
        return self

    def _build_user_token(self, user_cf: torch.Tensor, is_cold: torch.Tensor) -> torch.Tensor:
        if not self.cfg.use_routing:
            B = user_cf.size(0)
            return self.shared_user_token.unsqueeze(0).expand(B, self._H)
        warm = self.user_proj(user_cf)
        cold = self.cold_user_token.unsqueeze(0).expand_as(warm)
        return torch.where(is_cold.unsqueeze(-1), cold, warm)

    def _embed_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "get_input_embeddings"):
            return self.backbone.get_input_embeddings()(input_ids)
        return self.backbone.base_model.model.get_input_embeddings()(input_ids)

    def forward(self, texts, user_cf, is_cold, device=None):
        if device is None:
            device = next(self.parameters()).device

        tok = self.tokenizer(
            list(texts), padding=True, truncation=True,
            max_length=self.cfg.max_length, return_tensors="pt",
        ).to(device)
        ids: torch.Tensor = tok["input_ids"]            # (B, L)
        attn: torch.Tensor = tok["attention_mask"]      # (B, L)
        B, L = ids.shape

        tok_emb = self._embed_text(ids).to(torch.bfloat16)
        user_tok = self._build_user_token(
            user_cf.to(device), is_cold.to(device),
        ).to(torch.bfloat16).unsqueeze(1)               # (B, 1, H)

        type_ids = torch.cat([
            torch.zeros(B, 1, dtype=torch.long, device=device),
            torch.ones(B, L, dtype=torch.long, device=device),
        ], dim=1)
        type_e = self.type_emb(type_ids).to(torch.bfloat16)

        full_emb = torch.cat([user_tok, tok_emb], dim=1) + type_e   # (B, 1+L, H)
        full_attn = torch.cat([
            torch.ones(B, 1, dtype=attn.dtype, device=device), attn,
        ], dim=1)                                                    # (B, 1+L)

        out = self.backbone(
            inputs_embeds=full_emb,
            attention_mask=full_attn,
            output_hidden_states=False,
            return_dict=True,
        )
        last = out.last_hidden_state                                 # (B, 1+L, H)

        # Concat pool: position-0 (user/shared token after self-attention) + mean of text
        user_repr = last[:, 0].float()                                          # (B, H)
        text_mask = attn.float().unsqueeze(-1)                                  # (B, L, 1)
        text_repr = (last[:, 1:].float() * text_mask).sum(dim=1) \
                    / text_mask.sum(dim=1).clamp(min=1.0)                       # (B, H)
        pooled = torch.cat([user_repr, text_repr], dim=-1)                      # (B, 2H)

        z = self.proj(pooled)
        return F.normalize(z, p=2, dim=-1)

    # ── save / load ────────────────────────────────────────────────────────
    def save_adapter(self, path: Path) -> None:
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        torch.save(self.cfg, path / "cfg.pt")
        torch.save({
            "proj":              self.proj.state_dict(),
            "user_proj":         self.user_proj.state_dict(),
            "cold_user_token":   self.cold_user_token.detach().cpu(),
            "shared_user_token": self.shared_user_token.detach().cpu(),
            "type_emb":          self.type_emb.state_dict(),
        }, path / "head.pt")
        if self.cfg.use_lora:
            self.backbone.save_pretrained(path / "lora")

    @classmethod
    def load_adapter(cls, path: Path, device: torch.device) -> "HFProjectedEncoder":
        path = Path(path)
        cfg: HFProjectedConfig = torch.load(
            path / "cfg.pt", map_location="cpu", weights_only=False,
        )
        instance = cls(cfg)
        if cfg.use_lora:
            from peft import PeftModel
            instance.backbone = PeftModel.from_pretrained(
                instance.backbone.get_base_model(), path / "lora",
            )
            for n, p in instance.backbone.named_parameters():
                if "lora_" not in n:
                    p.requires_grad = False
        head = torch.load(path / "head.pt", map_location="cpu", weights_only=False)
        instance.proj.load_state_dict(head["proj"])
        instance.user_proj.load_state_dict(head["user_proj"])
        with torch.no_grad():
            instance.cold_user_token.copy_(head["cold_user_token"])
            instance.shared_user_token.copy_(head["shared_user_token"])
        instance.type_emb.load_state_dict(head["type_emb"])
        return instance.to(device)


# ── Convenience constructors ────────────────────────────────────────────────
# Each builder explicitly sets output_dim=768 — the native hidden size of all
# three backbones — so the matching-tower dimension is unambiguous.
def bert_proj(use_routing: bool, use_lora: bool) -> HFProjectedEncoder:
    return HFProjectedEncoder(HFProjectedConfig(
        base_model="bert-base-uncased",
        output_dim=768,
        max_length=256,
        use_routing=use_routing, use_lora=use_lora,
        lora_target_modules=("query", "key", "value", "dense"),
    ))


def sbert_proj(use_routing: bool, use_lora: bool) -> HFProjectedEncoder:
    # MPNet attention naming: q / k / v / o
    return HFProjectedEncoder(HFProjectedConfig(
        base_model="sentence-transformers/all-mpnet-base-v2",
        output_dim=768,
        max_length=384,
        use_routing=use_routing, use_lora=use_lora,
        lora_target_modules=("q", "k", "v", "o"),
    ))


def modernbert_proj(use_routing: bool, use_lora: bool) -> HFProjectedEncoder:
    return HFProjectedEncoder(HFProjectedConfig(
        base_model="answerdotai/ModernBERT-base",
        output_dim=768,
        max_length=2048,
        use_routing=use_routing, use_lora=use_lora,
        lora_target_modules=("Wqkv", "Wo"),
    ))