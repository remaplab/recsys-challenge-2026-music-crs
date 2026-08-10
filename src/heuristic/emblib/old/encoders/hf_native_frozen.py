"""Single source of truth for building any encoder by name.

Naming convention:
    qwen3_frozen
    qwen3_lora                                   (loaded from --adapter; cfg.pt has use_routing)
    {bert,sbert,modernbert}_native_frozen        (no training; same pipeline as text track tower)
    {bert,sbert,modernbert}_proj_{frozen,lora}_{routing,no_routing}
    keyword_{bert,sbert,modernbert}_qwen3
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Encoders that need an --adapter path to load trained weights
NEEDS_ADAPTER = {"qwen3_lora"}
HF_PROJ_PREFIXES = ("bert_proj_", "sbert_proj_", "modernbert_proj_")
NATIVE_FROZEN_NAMES = {
    "bert_native_frozen",
    "sbert_native_frozen",
    "modernbert_native_frozen",
}


def build_encoder(name: str, adapter: Path | None = None):
    """Returns a torch.nn.Module whose forward(texts, user_cf, is_cold) -> (B, D) L2-normed.

    Output dim D depends on the encoder family:
      - qwen3_*, keyword_*_qwen3      -> 1024  (Qwen3-Embedding-0.6B)
      - {bert,sbert,modernbert}_*     -> 768   (native HF hidden size)
    """
    # ── Qwen3 frozen — works out of the box, no training ────────────────
    if name == "qwen3_frozen":
        from emblib.encoders.qwen3_frozen import Qwen3QueryEncoder
        legacy = Qwen3QueryEncoder()

        class _Wrap(torch.nn.Module):
            output_dim = 1024
            def __init__(self): super().__init__(); self.legacy = legacy
            def forward(self, texts, user_cf, is_cold, device=None):
                emb = self.legacy.encode(list(texts), show_progress=False)
                t = torch.from_numpy(emb).to(device or _device(), dtype=torch.float32)
                return F.normalize(t, p=2, dim=-1)
        return _Wrap()

    # ── Qwen3 LoRA — load trained adapter ───────────────────────────────
    if name == "qwen3_lora":
        assert adapter is not None, "qwen3_lora requires --adapter"
        from emblib.encoders.qwen3_lora import Qwen3LoRAQueryEncoder
        return Qwen3LoRAQueryEncoder.load_adapter(adapter, _device())

    # ── HF native-frozen family — same pipeline as the text track tower ─
    # No training, no projection, no soft-prompt token. user_cf / is_cold
    # are accepted by the API and silently dropped (no destination).
    if name in NATIVE_FROZEN_NAMES:
        from emblib.encoders.native_frozen import NativeFrozenEncoder
        backbone = name.replace("_native_frozen", "")
        return NativeFrozenEncoder(backbone=backbone).to(_device())

    # ── HF-projected family (BERT / SBERT / ModernBERT) ─────────────────
    if name.startswith(HF_PROJ_PREFIXES):
        from emblib.encoders.hf_projected import HFProjectedEncoder
        if adapter is not None:
            return HFProjectedEncoder.load_adapter(adapter, _device())
        return _build_untrained_hf_proj(name).to(_device())

    # ── Keyword-extractor + Qwen3 (no training) ─────────────────────────
    if name in ("keyword_bert_qwen3", "keyword_sbert_qwen3", "keyword_modernbert_qwen3"):
        from emblib.encoders.keyword_qwen3 import KeywordQwen3Encoder
        ext = {
            "keyword_bert_qwen3":       "bert-base-uncased",
            "keyword_sbert_qwen3":      "sentence-transformers/all-mpnet-base-v2",
            "keyword_modernbert_qwen3": "answerdotai/ModernBERT-base",
        }[name]
        return KeywordQwen3Encoder(extractor_model=ext).to(_device())

    raise ValueError(f"unknown encoder {name!r}")


def _build_untrained_hf_proj(kind: str):
    """Build a fresh untrained HF-projected encoder. Used only by 05_train_encoder.

    Examples of valid `kind`:
        bert_proj_frozen_routing
        bert_proj_frozen_no_routing
        bert_proj_lora_routing
        sbert_proj_lora_no_routing
        modernbert_proj_frozen_routing
    """
    from emblib.encoders.hf_projected import bert_proj, sbert_proj, modernbert_proj
    parts = kind.split("_")
    if len(parts) < 4:
        raise ValueError(f"bad kind {kind!r}; want e.g. bert_proj_frozen_routing")
    base = parts[0]                            # bert | sbert | modernbert
    backbone_mode = parts[2]                   # frozen | lora
    routing_mode = "_".join(parts[3:])         # routing | no_routing
    use_routing = (routing_mode == "routing")
    use_lora = (backbone_mode == "lora")
    factory = {"bert": bert_proj, "sbert": sbert_proj, "modernbert": modernbert_proj}[base]
    return factory(use_routing=use_routing, use_lora=use_lora)