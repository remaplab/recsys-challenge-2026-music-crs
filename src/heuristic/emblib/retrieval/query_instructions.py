"""emblib/retrieval/query_instructions.py

LEVEL 2 — per-bucket Qwen query INSTRUCTIONS, resolved per row from the
session's conversation_goal (Topic A-K x TARGET-specificity letter L/H).

SINGLE SOURCE OF TRUTH
======================
Both scripts/12_encode_gambling_caches.py (which APPLIES the instruction at
encode time) and scripts/launchers/gambling_updated.py + scripts/13 (which
VALIDATE the cache) import THIS module and THIS prompts file, so the
instruction a cache was built with is always checkable against the one the
runtime would choose. The instruction is NOT part of query_text, so the
byte-for-byte query_text check cannot see it — that is why scripts/12 stores
the applied instruction string in query_meta.parquet ("instruction" column)
and the loaders compare it row by row.

PROMPTS FILE (default: emblib/retrieval/instruction_prompts.json)
=================================================================
    {"base": "<instruction for every bucket NOT listed>",
     "per_bucket": {"GH": "...", "GL": "...", "JH": "...", ...}}

Bucket key = category letter + TARGET letter (2nd char of specificity):
"KH" = temporal goal, one-specific-track target. Buckets not listed use
"base" — so listing ONLY the weak buckets you flagged changes nothing
anywhere else. If the prompts file does not exist (or path == "none"),
EVERY row gets the legacy fixed instruction ("catalog") — byte-identical
behaviour to the pre-Level-2 pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from emblib.retrieval.qwen_embeddings import QWEN_INSTRUCTIONS

DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parent / "instruction_prompts.json"
LEGACY_INSTRUCTION = QWEN_INSTRUCTIONS["catalog"]


def load_prompts(path: str | Path | None = "default") -> dict | None:
    """None or 'none' or a missing file -> legacy mode (fixed catalog instruction)."""
    if path is None or str(path).lower() == "none":
        return None
    p = DEFAULT_PROMPTS_PATH if str(path) == "default" else Path(path)
    if not p.exists():
        return None
    cfg = json.loads(p.read_text(encoding="utf-8"))
    if "per_bucket" not in cfg:
        raise ValueError(f"{p}: prompts JSON must contain 'per_bucket'")
    cfg.setdefault("base", LEGACY_INSTRUCTION)
    cfg["_path"] = str(p)
    return cfg


def bucket_key(row: dict[str, Any]) -> str:
    g = row.get("conversation_goal") or {}
    cat = (g.get("category") or "?").strip().upper()
    spec = (g.get("specificity") or "").strip().upper()
    tgt = spec[1] if len(spec) == 2 else "?"
    return f"{cat}{tgt}"


def resolve_instruction(row: dict[str, Any], cfg: dict | None) -> str:
    if cfg is None:
        return LEGACY_INSTRUCTION
    return cfg["per_bucket"].get(bucket_key(row), cfg["base"])


def resolve_instructions(task_rows: list[dict[str, Any]], cfg: dict | None) -> list[str]:
    """One instruction STRING per task row (task_rows carry the full session 'row')."""
    return [resolve_instruction(t["row"], cfg) for t in task_rows]


def describe(cfg: dict | None) -> str:
    if cfg is None:
        return "legacy fixed instruction ('catalog') for every row"
    return (f"{cfg.get('_path', '<inline>')} — {len(cfg['per_bucket'])} bucket prompts "
            f"({sorted(cfg['per_bucket'])}), others use base")