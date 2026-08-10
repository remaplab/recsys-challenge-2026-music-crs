"""emblib/retrieval/rerankers.py

Second-stage rerankers shared by scripts/launchers/gambling_updated.py and
scripts/13_precompute_reranked_qwen_recs.py (and scripts/15_fuse_rerank_diagnose.py).
Ported from scripts/10_rerank.py so the scoring is byte-for-byte what was tuned offline.

torch / transformers are imported LAZILY inside each scorer's __init__, so importing
this module is cheap (no GPU, no torch) until you actually build a scorer -- this
keeps gambling_updated's default cache-only path model-free.

Each scorer exposes a uniform interface:
  .listwise                         : bool
  .truncate_query(q, q_max_tokens)  -> str
  pointwise (qwen3, bge)            : .score_pairs(list[(query, doc)]) -> np.ndarray  (higher = better)
  listwise  (jina)                  : .score(query, docs)             -> np.ndarray  (higher = better)

RERANKERS registry keys (24 GB-safe defaults from script 10):
  qwen3_reranker_0p6b / _4b / _8b   pointwise yes/no LM (Qwen3-Reranker family)
  bge_v2_m3                         BAAI/bge-reranker-v2-m3 (sequence-classification)
  jina_v3                           jinaai/jina-reranker-v3 (LISTWISE, <=64 docs/pass)

ATTENTION KERNEL
================
Pass attn='sdpa' (PyTorch fused attention, no extra package -- recommended) or
'flash_attention_2' (needs the flash-attn package). If a requested kernel is not
installed/usable, loading falls back automatically: flash_attention_2 -> sdpa -> eager,
so a missing FlashAttention never crashes a long run.
"""
from __future__ import annotations

import numpy as np

RERANK_INSTRUCTION = (
    "Given a music recommendation conversation, judge whether the track satisfies "
    "what the listener wants next"
)

RERANKERS: dict[str, dict] = {
    "qwen3_reranker_0p6b": dict(kind="qwen3", model_id="Qwen/Qwen3-Reranker-0.6B",
                                dtype="float16", batch_size=128, max_length=384,
                                trust_remote_code=False, listwise=False),
    "qwen3_reranker_4b":   dict(kind="qwen3", model_id="Qwen/Qwen3-Reranker-4B",
                                dtype="float16", batch_size=48, max_length=384,
                                trust_remote_code=False, listwise=False),
    "qwen3_reranker_8b":   dict(kind="qwen3", model_id="Qwen/Qwen3-Reranker-8B",
                                dtype="float16", batch_size=24, max_length=384,
                                trust_remote_code=False, listwise=False),
    "bge_v2_m3":           dict(kind="bge", model_id="BAAI/bge-reranker-v2-m3",
                                dtype="float16", batch_size=256, max_length=384,
                                trust_remote_code=False, listwise=False),
    "jina_v3":             dict(kind="jina", model_id="jinaai/jina-reranker-v3",
                                dtype="auto", batch_size=64, max_length=None,
                                trust_remote_code=True, listwise=True, max_docs=64),
}


def _resolve_dtype(spec_dtype: str, override: str):
    import torch
    name = override if (override and override != "auto") else spec_dtype
    if name == "auto":
        name = "float16"                                  # rerankers default to fp16 on GPU
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def _attn_fallback_chain(attn):
    """Ordered attention kernels to try. A requested kernel that isn't installed
    falls back to the next, so a missing flash-attn never crashes the run."""
    if attn in (None, "", "auto"):
        return ["sdpa", "eager"]                          # sdpa = built-in fused attention
    if attn == "flash_attention_2":
        return ["flash_attention_2", "sdpa", "eager"]
    if attn == "sdpa":
        return ["sdpa", "eager"]
    return [attn]                                         # explicit 'eager' or anything else


def _from_pretrained(loader, model_id, dtype, trust_remote_code=False, attn=None, local_files_only=True):
    """Load with dtype-kwarg compatibility (dtype vs torch_dtype) AND attention-kernel
    fallback. Returns the first (dtype_key, attn) combination that loads."""
    base = {"trust_remote_code": trust_remote_code, "local_files_only": local_files_only}
    last = None
    for impl in _attn_fallback_chain(attn):
        kw = dict(base)
        if impl:
            kw["attn_implementation"] = impl
        for dkey in ("dtype", "torch_dtype"):             # transformers renamed the kwarg
            try:
                model = loader(model_id, **{dkey: dtype}, **kw)
                if impl and impl != (attn or "sdpa"):
                    print(f"    [attn] requested {attn!r} unavailable; using {impl!r}")
                return model
            except TypeError as e:
                last = e                                  # wrong dtype kwarg name -> try the other
            except (ImportError, ValueError) as e:
                last = e                                  # attn kernel not usable -> try next impl
                break                                     # don't retry dtype keys for a dead kernel
    raise last


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# --- pointwise scorers --------------------------------------------------------
class _BasePointwise:
    listwise = False

    def truncate_query(self, q: str, q_max: int) -> str:
        """Keep the TAIL of the (long) query -- the current user request lives at the
        bottom of the build_variant_query layout, so we must not drop it."""
        ids = self.tok(q, add_special_tokens=False)["input_ids"]
        if len(ids) > q_max:
            ids = ids[-q_max:]
        return self.tok.decode(ids)

    def score_pairs(self, pairs):
        """pairs: list[(query, doc)]. Global length-sorted batching keeps the GPU
        saturated and padding minimal."""
        if not pairs:
            return np.empty(0, dtype=np.float32)
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        scores = np.empty(len(pairs), dtype=np.float32)
        try:
            from tqdm import tqdm
            it = tqdm(list(_chunks(order, self.bs)), desc=f"{self.name} score")
        except Exception:
            it = _chunks(order, self.bs)
        for chunk in it:
            s = self._score_batch([pairs[i] for i in chunk])
            for k, i in enumerate(chunk):
                scores[i] = s[k]
        return scores


class Qwen3Reranker(_BasePointwise):
    name = "qwen3-rr"

    def __init__(self, spec, device, batch_size=None, max_length=None, attn=None,
                 dtype="auto", local_files_only=True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device
        self.bs = batch_size or spec["batch_size"]
        self.max_length = max_length or spec["max_length"]
        dt = _resolve_dtype(spec["dtype"], dtype)
        if device.type == "cpu":
            dt = torch.float32
        self.tok = AutoTokenizer.from_pretrained(
            spec["model_id"], padding_side="left",
            trust_remote_code=spec.get("trust_remote_code", False),
            local_files_only=local_files_only,
        )
        self.model = _from_pretrained(
            AutoModelForCausalLM.from_pretrained, spec["model_id"], dt,
            trust_remote_code=spec.get("trust_remote_code", False),
            attn=attn, local_files_only=local_files_only,
        ).to(device).eval()
        self.true_id = self.tok.convert_tokens_to_ids("yes")
        self.false_id = self.tok.convert_tokens_to_ids("no")
        prefix = ("<|im_start|>system\nJudge whether the Document meets the "
                  "requirements based on the Query and the Instruct provided. Note "
                  "that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
                  "<|im_start|>user\n")
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_ids = self.tok.encode(prefix, add_special_tokens=False)
        self.suffix_ids = self.tok.encode(suffix, add_special_tokens=False)
        self.body_max = self.max_length - len(self.prefix_ids) - len(self.suffix_ids)

    def _fmt(self, q, d):
        return f"<Instruct>: {RERANK_INSTRUCTION}\n<Query>: {q}\n<Document>: {d}"

    def _forward_ids(self, ids_list):
        """Pad a list of token-id sequences and return p(yes) per row."""
        torch = self.torch
        padded = self.tok.pad({"input_ids": ids_list}, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            # Only materialize logits for the LAST position; otherwise the model builds
            # a (batch x seqlen x vocab~152k) tensor and OOMs. Left padding guarantees
            # position -1 is the real last token for every row.
            try:
                logits = self.model(**padded, logits_to_keep=1).logits[:, -1, :]
            except TypeError:
                logits = self.model(**padded, num_logits_to_keep=1).logits[:, -1, :]
            pair = torch.stack([logits[:, self.false_id], logits[:, self.true_id]], dim=1)
            p_yes = torch.log_softmax(pair.float(), dim=1)[:, 1].exp()
        return p_yes.cpu().tolist()

    def score_pairs(self, pairs):
        """Optimized pointwise scoring (per-pair scores identical to the old path):
          1. tokenize ALL bodies ONCE (vectorized), build prefix+body+suffix ids;
          2. sort by TRUE token length so each batch pads tightly (minimal waste);
          3. dynamic token-budget batching -> short sequences pack into bigger batches,
             long sequences into smaller ones, keeping GPU throughput high.
        Batching/order do not affect per-pair scores (each pair is scored independently;
        left-padding is masked), so results match the unbatched reference exactly."""
        if not pairs:
            return np.empty(0, dtype=np.float32)
        # 1) tokenize once
        bodies = [self._fmt(q, d) for (q, d) in pairs]
        enc = self.tok(bodies, add_special_tokens=False, truncation=True,
                       max_length=self.body_max)["input_ids"]
        ids_all = [self.prefix_ids + x + self.suffix_ids for x in enc]
        lens = [len(x) for x in ids_all]
        # 2) sort by length
        order = sorted(range(len(ids_all)), key=lambda i: lens[i])
        scores = np.empty(len(pairs), dtype=np.float32)
        # 3) dynamic token budget: a full-length batch holds ~self.bs rows; shorter rows
        #    pack denser. Cap rows/batch so the last-token logits tensor stays bounded.
        budget = max(self.bs * self.max_length, max(lens))
        max_rows = self.bs * 4
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(order), desc=f"{self.name} score")
        except Exception:
            pbar = None
        i = 0
        while i < len(order):
            j = i
            cur_max = 0
            while j < len(order):
                cand_max = cur_max if lens[order[j]] <= cur_max else lens[order[j]]
                n = j - i + 1
                if n > 1 and (cand_max * n > budget or n > max_rows):
                    break
                cur_max = cand_max
                j += 1
            chunk = order[i:j]
            s = self._forward_ids([ids_all[k] for k in chunk])
            for pos, k in enumerate(chunk):
                scores[k] = s[pos]
            if pbar is not None:
                pbar.update(len(chunk))
            i = j
        if pbar is not None:
            pbar.close()
        return scores


class BGEReranker(_BasePointwise):
    name = "bge-rr"

    def __init__(self, spec, device, batch_size=None, max_length=None, attn=None,
                 dtype="auto", local_files_only=True):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = device
        self.bs = batch_size or spec["batch_size"]
        self.max_length = max_length or spec["max_length"]
        dt = _resolve_dtype(spec["dtype"], dtype)
        if device.type == "cpu":
            dt = torch.float32
        self.tok = AutoTokenizer.from_pretrained(
            spec["model_id"], trust_remote_code=spec.get("trust_remote_code", False),
            local_files_only=local_files_only,
        )
        self.model = _from_pretrained(
            AutoModelForSequenceClassification.from_pretrained, spec["model_id"], dt,
            trust_remote_code=spec.get("trust_remote_code", False),
            attn=attn, local_files_only=local_files_only,
        ).to(device).eval()

    def _score_batch(self, batch):
        torch = self.torch
        qs = [q for (q, _) in batch]
        ds = [d for (_, d) in batch]
        inp = self.tok(qs, ds, padding=True, truncation=True,
                       max_length=self.max_length, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inp).logits.view(-1).float()
        return logits.cpu().tolist()


# --- listwise scorer ----------------------------------------------------------
class JinaV3Reranker:
    listwise = True
    name = "jina-v3"

    def __init__(self, spec, device, batch_size=None, max_length=None, attn=None,
                 dtype="auto", local_files_only=True):
        import torch
        from transformers import AutoModel
        self.torch = torch
        self.max_docs = spec.get("max_docs", 64)
        load_dtype = "auto" if (dtype in (None, "auto") and spec["dtype"] == "auto") \
            else _resolve_dtype(spec["dtype"], dtype)
        try:
            self.model = AutoModel.from_pretrained(
                spec["model_id"], dtype=load_dtype, trust_remote_code=True,
                local_files_only=local_files_only,
            ).eval()
        except TypeError:
            self.model = AutoModel.from_pretrained(
                spec["model_id"], torch_dtype=load_dtype, trust_remote_code=True,
                local_files_only=local_files_only,
            ).eval()
        if device.type == "cuda" and torch.cuda.is_available():
            self.model = self.model.cuda()
        self.device = next(self.model.parameters()).device

    def truncate_query(self, q: str, q_max: int) -> str:
        return q[-(q_max * 6):]                            # rough char cap; jina tokenizer is internal

    def score(self, query, docs):
        docs = list(docs)
        scores = np.full(len(docs), -1e30, dtype=np.float32)
        with self.torch.no_grad():
            results = self.model.rerank(query, docs, top_n=len(docs))
        for res in results:
            idx = res.get("index", res.get("corpus_id"))
            scores[idx] = float(res.get("relevance_score", res.get("score", 0.0)))
        return scores


_CLASSES = {"qwen3": Qwen3Reranker, "bge": BGEReranker, "jina": JinaV3Reranker}


def build_scorer(key, *, device, batch_size=None, max_length=None, attn=None,
                 dtype="auto", local_files_only=True):
    if key not in RERANKERS:
        raise ValueError(f"unknown reranker {key!r}; choices: {sorted(RERANKERS)}")
    spec = RERANKERS[key]
    print(f"  loading reranker {key} ({spec['model_id']})  attn={attn or 'sdpa'}  "
          f"dtype={dtype}  local_files_only={local_files_only}")
    return _CLASSES[spec["kind"]](
        spec, device, batch_size=batch_size, max_length=max_length, attn=attn,
        dtype=dtype, local_files_only=local_files_only,
    )