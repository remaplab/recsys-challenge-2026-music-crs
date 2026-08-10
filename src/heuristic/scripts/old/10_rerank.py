"""scripts/10_rerank.py  (v2 — fast)

Unified second-stage reranking on top of any bi-encoder from
scripts/08_try_embedding_models.py. Supersedes 09_rerank_jina_v3.py.

v2 SPEED CHANGES (vs v1)
  * Query truncation: --q-max-tokens keeps the TAIL of the (long) v2 query and
    leaves the short track-metadata doc intact, so sequences shrink ~3x.
  * Global, length-sorted batching across ALL (query, doc) pairs (not per-row),
    so the GPU stays saturated and padding is minimal.
  * --subsample to tune N on a random subset of val (then confirm best N on full).
  * --batch-size / --max-length / --attn overrides.

WHY N MATTERS (recap)
  recall@N of the BASE retriever is the hard ceiling on post-rerank NDCG@20:
  the reranker can only reorder what was retrieved. Reranking N=20 can only
  the reranker can only reorder what was retrieved. Reranking N=20 can only
  reshuffle the same 20 the bi-encoder already returned, so it rarely beats the
  baseline — the gains come from larger N (pulling the GT from positions 21..N
  into the top-20). Use --recall-only to see the ceiling, then sweep larger N.

USAGE
  uv run python scripts/10_rerank.py --base qwen3_0p6b --recall-only

  # fast tuning on a subset, cheap reranker:
  uv run python scripts/10_rerank.py --base qwen3_0p6b --reranker qwen3_reranker_0p6b \
      --sweep 50 100 200 --subsample 800

  # confirm the winning N on full val with the strong reranker:
  uv run python scripts/10_rerank.py --base qwen3_0p6b --reranker qwen3_reranker_4b \
      --sweep 100

RERANKERS (24 GB-safe):
  qwen3_reranker_0p6b / _4b / _8b   pointwise yes/no LM (same family as your embedder)
  bge_v2_m3                         BAAI/bge-reranker-v2-m3 (568M, fast, diverse)
  jina_v3                           jinaai/jina-reranker-v3 (LISTWISE, <=64 docs/pass)

Output: models/eval_results/<base>__<reranker>/val/ndcg.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emblib.qwen.qwen_embeddings import track_metadata_text

DATA        = Path("./data/talkpl-ai")
TRACK_META  = DATA / "TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet"
TRACK_CACHE = Path("./models/track_tower_generic_cache")
QUERY_CACHE = Path("./models/query_emb_generic_cache")
CAND_CACHE  = Path("./models/rerank_candidates")
EVAL_OUT    = Path("./models/eval_results")
NDCG_K      = 20

RERANK_INSTRUCTION = (
    "Given a music recommendation conversation, judge whether the track satisfies "
    "what the listener wants next"
)

RERANKERS: dict[str, dict] = {
    "qwen3_reranker_0p6b": dict(kind="qwen3", model_id="Qwen/Qwen3-Reranker-0.6B",
                                dtype="float16", batch_size=128, max_length=384),
    "qwen3_reranker_4b":   dict(kind="qwen3", model_id="Qwen/Qwen3-Reranker-4B",
                                dtype="float16", batch_size=48,  max_length=384),
    "qwen3_reranker_8b":   dict(kind="qwen3", model_id="Qwen/Qwen3-Reranker-8B",
                                dtype="float16", batch_size=24,  max_length=384),
    "bge_v2_m3":           dict(kind="bge",   model_id="BAAI/bge-reranker-v2-m3",
                                dtype="float16", batch_size=256, max_length=384),
    "jina_v3":             dict(kind="jina",  model_id="jinaai/jina-reranker-v3",
                                dtype="auto",  batch_size=64, max_length=None,
                                listwise=True, max_docs=64),
}


def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _from_pretrained(loader, model_id, dtype, trust_remote_code=False, attn=None):
    kw = {"trust_remote_code": trust_remote_code}
    if attn:
        kw["attn_implementation"] = attn
    last = None
    for dkey in ("dtype", "torch_dtype"):
        try:
            return loader(model_id, **{dkey: dtype}, **kw)
        except TypeError as e:
            last = e
    raise last


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# ─── reranker scorers ────────────────────────────────────────────────────────
class _BasePointwise:
    listwise = False

    def truncate_query(self, q: str, q_max: int) -> str:
        """Keep the TAIL of the query (current user request lives at the bottom
        of the v2 layout), so we don't drop the salient part."""
        ids = self.tok(q, add_special_tokens=False)["input_ids"]
        if len(ids) > q_max:
            ids = ids[-q_max:]
        return self.tok.decode(ids)

    def score_pairs(self, pairs):
        """pairs: list[(query, doc)]. Global length-sorted batching."""
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        scores = np.empty(len(pairs), dtype=np.float32)
        from tqdm import tqdm
        for chunk in tqdm(list(_chunks(order, self.bs)), desc=f"{self.name} score"):
            s = self._score_batch([pairs[i] for i in chunk])
            for k, i in enumerate(chunk):
                scores[i] = s[k]
        return scores


class Qwen3Reranker(_BasePointwise):
    name = "qwen3-rr"

    def __init__(self, spec, device, batch_size=None, max_length=None, attn=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device
        self.bs = batch_size or spec["batch_size"]
        self.max_length = max_length or spec["max_length"]
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}[spec["dtype"]]
        if device.type == "cpu":
            dtype = torch.float32
        self.tok = AutoTokenizer.from_pretrained(spec["model_id"], padding_side="left")
        self.model = _from_pretrained(AutoModelForCausalLM.from_pretrained,
                                      spec["model_id"], dtype, attn=attn).to(device).eval()
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

    def _score_batch(self, batch):
        torch = self.torch
        bodies = [self._fmt(q, d) for (q, d) in batch]
        enc = self.tok(bodies, add_special_tokens=False, truncation=True,
                       max_length=self.body_max)["input_ids"]
        ids = [self.prefix_ids + x + self.suffix_ids for x in enc]
        padded = self.tok.pad({"input_ids": ids}, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            # Only materialize logits for the LAST position. Without this the
            # model builds a (batch x seqlen x vocab~152k) tensor and OOMs; we
            # only need the final token's yes/no logits anyway. Left padding
            # guarantees position -1 is the real last token for every row.
            try:
                logits = self.model(**padded, logits_to_keep=1).logits[:, -1, :]
            except TypeError:
                logits = self.model(**padded, num_logits_to_keep=1).logits[:, -1, :]
            pair = torch.stack([logits[:, self.false_id], logits[:, self.true_id]], dim=1)
            p_yes = torch.log_softmax(pair.float(), dim=1)[:, 1].exp()
        return p_yes.cpu().tolist()


class BGEReranker(_BasePointwise):
    name = "bge-rr"

    def __init__(self, spec, device, batch_size=None, max_length=None, attn=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = device
        self.bs = batch_size or spec["batch_size"]
        self.max_length = max_length or spec["max_length"]
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}[spec["dtype"]]
        if device.type == "cpu":
            dtype = torch.float32
        self.tok = AutoTokenizer.from_pretrained(spec["model_id"])
        self.model = _from_pretrained(AutoModelForSequenceClassification.from_pretrained,
                                      spec["model_id"], dtype, attn=attn).to(device).eval()

    def _score_batch(self, batch):
        torch = self.torch
        qs = [q for (q, _) in batch]
        ds = [d for (_, d) in batch]
        inp = self.tok(qs, ds, padding=True, truncation=True,
                       max_length=self.max_length, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inp).logits.view(-1).float()
        return logits.cpu().tolist()


class JinaV3Reranker:
    listwise = True
    name = "jina-v3"

    def __init__(self, spec, device, batch_size=None, max_length=None, attn=None):
        import torch
        from transformers import AutoModel
        self.torch = torch
        self.max_docs = spec.get("max_docs", 64)
        self.model = AutoModel.from_pretrained(spec["model_id"], dtype="auto",
                                               trust_remote_code=True).eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def truncate_query(self, q: str, q_max: int) -> str:
        return q[-(q_max * 6):]            # rough char cap; jina tokenizer is internal

    def score(self, query, docs):
        docs = list(docs)
        scores = np.full(len(docs), -1e30, dtype=np.float32)
        with self.torch.no_grad():
            results = self.model.rerank(query, docs, top_n=len(docs))
        for res in results:
            idx = res.get("index", res.get("corpus_id"))
            scores[idx] = float(res.get("relevance_score", res.get("score", 0.0)))
        return scores


def build_scorer(key, device, batch_size, max_length, attn):
    spec = RERANKERS[key]
    print(f"  loading reranker {key} ({spec['model_id']})  attn={attn or 'default'}")
    cls = {"qwen3": Qwen3Reranker, "bge": BGEReranker, "jina": JinaV3Reranker}[spec["kind"]]
    return cls(spec, device, batch_size=batch_size, max_length=max_length, attn=attn)


# ─── base loading + candidate retrieval ──────────────────────────────────────
def load_base(base):
    tc = TRACK_CACHE / base
    ids = [str(t) for t in np.load(tc / "track_ids.npy", allow_pickle=True).tolist()]
    # A fused base (built by 11_fuse_candidates.py) has track_ids + val_meta but
    # no tower / query embeddings — those are only needed to BUILD candidates,
    # which a fused base already has cached. Load what exists.
    emb  = np.load(tc / "emb.npy")  if (tc / "emb.npy").exists()  else None
    mask = np.load(tc / "mask.npy") if (tc / "mask.npy").exists() else None
    qc = QUERY_CACHE / base
    q   = np.load(qc / "val.npy") if (qc / "val.npy").exists() else None
    meta = pl.read_parquet(qc / "val_meta.parquet").to_dicts()
    return emb, mask, ids, q, meta


def build_candidates(base, emb, mask, ids, q, meta, id_to_idx, built_n, device):
    import torch
    CAND_CACHE.mkdir(parents=True, exist_ok=True)
    cpath = CAND_CACHE / f"{base}_cand_{built_n}.npz"
    if cpath.exists():
        z = np.load(cpath)
        print(f"  loaded cached candidates {z['cand'].shape} from {cpath}")
        return z["cand"], z["gt_idx"], z["turns"], z["scorable"]

    # Fallback: reuse a larger cached candidate file and slice (e.g. a fused
    # base cached at 500 while this run only needs 200).
    larger = []
    for f in CAND_CACHE.glob(f"{base}_cand_*.npz"):
        try:
            M = int(f.stem.split("_cand_")[1])
        except (IndexError, ValueError):
            continue
        if M >= built_n:
            larger.append((M, f))
    if larger:
        M, f = min(larger)
        z = np.load(f)
        print(f"  loaded cached candidates {z['cand'].shape} from {f}; slicing to {built_n}")
        return z["cand"][:, :built_n].copy(), z["gt_idx"], z["turns"], z["scorable"]

    if emb is None:
        raise FileNotFoundError(
            f"No candidate cache for base {base!r} and no tower to build from. "
            f"If this is a fused base, run scripts/11_fuse_candidates.py first."
        )

    n = q.shape[0]
    tower = torch.from_numpy(emb).to(device, dtype=torch.float32)
    masked_cols = torch.from_numpy(~mask).to(device)
    cand = np.full((n, built_n), -1, dtype=np.int32)
    gt_idx = np.full(n, -1, dtype=np.int64)
    turns = np.full(n, -1, dtype=np.int64)
    scorable = np.zeros(n, dtype=bool)

    CH = 256
    for s in range(0, n, CH):
        e = min(s + CH, n)
        qb = torch.from_numpy(q[s:e]).to(device, dtype=torch.float32)
        scores = qb @ tower.T
        scores[:, masked_cols] = float("-inf")
        for bi in range(e - s):
            ri = s + bi
            r = meta[ri]
            if r.get("turn_number") is not None:
                turns[ri] = int(r["turn_number"])
            gid = r.get("gt_track_id")
            if gid is None or gid not in id_to_idx:
                continue
            gi = id_to_idx[gid]
            row = scores[bi]
            gt_score = row[gi].item()
            if gt_score == float("-inf"):
                continue
            gt_idx[ri] = gi
            scorable[ri] = True
            for tid in (r.get("prior_track_ids") or []):
                j = id_to_idx.get(tid)
                if j is not None:
                    row[j] = float("-inf")
            row[gi] = gt_score
            top = torch.topk(row, k=min(built_n, row.shape[0]), dim=0).indices
            cand[ri, :top.shape[0]] = top.cpu().numpy().astype(np.int32)

    np.savez(cpath, cand=cand, gt_idx=gt_idx, turns=turns, scorable=scorable)
    print(f"  built + cached candidates {cand.shape} -> {cpath}")
    return cand, gt_idx, turns, scorable


def recall_at(cand, gt_idx, scorable, rows_eval, ns):
    out = {}
    for N in ns:
        hit = tot = 0
        for ri in rows_eval:
            if not scorable[ri]:
                continue
            tot += 1
            if gt_idx[ri] in cand[ri, :N]:
                hit += 1
        out[N] = hit / max(tot, 1)
    return out


def _ndcg(rank):
    return (1.0 / np.log2(1 + rank)) if rank <= NDCG_K else 0.0


def macro_micro(per_turn):
    ptm = {k: float(np.mean(v)) for k, v in sorted(per_turn.items())}
    macro = float(np.mean(list(ptm.values()))) if ptm else 0.0
    allv = [v for vs in per_turn.values() for v in vs]
    micro = float(np.mean(allv)) if allv else 0.0
    return macro, micro, ptm


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True)
    p.add_argument("--reranker", choices=list(RERANKERS))
    p.add_argument("--sweep", type=int, nargs="+", default=[50, 100, 200])
    p.add_argument("--recall-ns", type=int, nargs="+", default=[20, 50, 100, 200, 500])
    p.add_argument("--recall-only", action="store_true")
    p.add_argument("--subsample", type=int, default=0,
                   help="evaluate on a random subset of scorable rows (0 = all). "
                        "Use for fast N-tuning, then confirm best N on full val.")
    p.add_argument("--q-max-tokens", type=int, default=224,
                   help="truncate the query to its last N tokens before reranking")
    p.add_argument("--batch-size", type=int, default=None, help="override reranker batch size")
    p.add_argument("--max-length", type=int, default=None, help="override reranker max seq length")
    p.add_argument("--attn", default=None, choices=["flash_attention_2", "sdpa", "eager"],
                   help="attention impl (try flash_attention_2 if installed)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = _device()
    print(f"device: {device}")
    emb, mask, ids, q, meta = load_base(args.base)
    id_to_idx = {t: i for i, t in enumerate(ids)}
   # print(f"base={args.base}  val={q.shape}  tower={emb.shape}  mask={mask.mean():.3f}")
    if emb is None or q is None:
        print(f"base={args.base}  (fused base: no tower/query emb — "
              f"reranking cached candidates only)  rows={len(meta)}")
    else:
        print(f"base={args.base}  val={q.shape}  tower={emb.shape}  mask={mask.mean():.3f}")

    built_n = max(args.sweep + args.recall_ns)
    cand, gt_idx, turns, scorable = build_candidates(
        args.base, emb, mask, ids, q, meta, id_to_idx, built_n, device)

    scorable_rows = [ri for ri in range(len(meta)) if scorable[ri]]
    if args.subsample and args.subsample < len(scorable_rows):
        rng = np.random.default_rng(args.seed)
        scorable_rows = sorted(rng.choice(scorable_rows, size=args.subsample, replace=False).tolist())
        print(f"  subsample: evaluating on {len(scorable_rows)} of "
              f"{int(scorable.sum())} scorable rows (seed={args.seed})")
    n_eval = len(scorable_rows)

    rec = recall_at(cand, gt_idx, scorable, scorable_rows, sorted(set(args.recall_ns + args.sweep)))
    print(f"\n  recall@N ceiling (rows={n_eval}):")
    for N in sorted(rec):
        print(f"    recall@{N:<4} = {rec[N]:.4f}")

    base_json = EVAL_OUT / args.base / "val" / "ndcg.json"
    base_ndcg = json.loads(base_json.read_text()).get("macro_by_turn") if base_json.exists() else None
    if base_ndcg is not None:
        print(f"  BASE bi-encoder NDCG@{NDCG_K} (full catalog) macro = {base_ndcg:.4f}")

    if args.recall_only or not args.reranker:
        print("\n[recall-only] sweep larger N where recall is still rising.")
        return

    md_by_id = {str(r["track_id"]): r for r in pl.read_parquet(TRACK_META).to_dicts()}
    track_text = [track_metadata_text(md_by_id[t]) if t in md_by_id else "" for t in ids]

    scorer = build_scorer(args.reranker, device, args.batch_size, args.max_length, args.attn)
    spec = RERANKERS[args.reranker]
    sweep = sorted(args.sweep)
    if scorer.listwise:
        cap = spec.get("max_docs", 64)
        if max(sweep) > cap:
            print(f"  [note] {args.reranker} listwise; capping N at {cap}")
        sweep = [N for N in sweep if N <= cap] or [cap]
    rerank_n = max(sweep)

    # query truncation once per row
    qtrunc = {ri: scorer.truncate_query(meta[ri].get("query_text") or "", args.q_max_tokens)
              for ri in scorable_rows}

    precomp = {}
    if not scorer.listwise:
        # flatten ALL (query, doc) pairs for top-rerank_n, score once, scatter back
        pair_list = []
        row_cand = {}
        for ri in scorable_rows:
            c = cand[ri, :rerank_n]; c = c[c >= 0]
            row_cand[ri] = c
            for j in c:
                pair_list.append((qtrunc[ri], track_text[j]))
        print(f"\n  scoring {len(pair_list)} (query,doc) pairs "
              f"[{n_eval} rows x up to {rerank_n}] with {args.reranker} ...")
        flat = scorer.score_pairs(pair_list)
        pos = 0
        for ri in scorable_rows:
            m = len(row_cand[ri])
            precomp[ri] = flat[pos:pos + m]
            pos += m

    per_n_report = {}
    for N in sweep:
        per_turn = defaultdict(list)
        for ri in scorable_rows:
            c = cand[ri, :N]; c = c[c >= 0]
            gi = gt_idx[ri]; t = int(turns[ri])
            if gi not in c:
                per_turn[t].append(0.0); continue
            if scorer.listwise:
                s = scorer.score(qtrunc[ri], [track_text[j] for j in c])
            else:
                s = precomp[ri][:len(c)]
            reranked = c[np.argsort(-s)]
            rank = int(np.where(reranked == gi)[0][0]) + 1
            per_turn[t].append(_ndcg(rank))
        macro, micro, ptm = macro_micro(per_turn)
        per_n_report[N] = {"macro_by_turn": macro, "micro": micro, "recall_at_n": rec.get(N)}
        delta = f"  (base {base_ndcg:.4f}, Δ {macro - base_ndcg:+.4f})" if base_ndcg is not None else ""
        print(f"  N={N:<4} recall@N={rec.get(N, float('nan')):.4f}  "
              f"NDCG@{NDCG_K} macro={macro:.4f} micro={micro:.4f}{delta}")

    best_N = max(per_n_report, key=lambda k: per_n_report[k]["macro_by_turn"])
    best = per_n_report[best_N]["macro_by_turn"]
    print(f"\n  BEST: N={best_N}  NDCG@{NDCG_K} macro = {best:.4f}"
          + (f"  (base {base_ndcg:.4f}, Δ {best - base_ndcg:+.4f})" if base_ndcg is not None else ""))

    key = f"{args.base}__{args.reranker}"
    out = EVAL_OUT / key / "val"; out.mkdir(parents=True, exist_ok=True)
    (out / "ndcg.json").write_text(json.dumps({
        "base": args.base, "reranker": args.reranker, "fold": "val",
        "metric": f"ndcg@{NDCG_K}", "n_eval": n_eval, "subsample": args.subsample,
        "q_max_tokens": args.q_max_tokens,
        "base_bi_encoder_macro_by_turn": base_ndcg, "recall_at_n": rec,
        "best_n": best_N, "best_macro_by_turn": best, "per_n": per_n_report,
    }, indent=2))
    print(f"  wrote {out / 'ndcg.json'}")


if __name__ == "__main__":
    main()