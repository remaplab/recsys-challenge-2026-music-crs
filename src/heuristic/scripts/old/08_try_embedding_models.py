# """scripts/08_try_embedding_models.py
#
# Try alternative *bi-encoder* embedding models for the TalkPlay retrieval task,
# keeping the exact property your Qwen-native pipeline relies on:
#
#     query and tracks are encoded by the SAME model and L2-normalized,
#     so a dot product IS cosine similarity in one shared space.
#
# For each model in the registry this:
#   1. builds a TRACK tower  — encodes `track_metadata_text(row)` for every track in
#      the canonical organizer order (identical text to your current Qwen pipeline),
#      DOC side, no instruction prefix;
#   2. encodes the VAL queries (one row per val session at its pinned predict-turn) and
#      the BLIND-A queries, QUERY side, with an instruction prefix for instruction-aware
#      models (none for symmetric models like bge-m3);
#   3. computes macro-by-turn NDCG@20 on VAL — the SAME definition as
#      scripts/06b_evaluate_ndcg.py (per-K mean, then unweighted mean across K).
#
# Everything is cached, so eval can be re-run without re-encoding.
#
# Backend = sentence-transformers (already in your deps). ST loads each model's own
# pooling (last-token for Qwen3/gte-Qwen2/e5-mistral, CLS for bge-m3, etc.), handles
# EOS/padding, and lets us prepend a per-model query instruction. We only manage:
# prompts, L2-normalization, max length, and LEFT truncation for queries (so the
# [CURRENT USER] tail survives, exactly like the rest of your pipeline).
#
# VRAM: every model below is chosen to fit in 24 GB at fp16/bf16. The 8B model is the
# tightest (~16 GB weights); drop its batch size if you OOM.
#
# USAGE
# -----
#   # one model, full pipeline (tracks -> val+blindA queries -> eval):
#   uv run python scripts/08_try_embedding_models.py --model qwen3_4b --stage all
#
#   # only (re)evaluate from existing caches:
#   uv run python scripts/08_try_embedding_models.py --model qwen3_4b --stage eval
#
#   # build the side-by-side table once several have been scored:
#   uv run python scripts/08_try_embedding_models.py --compare \
#       qwen3_0p6b qwen3_4b qwen3_8b gte_qwen2_7b e5_mistral_7b bge_m3
#
# Outputs
#   models/track_tower_generic_cache/<key>/{track_ids.npy, emb.npy, mask.npy}
#   models/query_emb_generic_cache/<key>/{val,blind_a}.npy + *_meta.parquet
#   models/eval_results/<key>/val/ndcg.json
#   models/eval_results/ndcg_generic_comparison_val.csv
# """
# from __future__ import annotations
#
# import argparse
# import gc
# import json
# import sys
# from collections import defaultdict
# from pathlib import Path
#
# import numpy as np
# import polars as pl
#
# sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
#
# from emblib.data.parsing import build_query_text_v2
# from emblib.data.split import load_blinda_split, turn_pairs_for_fold
# from emblib.qwen.qwen_embeddings import track_metadata_text
#
#
# # ─── paths ───────────────────────────────────────────────────────────────────
# DATA          = Path("./data/talkpl-ai")
# TRACK_META    = DATA / "TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet"
# EMBED_SHARDS  = DATA / "TalkPlayData-Challenge-Track-Embeddings/data"
# TRAIN_CONV    = DATA / "TalkPlayData-Challenge-Dataset/data/train-00000-of-00001.parquet"
# TEST_CONV     = DATA / "TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet"
# BLIND_CONV    = DATA / "TalkPlayData-Challenge-Blind-A/data/test-00000-of-00001.parquet"
# SPLITS_PATH   = Path("./models/splits/train_val_test_blinda_matched.parquet")
#
# TRACK_CACHE   = Path("./models/track_tower_generic_cache")
# QUERY_CACHE   = Path("./models/query_emb_generic_cache")
# EVAL_OUT      = Path("./models/eval_results")
#
# NDCG_K        = 20
# SCORE_CHUNK   = 512
#
#
# # ─── instruction shared by all instruction-aware models (fair comparison) ─────
# MUSIC_INSTRUCTION = (
#     "Given a music recommendation conversation, retrieve the track from the "
#     "catalog that best matches what the listener wants next"
# )
# def instruct(q_instr: str) -> str:
#     return f"Instruct: {q_instr}\nQuery: "
#
#
# # ─── model registry ──────────────────────────────────────────────────────────
# # pooling/EOS are handled by each model's sentence-transformers config; we only
# # set the query prompt (None = symmetric / no instruction), the doc prompt
# # (None = raw metadata, the retrieval convention), max lengths and batch size.
# #
# # dim is informational only (we read the true dim from the encoder output).
# EMBEDDING_MODELS: dict[str, dict] = {
#     # ── Qwen3-Embedding family (last-token + instruction). Drop-in upgrades of
#     #    your current 0.6B. 0.6B included so the baseline is produced by THIS
#     #    same harness (apples-to-apples).
#     "qwen3_0p6b": dict(
#         model_id="Qwen/Qwen3-Embedding-0.6B", dim=1024,
#         query_prompt=instruct(MUSIC_INSTRUCTION), doc_prompt=None,
#         query_max=512, doc_max=256, dtype="float16",
#         trust_remote_code=False, batch_size=16,
#     ),
#     "qwen3_4b": dict(
#         model_id="Qwen/Qwen3-Embedding-4B", dim=2560,
#         query_prompt=instruct(MUSIC_INSTRUCTION), doc_prompt=None,
#         query_max=512, doc_max=256, dtype="float16",
#         trust_remote_code=False, batch_size=8,
#     ),
#     "qwen3_8b": dict(
#         model_id="Qwen/Qwen3-Embedding-8B", dim=4096,
#         query_prompt=instruct(MUSIC_INSTRUCTION), doc_prompt=None,
#         query_max=512, doc_max=256, dtype="float16",
#         trust_remote_code=False, batch_size=4,        # ~16 GB weights — tightest
#     ),
#
#     # ── gte-Qwen2-7B-instruct (previous-gen SOTA; last-token + instruction).
#     "gte_qwen2_7b": dict(
#         model_id="Alibaba-NLP/gte-Qwen2-7B-instruct", dim=3584,
#         query_prompt=instruct(MUSIC_INSTRUCTION), doc_prompt=None,
#         query_max=512, doc_max=256, dtype="float16",
#         trust_remote_code=True, batch_size=8,
#     ),
#
#     # ── e5-mistral-7b-instruct (Mistral lineage; last-token + instruction).
#     "e5_mistral_7b": dict(
#         model_id="intfloat/e5-mistral-7b-instruct", dim=4096,
#         query_prompt=instruct(MUSIC_INSTRUCTION), doc_prompt=None,
#         query_max=512, doc_max=256, dtype="float16",
#         trust_remote_code=False, batch_size=8,
#     ),
#
#     # ── bge-m3 (XLM-RoBERTa, CLS pooling, NO instruction). Tiny (568M), fast,
#     #    architecturally different — a good sanity/diversity baseline.
#     "bge_m3": dict(
#         model_id="BAAI/bge-m3", dim=1024,
#         query_prompt=None, doc_prompt=None,
#         query_max=512, doc_max=256, dtype="float16",
#         trust_remote_code=False, batch_size=64,
#     ),
#
#     # ── OPTIONAL extras (uncomment to try). ───────────────────────────────────
#     # stella: strong 1.5B, mean pooling, has its own s2p query prompt.
#     # "stella_1p5b": dict(
#     #     model_id="NovaSearch/stella_en_1.5B_v5", dim=1024,
#     #     query_prompt="Instruct: Retrieve the music track that best satisfies the "
#     #                  "listener's request.\nQuery: ",
#     #     doc_prompt=None, query_max=512, doc_max=256, dtype="float16",
#     #     trust_remote_code=True, batch_size=16,
#     # ),
#     # NV-Embed-v2: 7B, 4096-d, near-SOTA. Needs trust_remote_code + a recent
#     # transformers; ST works but can be version-sensitive.
#     # "nvembed_v2": dict(
#     #     model_id="nvidia/NV-Embed-v2", dim=4096,
#     #     query_prompt=instruct(MUSIC_INSTRUCTION), doc_prompt=None,
#     #     query_max=512, doc_max=256, dtype="float16",
#     #     trust_remote_code=True, batch_size=4,
#     # ),
# }
#
#
# # ─── helpers ─────────────────────────────────────────────────────────────────
# def _device():
#     import torch
#     return torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# def canonical_track_ids(shards_dir: Path) -> list[str]:
#     shards = sorted(Path(shards_dir).glob("all_tracks-*.parquet"))
#     if not shards:
#         raise FileNotFoundError(f"No all_tracks-*.parquet in {shards_dir}")
#     ids: list[str] = []
#     for shard in shards:
#         ids.extend(str(t) for t in pl.read_parquet(shard, columns=["track_id"])["track_id"].to_list())
#     return ids
#
#
# def load_track_lookup(path: Path) -> dict[str, dict]:
#     md = pl.read_parquet(path)
#     return {row["track_id"]: row for row in md.to_dicts()}
#
#
# def load_st_model(spec: dict, device):
#     import torch
#     from sentence_transformers import SentenceTransformer
#     dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
#              "float32": torch.float32}[spec["dtype"]]
#     if device.type == "cpu" and dtype != torch.float32:
#         print("  [warn] CPU detected — forcing float32 (fp16 matmul unsupported on CPU)")
#         dtype = torch.float32
#     print(f"  loading {spec['model_id']}  (dtype={dtype}, trust_remote_code={spec['trust_remote_code']})")
#     # transformers renamed the from_pretrained dtype kwarg (torch_dtype -> dtype)
#     # in 5.x. Try the new name first, fall back to the old one.
#     last_err = None
#     for dkey in ("dtype", "torch_dtype"):
#         try:
#             return SentenceTransformer(
#                 spec["model_id"],
#                 trust_remote_code=spec["trust_remote_code"],
#                 model_kwargs={dkey: dtype},
#                 device=str(device),
#             )
#         except TypeError as e:
#             last_err = e
#     raise last_err
#
#
# def encode_texts(model, texts, prompt, max_len, batch_size, truncation_side):
#     model.max_seq_length = max_len
#     try:
#         model.tokenizer.truncation_side = truncation_side
#     except Exception:
#         pass
#     emb = model.encode(
#         texts,
#         prompt=prompt,                 # None => no prompt prepended
#         batch_size=batch_size,
#         normalize_embeddings=True,     # => dot product is cosine
#         convert_to_numpy=True,
#         show_progress_bar=True,
#     )
#     return np.asarray(emb, dtype=np.float32)
#
#
# def free_model(model):
#     import torch
#     del model
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#
#
# # ─── query-row reconstruction (mirrors 03_encode_queries.parse_session) ───────
# def build_query_row(sess_row: dict, K: int, track_lookup: dict, need_gt: bool):
#     convs = sess_row.get("conversations") or []
#     up    = sess_row.get("user_profile") or {}
#     cg    = sess_row.get("conversation_goal") or {}
#     sdate = sess_row.get("session_date")
#     sdate = str(sdate) if sdate is not None else None
#
#     by_turn = defaultdict(list)
#     for t in convs:
#         by_turn[t["turn_number"]].append(t)
#
#     umsgs = [t for t in by_turn.get(K, []) if t.get("role") == "user"]
#     if not umsgs:
#         return None
#
#     gt = None
#     if need_gt:
#         mmsgs = [t for t in by_turn.get(K, []) if t.get("role") == "music"]
#         gt = mmsgs[0].get("content") if mmsgs else None
#         if gt is None:
#             return None
#
#     history, prior = [], []
#     for t in sorted(by_turn):
#         if t >= K:
#             break
#         for prev in by_turn[t]:
#             history.append({
#                 "role": prev.get("role"),
#                 "content": prev.get("content") or "",
#                 "thought": prev.get("thought") or "",
#             })
#             if prev.get("role") == "music" and prev.get("content"):
#                 prior.append(prev["content"])
#
#     text = build_query_text_v2(
#         chat_history=history, user_query=umsgs[0]["content"],
#         user_profile=up, conversation_goal=cg, session_date=sdate,
#         track_lookup=track_lookup, use_thoughts=True,
#     )
#     return {
#         "session_id": sess_row["session_id"], "turn_number": int(K),
#         "gt_track_id": gt, "category": cg.get("category") or "",
#         "specificity": cg.get("specificity") or "",
#         "prior_track_ids": prior, "query_text": text,
#     }
#
#
# def build_val_rows(track_lookup: dict) -> list[dict]:
#     if not SPLITS_PATH.exists():
#         raise FileNotFoundError(f"{SPLITS_PATH} not found. Run scripts/01b_rebuild_split.py first.")
#     split_df = load_blinda_split(SPLITS_PATH)
#     pairs = turn_pairs_for_fold(split_df, "val")            # {(session_id, predict_turn)}
#
#     # val sessions can come from organizer train OR test (01b merges both)
#     sessions: dict[str, dict] = {}
#     for path in (TRAIN_CONV, TEST_CONV):
#         if path.exists():
#             for r in pl.read_parquet(path).to_dicts():
#                 sessions[str(r["session_id"])] = r
#
#     rows, miss = [], 0
#     for sid, K in sorted(pairs):
#         s = sessions.get(str(sid))
#         if s is None:
#             miss += 1
#             continue
#         row = build_query_row(s, int(K), track_lookup, need_gt=True)
#         if row is not None:
#             rows.append(row)
#     print(f"  val rows built: {len(rows)}  (pairs={len(pairs)}, sessions missing={miss})")
#     return rows
#
#
# def build_blind_rows(track_lookup: dict) -> list[dict]:
#     if not BLIND_CONV.exists():
#         print(f"  [skip] blind conv not found at {BLIND_CONV}")
#         return []
#     rows = []
#     for s in pl.read_parquet(BLIND_CONV).to_dicts():
#         by_turn = defaultdict(list)
#         for t in (s.get("conversations") or []):
#             by_turn[t["turn_number"]].append(t)
#         user_turns = [tn for tn, ts in by_turn.items() if any(x.get("role") == "user" for x in ts)]
#         if not user_turns:
#             continue
#         K = max(user_turns)                                  # predict the final user request
#         row = build_query_row(s, int(K), track_lookup, need_gt=False)
#         if row is not None:
#             rows.append(row)
#     print(f"  blind_a rows built: {len(rows)}")
#     return rows
#
#
# def write_query_cache(key: str, fold: str, rows: list[dict], emb: np.ndarray):
#     out = QUERY_CACHE / key
#     out.mkdir(parents=True, exist_ok=True)
#     np.save(out / f"{fold}.npy", emb)
#     pl.DataFrame({
#         "session_id":      [r["session_id"] for r in rows],
#         "turn_number":     [r["turn_number"] for r in rows],
#         "gt_track_id":     [r["gt_track_id"] for r in rows],
#         "category":        [r["category"] for r in rows],
#         "specificity":     [r["specificity"] for r in rows],
#         "prior_track_ids": [r["prior_track_ids"] for r in rows],
#         # query_text is persisted so the second-stage reranker (script 09) has
#         # the real user text to score against, not just an id.
#         "query_text":      [r["query_text"] for r in rows],
#     }, schema={
#         "session_id": pl.String, "turn_number": pl.Int64, "gt_track_id": pl.String,
#         "category": pl.String, "specificity": pl.String,
#         "prior_track_ids": pl.List(pl.String), "query_text": pl.String,
#     }).write_parquet(out / f"{fold}_meta.parquet")
#     print(f"  wrote {emb.shape} -> {out / (fold + '.npy')}")
#
#
# # ─── track tower ─────────────────────────────────────────────────────────────
# def load_track_tower(key: str):
#     c = TRACK_CACHE / key
#     return (np.load(c / "emb.npy"), np.load(c / "mask.npy"),
#             [str(t) for t in np.load(c / "track_ids.npy", allow_pickle=True).tolist()])
#
#
# def track_tower_exists(key: str) -> bool:
#     c = TRACK_CACHE / key
#     return (c / "emb.npy").exists() and (c / "mask.npy").exists() and (c / "track_ids.npy").exists()
#
#
# def encode_track_tower(key: str, spec: dict, model):
#     c = TRACK_CACHE / key
#     c.mkdir(parents=True, exist_ok=True)
#     canonical = canonical_track_ids(EMBED_SHARDS)
#     n = len(canonical)
#     md_by_id = {str(r["track_id"]): r for r in pl.read_parquet(TRACK_META).to_dicts()}
#     print(f"  canonical tracks: {n}  metadata rows: {len(md_by_id)}")
#
#     texts, mask = [], np.zeros(n, dtype=bool)
#     for i, tid in enumerate(canonical):
#         r = md_by_id.get(tid)
#         if r is None:
#             texts.append("Unknown track")
#         else:
#             doc = track_metadata_text(r)
#             texts.append(doc if doc else "Unknown track")
#             mask[i] = bool(doc)
#     print(f"  text coverage: {mask.mean():.3f}")
#
#     emb = encode_texts(model, texts, spec["doc_prompt"], spec["doc_max"],
#                        spec["batch_size"], truncation_side="right")
#     emb[~mask] = 0.0
#     np.save(c / "track_ids.npy", np.asarray(canonical, dtype=object))
#     np.save(c / "emb.npy", emb)
#     np.save(c / "mask.npy", mask)
#     print(f"  cached track tower {emb.shape} -> {c}")
#
#
# # ─── NDCG@20 (same definition as 06b_evaluate_ndcg.py) ───────────────────────
# def macro_by_turn_ndcg(q_emb, track_emb, track_mask, meta_rows, id_to_idx, mask_played=True):
#     if track_emb.shape[1] != q_emb.shape[1]:
#         raise ValueError(f"DIM MISMATCH q={q_emb.shape[1]} vs tower={track_emb.shape[1]}")
#     inv_log = 1.0 / np.log2(np.arange(2, NDCG_K + 2))
#     masked_cols = ~track_mask
#     n = len(meta_rows)
#     ndcg = np.zeros(n); turns = np.full(n, -1, np.int64); scorable = np.zeros(n, bool)
#     n_skip = 0
#     for start in range(0, n, SCORE_CHUNK):
#         end = min(start + SCORE_CHUNK, n)
#         scores = q_emb[start:end] @ track_emb.T
#         scores[:, masked_cols] = -np.inf
#         for bi in range(end - start):
#             ri = start + bi
#             r = meta_rows[ri]
#             if r.get("turn_number") is not None:
#                 turns[ri] = int(r["turn_number"])
#             gid = r.get("gt_track_id")
#             if gid is None or gid not in id_to_idx:
#                 n_skip += 1
#                 continue
#             gidx = id_to_idx[gid]
#             row = scores[bi]
#             gt_score = row[gidx]
#             if not np.isfinite(gt_score):
#                 n_skip += 1
#                 continue
#             if mask_played:
#                 for tid in (r.get("prior_track_ids") or []):
#                     j = id_to_idx.get(tid)
#                     if j is not None:
#                         row[j] = -np.inf
#                 row[gidx] = gt_score
#             rank = 1 + int(np.sum(row > gt_score))
#             scorable[ri] = True
#             if rank <= NDCG_K:
#                 ndcg[ri] = inv_log[rank - 1]
#     per_turn = defaultdict(list); vals = []
#     for v, k, ok in zip(ndcg, turns, scorable):
#         if not ok:
#             continue
#         vals.append(v); per_turn[int(k)].append(v)
#     per_turn_mean = {k: float(np.mean(v)) for k, v in sorted(per_turn.items())}
#     per_turn_n    = {k: len(v) for k, v in sorted(per_turn.items())}
#     macro = float(np.mean(list(per_turn_mean.values()))) if per_turn_mean else 0.0
#     micro = float(np.mean(vals)) if vals else 0.0
#     return macro, micro, per_turn_mean, per_turn_n, int(scorable.sum()), n_skip
#
#
# def evaluate(key: str):
#     if not track_tower_exists(key):
#         raise FileNotFoundError(f"No track tower for {key!r}. Run --stage tracks (or all) first.")
#     qc = QUERY_CACHE / key
#     if not (qc / "val.npy").exists():
#         raise FileNotFoundError(f"No val queries for {key!r}. Run --stage queries (or all) first.")
#     q_emb = np.load(qc / "val.npy")
#     meta_rows = pl.read_parquet(qc / "val_meta.parquet").to_dicts()
#     track_emb, track_mask, track_ids = load_track_tower(key)
#     id_to_idx = {tid: i for i, tid in enumerate(track_ids)}
#     print(f"  val queries {q_emb.shape}  tower {track_emb.shape}  mask {track_mask.mean():.3f}")
#
#     macro, micro, ptm, ptn, n_scor, n_skip = macro_by_turn_ndcg(
#         q_emb, track_emb, track_mask, meta_rows, id_to_idx)
#     print(f"\n  [{key}] NDCG@{NDCG_K}  macro-by-turn = {macro:.4f}   micro = {micro:.4f}"
#           f"   (scorable {n_scor}, skipped {n_skip})")
#     for k in sorted(ptm):
#         print(f"    turn {k:>2}: {ptm[k]:.4f}  (n={ptn[k]})")
#
#     report = {"encoder": key, "fold": "val", "metric": f"ndcg@{NDCG_K}",
#               "macro_by_turn": macro, "micro": micro, "n_scorable": n_scor,
#               "n_skipped": n_skip, "per_turn_mean": ptm, "per_turn_n": ptn}
#     out = EVAL_OUT / key / "val"
#     out.mkdir(parents=True, exist_ok=True)
#     (out / "ndcg.json").write_text(json.dumps(report, indent=2))
#     print(f"  wrote {out / 'ndcg.json'}")
#     return report
#
#
# # ─── orchestration ───────────────────────────────────────────────────────────
# def ensure_caches(key: str, spec: dict, want_tracks: bool, want_queries: bool, device):
#     need_tracks  = want_tracks  and not track_tower_exists(key)
#     qc = QUERY_CACHE / key
#     need_val     = want_queries and not (qc / "val.npy").exists()
#     need_blind   = want_queries and not (qc / "blind_a.npy").exists()
#     if not (need_tracks or need_val or need_blind):
#         print("  all requested caches already present — nothing to encode.")
#         return
#
#     model = load_st_model(spec, device)
#     try:
#         if need_tracks:
#             print("\n-- encoding TRACK tower --")
#             encode_track_tower(key, spec, model)
#
#         track_lookup = None
#         if need_val or need_blind:
#             track_lookup = load_track_lookup(TRACK_META)
#
#         if need_val:
#             print("\n-- encoding VAL queries --")
#             rows = build_val_rows(track_lookup)
#             emb = encode_texts(model, [r["query_text"] for r in rows], spec["query_prompt"],
#                                spec["query_max"], spec["batch_size"], truncation_side="left")
#             write_query_cache(key, "val", rows, emb)
#
#         if need_blind:
#             print("\n-- encoding BLIND-A queries --")
#             rows = build_blind_rows(track_lookup)
#             if rows:
#                 emb = encode_texts(model, [r["query_text"] for r in rows], spec["query_prompt"],
#                                    spec["query_max"], spec["batch_size"], truncation_side="left")
#                 write_query_cache(key, "blind_a", rows, emb)
#     finally:
#         free_model(model)
#
#
# def compare(keys: list[str]):
#     rows = []
#     for key in keys:
#         p = EVAL_OUT / key / "val" / "ndcg.json"
#         if not p.exists():
#             print(f"  [warn] no eval for {key} ({p}) — run --stage all first")
#             continue
#         d = json.loads(p.read_text())
#         rows.append({"encoder": key,
#                      f"ndcg@{NDCG_K}_macro_by_turn": d["macro_by_turn"],
#                      f"ndcg@{NDCG_K}_micro": d["micro"],
#                      "n_scorable": d["n_scorable"]})
#     if not rows:
#         print("nothing to compare."); return
#     rows.sort(key=lambda r: -r[f"ndcg@{NDCG_K}_macro_by_turn"])
#     print("\n" + "#" * 74)
#     print(f"  COMPARISON — NDCG@{NDCG_K} on val")
#     print("#" * 74)
#     print(f"  {'encoder':<20}{'macro-by-turn':>16}{'micro':>10}{'n':>10}")
#     for r in rows:
#         print(f"  {r['encoder']:<20}{r[f'ndcg@{NDCG_K}_macro_by_turn']:>16.4f}"
#               f"{r[f'ndcg@{NDCG_K}_micro']:>10.4f}{r['n_scorable']:>10}")
#     EVAL_OUT.mkdir(parents=True, exist_ok=True)
#     out = EVAL_OUT / "ndcg_generic_comparison_val.csv"
#     pl.DataFrame(rows).write_csv(out)
#     print(f"\n  wrote {out}")
#
#
# def main():
#     p = argparse.ArgumentParser(description=__doc__,
#                                 formatter_class=argparse.RawDescriptionHelpFormatter)
#     p.add_argument("--model", choices=list(EMBEDDING_MODELS),
#                    help="Registry key to encode/evaluate.")
#     p.add_argument("--stage", default="all",
#                    choices=["all", "tracks", "queries", "eval"])
#     p.add_argument("--compare", nargs="+", metavar="KEY",
#                    help="Print a table of already-scored models and exit.")
#     args = p.parse_args()
#
#     if args.compare:
#         compare(args.compare); return
#     if not args.model:
#         p.error("give --model KEY (or --compare KEY [KEY ...])")
#
#     spec = EMBEDDING_MODELS[args.model]
#     print("#" * 74)
#     print(f"  MODEL = {args.model}  ({spec['model_id']})   stage = {args.stage}")
#     print("#" * 74)
#     device = _device()
#     print(f"  device: {device}")
#
#     if args.stage in ("tracks", "queries", "all"):
#         ensure_caches(args.model, spec,
#                       want_tracks=args.stage in ("tracks", "all"),
#                       want_queries=args.stage in ("queries", "all"),
#                       device=device)
#     if args.stage in ("eval", "all"):
#         evaluate(args.model)
#
#
# if __name__ == "__main__":
#     main()
"""scripts/08_try_embedding_models.py

Try alternative *bi-encoder* embedding models for the TalkPlay retrieval task,
keeping the exact property your Qwen-native pipeline relies on:

    query and tracks are encoded by the SAME model and L2-normalized,
    so a dot product IS cosine similarity in one shared space.

For each model in the registry this:
  1. builds a TRACK tower  — encodes `track_metadata_text(row)` for every track in
     the canonical organizer order (identical text to your current Qwen pipeline),
     DOC side, no instruction prefix;
  2. encodes the VAL queries (one row per val session at its pinned predict-turn) and
     the BLIND-A queries, QUERY side, with an instruction prefix for instruction-aware
     models (none for symmetric models like bge-m3);
  3. computes macro-by-turn NDCG@20 on VAL — the SAME definition as
     scripts/06b_evaluate_ndcg.py (per-K mean, then unweighted mean across K).

Everything is cached, so eval can be re-run without re-encoding.

Backend = sentence-transformers (already in your deps). ST loads each model's own
pooling (last-token for Qwen3/gte-Qwen2/e5-mistral, CLS for bge-m3, etc.), handles
EOS/padding, and lets us prepend a per-model query instruction. We only manage:
prompts, L2-normalization, max length, and LEFT truncation for queries (so the
[CURRENT USER] tail survives, exactly like the rest of your pipeline).

VRAM: every model below is chosen to fit in 24 GB at fp16/bf16. The 8B model is the
tightest (~16 GB weights); drop its batch size if you OOM.

USAGE
-----
  # one model, full pipeline (tracks -> val+blindA queries -> eval):
  uv run python scripts/08_try_embedding_models.py --model qwen3_4b --stage all

  # only (re)evaluate from existing caches:
  uv run python scripts/08_try_embedding_models.py --model qwen3_4b --stage eval

  # build the side-by-side table once several have been scored:
  uv run python scripts/08_try_embedding_models.py --compare \
      qwen3_0p6b qwen3_4b qwen3_8b gte_qwen2_7b e5_mistral_7b bge_m3

Outputs
  models/track_tower_generic_cache/<key>/{track_ids.npy, emb.npy, mask.npy}
  models/query_emb_generic_cache/<key>/{val,blind_a}.npy + *_meta.parquet
  models/eval_results/<key>/val/ndcg.json
  models/eval_results/ndcg_generic_comparison_val.csv
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emblib.data.parsing import build_query_text_v2
from emblib.data.split import load_blinda_split, turn_pairs_for_fold
from emblib.qwen.qwen_embeddings import track_metadata_text


# ─── paths ───────────────────────────────────────────────────────────────────
DATA          = Path("./data/talkpl-ai")
TRACK_META    = DATA / "TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet"
EMBED_SHARDS  = DATA / "TalkPlayData-Challenge-Track-Embeddings/data"
TRAIN_CONV    = DATA / "TalkPlayData-Challenge-Dataset/data/train-00000-of-00001.parquet"
TEST_CONV     = DATA / "TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet"
BLIND_CONV    = DATA / "TalkPlayData-Challenge-Blind-A/data/test-00000-of-00001.parquet"
SPLITS_PATH   = Path("./models/splits/train_val_test_blinda_matched.parquet")

TRACK_CACHE   = Path("./models/track_tower_generic_cache")
QUERY_CACHE   = Path("./models/query_emb_generic_cache")
EVAL_OUT      = Path("./models/eval_results")

NDCG_K        = 20
SCORE_CHUNK   = 512


# ─── instruction shared by all instruction-aware models (fair comparison) ─────
MUSIC_INSTRUCTION = (
    "Given a music recommendation conversation, retrieve the track from the "
    "catalog that best matches what the listener wants next"
)
def instruct(q_instr: str) -> str:
    return f"Instruct: {q_instr}\nQuery: "


# ─── model registry ──────────────────────────────────────────────────────────
# pooling/EOS are handled by each model's sentence-transformers config; we only
# set the query prompt (None = symmetric / no instruction), the doc prompt
# (None = raw metadata, the retrieval convention), max lengths and batch size.
#
# dim is informational only (we read the true dim from the encoder output).
EMBEDDING_MODELS: dict[str, dict] = {
    # ── Qwen3-Embedding family (last-token + instruction). Drop-in upgrades of
    #    your current 0.6B. 0.6B included so the baseline is produced by THIS
    #    same harness (apples-to-apples).
    "qwen3_0p6b": dict(
        model_id="Qwen/Qwen3-Embedding-0.6B", dim=1024,
        query_prompt=instruct(MUSIC_INSTRUCTION), doc_prompt=None,
        query_max=512, doc_max=256, dtype="float16",
        trust_remote_code=False, batch_size=16,
    ),
    "qwen3_4b": dict(
        model_id="Qwen/Qwen3-Embedding-4B", dim=2560,
        query_prompt=instruct(MUSIC_INSTRUCTION), doc_prompt=None,
        query_max=512, doc_max=256, dtype="float16",
        trust_remote_code=False, batch_size=8,
    ),
    "qwen3_8b": dict(
        model_id="Qwen/Qwen3-Embedding-8B", dim=4096,
        query_prompt=instruct(MUSIC_INSTRUCTION), doc_prompt=None,
        query_max=512, doc_max=256, dtype="float16",
        trust_remote_code=False, batch_size=4,        # ~16 GB weights — tightest
    ),

    # ── gte-Qwen2-7B-instruct (previous-gen SOTA; last-token + instruction).
    "gte_qwen2_7b": dict(
        model_id="Alibaba-NLP/gte-Qwen2-7B-instruct", dim=3584,
        query_prompt=instruct(MUSIC_INSTRUCTION), doc_prompt=None,
        query_max=512, doc_max=256, dtype="float16",
        trust_remote_code=True, batch_size=8,
    ),

    # ── e5-mistral-7b-instruct (Mistral lineage; last-token + instruction).
    "e5_mistral_7b": dict(
        model_id="intfloat/e5-mistral-7b-instruct", dim=4096,
        query_prompt=instruct(MUSIC_INSTRUCTION), doc_prompt=None,
        query_max=512, doc_max=256, dtype="float16",
        trust_remote_code=False, batch_size=8,
    ),

    # ── bge-m3 (XLM-RoBERTa, CLS pooling, NO instruction). Tiny (568M), fast,
    #    architecturally different — a good sanity/diversity baseline.
    "bge_m3": dict(
        model_id="BAAI/bge-m3", dim=1024,
        query_prompt=None, doc_prompt=None,
        query_max=512, doc_max=256, dtype="float16",
        trust_remote_code=False, batch_size=64,
    ),

    # ── OPTIONAL extras (uncomment to try). ───────────────────────────────────
    # stella: strong 1.5B, mean pooling, has its own s2p query prompt.
    # "stella_1p5b": dict(
    #     model_id="NovaSearch/stella_en_1.5B_v5", dim=1024,
    #     query_prompt="Instruct: Retrieve the music track that best satisfies the "
    #                  "listener's request.\nQuery: ",
    #     doc_prompt=None, query_max=512, doc_max=256, dtype="float16",
    #     trust_remote_code=True, batch_size=16,
    # ),
    # NV-Embed-v2: 7B, 4096-d, near-SOTA. Needs trust_remote_code + a recent
    # transformers; ST works but can be version-sensitive.
    # "nvembed_v2": dict(
    #     model_id="nvidia/NV-Embed-v2", dim=4096,
    #     query_prompt=instruct(MUSIC_INSTRUCTION), doc_prompt=None,
    #     query_max=512, doc_max=256, dtype="float16",
    #     trust_remote_code=True, batch_size=4,
    # ),
}


# ─── helpers ─────────────────────────────────────────────────────────────────
def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def canonical_track_ids(shards_dir: Path) -> list[str]:
    shards = sorted(Path(shards_dir).glob("all_tracks-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No all_tracks-*.parquet in {shards_dir}")
    ids: list[str] = []
    for shard in shards:
        ids.extend(str(t) for t in pl.read_parquet(shard, columns=["track_id"])["track_id"].to_list())
    return ids


def load_track_lookup(path: Path) -> dict[str, dict]:
    md = pl.read_parquet(path)
    return {row["track_id"]: row for row in md.to_dicts()}


def load_st_model(spec: dict, device):
    import torch
    from sentence_transformers import SentenceTransformer
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[spec["dtype"]]
    if device.type == "cpu" and dtype != torch.float32:
        print("  [warn] CPU detected — forcing float32 (fp16 matmul unsupported on CPU)")
        dtype = torch.float32
    print(f"  loading {spec['model_id']}  (dtype={dtype}, trust_remote_code={spec['trust_remote_code']})")
    # transformers renamed the from_pretrained dtype kwarg (torch_dtype -> dtype)
    # in 5.x. Try the new name first, fall back to the old one.
    last_err = None
    for dkey in ("dtype", "torch_dtype"):
        try:
            return SentenceTransformer(
                spec["model_id"],
                trust_remote_code=spec["trust_remote_code"],
                model_kwargs={dkey: dtype},
                device=str(device),
            )
        except TypeError as e:
            last_err = e
    raise last_err


def encode_texts(model, texts, prompt, max_len, batch_size, truncation_side):
    model.max_seq_length = max_len
    try:
        model.tokenizer.truncation_side = truncation_side
    except Exception:
        pass
    emb = model.encode(
        texts,
        prompt=prompt,                 # None => no prompt prepended
        batch_size=batch_size,
        normalize_embeddings=True,     # => dot product is cosine
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return np.asarray(emb, dtype=np.float32)


def free_model(model):
    import torch
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─── query-row reconstruction (mirrors 03_encode_queries.parse_session) ───────
def build_query_row(sess_row: dict, K: int, track_lookup: dict, need_gt: bool):
    convs = sess_row.get("conversations") or []
    up    = sess_row.get("user_profile") or {}
    cg    = sess_row.get("conversation_goal") or {}
    sdate = sess_row.get("session_date")
    sdate = str(sdate) if sdate is not None else None

    by_turn = defaultdict(list)
    for t in convs:
        by_turn[t["turn_number"]].append(t)

    umsgs = [t for t in by_turn.get(K, []) if t.get("role") == "user"]
    if not umsgs:
        return None

    gt = None
    if need_gt:
        mmsgs = [t for t in by_turn.get(K, []) if t.get("role") == "music"]
        gt = mmsgs[0].get("content") if mmsgs else None
        if gt is None:
            return None

    history, prior = [], []
    for t in sorted(by_turn):
        if t >= K:
            break
        for prev in by_turn[t]:
            history.append({
                "role": prev.get("role"),
                "content": prev.get("content") or "",
                "thought": prev.get("thought") or "",
            })
            if prev.get("role") == "music" and prev.get("content"):
                prior.append(prev["content"])

    text = build_query_text_v2(
        chat_history=history, user_query=umsgs[0]["content"],
        user_profile=up, conversation_goal=cg, session_date=sdate,
        track_lookup=track_lookup, use_thoughts=True,
    )
    return {
        "session_id": sess_row["session_id"], "turn_number": int(K),
        "gt_track_id": gt, "category": cg.get("category") or "",
        "specificity": cg.get("specificity") or "",
        "prior_track_ids": prior, "query_text": text,
    }


def build_val_rows(track_lookup: dict) -> list[dict]:
    if not SPLITS_PATH.exists():
        raise FileNotFoundError(f"{SPLITS_PATH} not found. Run scripts/01b_rebuild_split.py first.")
    split_df = load_blinda_split(SPLITS_PATH)
    pairs = turn_pairs_for_fold(split_df, "val")            # {(session_id, predict_turn)}

    # val sessions can come from organizer train OR test (01b merges both)
    sessions: dict[str, dict] = {}
    for path in (TRAIN_CONV, TEST_CONV):
        if path.exists():
            for r in pl.read_parquet(path).to_dicts():
                sessions[str(r["session_id"])] = r

    rows, miss = [], 0
    for sid, K in sorted(pairs):
        s = sessions.get(str(sid))
        if s is None:
            miss += 1
            continue
        row = build_query_row(s, int(K), track_lookup, need_gt=True)
        if row is not None:
            rows.append(row)
    print(f"  val rows built: {len(rows)}  (pairs={len(pairs)}, sessions missing={miss})")
    return rows


def build_blind_rows(track_lookup: dict) -> list[dict]:
    if not BLIND_CONV.exists():
        print(f"  [skip] blind conv not found at {BLIND_CONV}")
        return []
    rows = []
    for s in pl.read_parquet(BLIND_CONV).to_dicts():
        by_turn = defaultdict(list)
        for t in (s.get("conversations") or []):
            by_turn[t["turn_number"]].append(t)
        # ONE ROW PER TURN that has a user message — matches the legacy
        # 03_encode_queries.parse_session layout (all blind prediction turns,
        # keyed by (session_id, turn_number) for downstream lookup). Emitting
        # only the last turn per session is what produced the truncated
        # 80-row cache; per-turn gives the full ~290 rows.
        for tn in sorted(by_turn):
            row = build_query_row(s, int(tn), track_lookup, need_gt=False)
            if row is not None:
                rows.append(row)
    print(f"  blind_a rows built: {len(rows)} (per-turn)")
    return rows


def write_query_cache(key: str, fold: str, rows: list[dict], emb: np.ndarray):
    out = QUERY_CACHE / key
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{fold}.npy", emb)
    pl.DataFrame({
        "session_id":      [r["session_id"] for r in rows],
        "turn_number":     [r["turn_number"] for r in rows],
        "gt_track_id":     [r["gt_track_id"] for r in rows],
        "category":        [r["category"] for r in rows],
        "specificity":     [r["specificity"] for r in rows],
        "prior_track_ids": [r["prior_track_ids"] for r in rows],
        # query_text is persisted so the second-stage reranker (script 09) has
        # the real user text to score against, not just an id.
        "query_text":      [r["query_text"] for r in rows],
    }, schema={
        "session_id": pl.String, "turn_number": pl.Int64, "gt_track_id": pl.String,
        "category": pl.String, "specificity": pl.String,
        "prior_track_ids": pl.List(pl.String), "query_text": pl.String,
    }).write_parquet(out / f"{fold}_meta.parquet")
    print(f"  wrote {emb.shape} -> {out / (fold + '.npy')}")


# ─── track tower ─────────────────────────────────────────────────────────────
def load_track_tower(key: str):
    c = TRACK_CACHE / key
    return (np.load(c / "emb.npy"), np.load(c / "mask.npy"),
            [str(t) for t in np.load(c / "track_ids.npy", allow_pickle=True).tolist()])


def track_tower_exists(key: str) -> bool:
    c = TRACK_CACHE / key
    return (c / "emb.npy").exists() and (c / "mask.npy").exists() and (c / "track_ids.npy").exists()


def encode_track_tower(key: str, spec: dict, model):
    c = TRACK_CACHE / key
    c.mkdir(parents=True, exist_ok=True)
    canonical = canonical_track_ids(EMBED_SHARDS)
    n = len(canonical)
    md_by_id = {str(r["track_id"]): r for r in pl.read_parquet(TRACK_META).to_dicts()}
    print(f"  canonical tracks: {n}  metadata rows: {len(md_by_id)}")

    texts, mask = [], np.zeros(n, dtype=bool)
    for i, tid in enumerate(canonical):
        r = md_by_id.get(tid)
        if r is None:
            texts.append("Unknown track")
        else:
            doc = track_metadata_text(r)
            texts.append(doc if doc else "Unknown track")
            mask[i] = bool(doc)
    print(f"  text coverage: {mask.mean():.3f}")

    emb = encode_texts(model, texts, spec["doc_prompt"], spec["doc_max"],
                       spec["batch_size"], truncation_side="right")
    emb[~mask] = 0.0
    np.save(c / "track_ids.npy", np.asarray(canonical, dtype=object))
    np.save(c / "emb.npy", emb)
    np.save(c / "mask.npy", mask)
    print(f"  cached track tower {emb.shape} -> {c}")


# ─── NDCG@20 (same definition as 06b_evaluate_ndcg.py) ───────────────────────
def macro_by_turn_ndcg(q_emb, track_emb, track_mask, meta_rows, id_to_idx, mask_played=True):
    if track_emb.shape[1] != q_emb.shape[1]:
        raise ValueError(f"DIM MISMATCH q={q_emb.shape[1]} vs tower={track_emb.shape[1]}")
    inv_log = 1.0 / np.log2(np.arange(2, NDCG_K + 2))
    masked_cols = ~track_mask
    n = len(meta_rows)
    ndcg = np.zeros(n); turns = np.full(n, -1, np.int64); scorable = np.zeros(n, bool)
    n_skip = 0
    for start in range(0, n, SCORE_CHUNK):
        end = min(start + SCORE_CHUNK, n)
        scores = q_emb[start:end] @ track_emb.T
        scores[:, masked_cols] = -np.inf
        for bi in range(end - start):
            ri = start + bi
            r = meta_rows[ri]
            if r.get("turn_number") is not None:
                turns[ri] = int(r["turn_number"])
            gid = r.get("gt_track_id")
            if gid is None or gid not in id_to_idx:
                n_skip += 1
                continue
            gidx = id_to_idx[gid]
            row = scores[bi]
            gt_score = row[gidx]
            if not np.isfinite(gt_score):
                n_skip += 1
                continue
            if mask_played:
                for tid in (r.get("prior_track_ids") or []):
                    j = id_to_idx.get(tid)
                    if j is not None:
                        row[j] = -np.inf
                row[gidx] = gt_score
            rank = 1 + int(np.sum(row > gt_score))
            scorable[ri] = True
            if rank <= NDCG_K:
                ndcg[ri] = inv_log[rank - 1]
    per_turn = defaultdict(list); vals = []
    for v, k, ok in zip(ndcg, turns, scorable):
        if not ok:
            continue
        vals.append(v); per_turn[int(k)].append(v)
    per_turn_mean = {k: float(np.mean(v)) for k, v in sorted(per_turn.items())}
    per_turn_n    = {k: len(v) for k, v in sorted(per_turn.items())}
    macro = float(np.mean(list(per_turn_mean.values()))) if per_turn_mean else 0.0
    micro = float(np.mean(vals)) if vals else 0.0
    return macro, micro, per_turn_mean, per_turn_n, int(scorable.sum()), n_skip


def evaluate(key: str):
    if not track_tower_exists(key):
        raise FileNotFoundError(f"No track tower for {key!r}. Run --stage tracks (or all) first.")
    qc = QUERY_CACHE / key
    if not (qc / "val.npy").exists():
        raise FileNotFoundError(f"No val queries for {key!r}. Run --stage queries (or all) first.")
    q_emb = np.load(qc / "val.npy")
    meta_rows = pl.read_parquet(qc / "val_meta.parquet").to_dicts()
    track_emb, track_mask, track_ids = load_track_tower(key)
    id_to_idx = {tid: i for i, tid in enumerate(track_ids)}
    print(f"  val queries {q_emb.shape}  tower {track_emb.shape}  mask {track_mask.mean():.3f}")

    macro, micro, ptm, ptn, n_scor, n_skip = macro_by_turn_ndcg(
        q_emb, track_emb, track_mask, meta_rows, id_to_idx)
    print(f"\n  [{key}] NDCG@{NDCG_K}  macro-by-turn = {macro:.4f}   micro = {micro:.4f}"
          f"   (scorable {n_scor}, skipped {n_skip})")
    for k in sorted(ptm):
        print(f"    turn {k:>2}: {ptm[k]:.4f}  (n={ptn[k]})")

    report = {"encoder": key, "fold": "val", "metric": f"ndcg@{NDCG_K}",
              "macro_by_turn": macro, "micro": micro, "n_scorable": n_scor,
              "n_skipped": n_skip, "per_turn_mean": ptm, "per_turn_n": ptn}
    out = EVAL_OUT / key / "val"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ndcg.json").write_text(json.dumps(report, indent=2))
    print(f"  wrote {out / 'ndcg.json'}")
    return report


# ─── orchestration ───────────────────────────────────────────────────────────
def ensure_caches(key: str, spec: dict, want_tracks: bool, want_queries: bool, device):
    need_tracks  = want_tracks  and not track_tower_exists(key)
    qc = QUERY_CACHE / key
    need_val     = want_queries and not (qc / "val.npy").exists()
    need_blind   = want_queries and not (qc / "blind_a.npy").exists()
    if not (need_tracks or need_val or need_blind):
        print("  all requested caches already present — nothing to encode.")
        return

    model = load_st_model(spec, device)
    try:
        if need_tracks:
            print("\n-- encoding TRACK tower --")
            encode_track_tower(key, spec, model)

        track_lookup = None
        if need_val or need_blind:
            track_lookup = load_track_lookup(TRACK_META)

        if need_val:
            print("\n-- encoding VAL queries --")
            rows = build_val_rows(track_lookup)
            emb = encode_texts(model, [r["query_text"] for r in rows], spec["query_prompt"],
                               spec["query_max"], spec["batch_size"], truncation_side="left")
            write_query_cache(key, "val", rows, emb)

        if need_blind:
            print("\n-- encoding BLIND-A queries --")
            rows = build_blind_rows(track_lookup)
            if rows:
                emb = encode_texts(model, [r["query_text"] for r in rows], spec["query_prompt"],
                                   spec["query_max"], spec["batch_size"], truncation_side="left")
                write_query_cache(key, "blind_a", rows, emb)
    finally:
        free_model(model)


def compare(keys: list[str]):
    rows = []
    for key in keys:
        p = EVAL_OUT / key / "val" / "ndcg.json"
        if not p.exists():
            print(f"  [warn] no eval for {key} ({p}) — run --stage all first")
            continue
        d = json.loads(p.read_text())
        rows.append({"encoder": key,
                     f"ndcg@{NDCG_K}_macro_by_turn": d["macro_by_turn"],
                     f"ndcg@{NDCG_K}_micro": d["micro"],
                     "n_scorable": d["n_scorable"]})
    if not rows:
        print("nothing to compare."); return
    rows.sort(key=lambda r: -r[f"ndcg@{NDCG_K}_macro_by_turn"])
    print("\n" + "#" * 74)
    print(f"  COMPARISON — NDCG@{NDCG_K} on val")
    print("#" * 74)
    print(f"  {'encoder':<20}{'macro-by-turn':>16}{'micro':>10}{'n':>10}")
    for r in rows:
        print(f"  {r['encoder']:<20}{r[f'ndcg@{NDCG_K}_macro_by_turn']:>16.4f}"
              f"{r[f'ndcg@{NDCG_K}_micro']:>10.4f}{r['n_scorable']:>10}")
    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    out = EVAL_OUT / "ndcg_generic_comparison_val.csv"
    pl.DataFrame(rows).write_csv(out)
    print(f"\n  wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=list(EMBEDDING_MODELS),
                   help="Registry key to encode/evaluate.")
    p.add_argument("--stage", default="all",
                   choices=["all", "tracks", "queries", "eval"])
    p.add_argument("--compare", nargs="+", metavar="KEY",
                   help="Print a table of already-scored models and exit.")
    args = p.parse_args()

    if args.compare:
        compare(args.compare); return
    if not args.model:
        p.error("give --model KEY (or --compare KEY [KEY ...])")

    spec = EMBEDDING_MODELS[args.model]
    print("#" * 74)
    print(f"  MODEL = {args.model}  ({spec['model_id']})   stage = {args.stage}")
    print("#" * 74)
    device = _device()
    print(f"  device: {device}")

    if args.stage in ("tracks", "queries", "all"):
        ensure_caches(args.model, spec,
                      want_tracks=args.stage in ("tracks", "all"),
                      want_queries=args.stage in ("queries", "all"),
                      device=device)
    if args.stage in ("eval", "all"):
        evaluate(args.model)


if __name__ == "__main__":
    main()