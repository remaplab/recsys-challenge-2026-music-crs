"""
gambling_updated.py -- heuristic + Qwen-in-every-fallback, from PRECOMPUTED embeddings.
(v3: validates the Level-2 per-bucket query INSTRUCTION as well as the query text.)

Logically identical to the reference launcher. The Qwen TRACK and QUERY embeddings are
loaded from caches built by scripts/12_encode_gambling_caches.py. Every other step uses
the exact same emblib.retrieval functions, so scoring, future-release filtering,
popularity padding, and the fallback fusion are byte-for-byte the same.

SELF-DIAGNOSING CACHE CHECK (extended)
======================================
The query cache must match the runtime on TWO axes now:
  1. query_text       -- byte-for-byte, rebuilt from emblib.retrieval.core (as before);
  2. instruction      -- the Level-2 per-bucket instruction the cache was ENCODED with
                         (stored by scripts/12 in query_meta.parquet) must equal the one
                         the CURRENT prompts config (emblib.retrieval.query_instructions
                         + --instruction-prompts) resolves for that row. The instruction
                         is not part of query_text, so without this column an
                         instruction change would silently pass validation.
A cache written before v3 has no 'instruction' column: it is accepted ONLY if the
current config is the legacy fixed instruction for every row; otherwise re-encode.

RUN
===
    uv run python scripts/12_encode_gambling_caches.py --stages blind --models qwen3_0p6b
    uv run python scripts/launchers/gambling_updated.py --model 0.6

    # with the dense 8th term + goal-direction, and a provenance sidecar:
    uv run python scripts/launchers/gambling_updated.py --model 0.6 \
        --w-dense 1.5 --goal-direction-json models/eval_results/diagnosis/goal_direction.json \
        --output-path exp/inference/blind_a/gambling/sub_06_dense_goaldir.json \
        --dump-provenance auto
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import emblib.retrieval.core as _core
from emblib.retrieval.core import (
    build_train_popularity,
    build_variant_query,
    dense_score_rows,
    load_blind_task_rows,
    load_neural_track_metadata,
    make_tasks_from_queries,
    metadata_text,
    neural_metadata_doc,
    rank_rows,
    release_ordinals,
)
from emblib.retrieval.gambling import TrackIndex
from emblib.retrieval.heuristic_qwen_fallback import generate_heuristic_with_qwen_fallback
from emblib.retrieval.query_instructions import (
    LEGACY_INSTRUCTION,
    describe as describe_prompts,
    load_prompts,
    resolve_instructions,
)
from emblib.retrieval.paths import (
    BLIND_A_PATH,
    BLIND_A_INFERENCE_DIR,
    QWEN_TRACK_CACHE_DIR,
    TRACK_METADATA_PATH,
    TRAIN_PATH,
)

_YEAR_PROBE_ROW = {
    "track_name": "Take Me",
    "artist_name": "ONE OK ROCK",
    "album_name": "35xxxv (Deluxe Edition)",
    "tag_list": ["addictive", "one ok rock", "rock"],
    "release_date": "2015-02-11",
}


def runtime_core_emits_year() -> bool:
    return "Year:" in neural_metadata_doc(_YEAR_PROBE_ROW)


TOWERS_ROOT = QWEN_TRACK_CACHE_DIR.parent.parent
TRACK_SUBDIR = QWEN_TRACK_CACHE_DIR.name
QUERY_SUBDIR = "dense_blinda_query_len512_poollast"
MODEL_FOLDER = {
    "0.6": "Qwen__Qwen3-Embedding-0.6B",
    "4": "Qwen__Qwen3-Embedding-4B",
    "8": "Qwen__Qwen3-Embedding-8B",
}
QWEN_QUERY_CACHE_DIR = QWEN_TRACK_CACHE_DIR.parent / QUERY_SUBDIR
DEFAULT_OUTPUT_PATH = BLIND_A_INFERENCE_DIR / "gambling/qwen_fallback_tracks.json"


def model_track_cache_dir(model: str) -> Path:
    return TOWERS_ROOT / MODEL_FOLDER[model] / TRACK_SUBDIR


def model_query_cache_dir(model: str) -> Path:
    return TOWERS_ROOT / MODEL_FOLDER[model] / QUERY_SUBDIR


def _rrf_fuse_rows(rowsA, rowsB, *, rrf_k: int = 60, depth: int = 2000,
                   w_primary: float = 0.5, w_fuse: float = 0.5):
    """RRF-fuse two per-row dense score collections into one. Accepts a list of
    1-D arrays (as dense_score_rows returns) OR a 2-D array; returns a LIST of
    1-D float32 arrays so downstream consumers behave like the non-fused path.

    Per-model weighted RRF (scale-free, rank-only):
      fused[r][t] = w_primary/(rrf_k+rank_A) + w_fuse/(rrf_k+rank_B)
    for t in each model's top-`depth`."""
    import numpy as _np
    n_rows = len(rowsA)
    out = []
    for r in range(n_rows):
        a = _np.asarray(rowsA[r], dtype=_np.float32)
        b = _np.asarray(rowsB[r], dtype=_np.float32)
        n_tracks = a.shape[0]
        d = min(depth, n_tracks)
        fused = _np.zeros(n_tracks, dtype=_np.float32)
        rr = (1.0 / (rrf_k + _np.arange(d))).astype(_np.float32)
        for sc, wt in ((a, w_primary), (b, w_fuse)):
            if wt == 0.0:
                continue
            top = _np.argpartition(-sc, d - 1)[:d]
            order = top[_np.argsort(-sc[top])]
            fused[order] += wt * rr
        out.append(fused)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="heuristic + Qwen-fallback track predictions from precomputed embeddings."
    )
    parser.add_argument("--model", choices=["0.6", "4", "8"], default="0.6")
    parser.add_argument("--fuse-key", choices=["0.6", "4", "8"], default=None,
                        help="RRF-fuse the dense backbone with this 2nd model (e.g. 0.6 "
                             "when --model 8). Raises retrieval recall (scripts/46). Needs "
                             "the fuse model's track + blind-query caches.")
    parser.add_argument("--fuse-query-cache-dir", type=Path, default=None,
                        help="override the fuse model's blind query cache dir.")
    parser.add_argument("--fuse-rrf-k", type=int, default=10,
                        help="RRF constant. val sweep best=10 (k=60 degraded recall@20).")
    parser.add_argument("--fuse-depth", type=int, default=200,
                        help="RRF rank depth per model. val sweep: flat above 200.")
    parser.add_argument("--fuse-w-primary", type=float, default=0.4,
                        help="RRF weight on the PRIMARY model (--model). val sweep best=0.4.")
    parser.add_argument("--fuse-w-fuse", type=float, default=0.6,
                        help="RRF weight on the FUSE model (--fuse-key). val sweep best=0.6 "
                             "(0.6B beats 8B at the top, so lean toward it).")
    parser.add_argument("--instruction-prompts", default="default",
                        help="Level-2 per-bucket prompts JSON used to VALIDATE the query "
                             "cache's instruction column. 'default' = emblib/retrieval/"
                             "instruction_prompts.json (legacy if absent); 'none' = legacy. "
                             "MUST be the same config scripts/12 encoded with.")
    parser.add_argument("--fuse-dirs", type=Path, nargs="+", default=None,
                        help="RRF-fuse the dense backbone with these explicit FT cache dirs "
                             "(each holding dense_tracks_len256_poollast + "
                             "dense_blinda_query_len512_poollast). Fuses the PRIMARY model "
                             "(--qwen-track-cache-dir/--query-cache-dir) with all of them.")
    parser.add_argument("--fuse-dirs-weights", type=float, nargs="+", default=None,
                        help="per-model RRF weights: primary first, then one per --fuse-dirs.")
    parser.add_argument("--blind-path", type=Path, default=BLIND_A_PATH)
    parser.add_argument("--train-path", type=Path, default=TRAIN_PATH)
    parser.add_argument("--track-metadata-path", type=Path, default=TRACK_METADATA_PATH)
    parser.add_argument("--qwen-track-cache-dir", type=Path, default=None)
    parser.add_argument("--query-cache-dir", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=20)

    from emblib.retrieval.rerankers import RERANKERS as _RR
    parser.add_argument("--reranker", choices=["none", *sorted(_RR)], default="none")
    parser.add_argument("--candidate-n", type=int, default=64)
    parser.add_argument("--rerank-doc", choices=["neural", "metadata"], default="neural")
    parser.add_argument("--q-max-tokens", type=int, default=224)
    parser.add_argument("--rerank-batch-size", type=int, default=None)
    parser.add_argument("--rerank-max-length", type=int, default=None)
    parser.add_argument("--rerank-attn", default=None, choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--rerank-device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--rerank-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--rerank-allow-downloads", dest="rerank_local_files_only",
                        action="store_false", default=True)

    parser.add_argument("--reranked-recs-path", type=Path, default=None)
    parser.add_argument("--qwen-fill-k", type=int, default=None)
    parser.add_argument("--qwen-pool-k", type=int, default=0,
                        help="add top-K dense Qwen query->track candidates to the heuristic pool (0=off)")
    parser.add_argument("--no-decade", dest="use_decade", action="store_false", default=True,
                        help="drop the decade-neighbourhood source from the heuristic pool")
    parser.add_argument("--dump-pools", type=Path, default=None,
                        help="write per-row pool composition to this JSON")
    parser.add_argument("--w-dense", type=float, default=0.0,
                        help="weight on the min-max 8B query->candidate cosine 8th term "
                             "(0 = off = exact 0.59 baseline). Recommended 1.5.")
    parser.add_argument("--goal-direction-json", type=Path, default=None,
                        help="config from scripts/42_mine_goal_direction.py: shrinks the "
                             "prior terms when the query moves away from the priors.")
    parser.add_argument("--w-audio", type=float, default=0.0,
                        help="weight on the prior-mean LAION-CLAP audio cosine 9th term "
                             "(0 = off = identical to current). Centered at 0.42 like v2_plus.")
    parser.add_argument("--audio-center", type=float, default=0.42,
                        help="center subtracted from the audio cosine (v2_plus SIM_CENTERS p_audio).")
    parser.add_argument("--w-dense-anchor", type=float, default=0.0,
                        help="weight on the prior-mean QWEN-8B DENSE cosine continuity term: "
                             "w * (cos(mean_endorsed_prior_dense, candidate) - center). "
                             "Mirrors --w-audio but in the Qwen text-embedding space (the "
                             "missing prior-mean DENSE continuity; the query is already the "
                             "dense backbone). 0 = off. Tune on val; experiments suggest ~2.5.")
    parser.add_argument("--dense-anchor-center", type=float, default=0.0,
                        help="center subtracted from the dense-anchor cosine (tune on val; "
                             "0.0 = no centering, like image).")
    parser.add_argument("--w-anchor-whitened", type=float, default=0.0,
                        help="weight on the WHITENED dense-anchor term: cosine of the "
                             "pool-mean-removed prior-mean direction against pool-mean-"
                             "removed candidates. 0=off. Guess ~1.5; tune on val. "
                             "Pairs with --w-dense-anchor.")
    parser.add_argument("--anchor-whitened-center", type=float, default=0.0,
                        help="center subtracted from the whitened-anchor cosine (default 0).")
    parser.add_argument("--w-pop", type=float, default=0.0,
                        help="weight on the split-clean train POPULARITY prior, applied in "
                             "the no-prior (COLD/turn-1) fallback reranker only. 0=off. "
                             "Requires --fallback-content-rerank.")
    parser.add_argument("--w-cooc", type=float, default=0.0,
                        help="weight on the split-clean train CO-OCCURRENCE term in the "
                             "heuristic path (history rows). 0=off.")
    parser.add_argument("--dense-anchor-recency", action="store_true",
                        help="use a RECENCY-WEIGHTED prior mean for the dense-anchor term "
                             "(recent priors weighted exp(rank/tau) heavier). Validated to "
                             "beat the flat mean, gain grows with turn. Off = flat mean.")
    parser.add_argument("--recency-tau", type=float, default=2.0,
                        help="recency half-life-ish constant for --dense-anchor-recency "
                             "(larger = flatter). Validated ~2.0.")
    parser.add_argument("--default-weights-json", type=Path, default=None,
                        help="JSON file overriding DEFAULT_WEIGHTS (album_last/artist_last/"
                             "album_any/artist_any/year/pop_match/pop_z). Lets the optimizer "
                             "sweep these without editing source. Missing keys fall back to "
                             "the built-in DEFAULT_WEIGHTS.")
    parser.add_argument("--candidates-parquet", type=Path, default=None,
                        help="external candidate file (session_id,turn_number,track_id,rank,kind). "
                             "Takes top --candidates-topn by rank at each session's PREDICT turn "
                             "(kind=blind_b), adds at most --candidates-max NEW tracks (not already "
                             "in the heuristic pool) to the candidate set, re-ranked by qwen dense.")
    parser.add_argument("--candidates-submission", type=Path, nargs="+", default=None,
                        help="one or more SUBMISSION-format JSON files (list of {session_id, "
                             "turn_number, predicted_track_ids}) to use as an ADDITIONAL candidate "
                             "generator. The predicted_track_ids per session are injected into the "
                             "pool (top --candidates-topn each) and re-ranked by the heuristic, just "
                             "like --candidates-parquet. Use this to feed an XGBoost pipeline's "
                             "already-extracted top-20 as a small, high-precision candidate source. "
                             "Combines additively with --candidates-parquet (union, deduped).")
    parser.add_argument("--candidates-topn", type=int, default=200,
                        help="take top-N by rank (rank 1=best) per session from the candidates file.")
    parser.add_argument("--candidates-max", type=int, default=50,
                        help="add AT MOST this many NEW candidates per session (after excluding "
                             "tracks already extracted by the heuristic).")
    parser.add_argument("--candidates-kind", type=str, default="blind_b",
                        help="filter candidates file to this kind.")
    parser.add_argument("--last-prior-pool-k", type=int, default=0,
                        help="last-prior GENERATOR: union top-K dense-NN to the LAST prior "
                             "track. Uses the dense_anchor (qwen8b track) tower.")
    parser.add_argument("--bb2-pool-k", type=int, default=0,
                        help="second-backbone GENERATOR: union top-K from a SECOND embedding "
                             "backbone (e.g. qwen3-4b). Needs --bb2-query-cache-dir and "
                             "--bb2-track-cache-dir.")
    parser.add_argument("--bb2-query-cache-dir", type=Path, default=None,
                        help="second-backbone blindb query cache dir (query_embeddings.npy + "
                             "query_meta.parquet), e.g. GEN__qwen3-4b/dense_blindb_all_...")
    parser.add_argument("--bb2-track-cache-dir", type=Path, default=None,
                        help="second-backbone track tower dir (embeddings.npy + track_ids.npy).")
    parser.add_argument("--anchor-pool-k", type=int, default=0,
                        help="anchor-pool GENERATOR: union the top-K recency-weighted "
                             "prior-mean nearest-neighbor tracks (dense-anchor space) into "
                             "the candidate pool (0=off). Converts the dense-anchor from "
                             "rerank-only into REACH. History rows only; cold rows no-op.")
    parser.add_argument("--audio-pool-k", type=int, default=0,
                        help="audio-pool GENERATOR: union the top-K recency-weighted "
                             "prior-mean nearest-neighbor tracks in the CLAP AUDIO space "
                             "into the candidate pool (0=off). History rows only.")
    parser.add_argument("--heuristic-scale", type=float, default=1.0,
                        help="global multiplier on ALL 7 heuristic base weights "
                             "(album/artist/year/pop). <1 leans on embeddings, >1 on "
                             "metadata. Validated ~0.1 (base is sparse over the pool).")
    parser.add_argument("--cat-spec-gamma", type=float, default=1.0,
                        help="strength of the cat/spec multipliers via boost**gamma. "
                             "0=stratification OFF, 1=shipped, >1=amplified. Validated 0.")
    parser.add_argument("--audio-tower-cache", type=Path,
                        default=Path("./models/track_tower_cache"),
                        help="dir holding audio-laion_clap.npy [+ __mask.npy] + track_ids.npy")
    parser.add_argument("--w-image", type=float, default=0.0,
                        help="weight on the prior-mean SigLIP2 album-art cosine term "
                             "(0 = off). Centered at 0.0 like v2_plus p_image. Track-only "
                             "space (query<->track not meaningful), prior-mean NN only.")
    parser.add_argument("--image-center", type=float, default=0.0,
                        help="center subtracted from the image cosine (v2_plus SIM_CENTERS p_image).")
    parser.add_argument("--image-tower-cache", type=Path,
                        default=Path("./models/track_tower_cache"),
                        help="dir holding image-siglip2.npy [+ __mask.npy] + track_ids.npy")
    parser.add_argument("--w-lyr", type=float, default=0.0,
                        help="weight on the lyrics query->track term (0.6B space). 0 = off. "
                             "Uses the 0.6B blind query cache (lyrics tower is 0.6B, not 8B).")
    parser.add_argument("--w-attr", type=float, default=0.0,
                        help="weight on the attributes query->track term (0.6B space). 0 = off.")
    parser.add_argument("--lyr-pool-k", type=int, default=0,
                        help="lyrics RETRIEVAL ARM: union the top-K full-catalog 0.6B "
                             "lyrics query->track tracks into the candidate pool (0=off). "
                             "Targets lyrical answers dense can't reach (B/G recall holes).")
    parser.add_argument("--attr-pool-k", type=int, default=0,
                        help="attributes RETRIEVAL ARM: same as --lyr-pool-k for the "
                             "attributes tower (F/A/K/D recall holes).")
    parser.add_argument("--lyr-attr-tower-cache", type=Path,
                        default=Path("./models/track_tower_cache"),
                        help="dir with lyrics-qwen3_embedding_0.6b.npy + "
                             "attributes-qwen3_embedding_0.6b.npy [+ __mask.npy] + track_ids.npy")
    parser.add_argument("--use-progress", action="store_true",
                        help="use goal_progress_assessments: partition priors into "
                             "endorsed (MOVES/None) vs rejected (DOES_NOT). Identity sets, "
                             "prior-means, and the album_last/artist_last anchor are built "
                             "from ENDORSED priors only. Off = old all-priors behavior.")
    parser.add_argument("--w-reject", type=float, default=0.0,
                        help="strength of the rejected-prior REPULSION term: subtract "
                             "w_reject * cos(mean_rejected_modality, candidate) for audio/image. "
                             "0 = off. Implies the endorsed/rejected partition (auto-enables "
                             "the progress join). Fires only on rows with >=1 rejected prior.")
    parser.add_argument("--release-date-weight", type=float, default=1.0,
                        help="re-ranking weight applied to pool tracks that have NO release_date ")
    parser.add_argument("--w-rel-artist", type=float, default=0.0,
                        help="relation scorer: bonus for candidates sharing ARTIST with a prior "
                             "track. Discriminates among dateless tracks. 0 = off. Additive, "
                             "scaled to the per-row score spread.")
    parser.add_argument("--w-rel-registrant", type=float, default=0.0,
                        help="relation scorer: bonus for candidates sharing ISRC REGISTRANT "
                             "(label, first 5 chars) with a prior track. 0 = off.")
    parser.add_argument("--w-rel-raretag", type=float, default=0.0,
                        help="relation scorer: bonus PER shared RARE tag (tag on <= --rel-tag-df "
                             "catalog tracks) with a prior track. 0 = off.")
    parser.add_argument("--rel-tag-df", type=int, default=1000,
                        help="a tag is 'rare' if it appears on <= this many catalog tracks "
                             "(default 1000, the strongest-lift band).")
    parser.add_argument("--measured-config", action="store_true",
                        help="apply the val-ablation verdict (scripts/47): skip lyr/attr "
                             "AND image in the HEURISTIC path (they hurt/are dead there) "
                             "while keeping them in the FALLBACK rerank (where they help). "
                             "audio+dense stay everywhere. Use with --fallback-content-rerank "
                             "and --fuse-key 0.6.")
    parser.add_argument("--fallback-content-rerank", action="store_true",
                        help="re-rank the no-prior Qwen fallback recs by a BLEND of Qwen's "
                             "order (backbone) + query->track content sims (8B dense, 0.6B "
                             "lyrics/attributes), spec/category-gated. Off = verbatim Qwen "
                             "order. Targets the pure-dense lookup rows that bypass the "
                             "heuristic content terms.")
    parser.add_argument("--fallback-depth", type=int, default=None,
                        help="depth of the no-prior / empty-pool fallback candidate list that "
                             "--fallback-content-rerank reorders. Default = --top-k (verbatim "
                             "top-20, old behavior). Set ~200 so dense-rank 21-200 answers can "
                             "be pulled into the top-20 by the content rerank. Capped at 200 by "
                             "_rerank_fallback_by_content's n_consider.")
    parser.add_argument("--fallback-backbone-weight", type=float, default=1.0,
                        help="scalar on the Qwen rank backbone in the no-prior/empty-pool "
                             "content rerank. 1.0 = current (Qwen order dominates). <1.0 lets "
                             "the content terms (dense/lyr/attr) decide, so a deep-but-correct "
                             "dense-rank-21-200 answer can climb into the top-20. Try 0.2.")
    parser.add_argument("--spec-reweight", action="store_true",
                        help="specificity-driven prior-vs-dense reweighting (Axis 2). "
                             "LH/HH (specific-target lookups) lean dense+modality and "
                             "shrink prior continuity; HL/LL stay prior-led. Off = baseline. "
                             "Profiles in SPEC_REWEIGHT.")
    parser.add_argument("--anchor-last-endorsed", action="store_true",
                        help="anchor album_last/artist_last on the last ENDORSED prior "
                             "instead of the literal last track. Separate knob from "
                             "--use-progress so its effect can be isolated. Default off "
                             "(literal-last = freshest context).")
    parser.add_argument("--q06-query-cache-dir", type=Path, default=None,
                        help="0.6B blind query cache dir (query_embeddings.npy + query_meta.parquet) "
                             "used for the lyrics/attr query vectors. Default: the 0.6B model's "
                             "dense_blinda_query_len512_poollast.")
    parser.add_argument("--qt-gate-floor", type=float, default=1.0,
                        help="category-affinity gate for the lyrics/attributes query->track "
                             "terms. 1.0 = flat (fire on every category). 0.0 = hard gate "
                             "(lyrics only on B/G/D, attributes only on F/A/K/E). Interpolates. "
                             "See LYR_AFFINITY / ATTR_AFFINITY.")
    parser.add_argument("--dump-provenance", type=Path, default=None,
                        help="write a per-slot provenance parquet (source, score_total, the "
                             "7 prior-term contributions, dense_contrib, sim01, cosine, md). "
                             "Pass 'auto' to write <output-path>.provenance.parquet")
    parser.add_argument("--restrict-to-catalog", type=Path, default=None,
                        help="allow-list parquet of track_ids: predictions are restricted "
                             "to tracks present in this file (e.g. test_tracks-...parquet). "
                             "Forbidden tracks are removed BEFORE scoring (dense set to -inf, "
                             "and dropped from the heuristic pool). Omit = no restriction "
                             "(use the full all_tracks catalog).")
    parser.add_argument("--restrict-id-column", default="track_id",
                        help="column name holding the track ids in --restrict-to-catalog.")

    args = parser.parse_args()
    if args.qwen_track_cache_dir is None:
        args.qwen_track_cache_dir = model_track_cache_dir(args.model)
    if args.query_cache_dir is None:
        args.query_cache_dir = model_query_cache_dir(args.model)
    if args.qwen_fill_k is None:
        args.qwen_fill_k = args.top_k
    if args.fallback_depth is None:
        args.fallback_depth = args.top_k
    if args.output_path is None:
        key_tag = MODEL_FOLDER[args.model].split("__")[-1]
        stem = f"{DEFAULT_OUTPUT_PATH.stem}_{key_tag}"
        if args.reranked_recs_path is not None:
            stem += "_reranked"
        elif args.reranker != "none":
            stem += f"_{args.reranker}"
        args.output_path = DEFAULT_OUTPUT_PATH.with_name(f"{stem}{DEFAULT_OUTPUT_PATH.suffix}")
    return args


def key(task: dict[str, Any]) -> tuple[str, str, int]:
    return (str(task["session_id"]), str(task["user_id"]), int(task["turn_number"]))


def _mtime(path: Path) -> str:
    try:
        return time.ctime(path.stat().st_mtime)
    except OSError:
        return "<missing>"


# --- cached-embedding loaders --------------------------------------------------
def load_cached_track_embeddings(cache_dir: Path, track_ids: list[str]) -> np.ndarray:
    ids_path = cache_dir / "track_ids.npy"
    embeddings_path = cache_dir / "embeddings.npy"
    if not ids_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError(
            f"missing track embedding cache in {cache_dir}. "
            f"Run scripts/12_encode_gambling_caches.py first."
        )
    cached_ids = [str(track_id) for track_id in np.load(ids_path, allow_pickle=True).tolist()]
    if cached_ids != track_ids:
        raise ValueError(
            f"cached track IDs mismatch in {cache_dir}; rerun scripts/12_encode_gambling_caches.py."
        )
    return np.load(embeddings_path, mmap_mode="r")


def load_audio_tower(cache_dir: Path, track_ids: list[str], mod_name: str = "audio-laion_clap"):
    """Load a per-track audio embedding tower (LAION-CLAP) aligned to track_ids,
    L2-normalized, with a validity mask. Mirrors heuristic_v2_plus EmbIndex:
    layout <cache_dir>/<mod>.npy [+ <mod>__mask.npy], track_ids.npy for alignment.
    Returns (emb[n_tracks, d] float32, mask[n_tracks] bool) or (None, None) if absent."""
    cache_dir = Path(cache_dir)
    ep = cache_dir / f"{mod_name}.npy"
    if not ep.exists():
        print(f"  audio tower      : {ep} not found -- audio term OFF")
        return None, None
    ids_path = cache_dir / "track_ids.npy"
    tower_ids = ([str(t) for t in np.load(ids_path, allow_pickle=True).tolist()]
                 if ids_path.exists() else None)
    emb = np.load(ep).astype(np.float32)
    mp = cache_dir / f"{mod_name}__mask.npy"
    mask = np.load(mp).astype(bool) if mp.exists() else np.ones(len(emb), bool)
    # L2-normalize if not already unit-norm
    samp = emb[mask][:200]
    if len(samp) and abs(float(np.linalg.norm(samp, axis=1).mean()) - 1.0) > 0.01:
        nn = np.linalg.norm(emb, axis=1, keepdims=True); nn[nn < 1e-12] = 1.0
        emb = emb / nn
    # align to track_ids if the tower has its own order
    if tower_ids is not None and tower_ids != track_ids:
        pos = {tid: i for i, tid in enumerate(tower_ids)}
        ne = np.zeros((len(track_ids), emb.shape[1]), np.float32)
        nm = np.zeros(len(track_ids), bool)
        for ix, tid in enumerate(track_ids):
            p = pos.get(tid)
            if p is not None and mask[p]:
                ne[ix] = emb[p]; nm[ix] = True
        emb, mask = ne, nm
    elif tower_ids is None and len(emb) != len(track_ids):
        raise ValueError(
            f"audio tower has {len(emb)} rows but {len(track_ids)} tracks and no "
            f"track_ids.npy to align on in {cache_dir}")
    emb = emb.copy(); emb[~mask] = 0.0
    print(f"  audio tower      : {ep.name} dim={emb.shape[1]} cov={mask.mean():.3f}")
    return emb, mask


def load_dense_anchor_tower(cache_dir: Path, track_ids: list[str]):
    """Load the Qwen DENSE track tower (dense_tracks_len256_poollast/embeddings.npy)
    for the prior-mean dense-anchor continuity term, L2-normalized, full coverage.
    Reuses load_cached_track_embeddings (knows the embeddings.npy layout + id check),
    then unit-normalizes rows so cos(mean_prior_dense, candidate) is a true cosine.
    Returns (emb[n_tracks, d] float32, mask[n_tracks] bool all-True)."""
    da = np.asarray(load_cached_track_embeddings(cache_dir, track_ids), dtype=np.float32)
    nn = np.linalg.norm(da, axis=1, keepdims=True); nn[nn < 1e-12] = 1.0
    emb = (da / nn).astype(np.float32)
    mask = np.ones(emb.shape[0], dtype=bool)
    print(f"  dense-anchor tower: qwen dense dim={emb.shape[1]} cov=1.000  ({cache_dir})")
    return emb, mask


def attach_prefix_progress(task_rows, blind_path) -> None:
    """Attach task['prefix_progress'], a list aligned 1:1 with task['prefix_track_ids'],
    each entry 'MOVES' / 'DOES_NOT' / None, derived from the blind parquet's
    goal_progress_assessments column. Mapping: a music track at turn K carries the
    assessment whose turn_number == K (confirmed against the data: turn-1 music is
    always null/None; rejections are 'DOES_NOT_MOVE_TOWARD_GOAL')."""
    import polars as _pl
    df = _pl.read_parquet(blind_path)
    # per-session: turn_number -> label, and track_id -> turn_number (music turns)
    sess_turn_label: dict[str, dict[int, str]] = {}
    sess_track_turn: dict[str, dict[str, int]] = {}
    for r in df.iter_rows(named=True):
        sid = str(r["session_id"])
        labels = {}
        for a in (r.get("goal_progress_assessments") or []):
            v = a.get("goal_progress_assessment")
            tn = a.get("turn_number")
            if tn is None:
                continue
            if v == "MOVES_TOWARD_GOAL":
                labels[int(tn)] = "MOVES"
            elif v == "DOES_NOT_MOVE_TOWARD_GOAL":
                labels[int(tn)] = "DOES_NOT"
            else:
                labels[int(tn)] = None
        sess_turn_label[sid] = labels
        tt = {}
        for t in (r.get("conversations") or []):
            if t.get("role") == "music" and t.get("content"):
                tt[str(t["content"])] = int(t.get("turn_number", -1))
        sess_track_turn[sid] = tt

    n_aligned = n_missing = 0
    for task in task_rows:
        sid = str(task["session_id"])
        track_turn = sess_track_turn.get(sid, {})
        labels = sess_turn_label.get(sid, {})
        prog = []
        for tid in task.get("prefix_track_ids", []):
            tn = track_turn.get(str(tid))
            lab = labels.get(tn) if tn is not None else None
            prog.append(lab)
            if lab is not None:
                n_aligned += 1
            elif tn is None:
                n_missing += 1
        task["prefix_progress"] = prog
    if n_missing:
        print(f"  NOTE: {n_missing} prior tracks had no turn match for progress "
              f"(treated as endorsed/None).")


def load_q06_query_by_key(cache_dir: Path, tasks) -> dict:
    """Load the 0.6B blind query cache and return {(sid,uid,turn): vec(0.6B)} aligned
    to `tasks` by key. Asserts every task has a 0.6B query (same rows as 8B cache)."""
    emb_path = cache_dir / "query_embeddings.npy"
    meta_path = cache_dir / "query_meta.parquet"
    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"missing 0.6B query cache in {cache_dir} (needed for --w-lyr/--w-attr). "
            f"Encode it: scripts/12_encode_gambling_caches.py --stages blind --models qwen3_0p6b "
            f"--instruction-prompts none")
    emb = np.load(emb_path).astype(np.float32)
    # L2-normalize so cos(q, track) is a true cosine (towers are unit-norm)
    nn = np.linalg.norm(emb, axis=1, keepdims=True); nn[nn < 1e-12] = 1.0
    emb = emb / nn
    meta = pl.read_parquet(meta_path).to_dicts()
    by_key = {(str(r["session_id"]), str(r["user_id"]), int(r["turn_number"])): emb[i]
              for i, r in enumerate(meta)}
    out = {}
    missing = 0
    for t in tasks:
        k = (str(t["session_id"]), str(t["user_id"]), int(t["turn_number"]))
        v = by_key.get(k)
        if v is None:
            missing += 1
        else:
            out[k] = v
    if missing:
        raise ValueError(
            f"0.6B query cache {cache_dir} is missing {missing}/{len(tasks)} blind rows; "
            f"the 0.6B and 8B blind caches must cover the SAME rows. Re-encode blind for 0.6B.")
    print(f"  0.6B query cache : {cache_dir.name} dim={emb.shape[1]} rows={len(out)}")
    return out


def load_cached_query_embeddings(
    cache_dir: Path,
    tasks: list[dict[str, Any]],
    query_texts: list[str],
    expected_instructions: list[str] | None = None,
) -> np.ndarray:
    """Load query_embeddings.npy and assert it aligns with the tasks / query_texts /
    per-row INSTRUCTIONS rebuilt here, so the cached vectors are exactly what
    encode_qwen_texts would produce for these queries under the current config."""
    emb_path = cache_dir / "query_embeddings.npy"
    meta_path = cache_dir / "query_meta.parquet"
    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"missing query embedding cache in {cache_dir}. "
            f"Run scripts/12_encode_gambling_caches.py first."
        )
    emb = np.load(emb_path)
    meta = pl.read_parquet(meta_path).to_dicts()
    has_instr_col = bool(meta) and ("instruction" in meta[0])
    if expected_instructions is not None and not has_instr_col:
        if any(ins != LEGACY_INSTRUCTION for ins in expected_instructions):
            raise ValueError(
                f"query cache {cache_dir} predates per-row instructions (no 'instruction' "
                f"column) but the current prompts config assigns per-bucket instructions. "
                f"Re-encode: scripts/12_encode_gambling_caches.py --stages blind "
                f"--models {_model_key_for_dir(cache_dir)}"
            )
    # KEY-BASED selection: the cache may be a SUPERSET of the tasks (e.g. a Blind-B
    # all-turns cache, from which we select the final-turn row per session). Build a
    # key -> cache-row-index map, then assemble outputs positionally aligned to `tasks`.
    cache_index = {}
    for ri, rec in enumerate(meta):
        cache_index[(str(rec["session_id"]), str(rec["user_id"]), int(rec["turn_number"]))] = ri
    sel = []
    missing_keys = []
    for task in tasks:
        ri = cache_index.get(key(task))
        if ri is None:
            missing_keys.append(key(task))
        sel.append(ri)
    if missing_keys:
        ex = missing_keys[:3]
        raise ValueError(
            f"query cache {cache_dir} is missing {len(missing_keys)}/{len(tasks)} task "
            f"keys (final-turn rows). examples: {ex}. The cache must contain a row for "
            f"each (session_id, user_id, turn_number) the launcher predicts. If this is a "
            f"Blind-B all-turns cache, confirm the final predict turn per session is present "
            f"and that user_id normalization matches (null -> '')."
        )
    emb = emb[np.asarray(sel, dtype=np.int64)]
    meta = [meta[ri] for ri in sel]
    # from here meta is positionally aligned to tasks (one final-turn row each)
    for i, (task, rec) in enumerate(zip(tasks, meta)):
        if key(task) != (str(rec["session_id"]), str(rec["user_id"]), int(rec["turn_number"])):
            raise ValueError(
                f"query cache row {i} key mismatch after selection "
                f"(cache={(rec['session_id'], rec['user_id'], rec['turn_number'])}, "
                f"runtime={key(task)})."
            )
        if expected_instructions is not None and has_instr_col:
            if str(rec.get("instruction") or "") != expected_instructions[i]:
                raise ValueError(
                    f"query cache row {i} INSTRUCTION differs from the current prompts config.\n"
                    f"  cache  : {str(rec.get('instruction'))[:140]!r}\n"
                    f"  runtime: {expected_instructions[i][:140]!r}\n"
                    f"  query cache: {meta_path} (mtime {_mtime(meta_path)})\n"
                    f"  WHAT THIS MEANS: the cache was ENCODED under a different per-bucket\n"
                    f"  prompts JSON than the one this run resolves (--instruction-prompts).\n"
                    f"  FIX: re-encode with the current config:\n"
                    f"    uv run python scripts/12_encode_gambling_caches.py --stages blind "
                    f"--models {_model_key_for_dir(cache_dir)}\n"
                    f"  or pass the SAME --instruction-prompts the cache was built with."
                )
        if query_texts[i] != rec["query_text"]:
            cached = rec["query_text"] or ""
            runtime = query_texts[i] or ""
            j = next((k for k in range(min(len(cached), len(runtime))) if cached[k] != runtime[k]),
                     min(len(cached), len(runtime)))
            lo = max(0, j - 40)
            cache_year = "Year:" in cached
            runtime_year = "Year:" in runtime
            core_year = runtime_core_emits_year()
            raise ValueError(
                f"query cache row {i} text differs from what gambling_updated rebuilds.\n"
                f"  First difference at char {j} (len cache={len(cached)}, runtime={len(runtime)}):\n"
                f"    cache  : ...{cached[lo:j+40]!r}\n"
                f"    runtime: ...{runtime[lo:j+40]!r}\n"
                f"\n"
                f"  SELF-DIAGNOSIS (same process):\n"
                f"    runtime core.py           : {_core.__file__}\n"
                f"    runtime core emits Year:  : {core_year}\n"
                f"    this query has Year (cache/runtime): {cache_year}/{runtime_year}\n"
                f"    query cache file          : {meta_path}\n"
                f"    query cache mtime         : {_mtime(meta_path)}\n"
                f"\n"
                f"  FIX:\n"
                f"    1) find . -type d -name __pycache__ -exec rm -rf {{}} + 2>/dev/null\n"
                f"    2) rm -f {emb_path} {meta_path}\n"
                f"    3) uv run python scripts/12_encode_gambling_caches.py --stages blind "
                f"--models {_model_key_for_dir(cache_dir)} && \\\n"
                f"       uv run python scripts/launchers/gambling_updated.py --model "
                f"{_model_flag_for_dir(cache_dir)}"
            )
    return emb


def _model_key_for_dir(cache_dir: Path) -> str:
    folder = cache_dir.parent.name
    return {
        "Qwen__Qwen3-Embedding-0.6B": "qwen3_0p6b",
        "Qwen__Qwen3-Embedding-4B": "qwen3_4b",
        "Qwen__Qwen3-Embedding-8B": "qwen3_8b",
    }.get(folder, "qwen3_0p6b")


def _model_flag_for_dir(cache_dir: Path) -> str:
    folder = cache_dir.parent.name
    return {
        "Qwen__Qwen3-Embedding-0.6B": "0.6",
        "Qwen__Qwen3-Embedding-4B": "4",
        "Qwen__Qwen3-Embedding-8B": "8",
    }.get(folder, "0.6")


# --- optional second-stage reranking ------------------------------------------
def _resolve_rerank_device(device_arg: str):
    import torch
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _rerank_doc_map(args: argparse.Namespace, track_text_by_id: dict[str, str]) -> dict[str, str]:
    if args.rerank_doc == "neural":
        return track_text_by_id
    rows = pl.read_parquet(args.track_metadata_path).to_dicts()
    return {str(r["track_id"]): metadata_text(r) for r in rows}


def _dense_candidates(score_rows, tasks, track_index, release_ord, candidate_n,
                      filter_future_releases):
    cand_by_row: list[np.ndarray] = []
    for i, task in enumerate(tasks):
        dense = np.asarray(score_rows[i], dtype=np.float32).copy()
        for tid in task["seen_tracks"]:
            j = track_index.get(tid)
            if j is not None:
                dense[j] = -np.inf
        if filter_future_releases and task["session_date"] is not None:
            dense[release_ord > task["session_date"].toordinal()] = -np.inf
        finite = int(np.isfinite(dense).sum())
        if finite == 0:
            cand_by_row.append(np.empty(0, dtype=np.int64))
            continue
        n = min(candidate_n, finite)
        cand = np.argpartition(-dense, n - 1)[:n]
        cand = cand[np.argsort(-dense[cand])]
        cand_by_row.append(cand.astype(np.int64))
    return cand_by_row


def rerank_dense_score_rows(*, score_rows, tasks, track_ids, track_index, release_ord,
                            candidate_n, doc_by_id, scorer, q_max_tokens,
                            filter_future_releases):
    n_tracks = len(track_ids)
    cand_by_row = _dense_candidates(
        score_rows, tasks, track_index, release_ord, candidate_n, filter_future_releases,
    )
    qtrunc = [scorer.truncate_query(task["query"] or "", q_max_tokens) for task in tasks]
    order_by_row: list[np.ndarray | None] = [None] * len(tasks)

    if scorer.listwise:
        for i, cand in enumerate(cand_by_row):
            if len(cand) == 0:
                continue
            docs = [doc_by_id.get(track_ids[int(j)], track_ids[int(j)]) for j in cand]
            s = np.asarray(scorer.score(qtrunc[i], docs), dtype=np.float32)
            order_by_row[i] = cand[np.argsort(-s)]
    else:
        pairs: list[tuple[str, str]] = []
        spans: list[tuple[int, int]] = []
        for i, cand in enumerate(cand_by_row):
            start = len(pairs)
            for j in cand:
                pairs.append((qtrunc[i], doc_by_id.get(track_ids[int(j)], track_ids[int(j)])))
            spans.append((start, len(pairs)))
        flat = scorer.score_pairs(pairs)
        for i, cand in enumerate(cand_by_row):
            if len(cand) == 0:
                continue
            a, b = spans[i]
            order_by_row[i] = cand[np.argsort(-flat[a:b])]

    out: list[np.ndarray] = []
    for i in range(len(tasks)):
        new = np.full(n_tracks, -np.inf, dtype=np.float32)
        order = order_by_row[i]
        if order is not None:
            for rank_pos, j in enumerate(order):
                new[int(j)] = float(len(order) - rank_pos)
        out.append(new)
    return out


def build_all_qwen_recs_cached(args: argparse.Namespace, task_rows: list[dict[str, Any]],
                               prompts_cfg) -> dict[tuple[str, str, int], list[str]]:
    track_ids, track_docs, track_text_by_id, release_dates, _ = load_neural_track_metadata(args.track_metadata_path)
    track_embeddings = load_cached_track_embeddings(args.qwen_track_cache_dir, track_ids)
    _, popular_items = build_train_popularity(args.train_path, track_ids)
    # IMPORTANT: the cached query embeddings were encoded WITHOUT conversation_goal in the
    # query text. We recently populated conversation_goal (category/specificity) in the parquet
    # so the heuristic can apply CAT_MULT/SPEC_MULT -- but that must NOT change the encoded query
    # text, or it won't match the cache. So build the query text from a goal-STRIPPED copy of the
    # row. The original task_rows keep conversation_goal intact for the heuristic's boost path.
    def _row_no_goal(_r):
        if _r.get("conversation_goal") is None:
            return _r
        _c = dict(_r); _c["conversation_goal"] = None
        return _c
    queries_and_seen = [
        build_variant_query(_row_no_goal(task_row["row"]), "default", track_text_by_id, include_thoughts=False)
        for task_row in task_rows
    ]
    tasks, query_texts = make_tasks_from_queries(task_rows, queries_and_seen)
    expected_instructions = resolve_instructions(task_rows, prompts_cfg)
    query_embeddings = load_cached_query_embeddings(
        args.query_cache_dir, tasks, query_texts,
        expected_instructions=expected_instructions,
    )
    track_index = {track_id: idx for idx, track_id in enumerate(track_ids)}
    release_ord = release_ordinals(track_ids, release_dates)

    dense_rows  = dense_score_rows(query_embeddings, track_embeddings, batch_size=64)

    # --- RRF fusion with N explicit FT cache dirs (multi-model ensemble) ---
    if getattr(args, "fuse_dirs", None):
        import numpy as _np
        dirs = args.fuse_dirs
        ws = args.fuse_dirs_weights or [1.0] * (1 + len(dirs))
        if len(ws) != 1 + len(dirs):
            raise SystemExit(f"--fuse-dirs-weights needs {1 + len(dirs)} values "
                             f"(primary + {len(dirs)} dirs), got {len(ws)}")
        print(f"  RRF-fusing PRIMARY + {len(dirs)} dirs (rrf_k={args.fuse_rrf_k}, "
              f"depth={args.fuse_depth}, weights={ws})")
        # collect per-model dense_rows, all aligned to the SAME tasks/track order
        all_rows = [dense_rows]
        for d in dirs:
            td = d / "dense_tracks_len256_poollast"
            qd = d / "dense_blinda_query_len512_poollast"
            t_emb = load_cached_track_embeddings(td, track_ids)
            # align this dir's query cache to the current tasks by key (no instruction check:
            # these are FT caches encoded with the legacy instruction, same as primary)
            q_emb = load_cached_query_embeddings(qd, tasks, query_texts,
                                                 expected_instructions=None)
            all_rows.append(dense_score_rows(q_emb, t_emb, batch_size=64))
        # weighted N-way RRF, reusing the pairwise helper iteratively is wrong;
        # do it directly here:
        n_rows = len(dense_rows);
        n_tracks = len(track_ids)
        rrf_k = args.fuse_rrf_k;
        depth = min(args.fuse_depth, n_tracks)
        fused_out = []
        rr = (1.0 / (rrf_k + _np.arange(depth))).astype(_np.float32)
        for r in range(n_rows):
            fused = _np.zeros(n_tracks, dtype=_np.float32)
            for mi, rows_m in enumerate(all_rows):
                if ws[mi] == 0.0:
                    continue
                sc = _np.asarray(rows_m[r], dtype=_np.float32)
                top = _np.argpartition(-sc, depth - 1)[:depth]
                order = top[_np.argsort(-sc[top])]
                fused[order] += ws[mi] * rr
            fused_out.append(fused)
        dense_rows = fused_out
        print(f"  fused dense backbone ready ({len(dense_rows)} rows, {len(all_rows)} models)")
    # --- optional RRF fusion with a second model (e.g. 0.6B) for the dense backbone ---
    # The fused per-row scores replace the 8B scores everywhere downstream (heuristic
    # dense term AND fallback rerank backbone), raising retrieval recall (scripts/46).
    if getattr(args, "fuse_key", None):
        print(f"  RRF-fusing dense backbone with {args.fuse_key} (rrf_k={args.fuse_rrf_k}, "
              f"depth={args.fuse_depth})")
        fuse_track_dir = model_track_cache_dir(args.fuse_key)
        fuse_query_dir = args.fuse_query_cache_dir or model_query_cache_dir(args.fuse_key)
        f_track_emb = load_cached_track_embeddings(fuse_track_dir, track_ids)
        # the fuse model's query cache is keyed to the SAME blind rows
        f_expected = resolve_instructions(task_rows, prompts_cfg)
        f_query_emb = load_cached_query_embeddings(
            fuse_query_dir, tasks, query_texts, expected_instructions=f_expected)
        f_dense_rows = dense_score_rows(f_query_emb, f_track_emb, batch_size=64)
        dense_rows = _rrf_fuse_rows(dense_rows, f_dense_rows,
                                    rrf_k=args.fuse_rrf_k, depth=args.fuse_depth,
                                    w_primary=args.fuse_w_primary,
                                    w_fuse=args.fuse_w_fuse)
        print(f"  fused dense backbone ready ({len(dense_rows)} rows)")

    # raw dense query->track scores per row, for pool widening (pre-rerank)
    # qwen_scores_by_key = {key(t): dense_rows[i] for i, t in enumerate(tasks)}
    # score_rows = dense_rows

    if getattr(args, "restrict_to_catalog", None) is not None:
        allowed_mask = load_allowed_mask(
            args.restrict_to_catalog, track_ids, args.restrict_id_column)
        blocked = ~allowed_mask
        for i in range(len(dense_rows)):
            dr = np.asarray(dense_rows[i], dtype=np.float32).copy()
            dr[blocked] = -np.inf
            dense_rows[i] = dr
        # expose the mask so the heuristic path can drop blocked pool candidates too
        args._allowed_mask = allowed_mask
    else:
        args._allowed_mask = None

        # raw dense query->track scores per row, for pool widening (pre-rerank)
    qwen_scores_by_key = {key(t): dense_rows[i] for i, t in enumerate(tasks)}
    score_rows = dense_rows

    if args.reranker != "none":
        from emblib.retrieval.rerankers import build_scorer
        print(f"  reranking dense candidates with {args.reranker} "
              f"(candidate_n={args.candidate_n}, doc={args.rerank_doc}, q_max_tokens={args.q_max_tokens})")
        scorer = build_scorer(
            args.reranker,
            device=_resolve_rerank_device(args.rerank_device),
            batch_size=args.rerank_batch_size,
            max_length=args.rerank_max_length,
            attn=args.rerank_attn,
            dtype=args.rerank_dtype,
            local_files_only=args.rerank_local_files_only,
        )
        print(f"  reranker device: {scorer.device if hasattr(scorer, 'device') else 'n/a'}  listwise={scorer.listwise}")
        score_rows = rerank_dense_score_rows(
            score_rows=score_rows,
            tasks=tasks,
            track_ids=track_ids,
            track_index=track_index,
            release_ord=release_ord,
            candidate_n=args.candidate_n,
            doc_by_id=_rerank_doc_map(args, track_text_by_id),
            scorer=scorer,
            q_max_tokens=args.q_max_tokens,
            filter_future_releases=True,
        )

    recs_by_key = rank_rows(
        score_rows=score_rows,
        tasks=tasks,
        track_ids=track_ids,
        track_index=track_index,
        release_ord=release_ord,
        popular_items=popular_items,
        release_dates=release_dates,
        top_k=max(args.top_k, args.fallback_depth),
        filter_future_releases=True,
    )
    return recs_by_key, qwen_scores_by_key


def build_tracks_only_rows(task_rows, recs_by_key, *, top_k):
    rows: list[dict[str, Any]] = []
    for task in task_rows:
        task_key = key(task)
        recs = recs_by_key.get(task_key)
        if recs is None:
            raise ValueError(f"missing recommendations for {task_key}")
        if len(recs) != top_k:
            raise ValueError(f"wrong recommendation length for {task_key}: {len(recs)}")
        rows.append(
            {
                "session_id": task_key[0],
                "user_id": task_key[1],
                "turn_number": task_key[2],
                "predicted_track_ids": recs,
                "predicted_response": "",
            }
        )
    return rows


def load_reranked_qwen_recs(path, task_rows, *, model, fill_k):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    meta: dict[str, Any] = data.get("meta", {}) if isinstance(data, dict) else {}
    rows = data["rows"] if isinstance(data, dict) and "rows" in data else data

    cache_model = str(meta.get("model", ""))
    if cache_model and cache_model != str(model):
        raise ValueError(
            f"reranked recs cache at {path} was built for --model {cache_model!r}, but you passed "
            f"--model {model!r}."
        )

    by_key: dict[tuple[str, str, int], list[str]] = {}
    for rec in rows:
        kk = (str(rec["session_id"]), str(rec["user_id"]), int(rec["turn_number"]))
        by_key[kk] = [str(t) for t in rec["qwen_reranked_track_ids"]]

    recs_by_key: dict[tuple[str, str, int], list[str]] = {}
    for task in task_rows:
        kk = key(task)
        ids = by_key.get(kk)
        if ids is None:
            raise ValueError(f"reranked recs cache {path} is missing turn {kk}.")
        if len(ids) < fill_k:
            raise ValueError(
                f"reranked recs for {kk} hold only {len(ids)} ids but --qwen-fill-k={fill_k}."
            )
        recs_by_key[kk] = ids[:fill_k]
    return recs_by_key, meta


def load_allowed_mask(catalog_path: Path, track_ids: list[str], id_column: str) -> np.ndarray:
    """Boolean mask over `track_ids` (the full 47K catalog order): True where the
    track is present in `catalog_path` (the allow-list, e.g. test_tracks parquet).
    Predictions are later restricted to True entries. Raises if the column or file
    is missing, or if the allow-list and catalog share zero ids (a sign of a wrong
    column / wrong file)."""
    if not catalog_path.exists():
        raise FileNotFoundError(f"--restrict-to-catalog not found: {catalog_path}")
    df = pl.read_parquet(catalog_path)
    if id_column not in df.columns:
        raise ValueError(
            f"--restrict-id-column '{id_column}' not in {catalog_path} "
            f"(columns: {df.columns}).")
    allowed_ids = {str(t) for t in df[id_column].to_list()}
    mask = np.fromiter((tid in allowed_ids for tid in track_ids),
                       dtype=bool, count=len(track_ids))
    n_allowed = int(mask.sum())
    if n_allowed == 0:
        raise ValueError(
            f"allow-list {catalog_path} (col '{id_column}') matched 0 of "
            f"{len(track_ids)} catalog tracks -- wrong file or column?")
    print(f"  catalog restriction : {catalog_path.name} "
          f"-> {n_allowed}/{len(track_ids)} tracks allowed "
          f"({100.0*n_allowed/len(track_ids):.1f}%); {len(track_ids)-n_allowed} blocked")
    return mask


def build_modal_arm_pools(q06_by_key, emb, mask, k, chunk=512):
    """Full-catalog top-k retrieval per key: 0.6B query vec vs a modal tower.
    Returns {key: np.ndarray of catalog idxs}. emb/q06 are L2-normalized so dot=cosine."""
    if emb is None or k <= 0 or q06_by_key is None:
        return None
    keys = list(q06_by_key.keys())
    Q = np.stack([q06_by_key[kk] for kk in keys]).astype(np.float32)
    invalid = ~mask
    out = {}
    for s in range(0, len(keys), chunk):
        e = min(s + chunk, len(keys))
        sc = Q[s:e] @ emb.T
        sc[:, invalid] = -np.inf
        for bi in range(e - s):
            row = sc[bi]
            kk = min(int(k), int(np.isfinite(row).sum()))
            if kk <= 0:
                out[keys[s + bi]] = np.empty(0, dtype=np.int64)
                continue
            top = np.argpartition(-row, kk - 1)[:kk]
            out[keys[s + bi]] = top[np.argsort(-row[top])].astype(np.int64)
    return out


def main() -> None:
    args = parse_args()
    prompts_cfg = load_prompts(args.instruction_prompts)

    print("gambling_updated: scoring from precomputed embeddings")
    print(f"  emblib.retrieval.core   : {_core.__file__}")
    print(f"  runtime core Year    : {runtime_core_emits_year()}")
    print(f"  instruction prompts  : {describe_prompts(prompts_cfg)}")
    print(f"  track cache dir      : {args.qwen_track_cache_dir}")

    print(f"  query cache dir      : {args.query_cache_dir}  (mtime {_mtime(args.query_cache_dir / 'query_meta.parquet')})")
    if args.reranked_recs_path is not None:
        print(f"  qwen source          : precomputed reranked recs ({args.reranked_recs_path}), fill_k={args.qwen_fill_k}")
    elif args.reranker != "none":
        print(f"  qwen source          : dense + live {args.reranker} rerank (candidate_n={args.candidate_n}, doc={args.rerank_doc})")
    else:
        print("  qwen source          : dense Qwen (no rerank)")

    task_rows = load_blind_task_rows(args.blind_path)

    # --- Blind-B null-field graceful handling ---------------------------------
    # Blind-B has null conversation_goal (no category/specificity -> CAT_MULT/SPEC_MULT
    # already degrade to 1.0 via the "?" default in the heuristic), null
    # goal_progress_assessments (the progress loop is then empty), null session_date
    # (future-release filter is already guarded by `is not None`), and SOMETIMES null
    # user_id. We normalize user_id to a stable sentinel so the query-cache keys
    # (built by scripts/12 with the same normalization) line up exactly, and we make
    # sure session_date is None (not a bad string) so the guarded filter stays off.
    _nb_null_uid = 0
    for _t in task_rows:
        _uid = _t.get("user_id")
        if _uid is None or (isinstance(_uid, str) and _uid.strip() == "") or str(_uid) == "None":
            _t["user_id"] = ""               # stable sentinel, matches cache
            _nb_null_uid += 1
        _row = _t.get("row") or {}
        _ru = _row.get("user_id")
        if _ru is None or str(_ru) == "None":
            _row["user_id"] = ""
        # ensure goal is a dict so heuristic's row.get("conversation_goal") or {} is safe
        if _row.get("conversation_goal") is None:
            _row["conversation_goal"] = {}
        _t["row"] = _row
        # null session_date -> keep as None (guarded filter stays off, no crash)
        if _t.get("session_date") is not None and not hasattr(_t["session_date"], "toordinal"):
            _t["session_date"] = None
    if _nb_null_uid:
        print(f"  Blind-B: normalized {_nb_null_uid}/{len(task_rows)} null user_id -> '' sentinel")

    # --- attach progress labels (goal_progress_assessments) to each task's priors ---
    # For each session, music track at turn K is labelled by the assessment whose
    # turn_number == K. We align labels to task["prefix_track_ids"] by turn order.
    if args.use_progress or args.w_reject != 0.0:
        attach_prefix_progress(task_rows, args.blind_path)
        n_with = sum(1 for t in task_rows
                     if any(l == "DOES_NOT" for l in (t.get("prefix_progress") or [])))
        print(f"  progress signal     : use_progress={args.use_progress} "
              f"w_reject={args.w_reject}  ({n_with} task rows have >=1 rejected prior)")

    gd_cfg = None
    if args.goal_direction_json is not None:
        gd_cfg = json.loads(args.goal_direction_json.read_text())
        print(f"  goal-direction      : {args.goal_direction_json}  "
              f"(top_n={gd_cfg.get('top_n')}, t_lo={gd_cfg.get('t_lo')}, "
              f"t_hi={gd_cfg.get('t_hi')}, floor={gd_cfg.get('floor')}, "
              f"val Δ accepted={gd_cfg.get('accepted')})")
    w_dense = args.w_dense if args.w_dense > 0 else (float(gd_cfg.get("w_dense", 0.0)) if gd_cfg else 0.0)
    if (w_dense > 0 or gd_cfg is not None) and args.reranked_recs_path is not None:
        print("  NOTE: --w-dense / --goal-direction need the dense score rows, which the "
              "reranked-recs path does not produce; both are OFF on this path.")

    # resolve provenance sidecar path
    dump_prov = args.dump_provenance
    if dump_prov is not None and str(dump_prov) == "auto":
        dump_prov = args.output_path.with_suffix(".provenance.parquet")
    if dump_prov is not None:
        print(f"  provenance sidecar   : {dump_prov}")

    qwen_scores_by_key = None
    if args.reranked_recs_path is not None:
        qwen_recs_by_key, rmeta = load_reranked_qwen_recs(
            args.reranked_recs_path, task_rows, model=args.model, fill_k=args.qwen_fill_k,
        )
        print(f"loaded {len(qwen_recs_by_key)} precomputed reranked turns "
              f"(top-{args.qwen_fill_k} each); no model/embeddings loaded")
        if rmeta:
            print(f"reranked cache meta: {rmeta}")
        if args.qwen_pool_k > 0:
            print("  NOTE: --qwen-pool-k needs the dense score rows, which the "
                  "reranked-recs path does not produce; pool widening is OFF.")
    else:
        print("scoring all-Qwen fallback recommendations from cached embeddings")
        qwen_recs_by_key, qwen_scores_by_key = build_all_qwen_recs_cached(
            args, task_rows, prompts_cfg)

    print("applying heuristic with Qwen in every fallback slot")
    track_index = TrackIndex(args.track_metadata_path)

    
    _no_rd_mask = None
    if args.release_date_weight != 1.0:
        import numpy as _np_rd
        _rd_rows = pl.read_parquet(args.track_metadata_path).to_dicts()
        _rd_by_id = {}
        for _r in _rd_rows:
            _v = _r.get("release_date")
            _empty = (_v is None) or (isinstance(_v, str) and _v.strip() == "") or \
                     (isinstance(_v, float) and _v != _v)  # NaN
            _rd_by_id[str(_r.get("track_id"))] = bool(_empty)
        _no_rd_mask = _np_rd.array(
            [_rd_by_id.get(str(tid), True) for tid in track_index.track_ids], dtype=bool)


    # relation-scorer features (registrant / rare-tags) aligned to track_index.track_ids
    _track_registrant = None
    _track_raretags = None
    if args.w_rel_registrant != 0.0 or args.w_rel_raretag != 0.0:
        import re as _re_rel
        from collections import Counter as _Counter_rel
        _meta_rows = {str(r["track_id"]): r for r in pl.read_parquet(
            args.track_metadata_path,
            columns=["track_id", "ISRC", "tag_list"]).iter_rows(named=True)}
        def _one_rel(v):
            return (v[0] if v else "") if isinstance(v, (list, tuple)) else (v or "")
        def _sset_rel(v):
            return set(str(x).strip().lower() for x in v if x) if isinstance(v, (list, tuple)) else set()
        def _registrant(v):
            s = _re_rel.sub(r"[^A-Z0-9]", "", str(_one_rel(v)).upper())
            return s[:5] if len(s) >= 7 else ""
        # registrant per track id
        _reg_by_id = {t: _registrant(r.get("ISRC")) for t, r in _meta_rows.items()}
        _track_registrant = [_reg_by_id.get(str(tid), "") for tid in track_index.track_ids]
        # rare tags: tag DF over the (restricted) catalog the track_index covers
        _tag_by_id = {t: _sset_rel(r.get("tag_list")) for t, r in _meta_rows.items()}
        _dfc = _Counter_rel()
        for tid in track_index.track_ids:
            for x in _tag_by_id.get(str(tid), set()):
                _dfc[x] += 1
        _track_raretags = [
            frozenset(x for x in _tag_by_id.get(str(tid), set()) if _dfc.get(x, 0) <= args.rel_tag_df)
            for tid in track_index.track_ids]
        _nz_reg = sum(1 for r in _track_registrant if r)
        _nz_tag = sum(1 for s in _track_raretags if s)
        print(f"  relation scorer: w_rel_artist={args.w_rel_artist} "
              f"w_rel_registrant={args.w_rel_registrant} w_rel_raretag={args.w_rel_raretag} "
              f"(rel_tag_df<={args.rel_tag_df}) | {_nz_reg} tracks w/ registrant, "
              f"{_nz_tag} w/ >=1 rare tag")

    audio_emb = audio_mask = None
    if args.w_audio != 0.0 or args.audio_pool_k > 0:
        audio_emb, audio_mask = load_audio_tower(
            args.audio_tower_cache, list(track_index.track_ids))
        if audio_emb is None:
            print("  NOTE: --w-audio / --audio-pool-k set but audio tower missing; OFF.")

    # --- prior-mean DENSE-ANCHOR continuity term (Qwen-8B text space) ---
    # Mirrors the audio/image prior-mean terms but uses the SAME Qwen dense track
    # tower the backbone scores against. Adds the missing prior-mean DENSE continuity
    # (cos(mean_endorsed_prior_dense, candidate)); the query is already the backbone.
    dense_anchor_emb = dense_anchor_mask = None
    if args.w_dense_anchor != 0.0 or args.anchor_pool_k > 0:
        dense_anchor_emb, dense_anchor_mask = load_dense_anchor_tower(
            args.qwen_track_cache_dir, list(track_index.track_ids))
        if dense_anchor_emb is None:
            print("  NOTE: --w-dense-anchor / --anchor-pool-k set but dense tower missing; OFF.")

    image_emb = image_mask = None
    if args.w_image != 0.0:
        image_emb, image_mask = load_audio_tower(
            args.image_tower_cache, list(track_index.track_ids), mod_name="image-siglip2")
        if image_emb is None:
            print("  NOTE: --w-image set but image tower missing; image term will be OFF.")

    # --- lyrics / attributes query->track terms (0.6B space) ---
    lyr_emb = lyr_mask = attr_emb = attr_mask = None
    q06_by_key = None
    lyr_pool_by_key = attr_pool_by_key = None
    need_lyr = (args.w_lyr != 0.0) or (args.lyr_pool_k > 0)
    need_attr = (args.w_attr != 0.0) or (args.attr_pool_k > 0)
    if need_lyr or need_attr:
        track_ids_list = list(track_index.track_ids)
        if need_lyr:
            lyr_emb, lyr_mask = load_audio_tower(
                args.lyr_attr_tower_cache, track_ids_list,
                mod_name="lyrics-qwen3_embedding_0.6b")
            if lyr_emb is None:
                print("  NOTE: lyrics tower missing; lyrics term + arm OFF.")
        if need_attr:
            attr_emb, attr_mask = load_audio_tower(
                args.lyr_attr_tower_cache, track_ids_list,
                mod_name="attributes-qwen3_embedding_0.6b")
            if attr_emb is None:
                print("  NOTE: attributes tower missing; attributes term + arm OFF.")
        if lyr_emb is not None or attr_emb is not None:
            q06_dir = args.q06_query_cache_dir or model_query_cache_dir("0.6")
            q06_by_key = load_q06_query_by_key(q06_dir, task_rows)
            if args.lyr_pool_k > 0 and lyr_emb is not None:
                print(f"  building lyrics retrieval arm (top-{args.lyr_pool_k}) ...")
                lyr_pool_by_key = build_modal_arm_pools(q06_by_key, lyr_emb, lyr_mask, args.lyr_pool_k)
            if args.attr_pool_k > 0 and attr_emb is not None:
                print(f"  building attributes retrieval arm (top-{args.attr_pool_k}) ...")
                attr_pool_by_key = build_modal_arm_pools(q06_by_key, attr_emb, attr_mask, args.attr_pool_k)

    # ---- EXTRA GENERATORS: last-prior dense-NN + second-backbone (qwen3-4b) -------------
    # Both produce key -> candidate-idx arrays unioned into the pool (like the modal arms).
    # external candidates parquet -> per-session top-N-by-rank track ids (predict turn)
    cand_by_session = None
    if args.candidates_parquet is not None and args.candidates_parquet.exists():
        import polars as _plc
        _cdf = _plc.read_parquet(args.candidates_parquet)
        if "kind" in _cdf.columns and args.candidates_kind:
            _cdf = _cdf.filter(_plc.col("kind") == args.candidates_kind)
        # predict turn = MAX turn per session; take its rows, top-N by rank (rank 1 = best)
        _mt = _cdf.group_by("session_id").agg(_plc.col("turn_number").max().alias("_mt"))
        _cdf = _cdf.join(_mt, on="session_id").filter(_plc.col("turn_number") == _plc.col("_mt"))
        cand_by_session = {}
        for _sid, _sub in _cdf.group_by("session_id"):
            _sid = _sid[0] if isinstance(_sid, tuple) else _sid
            _rows = _sub.sort("rank").head(args.candidates_topn)
            cand_by_session[str(_sid)] = [str(t) for t in _rows["track_id"].to_list()]
        print(f"  candidates-parquet: loaded top-{args.candidates_topn} for "
              f"{len(cand_by_session)} sessions (kind={args.candidates_kind})")

    # additional candidate source: SUBMISSION-format JSON(s). predicted_track_ids is already
    # rank-ordered (position = rank), so take the first --candidates-topn per session. Merge
    # ADDITIVELY into cand_by_session (parquet tracks first, then submission tracks, deduped).
    if args.candidates_submission:
        import json as _json_cs
        if cand_by_session is None:
            cand_by_session = {}
        _n_sess_added = 0
        for _subpath in args.candidates_submission:
            if not _subpath.exists():
                print(f"  candidates-submission: WARNING {_subpath} not found, skipping"); continue
            _d = _json_cs.load(open(_subpath))
            if isinstance(_d, dict):
                _d = list(_d.values()) if all(isinstance(v, dict) for v in _d.values()) else _d
            for _e in _d:
                _sid = str(_e["session_id"])
                _preds = [str(t) for t in (_e.get("predicted_track_ids") or [])][:args.candidates_topn]
                if not _preds: continue
                _existing = cand_by_session.get(_sid, [])
                _seen = set(_existing)
                _merged = list(_existing) + [t for t in _preds if not (t in _seen or _seen.add(t))]
                if _sid not in cand_by_session: _n_sess_added += 1
                cand_by_session[_sid] = _merged
        print(f"  candidates-submission: merged {len(args.candidates_submission)} file(s); "
              f"cand_by_session now covers {len(cand_by_session)} sessions "
              f"(+{_n_sess_added} new from submissions)")

    extra_pool_by_key = None
    if args.last_prior_pool_k > 0 or args.bb2_pool_k > 0 or cand_by_session is not None:
        import numpy as _np2
        extra_pool_by_key = {}
        # last-prior generator: top-K dense-NN to the LAST prior track in the qwen8b space
        lp_emb = lp_mask = None
        if args.last_prior_pool_k > 0:
            lp_emb, lp_mask = (dense_anchor_emb, dense_anchor_mask)
            if lp_emb is None:
                print("  NOTE: --last-prior-pool-k set but dense_anchor tower missing; OFF.")
        # second backbone: load its track tower + query cache, score per task
        bb2_T = bb2_qbykey = bb2_ids = None
        if args.bb2_pool_k > 0 and args.bb2_query_cache_dir and args.bb2_track_cache_dir:
            import polars as _pl2
            _te = args.bb2_track_cache_dir / "embeddings.npy"
            _ti = args.bb2_track_cache_dir / "track_ids.npy"
            _qe = args.bb2_query_cache_dir / "query_embeddings.npy"
            _qm = args.bb2_query_cache_dir / "query_meta.parquet"
            if _te.exists() and _qe.exists():
                bb2_T = _np2.load(_te).astype(_np2.float32)
                bb2_T /= (_np2.linalg.norm(bb2_T, axis=1, keepdims=True) + 1e-12)
                bb2_btids = ([str(t) for t in _np2.load(_ti, allow_pickle=True).tolist()]
                             if _ti.exists() else list(track_index.track_ids))
                # map second-backbone track idx -> canonical catalog idx
                _canon = {t: i for i, t in enumerate(track_index.track_ids)}
                bb2_b2c = _np2.array([_canon.get(t, -1) for t in bb2_btids], dtype=_np2.int64)
                _qm_df = _pl2.read_parquet(_qm)
                bb2_Q = _np2.load(_qe).astype(_np2.float32)
                bb2_Q /= (_np2.linalg.norm(bb2_Q, axis=1, keepdims=True) + 1e-12)
                bb2_qbykey = {}
                for _i, (_s, _t) in enumerate(zip(_qm_df["session_id"].to_list(),
                                                  _qm_df["turn_number"].to_list())):
                    bb2_qbykey[(str(_s), int(_t))] = _i
                print(f"  second-backbone generator: tower {bb2_T.shape}, "
                      f"{len(bb2_qbykey)} query rows (top-{args.bb2_pool_k})")
            else:
                print("  NOTE: --bb2-pool-k set but bb2 caches missing; OFF.")
                bb2_T = None
        # per task, build the union of extra candidates
        for _task in task_rows:
            _key = (_task["session_id"], _task["user_id"], int(_task["turn_number"]))
            _cands = []
            _priors = [track_index.id_to_idx[t] for t in
                       (str(x) for x in _task["prefix_track_ids"]) if t in track_index.id_to_idx]
            # last-prior arm
            if lp_emb is not None and _priors:
                _last = _priors[-1]
                if lp_mask is None or lp_mask[_last]:
                    _sc = lp_emb @ lp_emb[_last]
                    _k = min(args.last_prior_pool_k, _sc.shape[0] - 1)
                    _top = _np2.argpartition(-_sc, _k)[:_k]
                    _cands.append(_top.astype(_np2.int64))
            # second-backbone arm
            if bb2_T is not None and bb2_qbykey is not None:
                _qi = bb2_qbykey.get((_key[0], _key[2]))
                if _qi is not None:
                    _scb = bb2_T @ bb2_Q[_qi]
                    _k = min(args.bb2_pool_k, _scb.shape[0] - 1)
                    _topb = _np2.argpartition(-_scb, _k)[:_k]
                    _cc = bb2_b2c[_topb]
                    _cands.append(_cc[_cc >= 0].astype(_np2.int64))
            # external candidates (top-N already; add at most candidates_max NEW catalog idx)
            if cand_by_session is not None:
                # split parquets (allobs) suffix session_id with __t{turn}; strip to match the
                # candidates file which is keyed by the ORIGINAL session_id.
                _csid = _key[0].split("__t")[0] if "__t" in _key[0] else _key[0]
                _clist = cand_by_session.get(_csid, [])
                _already = set()
                for _arr in _cands:
                    _already.update(int(x) for x in _arr.tolist())
                _already.update(_priors)
                _added = []
                for _tk in _clist:                       # already ordered best-first by rank
                    _ix = track_index.id_to_idx.get(str(_tk))
                    if _ix is None or _ix in _already:    # exclude common/already-extracted
                        continue
                    _added.append(_ix); _already.add(_ix)
                    if len(_added) >= args.candidates_max:   # cap at most N new
                        break
                if _added:
                    _cands.append(_np2.array(_added, dtype=_np2.int64))
            if _cands:
                extra_pool_by_key[_key] = _np2.unique(_np2.concatenate(_cands))
        print(f"  extra-generator pools built for {len(extra_pool_by_key)} task rows")

    _weights_override = None
    if args.default_weights_json is not None and args.default_weights_json.exists():
        import json as _json
        from emblib.retrieval.gambling import DEFAULT_WEIGHTS as _DW
        _ov = _json.loads(args.default_weights_json.read_text())
        _weights_override = dict(_DW); _weights_override.update(_ov)
        print(f"  DEFAULT_WEIGHTS override: {_weights_override}")

    final_recs_by_key, stats = generate_heuristic_with_qwen_fallback(
        task_rows=task_rows,
        qwen_recs_by_key=qwen_recs_by_key,
        track_index=track_index,
        top_k=args.top_k,
        qwen_scores_by_key=qwen_scores_by_key,
        qwen_pool_k=args.qwen_pool_k,
        use_decade=args.use_decade,
        dump_pools=args.dump_pools,
        w_dense=w_dense,
        goal_direction=gd_cfg,
        dump_provenance=dump_prov,
        audio_emb=audio_emb,
        audio_mask=audio_mask,
        w_audio=args.w_audio,
        audio_center=args.audio_center,
        dense_anchor_emb=dense_anchor_emb,
        dense_anchor_mask=dense_anchor_mask,
        w_dense_anchor=args.w_dense_anchor,
        dense_anchor_center=args.dense_anchor_center,
        w_anchor_whitened=args.w_anchor_whitened,
        anchor_whitened_center=args.anchor_whitened_center,
        w_pop=args.w_pop,
        w_cooc=args.w_cooc,
        dense_anchor_recency=args.dense_anchor_recency,
        recency_tau=args.recency_tau,
        anchor_pool_k=args.anchor_pool_k,
        audio_pool_k=args.audio_pool_k,
        heuristic_scale=args.heuristic_scale,
        cat_spec_gamma=args.cat_spec_gamma,
        image_emb=image_emb,
        image_mask=image_mask,
        w_image=args.w_image,
        image_center=args.image_center,
        q06_by_key=q06_by_key,
        extra_pool_by_key=extra_pool_by_key,
        weights_override=_weights_override,
        lyr_emb=lyr_emb,
        lyr_mask=lyr_mask,
        w_lyr=args.w_lyr,
        attr_emb=attr_emb,
        attr_mask=attr_mask,
        w_attr=args.w_attr,
        qt_gate_floor=args.qt_gate_floor,
        use_progress=(args.use_progress or args.w_reject != 0.0),
        w_reject=args.w_reject,
        anchor_last_endorsed=args.anchor_last_endorsed,
        spec_reweight=args.spec_reweight,
        fallback_content_rerank=args.fallback_content_rerank,
        heuristic_skip_mods=(frozenset({"lyr", "attr", "image"})
                             if args.measured_config else frozenset()),
        allowed_mask=getattr(args, "_allowed_mask", None),
        fallback_backbone_weight=args.fallback_backbone_weight,
        lyr_pool_by_key=lyr_pool_by_key,
        attr_pool_by_key=attr_pool_by_key,
        release_date_weight=args.release_date_weight,
        no_release_date_mask=_no_rd_mask,
        w_rel_artist=args.w_rel_artist,
        w_rel_registrant=args.w_rel_registrant,
        w_rel_raretag=args.w_rel_raretag,
        track_registrant=_track_registrant,
        track_raretags=_track_raretags,

    )

    rows = build_tracks_only_rows(task_rows, final_recs_by_key, top_k=args.top_k)
    output = args.output_path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"saved_tracks_only={output}")
    print(f"stats={stats}")
    if dump_prov is not None:
        print(f"provenance={Path(dump_prov).expanduser().resolve()}")


if __name__ == "__main__":
    main()