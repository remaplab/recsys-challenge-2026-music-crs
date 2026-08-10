from __future__ import annotations

from typing import Any

import json
from pathlib import Path

import numpy as np
import polars as pl

from emblib.retrieval.gambling import (
    CAT_MULT,
    DEFAULT_WEIGHTS,
    SCORE_TERMS,
    SPEC_MULT,
    TrackIndex,
    get_candidate_pool,
    score_candidates,
)

# Prior-mean track-embedding (continuity) terms: w * (cos(mean_prior, cand) - center).
# dense_anchor = prior-mean in the Qwen-8B DENSE track space (same tower the backbone
# scores against); the missing prior-mean DENSE continuity. Added as a first-class
# prior-sim modality so the scoring loop AND provenance handle it automatically.
PRIOR_SIM_MODS = ("audio", "image", "dense_anchor")
# Query->track (content) terms: w * minmax_pool(cos(query_06b, track_mod)).
# Built from the 0.6B query vector (lyrics/attr towers are 0.6B, NOT 8B-comparable).
QUERY_SIM_MODS = ("lyr", "attr")

# ── Category affinity for the query->track content terms (Table 2, Axis 1) ───
# A=Audio-Based, B=Lyrical, C=Visual-Musical, D=Contextual, E=Interactive,
# F=Metadata-Rich, G=Mood/Emotion, H=Artist/Discography, I=Cultural/Geographic,
# J=Social/Popularity, K=Temporal/Era.
# affinity in [0,1]: 1.0 = home category, partial = related, absent -> `floor`.
# LYRICS: lyrical content -> B(home), G(mood lives in lyrics), D(thematic, weaker).
#   Deliberately NOT on H/I/J (identity/geo/popularity = lyrics is noise there).
LYR_AFFINITY = {"B": 1.0, "G": 0.6, "D": 0.4}
# ATTRIBUTES: genre/metadata/energy -> F(home), A(sonic descriptors), K(era-as-meta),
#   E(genre navigation). Broader than lyrics because metadata is a more general axis.
ATTR_AFFINITY = {"F": 1.0, "A": 0.6, "K": 0.6, "E": 0.5}
QUERY_AFFINITY = {"lyr": LYR_AFFINITY, "attr": ATTR_AFFINITY}

# ── Specificity-driven prior-vs-dense reweighting (Table 3, Axis 2) ──────────
SPEC_REWEIGHT = {
    "LH": (0.5, 1.6),
    "HH": (0.7, 1.4),
    "HL": (1.0, 1.0),
    "LL": (1.0, 1.1),
}


def _spec_scales(spec: str, enabled: bool) -> tuple[float, float]:
    if not enabled:
        return 1.0, 1.0
    return SPEC_REWEIGHT.get(spec, (1.0, 1.0))


def _qt_affinity(cat: str, table: dict, floor: float) -> float:
    return max(float(table.get(cat, 0.0)), float(floor))


def _rerank_fallback_by_content(
    fallback_recs, *, exclude, top_k, track_index,
    qscores, q06, lyr_emb, lyr_mask, attr_emb, attr_mask,
    w_dense, w_lyr, w_attr, dense_scale, cat, qt_gate_floor, n_consider=200,
    prior_means=None, active_prior=None, cont_scale=1.0, backbone_weight=1.0, extra_track_ids=None,
    pop_log=None, w_pop=0.0,
):
    """Re-rank the Qwen fallback list by a BLEND of Qwen's own order (backbone) and
    similarity terms. Returns reranked top_k track-ids + per-pick breakdown."""
    cands, seen = [], set(exclude)
    for tid in fallback_recs:
        if tid in seen:
            continue
        idx = track_index.id_to_idx.get(tid)
        if idx is None:
            continue
        cands.append((tid, idx))
        seen.add(tid)
        if len(cands) >= n_consider:
            break

    if extra_track_ids:
        for tid in extra_track_ids:
            if tid in seen:
                continue
            ix = track_index.id_to_idx.get(tid)
            if ix is None:
                continue
            cands.append((tid, ix))
            seen.add(tid)
    if not cands:
        return [], []
    tids = [t for t, _ in cands]
    idxs = np.array([i for _, i in cands], dtype=np.int64)
    n = len(cands)

    backbone = np.linspace(1.0, 0.0, n, dtype=np.float64)
    scores = (backbone_weight * backbone).copy()
    breakdown = [{"qwen_rank": r, "backbone": float(backbone[r])} for r in range(n)]

    # CONTENT: 8B dense query->track
    if w_dense > 0.0 and qscores is not None:
        sim = np.asarray(qscores, dtype=np.float64)[idxs]
        lo, hi = float(sim.min()), float(sim.max())
        s01 = (sim - lo) / (hi - lo) if hi > lo else np.zeros_like(sim)
        scores = scores + (w_dense * dense_scale) * s01
        for r in range(n):
            breakdown[r]["dense01"] = float(s01[r])

    # POP: split-clean global train popularity prior (cold-row rescue)
    if w_pop > 0.0 and pop_log is not None:
        pv = np.asarray(pop_log, dtype=np.float64)[idxs]
        lo, hi = float(pv.min()), float(pv.max())
        p01 = (pv - lo) / (hi - lo) if hi > lo else np.zeros_like(pv)
        scores = scores + w_pop * p01
        for r in range(n):
            breakdown[r]["pop01"] = float(p01[r])

    # CONTENT: 0.6B lyrics / attributes query->track
    for m, emb, mask, w in (("lyr", lyr_emb, lyr_mask, w_lyr),
                            ("attr", attr_emb, attr_mask, w_attr)):
        if w == 0.0 or emb is None or q06 is None:
            continue
        aff = _qt_affinity(cat, QUERY_AFFINITY[m], qt_gate_floor)
        raw = emb[idxs] @ q06
        raw = np.where(mask[idxs], raw, -np.inf)
        finite = raw[np.isfinite(raw)]
        if finite.size == 0:
            continue
        lo, hi = float(finite.min()), float(finite.max())
        s01 = np.where(np.isfinite(raw), (raw - lo) / (hi - lo) if hi > lo else 0.0, 0.0)
        scores = scores + (w * dense_scale * aff) * s01
        for r in range(n):
            breakdown[r][f"{m}01"] = float(s01[r])

    # CONTINUITY: prior-mean track->track (empty-pool rows only).
    if prior_means and active_prior:
        for m, pm in prior_means.items():
            emb, mask, w, center = active_prior[m]
            raw = emb[idxs] @ pm
            cos = np.where(mask[idxs], raw, 0.0).astype(np.float64)
            scores = scores + (w * cont_scale) * (cos - center)
            for r in range(n):
                breakdown[r][f"{m}_priorcos"] = float(cos[r])

    order = np.argsort(-scores)[:top_k]
    return [tids[o] for o in order], [breakdown[o] for o in order]


def take_fallback_recs(
    fallback_recs: list[str],
    *,
    exclude: set[str],
    top_k: int,
    context: str,
) -> list[str]:
    out: list[str] = []
    seen = set(exclude)
    for track_id in fallback_recs:
        if track_id in seen:
            continue
        out.append(track_id)
        seen.add(track_id)
        if len(out) >= top_k:
            return out
    raise ValueError(f"fallback did not provide enough tracks for {context}: got {len(out)}, needed {top_k}")


def _move_away_factor(qscores, prior_idxs, cfg) -> float:
    if not prior_idxs or qscores is None:
        return 1.0
    a_prior = max(float(qscores[p]) for p in prior_idxs)
    v = np.asarray(qscores, dtype=np.float64).copy()
    for p in prior_idxs:
        v[p] = -np.inf
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return 1.0
    N = int(cfg.get("top_n", 50))
    a_top = float(np.mean(np.partition(finite, -N)[-N:] if N < finite.size else finite))
    gap = a_top - a_prior
    t_lo, t_hi, floor = float(cfg["t_lo"]), float(cfg["t_hi"]), float(cfg["floor"])
    if gap <= t_lo:
        return 1.0
    if gap >= t_hi:
        return floor
    return 1.0 - (gap - t_lo) / (t_hi - t_lo) * (1.0 - floor)


def _prior_mean(emb, mask, prior_idxs, recency_tau=None):
    """Unit-normalized mean of the prior tracks' embeddings. If recency_tau is set
    (>0), weight priors by exp(rank/tau) so RECENT priors count more (priors are in
    chronological prefix order; later index = more recent). Validated: recency beats
    the flat mean on the dense-anchor term, gain grows with turn (tau~2.0)."""
    if emb is None or not prior_idxs:
        return None
    valid = [p for p in prior_idxs if mask[p]]
    if not valid:
        return None
    if recency_tau is not None and recency_tau > 0 and len(valid) > 1:
        w = np.exp(np.arange(len(valid), dtype=np.float32) / float(recency_tau))
        mean = (emb[valid] * w[:, None]).sum(axis=0) / w.sum()
    else:
        mean = emb[valid].mean(axis=0)
    nn = float(np.linalg.norm(mean))
    return (mean / nn).astype(np.float32) if nn > 1e-12 else None


def _minmax_pool(values, cand_idxs):
    v = np.asarray(values, dtype=np.float64)[cand_idxs]
    lo, hi = float(v.min()), float(v.max())
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


def _prov_fallback_rows(key, recs, branch, *, cat, spec):
    nan = float("nan")
    rows = []
    for slot, tid in enumerate(recs):
        rec = {
            "session_id": key[0], "user_id": key[1], "turn_number": key[2],
            "branch": branch, "slot": slot, "track_id": tid,
            "source": "fallback_dense", "score_total": nan, "dense_contrib": 0.0,
            "sim01": nan, "cosine": nan, "md": 1.0, "cat": cat, "spec": spec, "n_pool": 0,
            **{f"t_{t}": nan for t in SCORE_TERMS},
        }
        for m in PRIOR_SIM_MODS:
            rec[f"{m}_contrib"] = 0.0
            rec[f"{m}_cos"] = nan
            rec[f"{m}_reject_contrib"] = 0.0
        for m in QUERY_SIM_MODS:
            rec[f"{m}_contrib"] = 0.0
            rec[f"{m}_sim01"] = nan
            rec[f"{m}_affinity"] = nan
        rows.append(rec)
    return rows


def generate_heuristic_with_qwen_fallback(
    *,
    task_rows: list[dict[str, Any]],
    qwen_recs_by_key: dict[tuple[str, str, int], list[str]],
    track_index: TrackIndex,
    top_k: int,
    qwen_scores_by_key: dict | None = None,
    qwen_pool_k: int = 0,
    use_decade: bool = True,
    dump_pools: Path | None = None,
    w_dense: float = 0.0,
    goal_direction: dict | None = None,
    dump_provenance: Path | None = None,
    audio_emb: np.ndarray | None = None,
    audio_mask: np.ndarray | None = None,
    w_audio: float = 0.0,
    audio_center: float = 0.42,
    dense_anchor_emb: np.ndarray | None = None,
    dense_anchor_mask: np.ndarray | None = None,
    w_dense_anchor: float = 0.0,
    dense_anchor_center: float = 0.0,
    pop_log: np.ndarray | None = None,
    w_pop: float = 0.0,
    cooc_map: dict | None = None,
    w_cooc: float = 0.0,
    dense_anchor_recency: bool = False,
    recency_tau: float = 2.0,
    w_anchor_whitened: float = 0.0,
    anchor_whitened_center: float = 0.0,
    anchor_pool_k: int = 0,
    audio_pool_k: int = 0,
    heuristic_scale: float = 1.0,
    cat_spec_gamma: float = 1.0,
    image_emb: np.ndarray | None = None,
    image_mask: np.ndarray | None = None,
    w_image: float = 0.0,
    image_center: float = 0.0,
    q06_by_key: dict | None = None,
    lyr_emb: np.ndarray | None = None,
    lyr_mask: np.ndarray | None = None,
    w_lyr: float = 0.0,
    attr_emb: np.ndarray | None = None,
    attr_mask: np.ndarray | None = None,
    w_attr: float = 0.0,
    qt_gate_floor: float = 1.0,
    use_progress: bool = False,
    w_reject: float = 0.0,
    anchor_last_endorsed: bool = False,
    spec_reweight: bool = False,
    fallback_content_rerank: bool = False,
    fallback_backbone_weight: float = 1.0,
    lyr_pool_by_key: dict | None = None,
    attr_pool_by_key: dict | None = None,
    extra_pool_by_key: dict | None = None,
    weights_override: dict | None = None,
    heuristic_skip_mods: frozenset = frozenset(),
    allowed_mask: np.ndarray | None = None,
    release_date_weight: float = 1.0,
    no_release_date_mask: np.ndarray | None = None,
    w_rel_artist: float = 0.0,
    w_rel_registrant: float = 0.0,
    w_rel_raretag: float = 0.0,
    track_registrant: list | None = None,
    track_raretags: list | None = None,
) -> tuple[dict[tuple[str, str, int], list[str]], dict[str, int]]:
    recs_by_key: dict[tuple[str, str, int], list[str]] = {}
    stats = {
        "rows": len(task_rows),
        "qwen_no_known_prior_fallback_rows": 0,
        "qwen_empty_candidate_pool_fallback_rows": 0,
        "qwen_fill_shortfall_rows": 0,
        "heuristic_ranked_rows": 0,
        "use_decade": bool(use_decade),
        "qwen_pool_k": int(qwen_pool_k),
        "w_dense": float(w_dense),
        "w_audio": float(w_audio),
        "w_dense_anchor": float(w_dense_anchor),
        "w_anchor_whitened": float(w_anchor_whitened),
        "anchor_whitened_center": float(anchor_whitened_center),
        "w_image": float(w_image),
        "w_lyr": float(w_lyr),
        "w_attr": float(w_attr),
        "qt_gate_floor": float(qt_gate_floor),
        "use_progress": bool(use_progress),
        "w_reject": float(w_reject),
        "anchor_last_endorsed": bool(anchor_last_endorsed),
        "spec_reweight": bool(spec_reweight),
        "spec_reweight_rows": 0,
        "fallback_content_rerank": bool(fallback_content_rerank),
        "fallback_reranked_rows": 0,
        "empty_pool_reranked_rows": 0,
        "rows_with_rejection": 0,
        "rejected_priors_total": 0,
        "reject_applied_rows": 0,
        "goal_direction": bool(goal_direction is not None),
        "dense_applied_rows": 0,
        "audio_applied_rows": 0,
        "dense_anchor_applied_rows": 0,
        "anchor_whitened_applied_rows": 0,
        "cooc_applied_rows": 0,
        "anchor_pool_rows": 0,
        "audio_pool_rows": 0,
        "image_applied_rows": 0,
        "lyr_applied_rows": 0,
        "attr_applied_rows": 0,
        "md_sum": 0.0,
        "md_rows": 0,
    }
    pool_records: list[dict] = []
    prov_records: list[dict] = []
    need_q = (qwen_pool_k > 0) or (w_dense > 0.0) or (goal_direction is not None) \
        or (dump_provenance is not None)
    want_components = dump_provenance is not None

    prior_mods = {
        "audio": (audio_emb, audio_mask, float(w_audio), float(audio_center)),
        "image": (image_emb, image_mask, float(w_image), float(image_center)),
        "dense_anchor": (dense_anchor_emb, dense_anchor_mask,
                         float(w_dense_anchor), float(dense_anchor_center)),
    }
    active_prior = {m: v for m, v in prior_mods.items() if v[0] is not None and v[2] != 0.0}

    query_mods = {
        "lyr": (lyr_emb, lyr_mask, float(w_lyr)),
        "attr": (attr_emb, attr_mask, float(w_attr)),
    }
    active_query = {m: v for m, v in query_mods.items()
                    if v[0] is not None and v[2] != 0.0 and q06_by_key is not None}

    for task in task_rows:
        key = (task["session_id"], task["user_id"], int(task["turn_number"]))
        qwen_fallback = qwen_recs_by_key.get(key)
        if qwen_fallback is None:
            raise ValueError(f"missing Qwen fallback recommendations for {key}")

        lyr_arm = np.asarray(lyr_pool_by_key.get(key)) \
            if (lyr_pool_by_key is not None and lyr_pool_by_key.get(key) is not None) \
            else np.empty(0, dtype=np.int64)
        attr_arm = np.asarray(attr_pool_by_key.get(key)) \
            if (attr_pool_by_key is not None and attr_pool_by_key.get(key) is not None) \
            else np.empty(0, dtype=np.int64)
        extra_arm = np.asarray(extra_pool_by_key.get(key)) \
            if (extra_pool_by_key is not None and extra_pool_by_key.get(key) is not None) \
            else np.empty(0, dtype=np.int64)
        _arm_parts = [a for a in (lyr_arm, attr_arm, extra_arm) if len(a)]
        if _arm_parts:
            arm_tids = [track_index.track_ids[int(j)]
                        for j in np.unique(np.concatenate(_arm_parts))]
        else:
            arm_tids = []

        row = task["row"]
        goal = row.get("conversation_goal") or {}
        cat = goal.get("category") or "?"
        spec = goal.get("specificity") or "?"

        qscores = qwen_scores_by_key.get(key) if (qwen_scores_by_key is not None and need_q) else None
        q06 = q06_by_key.get(key) if (q06_by_key is not None and (lyr_emb is not None or attr_emb is not None)) else None
        cont_scale, dense_scale = _spec_scales(spec, spec_reweight)

        prior_track_ids = [str(track_id) for track_id in task["prefix_track_ids"]]
        prior_progress = task.get("prefix_progress") or [None] * len(prior_track_ids)
        if len(prior_progress) != len(prior_track_ids):
            prior_progress = [None] * len(prior_track_ids)

        prior_pairs = [(track_index.id_to_idx[t], prior_progress[j])
                       for j, t in enumerate(prior_track_ids)
                       if t in track_index.id_to_idx]
        prior_idxs = [i for i, _ in prior_pairs]
        if use_progress:
            endorsed_idxs = [i for i, lab in prior_pairs if lab != "DOES_NOT"]
            rejected_idxs = [i for i, lab in prior_pairs if lab == "DOES_NOT"]
        else:
            endorsed_idxs = list(prior_idxs)
            rejected_idxs = []
        if not endorsed_idxs:
            endorsed_idxs = list(prior_idxs)
        if rejected_idxs:
            stats["rows_with_rejection"] += 1
            stats["rejected_priors_total"] += len(rejected_idxs)

        if not prior_idxs:
            stats["qwen_no_known_prior_fallback_rows"] += 1
            rerank_bd = None
            if fallback_content_rerank:
                reranked, rerank_bd = _rerank_fallback_by_content(
                    qwen_fallback, exclude=set(prior_track_ids), top_k=top_k,
                    track_index=track_index, qscores=qscores, q06=q06,
                    lyr_emb=lyr_emb, lyr_mask=lyr_mask, attr_emb=attr_emb, attr_mask=attr_mask,
                    w_dense=w_dense, w_lyr=w_lyr, w_attr=w_attr, dense_scale=dense_scale,
                    cat=cat, qt_gate_floor=qt_gate_floor, backbone_weight=fallback_backbone_weight,
                    pop_log=pop_log, w_pop=w_pop, extra_track_ids=arm_tids)
                if len(reranked) >= top_k:
                    recs_by_key[key] = reranked
                    stats["fallback_reranked_rows"] += 1
                else:
                    recs_by_key[key] = take_fallback_recs(
                        qwen_fallback, exclude=set(prior_track_ids), top_k=top_k,
                        context=f"no-known-prior {key}")
                    rerank_bd = None
            else:
                recs_by_key[key] = take_fallback_recs(
                    qwen_fallback, exclude=set(prior_track_ids), top_k=top_k,
                    context=f"no-known-prior {key}")
            if dump_provenance is not None:
                prov_records.extend(_prov_fallback_rows(
                    key, recs_by_key[key], "no_prior_fallback", cat=cat, spec=spec))
            continue

        weights = dict(weights_override) if weights_override else dict(DEFAULT_WEIGHTS)
        # cat/spec stratification strength: gamma=1 shipped, gamma=0 OFF (boost->1),
        # gamma>1 amplified. Validated gamma=0 (stratification doesn't help here).
        boost = (CAT_MULT.get(cat, 1.0) * SPEC_MULT.get(spec, 1.0)) ** cat_spec_gamma
        for weight_name in ("album_last", "artist_last", "album_any", "artist_any"):
            weights[weight_name] *= boost
        # heuristic-vs-embedding balance: scale ALL 7 base terms. Validated ~0.1
        # (the metadata base is sparse over the pool; embeddings should dominate).
        if heuristic_scale != 1.0:
            for weight_name in weights:
                weights[weight_name] *= heuristic_scale

        md = 1.0
        if goal_direction is not None and qscores is not None:
            md = _move_away_factor(qscores, prior_idxs, goal_direction)
            for weight_name in ("album_last", "artist_last", "album_any", "artist_any"):
                weights[weight_name] *= md
            stats["md_sum"] += md
            stats["md_rows"] += 1

        if spec_reweight and (cont_scale != 1.0 or dense_scale != 1.0):
            for weight_name in ("album_last", "artist_last", "album_any", "artist_any"):
                weights[weight_name] *= cont_scale
            stats["spec_reweight_rows"] += 1
        w_dense_eff = w_dense * dense_scale
        w_audio_eff = w_audio * dense_scale
        w_image_eff = w_image * dense_scale
        w_lyr_eff = w_lyr * dense_scale
        w_attr_eff = w_attr * dense_scale

        pool_artists = {track_index.artist[i] for i in prior_idxs if track_index.artist[i] is not None}
        pool_albums = {track_index.album[i] for i in prior_idxs if track_index.album[i] is not None}
        boost_artists = {track_index.artist[i] for i in endorsed_idxs if track_index.artist[i] is not None}
        boost_albums = {track_index.album[i] for i in endorsed_idxs if track_index.album[i] is not None}

        # relation-scorer pool sets (registrant / rare-tags shared with PRIOR tracks)
        pool_registrants = set()
        if w_rel_registrant != 0.0 and track_registrant is not None:
            pool_registrants = {track_registrant[i] for i in prior_idxs
                                if i < len(track_registrant) and track_registrant[i]}
        pool_raretags = set()
        if w_rel_raretag != 0.0 and track_raretags is not None:
            for i in prior_idxs:
                if i < len(track_raretags) and track_raretags[i]:
                    pool_raretags |= track_raretags[i]

        prior_means = {}
        for m, (emb, mask, w, center) in active_prior.items():
            tau = recency_tau if (dense_anchor_recency and m == "dense_anchor") else None
            pm = _prior_mean(emb, mask, endorsed_idxs, recency_tau=tau)
            if pm is not None:
                prior_means[m] = pm

        reject_means = {}
        if w_reject != 0.0 and rejected_idxs:
            for m, (emb, mask, w, center) in active_prior.items():
                rm = _prior_mean(emb, mask, rejected_idxs)
                if rm is not None:
                    reject_means[m] = rm

        qpool = qscores if (qwen_scores_by_key is not None and qwen_pool_k > 0) else None

        _extra_src = {}
        if len(lyr_arm):
            _extra_src["lyr_arm"] = lyr_arm
        if len(attr_arm):
            _extra_src["attr_arm"] = attr_arm
        if len(extra_arm):
            _extra_src["extra_gen"] = extra_arm

        # --- anchor-pool GENERATOR: top-K recency-weighted prior-mean nearest-neighbors
        #     in the dense-anchor track space, unioned into the candidate pool. Converts
        #     the dense-anchor's strength from rerank-only into REACH. Cold rows (no
        #     priors) add nothing. Respects the catalog restriction and excludes priors.
        if anchor_pool_k > 0 and dense_anchor_emb is not None and endorsed_idxs:
            valid = [p for p in endorsed_idxs
                     if dense_anchor_mask is None or dense_anchor_mask[p]]
            if valid:
                if dense_anchor_recency and len(valid) > 1:
                    wv = np.exp(np.arange(len(valid), dtype=np.float32) / float(recency_tau))
                    pm = (dense_anchor_emb[valid] * wv[:, None]).sum(0) / wv.sum()
                else:
                    pm = dense_anchor_emb[valid].mean(0)
                nrm = float(np.linalg.norm(pm))
                if nrm > 1e-12:
                    pm = (pm / nrm).astype(np.float32)
                    sims = np.asarray(dense_anchor_emb, dtype=np.float32) @ pm
                    if allowed_mask is not None:
                        sims = np.where(allowed_mask, sims, -np.inf)
                    sims[prior_idxs] = -np.inf            # don't re-suggest priors
                    finite = int(np.isfinite(sims).sum())
                    kk = min(anchor_pool_k, finite)
                    if kk > 0:
                        ap = np.argpartition(-sims, kk - 1)[:kk].astype(np.int64)
                        _extra_src["anchor_pool"] = ap
                        stats["anchor_pool_rows"] += 1

        # --- audio-pool GENERATOR
        if audio_pool_k > 0 and audio_emb is not None and endorsed_idxs:
            valid = [p for p in endorsed_idxs if audio_mask is None or audio_mask[p]]
            if valid:
                if len(valid) > 1:
                    wv = np.exp(np.arange(len(valid), dtype=np.float32) / float(recency_tau))
                    pm = (audio_emb[valid] * wv[:, None]).sum(0) / wv.sum()
                else:
                    pm = audio_emb[valid].mean(0)
                nrm = float(np.linalg.norm(pm))
                if nrm > 1e-12:
                    pm = (pm / nrm).astype(np.float32)
                    sims = np.asarray(audio_emb, dtype=np.float32) @ pm
                    if audio_mask is not None:
                        sims = np.where(audio_mask, sims, -np.inf)
                    if allowed_mask is not None:
                        sims = np.where(allowed_mask, sims, -np.inf)
                    sims[prior_idxs] = -np.inf
                    finite = int(np.isfinite(sims).sum())
                    kk = min(audio_pool_k, finite)
                    if kk > 0:
                        apa = np.argpartition(-sims, kk - 1)[:kk].astype(np.int64)
                        _extra_src["audio_pool"] = apa
                        stats["audio_pool_rows"] += 1

        cand_idxs, sources = get_candidate_pool(
            prior_idxs, pool_artists, pool_albums, track_index,
            qwen_scores=qpool, qwen_k=qwen_pool_k, use_decade=use_decade,
            return_sources=True, extra_sources=(_extra_src or None))

        forbidden_idxs = set(prior_idxs)
        keep_mask = np.fromiter((i not in forbidden_idxs for i in cand_idxs),
                                dtype=bool, count=len(cand_idxs))
        cand_idxs = cand_idxs[keep_mask]
        forbidden_track_ids = {track_index.track_ids[i] for i in forbidden_idxs}

        if allowed_mask is not None and len(cand_idxs):
            cand_idxs = cand_idxs[allowed_mask[cand_idxs]]

        if dump_pools is not None:
            union = sources["artist"] | sources["album"] | sources["decade"] | sources["qwen"]
            qonly = sources["qwen"] - sources["artist"] - sources["album"] - sources["decade"]
            pool_records.append({
                "key": list(key), "branch": "heuristic",
                "n_artist": len(sources["artist"]), "n_album": len(sources["album"]),
                "n_decade": len(sources["decade"]), "n_qwen": len(sources["qwen"]),
                "n_qwen_only": len(qonly), "n_union_raw": len(union),
                "n_after_mask": int(len(cand_idxs))})

        if len(cand_idxs) == 0:
            stats["qwen_empty_candidate_pool_fallback_rows"] += 1
            rerank_bd = None
            if fallback_content_rerank:
                reranked, rerank_bd = _rerank_fallback_by_content(
                    qwen_fallback, exclude=forbidden_track_ids, top_k=top_k,
                    track_index=track_index, qscores=qscores, q06=q06,
                    lyr_emb=lyr_emb, lyr_mask=lyr_mask, attr_emb=attr_emb, attr_mask=attr_mask,
                    w_dense=w_dense, w_lyr=w_lyr, w_attr=w_attr, dense_scale=dense_scale,
                    cat=cat, qt_gate_floor=qt_gate_floor,
                    prior_means=prior_means, active_prior=active_prior, cont_scale=cont_scale, backbone_weight=fallback_backbone_weight, extra_track_ids=None,)
                if len(reranked) >= top_k:
                    recs_by_key[key] = reranked
                    stats["fallback_reranked_rows"] += 1
                    stats["empty_pool_reranked_rows"] += 1
                else:
                    recs_by_key[key] = take_fallback_recs(
                        qwen_fallback, exclude=forbidden_track_ids, top_k=top_k,
                        context=f"empty-candidate-pool {key}")
            else:
                recs_by_key[key] = take_fallback_recs(
                    qwen_fallback, exclude=forbidden_track_ids, top_k=top_k,
                    context=f"empty-candidate-pool {key}")
            if dump_provenance is not None:
                prov_records.extend(_prov_fallback_rows(
                    key, recs_by_key[key], "empty_pool_fallback", cat=cat, spec=spec))
            continue

        if anchor_last_endorsed:
            last_anchor = endorsed_idxs[-1]
        else:
            last_anchor = prior_idxs[-1]

        if want_components:
            scores, comp = score_candidates(
                cand_idxs, last_anchor, prior_idxs, boost_artists, boost_albums,
                track_index, weights, return_components=True)
            scores = scores.astype(np.float64)
            comp = {t: np.asarray(v, dtype=np.float64) for t, v in comp.items()}
        else:
            scores = score_candidates(
                cand_idxs, last_anchor, prior_idxs, boost_artists, boost_albums,
                track_index, weights).astype(np.float64)
            comp = None

        # --- dense 8th term: min-max 8B query->candidate cosine over the pool ---
        sim01 = None
        if qscores is not None:
            sim = np.asarray(qscores, dtype=np.float64)[cand_idxs]
            lo, hi = float(sim.min()), float(sim.max())
            sim01 = (sim - lo) / (hi - lo) if hi > lo else np.zeros_like(sim)
        if w_dense_eff > 0.0 and sim01 is not None:
            scores = scores + w_dense_eff * sim01
            stats["dense_applied_rows"] += 1

        # --- co-occurrence (split-clean train CF): per-candidate co-occurrence count
        #     with this row's prior tracks, min-max over the pool, added like the dense term.
        if w_cooc > 0.0 and cooc_map is not None and prior_track_ids:
            acc = {}
            for p in prior_track_ids:
                c = cooc_map.get(p)
                if c:
                    for tt, cnt in c.items():
                        j = track_index.id_to_idx.get(tt)
                        if j is not None:
                            acc[j] = acc.get(j, 0) + cnt
            if acc:
                cc = np.array([acc.get(int(ci), 0) for ci in cand_idxs], dtype=np.float64)
                lo, hi = float(cc.min()), float(cc.max())
                if hi > lo:
                    scores = scores + w_cooc * ((cc - lo) / (hi - lo))
                    stats["cooc_applied_rows"] += 1

        # --- prior-mean continuity terms (audio, image, dense_anchor): centered cosine ---
        mod_cos = {}
        for m, pm in prior_means.items():
            if m in heuristic_skip_mods:
                continue
            emb, mask, w, center = active_prior[m]
            w = w * cont_scale
            raw = emb[cand_idxs] @ pm
            cos = np.where(mask[cand_idxs], raw, 0.0).astype(np.float64)
            scores = scores + w * (cos - center)
            mod_cos[m] = cos
            stats[f"{m}_applied_rows"] += 1

        # --- WHITENED dense-anchor term (pool common-mode removed) -----------------
        if (w_anchor_whitened != 0.0 and dense_anchor_emb is not None
                and "dense_anchor" in prior_means):
            pm_dir = prior_means["dense_anchor"]                 # unit prior-mean (dense)
            P = np.asarray(dense_anchor_emb[cand_idxs], dtype=np.float64)  # pool x d
            dmask = (dense_anchor_mask[cand_idxs] if dense_anchor_mask is not None
                     else np.ones(len(cand_idxs), dtype=bool))
            valid_rows = P[dmask]
            if valid_rows.shape[0] >= 2:
                mu = valid_rows.mean(0)                          # pool common-mode
                Pw = P - mu
                nP = np.linalg.norm(Pw, axis=1, keepdims=True); nP[nP < 1e-12] = 1.0
                Pw = Pw / nP
                cw = np.asarray(pm_dir, dtype=np.float64) - mu
                ncw = float(np.linalg.norm(cw))
                if ncw > 1e-12:
                    cw = cw / ncw
                    wcos = np.where(dmask, Pw @ cw, 0.0).astype(np.float64)
                    scores = scores + (w_anchor_whitened * cont_scale) * (wcos - anchor_whitened_center)
                    mod_cos["dense_anchor_whitened"] = wcos
                    stats["anchor_whitened_applied_rows"] += 1

        reject_cos = {}
        if reject_means:
            for m, rm in reject_means.items():
                emb, mask, w, center = active_prior[m]
                raw = emb[cand_idxs] @ rm
                rcos = np.where(mask[cand_idxs], raw, 0.0).astype(np.float64)
                scores = scores - w_reject * rcos
                reject_cos[m] = rcos
            stats["reject_applied_rows"] += 1

        # --- query->track content terms (lyr, attr): 0.6B query, min-max pool ---
        qmod_sim01 = {}
        qmod_aff = {}
        if q06 is not None:
            for m, (emb, mask, w) in active_query.items():
                if m in heuristic_skip_mods:
                    continue
                aff = _qt_affinity(cat, QUERY_AFFINITY[m], qt_gate_floor)
                qmod_aff[m] = aff
                raw = emb[cand_idxs] @ q06
                raw = np.where(mask[cand_idxs], raw, -np.inf)
                finite = raw[np.isfinite(raw)]
                if finite.size == 0:
                    qmod_sim01[m] = np.zeros(len(cand_idxs), dtype=np.float64)
                    continue
                lo, hi = float(finite.min()), float(finite.max())
                s01 = np.where(np.isfinite(raw),
                               (raw - lo) / (hi - lo) if hi > lo else 0.0, 0.0)
                scores = scores + (w * dense_scale * aff) * s01
                qmod_sim01[m] = s01.astype(np.float64)
                if aff > 0:
                    stats[f"{m}_applied_rows"] += 1

        # --- relation-overlap scorer (shared artist / registrant / rare-tag w/ priors) ----
        # Discriminates AMONG dateless tracks (which release-date-weight cannot). Additive,
        # scaled to the score spread. rare-tag overlap weighted by count of shared rare tags.
        if (w_rel_artist != 0.0 or w_rel_registrant != 0.0 or w_rel_raretag != 0.0):
            finite = scores[np.isfinite(scores)]
            spread = (float(finite.max() - finite.min()) or 1.0) if finite.size else 1.0
            rel_bonus = np.zeros(len(cand_idxs), dtype=np.float64)
            for _j, _ci in enumerate(cand_idxs):
                _ci = int(_ci)
                if w_rel_artist != 0.0 and pool_artists:
                    if track_index.artist[_ci] in pool_artists:
                        rel_bonus[_j] += w_rel_artist
                if w_rel_registrant != 0.0 and pool_registrants and track_registrant is not None:
                    if _ci < len(track_registrant) and track_registrant[_ci] in pool_registrants:
                        rel_bonus[_j] += w_rel_registrant
                if w_rel_raretag != 0.0 and pool_raretags and track_raretags is not None:
                    if _ci < len(track_raretags) and track_raretags[_ci]:
                        _shared = len(track_raretags[_ci] & pool_raretags)
                        if _shared:
                            rel_bonus[_j] += w_rel_raretag * _shared
            if rel_bonus.any():
                scores = scores + rel_bonus * spread
                stats["rel_scorer_applied_rows"] = stats.get("rel_scorer_applied_rows", 0) + 1

        # --- release-date feature weight ----------------------------------------
        if release_date_weight != 1.0 and no_release_date_mask is not None:
            nd = np.asarray(no_release_date_mask)[cand_idxs].astype(bool)
            if nd.any():
                finite = scores[np.isfinite(scores)]
                if finite.size:
                    spread = float(finite.max() - finite.min()) or 1.0
                else:
                    spread = 1.0
                scores = scores + np.where(nd, (release_date_weight - 1.0) * spread, 0.0)


        order = np.argsort(-scores)
        sel_order = order[:top_k]
        top = [track_index.track_ids[int(cand_idxs[oi])] for oi in sel_order]

        if dump_provenance is not None:
            def _src(ci):
                t = []
                for nm in ("artist", "album", "decade", "qwen", "global_pop", "lyr_arm", "attr_arm", "anchor_pool", "audio_pool"):
                    if ci in sources.get(nm, set()):
                        t.append(nm)
                return "+".join(t) if t else "?"
            for slot, oi in enumerate(sel_order):
                oi = int(oi)
                ci = int(cand_idxs[oi])
                dense_c = float(w_dense_eff * sim01[oi]) if (w_dense_eff > 0 and sim01 is not None) else 0.0
                rec = {
                    "session_id": key[0], "user_id": key[1], "turn_number": key[2],
                    "branch": "heuristic", "slot": slot,
                    "track_id": track_index.track_ids[ci], "source": _src(ci),
                    "score_total": float(scores[oi]), "dense_contrib": dense_c,
                    "sim01": float(sim01[oi]) if sim01 is not None else float("nan"),
                    "cosine": float(qscores[ci]) if qscores is not None else float("nan"),
                    "md": float(md), "cat": cat, "spec": spec,
                    "n_pool": int(len(cand_idxs)),
                }
                for t in SCORE_TERMS:
                    rec[f"t_{t}"] = float(comp[t][oi]) if comp is not None else float("nan")
                for m in PRIOR_SIM_MODS:
                    if m in mod_cos:
                        _, _, w, center = active_prior[m]
                        rec[f"{m}_cos"] = float(mod_cos[m][oi])
                        rec[f"{m}_contrib"] = float(w * (mod_cos[m][oi] - center))
                    else:
                        rec[f"{m}_cos"] = float("nan"); rec[f"{m}_contrib"] = 0.0
                    if m in reject_cos:
                        w = active_prior[m][2]
                        rec[f"{m}_reject_contrib"] = float(-w_reject * reject_cos[m][oi])
                    else:
                        rec[f"{m}_reject_contrib"] = 0.0
                for m in QUERY_SIM_MODS:
                    if m in qmod_sim01:
                        w = active_query[m][2]
                        aff = qmod_aff.get(m, 0.0)
                        rec[f"{m}_sim01"] = float(qmod_sim01[m][oi])
                        rec[f"{m}_affinity"] = float(aff)
                        rec[f"{m}_contrib"] = float((w * aff) * qmod_sim01[m][oi])
                    else:
                        rec[f"{m}_sim01"] = float("nan"); rec[f"{m}_affinity"] = float("nan")
                        rec[f"{m}_contrib"] = 0.0
                prov_records.append(rec)

        if len(top) < top_k:
            stats["qwen_fill_shortfall_rows"] += 1
            filled = list(top)
            extra = take_fallback_recs(
                qwen_fallback, exclude=set(filled) | forbidden_track_ids,
                top_k=top_k - len(filled), context=f"fill-shortfall {key}")
            if dump_provenance is not None:
                nan = float("nan")
                for j, tid in enumerate(extra):
                    rec = {
                        "session_id": key[0], "user_id": key[1], "turn_number": key[2],
                        "branch": "fill", "slot": len(filled) + j, "track_id": tid,
                        "source": "fallback_dense", "score_total": nan, "dense_contrib": 0.0,
                        "sim01": nan, "cosine": nan, "md": float(md), "cat": cat, "spec": spec,
                        "n_pool": int(len(cand_idxs)),
                        **{f"t_{t}": nan for t in SCORE_TERMS}}
                    for m in PRIOR_SIM_MODS:
                        rec[f"{m}_contrib"] = 0.0; rec[f"{m}_cos"] = nan; rec[f"{m}_reject_contrib"] = 0.0
                    for m in QUERY_SIM_MODS:
                        rec[f"{m}_contrib"] = 0.0; rec[f"{m}_sim01"] = nan; rec[f"{m}_affinity"] = nan
                    prov_records.append(rec)
            filled.extend(extra)
            top = filled

        stats["heuristic_ranked_rows"] += 1
        recs_by_key[key] = top[:top_k]

    if stats["md_rows"]:
        stats["mean_move_away_factor"] = round(stats["md_sum"] / stats["md_rows"], 3)
    stats.pop("md_sum", None)

    if dump_pools is not None:
        Path(dump_pools).parent.mkdir(parents=True, exist_ok=True)
        Path(dump_pools).write_text(json.dumps(pool_records, indent=2))

    if dump_provenance is not None:
        Path(dump_provenance).parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(prov_records).write_parquet(dump_provenance)
        stats["provenance_rows"] = len(prov_records)

    return recs_by_key, stats