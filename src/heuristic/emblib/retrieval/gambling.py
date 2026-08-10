"""
The score per candidate track is:

    s = w_album_last  * 1[same album as last_played]
      + w_artist_last * 1[same artist as last_played]
      + w_album_any   * 1[in album-set of any prior track]
      + w_artist_any  * 1[in artist-set of any prior track]
      + w_year        * exp(-|year(c) - year(last)| / 5)
      + w_pop_match   * 1[same popularity tertile as last]
      + epsilon * popularity_z   (final tie-breaker)

After scoring, played tracks are masked, then we take top-20.

Output JSON has the schema enforced by BasePipeline.

Run for dev:
    uv run python -m scripts.launchers.generate_gambling_submission --split dev
Run for blind_a:
    uv run python -m scripts.launchers.generate_gambling_submission --split blind_a
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

from emblib.retrieval.paths import (
    BLIND_A_PATH,
    HEURISTIC_ONLY_BLIND_A_PATH,
    ROOT,
    TRACK_METADATA_PATH,
)

DEV_PATH = ROOT / "data/talkpl-ai/TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet"
OUT_DEV = ROOT / "exp/inference/devset/heuristics/heuristic_v2.json"


DEFAULT_WEIGHTS = {
    "album_last": 0.7,
    "artist_last": 3.8,
    "album_any": 2.8,
    "artist_any": 1.3,
    "year": 1.7,
    "pop_match": 0.15,
    "pop_z": 0.4,
}

# DEFAULT_WEIGHTS = {
#     "album_last": 4.17,
#     "artist_last": 4.46,
#     "album_any": 0.63,
#     "artist_any": 3.0,
#     "year": 0.16,
#     "pop_match": 1.11,
#     "pop_z": 0.37
#   }

# DEFAULT_WEIGHTS = {
#     "album_last": 2.55,
#     "artist_last": 0.45,
#     "album_any": 3.0,
#     "artist_any": 1.3,
#     "year": 1.0,
#     "pop_match": 0.7,
#     "pop_z": 0.1
#   }


# Canonical order of the seven heuristic terms (used by return_components and
# by the provenance / explain tooling so column order is stable everywhere).
SCORE_TERMS = (
    "album_last", "artist_last", "album_any", "artist_any",
    "year", "pop_match", "pop_z",
)

CAT_MULT = {
    "C": 1.20,
    "H": 1.15,
    "F": 1.15,
    "E": 1.10,
    "A": 1.10,
    "D": 1.10,
    "B": 1.00,
    "G": 1.00,
    "K": 0.85,
    "J": 0.75,
    "I": 0.65,
}

SPEC_MULT = {
    "HH": 1.20,
    "LH": 1.05,
    "HL": 1.00,
    "LL": 1.00,
    "": 1.00,
    "?": 1.00,
}


class TrackIndex:
    def __init__(self, track_meta_path: Path):
        print(f"Loading track metadata from {track_meta_path}")
        df = pl.read_parquet(track_meta_path)
        n = df.shape[0]
        self.track_ids: list[str] = df["track_id"].to_list()
        self.id_to_idx = {tid: i for i, tid in enumerate(self.track_ids)}

        self.artist = [None] * n
        self.album = [None] * n
        self.year = np.zeros(n, dtype=np.int32)
        self.popularity = np.zeros(n, dtype=np.float32)
        self.pop_bucket = np.full(n, -1, dtype=np.int8)

        for i, r in enumerate(df.iter_rows(named=True)):
            aid_list = r.get("artist_id") or []
            artist = aid_list[0] if aid_list else None
            self.artist[i] = artist

            album_raw = r.get("album_name")
            if isinstance(album_raw, (list, tuple)):
                album_raw = album_raw[0] if album_raw else None
            if album_raw:
                an = str(album_raw).strip()
                if an and an.lower() not in {"", "unknown", "none"}:
                    self.album[i] = f"{artist or ''}::{an}"

            rd = r.get("release_date") or ""
            try:
                if rd and len(str(rd)) >= 4:
                    self.year[i] = int(str(rd)[:4])
            except (TypeError, ValueError):
                pass

            pop = r.get("popularity")
            if pop is not None:
                self.popularity[i] = float(pop)

        valid_pop = self.popularity[self.popularity > 0]
        if len(valid_pop) > 0:
            q33 = float(np.percentile(valid_pop, 33))
            q67 = float(np.percentile(valid_pop, 67))
            for i in range(n):
                p = self.popularity[i]
                if p <= 0:
                    self.pop_bucket[i] = -1
                elif p < q33:
                    self.pop_bucket[i] = 0
                elif p < q67:
                    self.pop_bucket[i] = 1
                else:
                    self.pop_bucket[i] = 2

        if len(valid_pop) > 0:
            mu = float(valid_pop.mean())
            sigma = float(valid_pop.std()) or 1.0
            self.pop_z = (self.popularity - mu) / sigma
        else:
            self.pop_z = np.zeros(n, dtype=np.float32)

        self.artist_to_idxs: dict[str, np.ndarray] = defaultdict(list)
        self.album_to_idxs: dict[str, np.ndarray] = defaultdict(list)
        for i in range(n):
            if self.artist[i]:
                self.artist_to_idxs[self.artist[i]].append(i)
            if self.album[i]:
                self.album_to_idxs[self.album[i]].append(i)
        self.artist_to_idxs = {k: np.asarray(v, dtype=np.int64) for k, v in self.artist_to_idxs.items()}
        self.album_to_idxs = {k: np.asarray(v, dtype=np.int64) for k, v in self.album_to_idxs.items()}

        decade_buckets: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            if self.year[i] > 0:
                decade_buckets[(int(self.year[i]) // 10) * 10].append(i)
        self.decade_to_top_idxs: dict[int, np.ndarray] = {}
        for d, idxs in decade_buckets.items():
            arr = np.asarray(idxs, dtype=np.int64)
            order = np.argsort(-self.popularity[arr])
            self.decade_to_top_idxs[d] = arr[order]

        self.global_top_idxs = np.argsort(-self.popularity)

        print(
            f"  n_tracks={n} | n_artists={len(self.artist_to_idxs)} | "
            f"n_albums={len(self.album_to_idxs)} | n_decades={len(self.decade_to_top_idxs)}"
        )


def score_candidates(
    cand_idxs: np.ndarray,
    last_idx: int,
    prior_idxs: list[int],
    prior_artists: set,
    prior_albums: set,
    idx: TrackIndex,
    weights: dict,
    return_components: bool = False,
):
    """Seven-term additive score over cand_idxs.

    return_components=False (default): returns the float32 score vector `s`,
      BYTE-IDENTICAL to the original implementation (same per-term arrays added
      in the same order; inactive terms contribute exact float32 zeros, which
      are no-ops). Existing submissions are unaffected.

    return_components=True: returns (s, comp) where comp is a dict keyed by
      SCORE_TERMS, each value the full weighted contribution array for that term.
      By construction sum(comp.values()) == s (up to float rounding), so the
      breakdown matches exactly what ranked.
    """
    n = len(cand_idxs)
    if n == 0:
        empty = np.zeros(0, dtype=np.float32)
        if return_components:
            return empty, {t: empty.copy() for t in SCORE_TERMS}
        return empty

    last_artist = idx.artist[last_idx]
    last_album = idx.album[last_idx]
    last_year = idx.year[last_idx]
    last_pop_bkt = idx.pop_bucket[last_idx]

    cand_artist = np.asarray([idx.artist[c] for c in cand_idxs])
    cand_album = np.asarray([idx.album[c] for c in cand_idxs])
    cand_year = idx.year[cand_idxs]
    cand_pop_bkt = idx.pop_bucket[cand_idxs]

    # Per-term contributions (inactive -> float32 zeros). Computed once, used
    # both to accumulate s (identically to the original) and to expose comp.
    t_album_last = np.zeros(n, dtype=np.float32)
    t_artist_last = np.zeros(n, dtype=np.float32)
    t_album_any = np.zeros(n, dtype=np.float32)
    t_artist_any = np.zeros(n, dtype=np.float32)
    t_year = np.zeros(n, dtype=np.float32)
    t_pop_match = np.zeros(n, dtype=np.float32)

    if last_album is not None:
        t_album_last = weights["album_last"] * (cand_album == last_album).astype(np.float32)
    if last_artist is not None:
        t_artist_last = weights["artist_last"] * (cand_artist == last_artist).astype(np.float32)

    if prior_albums:
        t_album_any = weights["album_any"] * np.fromiter(
            (a in prior_albums for a in cand_album),
            dtype=np.float32,
            count=len(cand_album),
        )
    if prior_artists:
        t_artist_any = weights["artist_any"] * np.fromiter(
            (a in prior_artists for a in cand_artist),
            dtype=np.float32,
            count=len(cand_artist),
        )

    if last_year > 0:
        dy = np.abs(cand_year - last_year).astype(np.float32)
        valid = cand_year > 0
        year_score = np.where(valid, np.exp(-dy / 5.0), 0.0)
        t_year = weights["year"] * year_score

    if last_pop_bkt >= 0:
        t_pop_match = weights["pop_match"] * (cand_pop_bkt == last_pop_bkt).astype(np.float32)

    t_pop_z = weights["pop_z"] * idx.pop_z[cand_idxs]

    # Accumulate in the original order; adding the zero arrays for inactive
    # terms is an exact float32 no-op, so this matches the original byte-for-byte.
    s = np.zeros(n, dtype=np.float32)
    s += t_album_last
    s += t_artist_last
    s += t_album_any
    s += t_artist_any
    s += t_year
    s += t_pop_match
    s += t_pop_z

    if return_components:
        comp = {
            "album_last": t_album_last,
            "artist_last": t_artist_last,
            "album_any": t_album_any,
            "artist_any": t_artist_any,
            "year": t_year,
            "pop_match": t_pop_match,
            "pop_z": t_pop_z,
        }
        return s, comp
    return s


def get_candidate_pool(
    prior_idxs,
    prior_artists,
    prior_albums,
    idx,
    pool_cap: int = 5000,
    *,
    qwen_scores=None,
    qwen_k: int = 0,
    use_decade: bool = True,
    return_sources: bool = False,
    extra_sources: dict | None = None,

):
    """Candidate pool for the heuristic.

    Sources (unioned):
      artist  : all tracks by any prior artist
      album   : all tracks on any prior album
      decade  : top-100-pop tracks of the last track's decade +/-1 (if use_decade)
      qwen    : top-`qwen_k` of the dense query->track scores for THIS row
                (qwen_scores = a 1-D array over the WHOLE catalog; pass None to skip)
      global_pop : popularity fallback only when everything else is empty

    return_sources=True -> (pool_array, {source_name: set(idx)}); else just pool_array.
    """
    sources = {"artist": set(), "album": set(), "decade": set(),
               "qwen": set(), "global_pop": set()}

    for a in prior_artists:
        if a in idx.artist_to_idxs:
            sources["artist"].update(idx.artist_to_idxs[a].tolist())
    for al in prior_albums:
        if al in idx.album_to_idxs:
            sources["album"].update(idx.album_to_idxs[al].tolist())

    if use_decade and prior_idxs:
        last_year = idx.year[prior_idxs[-1]]
        if last_year > 0:
            decade = (int(last_year) // 10) * 10
            for d in (decade - 10, decade, decade + 10):
                if d in idx.decade_to_top_idxs:
                    sources["decade"].update(idx.decade_to_top_idxs[d][:100].tolist())

    if qwen_scores is not None and qwen_k > 0:
        k = min(int(qwen_k), int(qwen_scores.shape[0]))
        if k > 0:
            top = np.argpartition(-qwen_scores, k - 1)[:k]
            sources["qwen"].update(int(j) for j in top)

    if extra_sources:
        for _name, _idxs in extra_sources.items():
            if _idxs is not None and len(_idxs):
                sources.setdefault(_name, set()).update(int(j) for j in _idxs)
    pool = set()
    for v in sources.values():
        pool |= v

    if not pool:
        gp = idx.global_top_idxs[:pool_cap].tolist()
        sources["global_pop"].update(gp)
        pool = set(gp)

    arr = np.asarray(sorted(pool), dtype=np.int64)
    return (arr, sources) if return_sources else arr



def _turn_one_fallback_idxs(
    idx: TrackIndex,
    k: int,
    exclude: set | None = None,
) -> np.ndarray:
    exclude = exclude or set()
    out: list[int] = []
    for decade in (2010, 2000, 1990, 2020, 1980, 1970):
        arr = idx.decade_to_top_idxs.get(decade)
        if arr is None:
            continue
        for i in arr.tolist():
            if i not in exclude:
                out.append(i)
                exclude.add(i)
                if len(out) >= k:
                    return np.asarray(out, dtype=np.int64)
    for i in idx.global_top_idxs.tolist():
        if i not in exclude:
            out.append(i)
            exclude.add(i)
            if len(out) >= k:
                break
    return np.asarray(out, dtype=np.int64)


def _turn_one_fallback(idx: TrackIndex, k: int, exclude: set | None = None) -> list[str]:
    arr = _turn_one_fallback_idxs(idx, k, exclude)
    return [idx.track_ids[i] for i in arr]


def predict_with_history(
    prior_track_ids: list[str],
    cat: str,
    spec: str,
    idx: TrackIndex,
    top_k: int = 20,
) -> list[str]:
    prior_idxs = [idx.id_to_idx[t] for t in prior_track_ids if t in idx.id_to_idx]

    if not prior_idxs:
        return _turn_one_fallback(idx, top_k)

    weights = dict(DEFAULT_WEIGHTS)
    cm = CAT_MULT.get(cat, 1.0)
    sm = SPEC_MULT.get(spec, 1.0)
    boost = cm * sm
    for k in ("album_last", "artist_last", "album_any", "artist_any"):
        weights[k] *= boost

    last_idx = prior_idxs[-1]
    prior_artists = {idx.artist[i] for i in prior_idxs if idx.artist[i] is not None}
    prior_albums = {idx.album[i] for i in prior_idxs if idx.album[i] is not None}

    cand_idxs = get_candidate_pool(prior_idxs, prior_artists, prior_albums, idx)

    forbidden = set(prior_idxs)
    keep_mask = np.fromiter((c not in forbidden for c in cand_idxs), dtype=bool, count=len(cand_idxs))
    cand_idxs = cand_idxs[keep_mask]

    if len(cand_idxs) == 0:
        return _turn_one_fallback(idx, top_k, exclude=forbidden)

    s = score_candidates(cand_idxs, last_idx, prior_idxs, prior_artists, prior_albums, idx, weights)
    order = np.argsort(-s)
    top = cand_idxs[order[:top_k]]

    if len(top) < top_k:
        seen = set(top.tolist()) | forbidden
        for fb in _turn_one_fallback_idxs(idx, top_k * 3, exclude=set(seen)):
            if len(top) >= top_k:
                break
            if fb not in seen:
                top = np.append(top, fb)
                seen.add(fb)

    return [idx.track_ids[i] for i in top[:top_k]]


def predict_dev(idx: TrackIndex, parquet_path: Path) -> list[dict]:
    print(f"\nLoading dev sessions from {parquet_path}")
    df = pl.read_parquet(parquet_path).to_dicts()
    print(f"  {len(df)} sessions")

    out: list[dict] = []
    for sess in df:
        sid = sess["session_id"]
        uid = sess["user_id"]
        cg = sess.get("conversation_goal") or {}
        cat = cg.get("category") or "?"
        spec = cg.get("specificity") or "?"

        convs = sorted(sess["conversations"], key=lambda t: t.get("turn_number", 0))
        user_turns = sorted({t["turn_number"] for t in convs if t.get("role") == "user"})

        for target in user_turns:
            prior_tids: list[str] = []
            for t in convs:
                if t.get("turn_number", 0) >= target:
                    break
                if t.get("role") == "music":
                    c = t.get("content")
                    if c:
                        prior_tids.append(str(c).strip())

            preds = predict_with_history(prior_tids, cat, spec, idx, top_k=20)
            out.append(
                {
                    "session_id": str(sid),
                    "user_id": str(uid),
                    "turn_number": int(target),
                    "predicted_track_ids": preds,
                    "predicted_response": "",
                }
            )
    print(f"  generated {len(out)} predictions")
    return out


def predict_blind(idx: TrackIndex, parquet_path: Path) -> list[dict]:
    print(f"\nLoading blind sessions from {parquet_path}")
    df = pl.read_parquet(parquet_path).to_dicts()
    print(f"  {len(df)} sessions")

    out: list[dict] = []
    for sess in df:
        sid = sess["session_id"]
        uid = sess["user_id"]
        cg = sess.get("conversation_goal") or {}
        cat = cg.get("category") or "?"
        spec = cg.get("specificity") or "?"

        convs = sorted(sess["conversations"], key=lambda t: t.get("turn_number", 0))
        if not convs:
            continue

        prior_tids = [str(t.get("content")).strip() for t in convs if t.get("role") == "music" and t.get("content")]
        target_turn = int(convs[-1].get("turn_number", 1))

        preds = predict_with_history(prior_tids, cat, spec, idx, top_k=20)
        out.append(
            {
                "session_id": str(sid),
                "user_id": str(uid),
                "turn_number": target_turn,
                "predicted_track_ids": preds,
                "predicted_response": "",
            }
        )
    print(f"  generated {len(out)} predictions")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=["dev", "blind_a", "both"])
    args = p.parse_args()

    idx = TrackIndex(TRACK_METADATA_PATH)
    splits = ["dev", "blind_a"] if args.split == "both" else [args.split]

    for split in splits:
        if split == "dev":
            preds = predict_dev(idx, DEV_PATH)
            out_path = OUT_DEV
        else:
            preds = predict_blind(idx, BLIND_A_PATH)
            out_path = HEURISTIC_ONLY_BLIND_A_PATH

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(preds, f, indent=2, ensure_ascii=False)
        print(f"\n[{split}] wrote {len(preds)} predictions to {out_path}")


if __name__ == "__main__":
    main()