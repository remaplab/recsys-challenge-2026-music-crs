"""Mine hard negatives for every training row, store offline.

──────────────────────────────────────────────────────────────────────────────
UPDATED FOR THE QWEN3-NATIVE PIPELINE
──────────────────────────────────────────────────────────────────────────────
Negatives are now mined in the SAME space the encoder is trained/evaluated in:
the `metadata-qwen3-native` track tower built by scripts/04b_encode_qwen_tracks.py
(our own Qwen3 last-token encoding of the track-metadata text). Query embeddings
are read from the new per-encoder cache written by scripts/03_encode_queries.py:

    models/query_emb_cache/<encoder>/<split>.npy
    models/query_emb_cache/<encoder>/<split>_meta.parquet

The default `--encoder` is `qwen3_native_frozen`, the frozen query mirror of that
tower — so the general top-pool bucket (#9) is scored with exactly the geometry
the LoRA encoder will later be trained against.

Output indices address the canonical track order (organizer shards), so they
stay valid across any tower with the same ordering.

Phase 13 update: per-row negative pool is nine buckets, prioritized to exploit
structural priors (Section 3 transition analysis on the train split):

  global transition rates that drive the budget:
    same-artist          : 0.575
    same-album           : 0.333
    same-popularity-bkt  : 0.707   ← orthogonal to artist/album
    within-5y release    : 0.779   ← orthogonal to artist/album
    same-primary-tag     : 0.163   (DROPPED — too noisy, median pool size 1)

  per-row mining buckets (in priority order):

  1. SAME-SESSION (n_session_neg, default 4) — the row's own
     `prior_track_ids`. These tracks "survived" a near-identical conversational
     context and were the assistant's PRIOR answers in this session, but they
     are NOT the goal-relevant track at this turn. Free, strongest negatives.

  2. SAME-ALBUM-AS-PRIOR (n_prior_album_neg, default 4) — tracks sharing an
     album with one of prior_track_ids. Attacks the same-album transition
     prior; what the heuristic baseline ranks high.

  3. SAME-ARTIST-AS-PRIOR (n_prior_artist_neg, default 4) — tracks sharing an
     artist with one of prior_track_ids. Attacks the same-artist transition
     prior. Reduced from the analyzer's recommendation of 7 to 4 because the
     prior pool is already the union over the entire history (median size
     57); 7 would double-count.

  4. SAME-ALBUM-AS-GT (n_album_neg, default 4) — within-album discrimination.
     Median pool 7 supports this; this is the hardest discrimination signal
     in the data.

  5. SAME-ARTIST-AS-GT (n_artist_neg, default 3) — partly redundant with
     bucket 3, but keeps the gradient sharp on cross-album / same-artist
     pairs that the prior union may not surface.

  6. SAME-POPULARITY-BUCKET (n_pop_bucket_neg, default 3) — orthogonal prior
     at 71% transition rate. Coarse 3-way split (low/mid/high) keyed by track
     popularity tertiles in the catalog.

  7. SAME-YEAR-±5 (n_year_5y_neg, default 2) — orthogonal prior at 78%
     transition rate. Pool is large (median 1656), so each sample is a
     weaker signal — keep budget low.

  8. SAME-CATEGORY-BY-LIFT (n_cat_lift_neg, default 2) — tracks whose tags
     have high token-tag lift for the row's `category` according to the
     Phase 7 CSV.

  9. GENERAL TOP-POOL (fills remaining n_negatives slots) — top-N by frozen
     metadata-qwen3-native score, minus the forbidden set. Diversity floor.

Output: models/hard_negatives/<split>_negs.npy of shape (N, K) of int32 track
indices, plus <split>_gt.npy. -1 fills unused slots when a bucket is empty.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

from emblib.tracks.qwen_track_loader import load_qwen_track_tower, QWEN_MODALITY


DATA = Path("./data/talkpl-ai")
QEMB = Path("./models/query_emb_cache")
QWEN_TOWER_CACHE = Path("./models/track_tower_qwen_cache")
OUT_DIR = Path("./models/hard_negatives")

MINING_MODALITY = QWEN_MODALITY


def build_track_index(track_meta_path: Path, tower_track_ids: list[str]):
    """Build per-track artist, album, year, popularity-bucket, and tag indices.

    Returns:
      track_to_artist:    (N,) int64, -1 if missing
      track_to_album:     (N,) int64, -1 if missing
      track_to_year:      (N,) int32, -1 if missing
      track_to_pop_bkt:   (N,) int8, 0=low, 1=mid, 2=high, -1 if missing
      artist_to_tracks:   dict[artist_id_int -> np.ndarray of track indices]
      album_to_tracks:    dict[album_id_int -> np.ndarray of track indices]
      year_to_tracks:     dict[year_int -> np.ndarray of track indices]
      pop_bkt_to_tracks:  dict[bucket_int -> np.ndarray of track indices]
      track_tags_full:    list[list[str]] of all (lowercased, stripped) tags
                           per track, used by the cat-lift ranking.
    """
    print(f"Loading track metadata from {track_meta_path}")
    df = pl.read_parquet(
        track_meta_path,
        columns=["track_id", "artist_id", "album_name", "tag_list",
                 "release_date", "popularity"],
    )

    id_to_idx = {str(tid): i for i, tid in enumerate(tower_track_ids)}
    n = len(tower_track_ids)

    track_to_artist = np.full(n, -1, dtype=np.int64)
    track_to_album = np.full(n, -1, dtype=np.int64)
    track_to_year = np.full(n, -1, dtype=np.int32)
    track_to_pop_raw = np.full(n, np.nan, dtype=np.float32)
    track_tags_full: list[list[str]] = [[] for _ in range(n)]

    artist_to_idx: dict[str, int] = {}
    album_to_idx: dict[str, int] = {}

    def _parse_year(rd) -> int:
        if not rd:
            return -1
        try:
            return int(str(rd)[:4])
        except (TypeError, ValueError):
            return -1

    for row in df.iter_rows(named=True):
        tid = str(row["track_id"])
        if tid not in id_to_idx:
            continue
        ti = id_to_idx[tid]

        aid_list = row.get("artist_id") or []
        if aid_list:
            aid = aid_list[0]
            if aid not in artist_to_idx:
                artist_to_idx[aid] = len(artist_to_idx)
            track_to_artist[ti] = artist_to_idx[aid]

        # Album: parquet exposes album_name as a string (possibly list-wrapped).
        # Use it as a string key. Empty / "Unknown" treated as missing.
        album_raw = row.get("album_name")
        if isinstance(album_raw, (list, tuple)):
            album_raw = album_raw[0] if album_raw else None
        if album_raw:
            album_name = str(album_raw).strip()
            if album_name and album_name.lower() not in {"", "unknown", "none"}:
                # Disambiguate by (artist, album) so two artists with an
                # album called "Greatest Hits" don't collapse into one bucket.
                artist_key = aid_list[0] if aid_list else ""
                album_key = f"{artist_key}::{album_name}"
                if album_key not in album_to_idx:
                    album_to_idx[album_key] = len(album_to_idx)
                track_to_album[ti] = album_to_idx[album_key]

        yr = _parse_year(row.get("release_date"))
        if yr > 0:
            track_to_year[ti] = yr

        pop = row.get("popularity")
        if pop is not None:
            track_to_pop_raw[ti] = float(pop)

        tags = row.get("tag_list") or []
        cleaned = [
            t.strip().lower() for t in tags
            if t and len(t.strip()) >= 2 and not t.strip().lower().endswith("of 10 stars")
        ]
        track_tags_full[ti] = cleaned

    # Popularity tertiles: bucket the catalog by the 33rd / 67th percentiles
    # of *known* popularity values, then assign each track to {low, mid, high}.
    track_to_pop_bkt = np.full(n, -1, dtype=np.int8)
    pop_known = ~np.isnan(track_to_pop_raw)
    if pop_known.sum() > 0:
        pop_vals = track_to_pop_raw[pop_known]
        q33 = float(np.percentile(pop_vals, 33))
        q67 = float(np.percentile(pop_vals, 67))
        # 0=low, 1=mid, 2=high
        bkt = np.zeros(n, dtype=np.int8)
        bkt[track_to_pop_raw < q33] = 0
        bkt[(track_to_pop_raw >= q33) & (track_to_pop_raw < q67)] = 1
        bkt[track_to_pop_raw >= q67] = 2
        bkt[~pop_known] = -1
        track_to_pop_bkt = bkt
        print(f"  popularity tertiles: low<{q33:.2f}  mid<{q67:.2f}  high≥{q67:.2f}")
    else:
        print("  [warn] no popularity values found — pop-bucket bucket disabled.")

    print("  building inverted indices...")
    artist_to_tracks: dict[int, list[int]] = {}
    album_to_tracks: dict[int, list[int]] = {}
    year_to_tracks: dict[int, list[int]] = {}
    pop_bkt_to_tracks: dict[int, list[int]] = {}
    for ti in range(n):
        a = track_to_artist[ti]
        if a >= 0:
            artist_to_tracks.setdefault(int(a), []).append(ti)
        al = track_to_album[ti]
        if al >= 0:
            album_to_tracks.setdefault(int(al), []).append(ti)
        yr = int(track_to_year[ti])
        if yr > 0:
            year_to_tracks.setdefault(yr, []).append(ti)
        pb = int(track_to_pop_bkt[ti])
        if pb >= 0:
            pop_bkt_to_tracks.setdefault(pb, []).append(ti)
    artist_to_tracks = {k: np.asarray(v, dtype=np.int64) for k, v in artist_to_tracks.items()}
    album_to_tracks = {k: np.asarray(v, dtype=np.int64) for k, v in album_to_tracks.items()}
    year_to_tracks = {k: np.asarray(v, dtype=np.int64) for k, v in year_to_tracks.items()}
    pop_bkt_to_tracks = {k: np.asarray(v, dtype=np.int64) for k, v in pop_bkt_to_tracks.items()}

    print(f"  {len(artist_to_idx)} unique artists, "
          f"{len(album_to_idx)} unique (artist, album) pairs")
    print(f"  {len(year_to_tracks)} distinct release years")
    print(f"  {len(pop_bkt_to_tracks)} popularity buckets active")
    n_with_album = int((track_to_album >= 0).sum())
    n_with_year = int((track_to_year > 0).sum())
    n_with_pop = int((track_to_pop_bkt >= 0).sum())
    print(f"  coverage: album={n_with_album}/{n} ({n_with_album/n:.3f})  "
          f"year={n_with_year}/{n} ({n_with_year/n:.3f})  "
          f"pop={n_with_pop}/{n} ({n_with_pop/n:.3f})")
    return (track_to_artist, track_to_album, track_to_year, track_to_pop_bkt,
            artist_to_tracks, album_to_tracks, year_to_tracks, pop_bkt_to_tracks,
            track_tags_full)


def build_category_lift_ranking(
    token_tag_csv: Path,
    track_tags_full: list[list[str]],
    pool_size: int = 2000,
    lift_clip: float = 20.0,
) -> dict[str, np.ndarray]:
    """For each category, rank tracks by sum-of-(max-token) lift over their tags.

    Score(track t, cat C) = sum_{tag in tags(t)} max_{token} lift[(C, token)][tag]

    Higher score = "this track's tags look like queries from category C
    typically retrieve". Returns dict[cat_str -> np.ndarray of top `pool_size`
    track indices, descending by score].
    """
    if not token_tag_csv.exists():
        print(f"  [warn] {token_tag_csv} missing — cat-lift bucket disabled.")
        return {}

    df = pl.read_csv(token_tag_csv)
    print(f"  loaded {df.shape[0]} (category, token, tag, lift) rows")

    # Build cat -> {tag -> max_lift_over_tokens}
    cat_tag_lift: dict[str, dict[str, float]] = defaultdict(dict)
    for r in df.iter_rows(named=True):
        cat = r["category"]
        tag = (r["tag"] or "").strip().lower()
        try:
            lift = float(r["lift"])
        except (TypeError, ValueError):
            continue
        if not cat or not tag:
            continue
        lift = min(lift, lift_clip)
        if lift > cat_tag_lift[cat].get(tag, 0.0):
            cat_tag_lift[cat][tag] = lift

    n = len(track_tags_full)
    out: dict[str, np.ndarray] = {}
    for cat, tag_lifts in cat_tag_lift.items():
        scores = np.zeros(n, dtype=np.float32)
        for ti, tags in enumerate(track_tags_full):
            if not tags:
                continue
            s = 0.0
            for tag in tags:
                v = tag_lifts.get(tag)
                if v is not None:
                    s += v
            scores[ti] = s
        # We only need top `pool_size`. Tracks with score 0 (no relevant tags)
        # never fire as cat-lift negatives — that's the desired behaviour.
        nz = int((scores > 0).sum())
        if nz == 0:
            continue
        top_n = min(pool_size, nz)
        ranked = np.argpartition(-scores, top_n - 1)[:top_n]
        ranked = ranked[np.argsort(-scores[ranked])]
        out[cat] = ranked.astype(np.int64)
        print(f"    cat {cat}: {nz} tracks with lift>0, kept top {len(ranked)}")
    return out


def mine_split(
    split_name: str,
    encoder: str,
    n_negatives: int,
    n_top_pool: int,
    n_session_neg: int,
    n_prior_album_neg: int,
    n_prior_artist_neg: int,
    n_album_neg: int,
    n_artist_neg: int,
    n_pop_bucket_neg: int,
    n_year_5y_neg: int,
    year_window: int,
    n_cat_lift_neg: int,
    token_tag_csv: Path | None,
    track_meta_path: Path,
    seed: int,
):
    print(f"\n=== Mining {split_name} (encoder={encoder}) ===")

    print("Loading Qwen3-native TrackTower...")
    tower = load_qwen_track_tower(QWEN_TOWER_CACHE, MINING_MODALITY)
    track_emb = tower.embeddings[MINING_MODALITY]
    track_mask = tower.masks[MINING_MODALITY]
    n_tracks = tower.n_tracks
    valid_indices = np.where(track_mask)[0]

    (track_to_artist, track_to_album, track_to_year, track_to_pop_bkt,
     artist_to_tracks, album_to_tracks, year_to_tracks, pop_bkt_to_tracks,
     track_tags_full) = build_track_index(track_meta_path, tower.track_ids)

    cat_lift_ranked: dict[str, np.ndarray] = {}
    if token_tag_csv is not None and n_cat_lift_neg > 0:
        print("Building category-lift ranking...")
        cat_lift_ranked = build_category_lift_ranking(
            token_tag_csv, track_tags_full, pool_size=2000,
        )

    enc_dir = QEMB / encoder
    q_path = enc_dir / f"{split_name}.npy"
    meta_path = enc_dir / f"{split_name}_meta.parquet"
    if not q_path.exists():
        raise FileNotFoundError(
            f"{q_path} not found. Run "
            f"`python scripts/03_encode_queries.py --encoder {encoder} --splits {split_name}` first."
        )
    queries = np.load(q_path)
    meta = pl.read_parquet(meta_path)
    print(f"  queries: {queries.shape}, meta rows: {meta.shape[0]}")

    rng = np.random.default_rng(seed)

    n_rows = meta.shape[0]
    hard_negs = np.full((n_rows, n_negatives), -1, dtype=np.int32)
    gt_indices = np.full(n_rows, -1, dtype=np.int64)

    n_skipped_no_gt = 0
    n_session_filled = 0
    n_prior_album_filled = 0
    n_prior_artist_filled = 0
    n_album_filled = 0
    n_artist_filled = 0
    n_pop_bucket_filled = 0
    n_year_5y_filled = 0
    n_cat_lift_filled = 0
    n_general_filled = 0

    chunk = 512
    rows = meta.to_dicts()
    for start in tqdm(range(0, n_rows, chunk), desc=f"Mining {split_name}"):
        end = min(start + chunk, n_rows)
        batch_rows = rows[start:end]
        q_batch = queries[start:end]

        scores = q_batch @ track_emb.T
        scores[:, ~track_mask] = -np.inf

        for bi, r in enumerate(batch_rows):
            row_idx = start + bi
            gt_id = r.get("gt_track_id")
            if gt_id is None or gt_id not in tower.id_to_idx:
                n_skipped_no_gt += 1
                continue
            gt_idx = tower.id_to_idx[gt_id]
            gt_indices[row_idx] = gt_idx

            forbidden = {gt_idx}  # only GT is unconditionally forbidden
            row_negs: list[int] = []

            # Resolve prior-track tower indices once; reused by buckets 1, 2, 3
            prior_idxs: list[int] = []
            for tid in (r.get("prior_track_ids") or []):
                if tid in tower.id_to_idx:
                    idx = tower.id_to_idx[tid]
                    if track_mask[idx]:
                        prior_idxs.append(idx)

            # 1) SAME-SESSION priors (free, strongest negatives)
            if n_session_neg > 0 and prior_idxs:
                avail = [i for i in prior_idxs if i not in forbidden]
                if avail:
                    pick = rng.choice(
                        avail,
                        size=min(n_session_neg, len(avail)),
                        replace=False,
                    )
                    row_negs.extend(int(x) for x in pick.tolist())
                    forbidden.update(int(x) for x in pick.tolist())
                    n_session_filled += len(pick)

            # 2) SAME-ALBUM-AS-PRIOR
            if n_prior_album_neg > 0 and prior_idxs:
                prior_albums = set()
                for pi in prior_idxs:
                    pa = track_to_album[pi]
                    if pa >= 0:
                        prior_albums.add(int(pa))
                if prior_albums:
                    cand_pool: list[int] = []
                    for pa in prior_albums:
                        cand_pool.extend(album_to_tracks.get(pa, np.array([], dtype=np.int64)).tolist())
                    if cand_pool:
                        cand = np.fromiter(
                            (t for t in cand_pool if t not in forbidden),
                            dtype=np.int64,
                        )
                        if len(cand) > 0:
                            pick = rng.choice(
                                cand,
                                size=min(n_prior_album_neg, len(cand)),
                                replace=False,
                            )
                            row_negs.extend(int(x) for x in pick.tolist())
                            forbidden.update(int(x) for x in pick.tolist())
                            n_prior_album_filled += len(pick)

            # 3) SAME-ARTIST-AS-PRIOR
            if n_prior_artist_neg > 0 and prior_idxs:
                prior_artists = set()
                for pi in prior_idxs:
                    pa = track_to_artist[pi]
                    if pa >= 0:
                        prior_artists.add(int(pa))
                if prior_artists:
                    cand_pool = []
                    for pa in prior_artists:
                        cand_pool.extend(artist_to_tracks.get(pa, np.array([], dtype=np.int64)).tolist())
                    if cand_pool:
                        cand = np.fromiter(
                            (t for t in cand_pool if t not in forbidden),
                            dtype=np.int64,
                        )
                        if len(cand) > 0:
                            pick = rng.choice(
                                cand,
                                size=min(n_prior_artist_neg, len(cand)),
                                replace=False,
                            )
                            row_negs.extend(int(x) for x in pick.tolist())
                            forbidden.update(int(x) for x in pick.tolist())
                            n_prior_artist_filled += len(pick)

            # 4) SAME-ALBUM-AS-GT
            if n_album_neg > 0:
                al = track_to_album[gt_idx]
                if al >= 0 and al in album_to_tracks:
                    cand = album_to_tracks[int(al)]
                    cand = cand[~np.isin(cand, list(forbidden))]
                    if len(cand) > 0:
                        pick = rng.choice(
                            cand,
                            size=min(n_album_neg, len(cand)),
                            replace=False,
                        )
                        row_negs.extend(int(x) for x in pick.tolist())
                        forbidden.update(int(x) for x in pick.tolist())
                        n_album_filled += len(pick)

            # 5) SAME-ARTIST-AS-GT
            if n_artist_neg > 0:
                a = track_to_artist[gt_idx]
                if a >= 0 and a in artist_to_tracks:
                    cand = artist_to_tracks[int(a)]
                    cand = cand[~np.isin(cand, list(forbidden))]
                    if len(cand) > 0:
                        pick = rng.choice(
                            cand,
                            size=min(n_artist_neg, len(cand)),
                            replace=False,
                        )
                        row_negs.extend(int(x) for x in pick.tolist())
                        forbidden.update(int(x) for x in pick.tolist())
                        n_artist_filled += len(pick)

            # 6) SAME-POPULARITY-BUCKET — orthogonal prior at 71% transition
            # rate. Bucket is coarse (low/mid/high), so the pool is huge —
            # we limit sampling to a 500-track random window for speed.
            if n_pop_bucket_neg > 0:
                pb = int(track_to_pop_bkt[gt_idx])
                if pb >= 0 and pb in pop_bkt_to_tracks:
                    cand = pop_bkt_to_tracks[pb]
                    if len(cand) > 500:
                        cand = rng.choice(cand, size=500, replace=False)
                    cand = cand[~np.isin(cand, list(forbidden))]
                    if len(cand) > 0:
                        pick = rng.choice(
                            cand,
                            size=min(n_pop_bucket_neg, len(cand)),
                            replace=False,
                        )
                        row_negs.extend(int(x) for x in pick.tolist())
                        forbidden.update(int(x) for x in pick.tolist())
                        n_pop_bucket_filled += len(pick)

            # 7) SAME-YEAR-±W — orthogonal prior at 78% transition rate for
            # window=5. Pool is large (median 1656 for window=0); cap at
            # 500 random samples to keep mining fast.
            if n_year_5y_neg > 0:
                yr = int(track_to_year[gt_idx])
                if yr > 0:
                    cand_pool: list[int] = []
                    for dy in range(-year_window, year_window + 1):
                        cand_pool.extend(
                            year_to_tracks.get(yr + dy, np.array([], dtype=np.int64)).tolist()
                        )
                    if cand_pool:
                        cand = np.asarray(cand_pool, dtype=np.int64)
                        if len(cand) > 500:
                            cand = rng.choice(cand, size=500, replace=False)
                        cand = cand[~np.isin(cand, list(forbidden))]
                        if len(cand) > 0:
                            pick = rng.choice(
                                cand,
                                size=min(n_year_5y_neg, len(cand)),
                                replace=False,
                            )
                            row_negs.extend(int(x) for x in pick.tolist())
                            forbidden.update(int(x) for x in pick.tolist())
                            n_year_5y_filled += len(pick)

            # 8) SAME-CATEGORY-BY-LIFT
            if n_cat_lift_neg > 0:
                cat = r.get("category") or ""
                if cat in cat_lift_ranked:
                    pool = cat_lift_ranked[cat]
                    cand = pool[~np.isin(pool, list(forbidden))]
                    if len(cand) > 0:
                        head = cand[: min(len(cand), 200)]  # sample from top-200
                        pick = rng.choice(
                            head,
                            size=min(n_cat_lift_neg, len(head)),
                            replace=False,
                        )
                        row_negs.extend(int(x) for x in pick.tolist())
                        forbidden.update(int(x) for x in pick.tolist())
                        n_cat_lift_filled += len(pick)

            # 9) GENERAL TOP-POOL — diversity floor
            row_scores = scores[bi]
            pool_size = min(n_top_pool + len(forbidden) + 50, n_tracks)
            top_pool_idx = np.argpartition(-row_scores, pool_size - 1)[:pool_size]
            top_pool_idx = top_pool_idx[~np.isin(top_pool_idx, list(forbidden))]
            need = n_negatives - len(row_negs)
            if need > 0 and len(top_pool_idx) > 0:
                k = min(need, len(top_pool_idx))
                pick = rng.choice(top_pool_idx, size=k, replace=False)
                row_negs.extend(int(x) for x in pick.tolist())
                forbidden.update(int(x) for x in pick.tolist())
                n_general_filled += len(pick)

            # 10) Random pad if still short
            need = n_negatives - len(row_negs)
            if need > 0:
                cand = valid_indices[~np.isin(valid_indices, list(forbidden))]
                pick = rng.choice(cand, size=min(need, len(cand)), replace=False)
                row_negs.extend(int(x) for x in pick.tolist())

            hard_negs[row_idx, : len(row_negs)] = row_negs[:n_negatives]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"{split_name}_negs.npy", hard_negs)
    np.save(OUT_DIR / f"{split_name}_gt.npy", gt_indices)
    print(f"  saved {OUT_DIR / f'{split_name}_negs.npy'} shape={hard_negs.shape}")
    print(f"  saved {OUT_DIR / f'{split_name}_gt.npy'} shape={gt_indices.shape}")
    print(f"  rows skipped (no gt): {n_skipped_no_gt}")
    print(f"  fills:")
    print(f"    same-session         : {n_session_filled}")
    print(f"    same-album-as-prior  : {n_prior_album_filled}")
    print(f"    same-artist-as-prior : {n_prior_artist_filled}")
    print(f"    same-album-as-gt     : {n_album_filled}")
    print(f"    same-artist-as-gt    : {n_artist_filled}")
    print(f"    same-pop-bucket      : {n_pop_bucket_filled}")
    print(f"    same-year-±{year_window}        : {n_year_5y_filled}")
    print(f"    same-category-lift   : {n_cat_lift_filled}")
    print(f"    general top-pool     : {n_general_filled}")
    valid_rows = (hard_negs[:, 0] != -1).sum()
    print(f"  rows with negatives: {valid_rows}/{n_rows}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", type=str, default="qwen3_native_frozen",
                   help="Query-embedding cache to mine against, under "
                        "models/query_emb_cache/<encoder>/. Must be a 1024-d "
                        "encoder matching the Qwen3-native tower "
                        "(default: qwen3_native_frozen).")
    p.add_argument("--n-negatives", type=int, default=32,
                   help="Total negatives per row (size of K).")
    # Bucket budgets — defaults match the post-analysis recommendation.
    p.add_argument("--n-session-neg", type=int, default=4,
                   help="Same-session priors (the row's own prior_track_ids). "
                        "Free, strongest negatives.")
    p.add_argument("--n-prior-album-neg", type=int, default=4,
                   help="Tracks sharing an album with a prior_track_ids entry. "
                        "Targets the 33%% same-album transition prior.")
    p.add_argument("--n-prior-artist-neg", type=int, default=4,
                   help="Tracks sharing an artist with a prior_track_ids entry. "
                        "Targets the 57%% same-artist transition prior. Held "
                        "at 4 (not 7 as raw rate would suggest) because the "
                        "prior pool is the union over the entire history.")
    p.add_argument("--n-album-neg", type=int, default=4,
                   help="Same-album-as-GT (within-album discrimination). "
                        "Median pool 7 supports up to ~6.")
    p.add_argument("--n-artist-neg", type=int, default=3,
                   help="Same-artist-as-GT. Partly redundant with "
                        "same-artist-as-prior; kept for cross-album signal.")
    p.add_argument("--n-pop-bucket-neg", type=int, default=3,
                   help="Same popularity tertile as GT. Orthogonal prior "
                        "at 71%% transition rate.")
    p.add_argument("--n-year-5y-neg", type=int, default=2,
                   help="Within --year-window years of GT. Orthogonal "
                        "prior at 78%% transition rate (window=5). Lower "
                        "budget than the rate suggests because the pool is "
                        "huge (median 1656), so each sample is weaker signal.")
    p.add_argument("--year-window", type=int, default=5,
                   help="Half-window for same-year bucket: ±W years.")
    p.add_argument("--n-cat-lift-neg", type=int, default=2,
                   help="Same-category-by-lift tracks. Requires --token-tag-csv.")
    p.add_argument("--n-top-pool", type=int, default=300,
                   help="Pool size for general hard negatives.")
    p.add_argument("--token-tag-csv", type=Path,
                   default=Path("analysis/output/phase07_tags_lexical/token_tag_associations.csv"),
                   help="Phase 7 token-tag-lift CSV. If missing, cat-lift "
                        "bucket is silently skipped and those slots fall "
                        "through to the general pool.")
    p.add_argument("--track-meta", type=Path,
                   default=DATA / "TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--splits", nargs="+", default=["train"])
    args = p.parse_args()

    curated_total = (
        args.n_session_neg + args.n_prior_album_neg + args.n_prior_artist_neg
        + args.n_album_neg + args.n_artist_neg + args.n_pop_bucket_neg
        + args.n_year_5y_neg + args.n_cat_lift_neg
    )
    print(f"Mining config: encoder={args.encoder}, K={args.n_negatives}, "
          f"top-pool={args.n_top_pool}")
    print(f"  session={args.n_session_neg}, "
          f"prior-album={args.n_prior_album_neg}, "
          f"prior-artist={args.n_prior_artist_neg},")
    print(f"  album-as-gt={args.n_album_neg}, "
          f"artist-as-gt={args.n_artist_neg},")
    print(f"  pop-bucket={args.n_pop_bucket_neg}, "
          f"year-±{args.year_window}={args.n_year_5y_neg}, "
          f"cat-lift={args.n_cat_lift_neg}")
    print(f"  curated buckets sum to {curated_total}; remaining "
          f"{max(0, args.n_negatives - curated_total)} slots → general top-pool")
    if curated_total > args.n_negatives:
        print("  [warn] curated buckets sum to more than n_negatives — later "
              "buckets will be truncated.")

    for split in args.splits:
        mine_split(
            split_name=split,
            encoder=args.encoder,
            n_negatives=args.n_negatives,
            n_top_pool=args.n_top_pool,
            n_session_neg=args.n_session_neg,
            n_prior_album_neg=args.n_prior_album_neg,
            n_prior_artist_neg=args.n_prior_artist_neg,
            n_album_neg=args.n_album_neg,
            n_artist_neg=args.n_artist_neg,
            n_pop_bucket_neg=args.n_pop_bucket_neg,
            n_year_5y_neg=args.n_year_5y_neg,
            year_window=args.year_window,
            n_cat_lift_neg=args.n_cat_lift_neg,
            token_tag_csv=args.token_tag_csv,
            track_meta_path=args.track_meta,
            seed=args.seed,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()