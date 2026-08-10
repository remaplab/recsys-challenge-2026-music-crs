"""Per augmented session, aggregate track-metadata stats over prior_track_ids.

Output columns (one row per (session_id, aug_id, max_turn)):
    n_prior, n_unique_artists, n_unique_albums,
    pop_mean, pop_std,
    year_mean, year_std, year_missing,
    dur_mean, dur_std,
    tag_entropy, tag_diversity, top_tag (categorical)

Missing aggregates (empty prior_track_ids): numerics imputed with global median,
boolean `prior_empty` flag is set.
"""
from __future__ import annotations

import re
from collections import Counter
from math import log2

import numpy as np
import polars as pl

from lbo.paths import QUERY_EMB_DIR, TRACKS_META

YEAR_RE = re.compile(r"(\d{4})")


def _parse_year(s: str | None) -> float | None:
    if s is None:
        return None
    m = YEAR_RE.match(s)
    return float(m.group(1)) if m else None


def _load_track_table() -> pl.DataFrame:
    t = pl.read_parquet(TRACKS_META).with_columns(
        pl.col("release_date").map_elements(_parse_year, return_dtype=pl.Float64).alias("year"),
        pl.col("artist_id").list.first().alias("artist_id_first"),
        pl.col("album_id").list.first().alias("album_id_first"),
        pl.col("tag_list").alias("tags"),
    ).select(
        "track_id", "popularity", "year", "duration",
        "artist_id_first", "album_id_first", "tags",
    )
    return t


def _build_prior_lookup() -> dict[tuple[str, int], list[str]]:
    metas = [pl.read_parquet(QUERY_EMB_DIR / f"{s}_meta.parquet") for s in ("train", "dev", "blind_a")]
    full = pl.concat(metas, how="vertical")
    return {
        (sid, int(tn)): list(p) if p is not None else []
        for sid, tn, p in zip(
            full["session_id"].to_list(),
            full["turn_number"].to_list(),
            full["prior_track_ids"].to_list(),
        )
    }


def aggregate_tracks(aug_df: pl.DataFrame) -> pl.DataFrame:
    tracks = _load_track_table()
    tid_to_row: dict[str, int] = {tid: i for i, tid in enumerate(tracks["track_id"].to_list())}
    pop = tracks["popularity"].to_numpy()
    year = tracks["year"].to_numpy()
    dur = tracks["duration"].to_numpy()
    artists = tracks["artist_id_first"].to_list()
    albums = tracks["album_id_first"].to_list()
    tags = tracks["tags"].to_list()

    year_median = float(np.nanmedian(year))
    pop_median = float(np.nanmedian(pop))
    dur_median = float(np.nanmedian(dur))

    prior = _build_prior_lookup()
    sids = aug_df["session_id"].to_list()
    augids = aug_df["aug_id"].to_list()
    mts = aug_df["max_turn"].to_list()

    n = len(sids)
    out = {
        "session_id": sids,
        "aug_id": augids,
        "n_prior": np.zeros(n, dtype=np.int32),
        "n_unique_artists": np.zeros(n, dtype=np.int32),
        "n_unique_albums": np.zeros(n, dtype=np.int32),
        "pop_mean": np.full(n, pop_median, dtype=np.float32),
        "pop_std": np.zeros(n, dtype=np.float32),
        "year_mean": np.full(n, year_median, dtype=np.float32),
        "year_std": np.zeros(n, dtype=np.float32),
        "year_missing": np.ones(n, dtype=np.int8),
        "dur_mean": np.full(n, dur_median, dtype=np.float32),
        "dur_std": np.zeros(n, dtype=np.float32),
        "tag_entropy": np.zeros(n, dtype=np.float32),
        "tag_diversity": np.zeros(n, dtype=np.float32),
        "top_tag": ["__missing__"] * n,
        "prior_empty": np.ones(n, dtype=np.int8),
    }

    for i in range(n):
        sid = sids[i]
        mt = int(mts[i])
        plist = prior.get((sid, mt + 1), [])
        if not plist:
            continue
        out["prior_empty"][i] = 0
        idxs = [tid_to_row[t] for t in plist if t in tid_to_row]
        if not idxs:
            continue
        out["n_prior"][i] = len(idxs)
        out["n_unique_artists"][i] = len({artists[j] for j in idxs if artists[j] is not None})
        out["n_unique_albums"][i] = len({albums[j] for j in idxs if albums[j] is not None})

        p = pop[idxs]
        p = p[~np.isnan(p)]
        if p.size:
            out["pop_mean"][i] = float(p.mean())
            out["pop_std"][i] = float(p.std())

        y = year[idxs]
        y = y[~np.isnan(y)]
        if y.size:
            out["year_mean"][i] = float(y.mean())
            out["year_std"][i] = float(y.std())
            out["year_missing"][i] = 0

        d = dur[idxs]
        d = d[~np.isnan(d)]
        if d.size:
            out["dur_mean"][i] = float(d.mean())
            out["dur_std"][i] = float(d.std())

        flat: list[str] = []
        for j in idxs:
            if tags[j]:
                flat.extend(tags[j])
        if flat:
            c = Counter(flat)
            total = sum(c.values())
            out["top_tag"][i] = c.most_common(1)[0][0]
            ps = np.array([v / total for v in c.values()], dtype=np.float64)
            out["tag_entropy"][i] = float(-(ps * np.log2(ps)).sum())
            out["tag_diversity"][i] = float(len(c) / max(out["n_prior"][i], 1))

    return pl.DataFrame(out)


if __name__ == "__main__":
    from lbo.shift.assemble import assemble
    from lbo.shift.augment import build_augmented

    df = assemble()
    aug = build_augmented(df, n_augs=3, seed=42)
    feats = aggregate_tracks(aug)
    print("shape:", feats.shape)
    print(feats.head(3))
    print("n_prior=0 count:", (feats["n_prior"] == 0).sum())
    print("year_missing rate:", feats["year_missing"].mean())
