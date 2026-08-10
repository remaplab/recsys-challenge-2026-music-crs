"""Data loaders for the reranker feature builder.

This module provides every "side table" the :class:`FeatureBuilder` needs:

- :func:`load_track_metadata` — full track catalog (artist/album/tags/...).
- :func:`load_user_metadata`  — full user catalog (age/country/gender/...).
- :func:`load_warm_user_ids`  — set of users that have a CF-BPR embedding
  (the project's definition of "warm").
- :func:`build_urm_for_split` — popularity URM, leakage-safe per split/fold.
- :func:`build_session_history` — assembled rows for the predicted split
  (provides past tracks per turn — i.e. ``ctx_track_ids``).
- :func:`cg_candidate_path` — locate per-CG candidate parquets on disk.

Leakage discipline
------------------
Building features without leaking the target track is critical. For each
split:

- ``train``  (CG OOF training rows = predictions on ``fold_k_cg_val``):
  URM = ``fold_k_cg_train``. Session history = ``fold_k_cg_val`` itself
  (gives context tracks per turn but never reveals the target).
- ``val``    (CG OOF val rows = predictions on ``fold_k_reranker_val``):
  URM = ``fold_k_cg_train + fold_k_cg_val``. Session history =
  ``fold_k_reranker_val``.
- ``holdout``:
  URM = everything that isn't holdout (``fold_0_cg_train+cg_val+reranker_val``).
  Session history = ``holdout_test``.
- ``blind_a`` (final submission): URM = full assembled (train + test +
  holdout); session history = blind-A assembled. Built only inside
  ``submit_blind_a.py``.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from .. import paths as _paths                # dynamic lookup of CG_STORE
from ..paths import (
    DATA_DIR,
    SPLITK_DIR,
    cg_folder,
)


# ---------------------------------------------------------------------------
# Track / user metadata
# ---------------------------------------------------------------------------

def load_track_metadata() -> pl.DataFrame:
    """Load the full track catalog.

    Prefers the ``.fixed.parquet`` variant if it exists (the dataset shipped
    a few corrected records in that file); falls back to the plain version.
    """
    fixed = (
        DATA_DIR
        / "TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.fixed.parquet"
    )
    plain = (
        DATA_DIR
        / "TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet"
    )
    path = fixed if fixed.exists() else plain
    return pl.read_parquet(path)


def load_user_metadata() -> pl.DataFrame:
    """Load the full user catalog (one row per user_id)."""
    return pl.read_parquet(
        DATA_DIR / "TalkPlayData-Challenge-User-Metadata/data/all_users-00000-of-00001.parquet"
    )


def load_warm_user_ids(include_cold: bool = False) -> set[str]:
    """Return the set of ``user_id``s that have a CF-BPR embedding (= warm).

    The embeddings themselves are not loaded — only the warm/cold flag is
    used downstream (as a feature ``is_warm_user``). Cold users can be
    included if you want to mark them as "warm-via-cold-side-channel" but
    the default is the strict CF-BPR-warm definition.
    """
    parts = [
        pl.read_parquet(
            DATA_DIR / "TalkPlayData-Challenge-User-Embeddings/data/train-00000-of-00001.parquet"
        )["user_id"],
        pl.read_parquet(
            DATA_DIR / "TalkPlayData-Challenge-User-Embeddings/data/test_warm-00000-of-00001.parquet"
        )["user_id"],
    ]
    if include_cold:
        parts.append(
            pl.read_parquet(
                DATA_DIR / "TalkPlayData-Challenge-User-Embeddings/data/test_cold-00000-of-00001.parquet"
            )["user_id"]
        )
    return set(pl.concat(parts).to_list())


# ---------------------------------------------------------------------------
# Per-split popularity URM
# ---------------------------------------------------------------------------

def build_urm_for_split(split: str, fold_idx: int) -> pl.DataFrame:
    """Return the (session_id, turn_number, track_id) interactions used for
    popularity statistics for the given split — leakage-safe by construction.

    See module docstring for the per-split data composition. The returned DF
    is used by :class:`feature_builder.FeatureBuilder` to compute track /
    artist / album play counts.
    """
    if split == "train":
        df = pl.read_parquet(SPLITK_DIR / f"fold_{fold_idx}_cg_train.parquet")
    elif split == "val":
        df = pl.concat([
            pl.read_parquet(SPLITK_DIR / f"fold_{fold_idx}_cg_train.parquet"),
            pl.read_parquet(SPLITK_DIR / f"fold_{fold_idx}_cg_val.parquet"),
        ])
    elif split == "holdout":
        df = pl.concat([
            pl.read_parquet(SPLITK_DIR / "fold_0_cg_train.parquet"),
            pl.read_parquet(SPLITK_DIR / "fold_0_cg_val.parquet"),
            pl.read_parquet(SPLITK_DIR / "fold_0_reranker_val.parquet"),
        ])
    else:
        raise ValueError(f"unknown split {split!r}")
    return df.select("session_id", "turn_number", "track_id")


def build_session_history(split: str, fold_idx: int) -> pl.DataFrame:
    """Return the assembled rows for the predicted split.

    "Assembled rows" means the long view ``(session_id, user_id, turn_number,
    track_id, user_query, ...)`` produced by
    ``src/splits/launchers/splitK_crossvalidation.py``. The FeatureBuilder
    uses this to derive ``ctx_track_ids`` (the past tracks of each session)
    and the text features.
    """
    if split == "train":
        # CG OOF training rows live on cg_val of the fold.
        return pl.read_parquet(SPLITK_DIR / f"fold_{fold_idx}_cg_val.parquet")
    if split == "val":
        # CG OOF val rows live on reranker_val of the fold.
        return pl.read_parquet(SPLITK_DIR / f"fold_{fold_idx}_reranker_val.parquet")
    if split == "holdout":
        return pl.read_parquet(SPLITK_DIR / "holdout_test.parquet")
    raise ValueError(f"unknown split {split!r}")


# ---------------------------------------------------------------------------
# Per-CG candidate file locator
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Full-dataset helpers (used only by the Blind-A submission launcher)
# ---------------------------------------------------------------------------

def build_full_urm() -> pl.DataFrame:
    """URM over EVERY known interaction (train + test + holdout).

    Used by the Blind-A submission feature builder so popularity stats see
    the entire history. Returns ``(session_id, turn_number, track_id)``.
    """
    parts = []
    for f in ("cg_train", "cg_val", "reranker_val"):
        parts.append(pl.read_parquet(SPLITK_DIR / f"fold_0_{f}.parquet"))
    parts.append(pl.read_parquet(SPLITK_DIR / "holdout_test.parquet"))
    return pl.concat(parts).select("session_id", "turn_number", "track_id")


def build_blind_a_session_history(blind_a_path: Path, target_turns: pl.DataFrame) -> pl.DataFrame:
    """Assemble Blind-A into the same shape as ``build_session_history``.

    For each Blind-A session we want:
      - one row per PAST music turn (``track_id`` filled, ``turn_number`` set),
      - one row at the target turn (``track_id=null``, ``user_query`` filled
        — used by family H).

    Parameters
    ----------
    blind_a_path
        Path to the raw Blind-A parquet
        (``data/talkpl-ai/.../test-00000-of-00001.parquet``).
    target_turns
        DataFrame with columns ``(session_id, turn_number)`` listing the
        target turn for each session — typically taken from any CG's
        ``blind_candidates.parquet``.

    Returns
    -------
    polars.DataFrame
        Columns: ``session_id``, ``user_id``, ``turn_number``, ``track_id``,
        ``user_query``, ``user_thought``, ``assistant_response``,
        ``assistant_thought``, ``conversation_goal`` (struct) and
        ``user_profile`` (struct) — so the goal / sentiment / culture / text-
        agreement families (H + K + L) populate at submission instead of going
        null.
    """
    blind = pl.read_parquet(blind_a_path)
    convs = blind.explode("conversations").unnest("conversations")
    # Past music turns — already complete utterances.
    music = (
        convs.filter(pl.col("role") == "music")
              .rename({"content": "track_id"})
              .select("session_id", "user_id", "turn_number", "track_id")
    )
    # User turns — one per turn: the query (content) + the user thought.
    user_q = (
        convs.filter(pl.col("role") == "user")
              .rename({"content": "user_query", "thought": "user_thought"})
              .select("session_id", "turn_number", "user_query", "user_thought")
    )
    # Assistant turns — response content + thought (= music_thought). Feed the
    # family-L prev_assistant_content / prev_music_thought features.
    assistant = (
        convs.filter(pl.col("role") == "assistant")
              .rename({"content": "assistant_response", "thought": "assistant_thought"})
              .select("session_id", "turn_number", "assistant_response", "assistant_thought")
    )
    # Session-level goal + profile structs (constant across a session's turns).
    sess_meta = blind.select("session_id", "conversation_goal", "user_profile")

    text_cols = ["user_query", "user_thought", "assistant_response",
                 "assistant_thought", "conversation_goal", "user_profile"]

    # Join the target-turn user query to produce the prediction-turn row.
    target_rows = (
        target_turns.unique(subset=["session_id", "turn_number"])
        .join(blind.select("session_id", "user_id"), on="session_id", how="left")
        .join(user_q, on=["session_id", "turn_number"], how="left")
        .join(assistant, on=["session_id", "turn_number"], how="left")
        .join(sess_meta, on="session_id", how="left")
        .with_columns(pl.lit(None, dtype=pl.Utf8).alias("track_id"))
        .select("session_id", "user_id", "turn_number", "track_id", *text_cols)
    )
    # Past-music rows also carry the text/goal/profile columns to keep the
    # schema uniform with the target rows.
    music = (
        music.join(user_q, on=["session_id", "turn_number"], how="left")
             .join(assistant, on=["session_id", "turn_number"], how="left")
             .join(sess_meta, on="session_id", how="left")
    )
    return pl.concat([music, target_rows]).sort("session_id", "turn_number")


def cg_candidate_path(cg: str, split: str, fold_idx: int) -> Path:
    """Return the on-disk parquet path that holds a CG's candidates.

    Resolves dynamically against :data:`paths.CG_STORE`,
    :data:`paths.CG_FOLDER_MAP`, and :data:`paths.FILENAMES_BY_SPLIT` — so a
    YAML config consumed through :func:`paths.apply_feature_builder_config`
    can redirect the CG store, rename a CG's on-disk folder, or change the
    per-split filename templates without code changes.

    Filename templates support a ``{fold}`` placeholder (used by the
    fold-dependent splits ``train`` and ``val``).
    """
    base = _paths.CG_STORE / cg_folder(cg) / "datasets"
    template = _paths.FILENAMES_BY_SPLIT.get(split)
    if template is None:
        raise ValueError(
            f"unknown split {split!r}; known: {sorted(_paths.FILENAMES_BY_SPLIT)}"
        )
    return base / template.format(fold=fold_idx)
