"""Per-set query-tower bundles (query_text + query embeddings) for text CGs.

The Qwen3-Embedding-8B query tower precomputes one embedding per
(session_id, turn_number) for every inference set, stored under
`models/retrieval_text_towers/Qwen__Qwen3-Embedding-8B/dense_*_query_*`:

  * dense_splitk_fold_{k}_cg_val_query_len512_poollast       (OOF / tuning)
  * dense_splitk_fold_{k}_reranker_val_query_len512_poollast (reranker val)
  * dense_splitk_holdout_query_len512_poollast               (holdout)
  * dense_blinda_query_len512_poollast                       (blind-A)

Each dir holds `query_meta.parquet` (session_id, turn_number, user_id,
query_text) row-aligned with `query_embeddings.npy` (n, d) L2-normed. The
embeddings live in the SAME space as the `dense_tracks_*` track tower, so a
query→track cosine signal drops straight into the existing `_emb_topk`.

`attach_query` left-joins a bundle onto a recommender's `sess_info` on
(session_id, turn_number), supplying both the text (tfidf/bm25 query) and the
vector (query→track emb signal). Rows with no matching embedding keep a null
`query_emb`, which `recommend_text` treats as "signal inactive".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

QUERY_TOWER_BASE = "models/retrieval_text_towers/Qwen__Qwen3-Embedding-8B"

_SET_DIRS = {
    "cg_val":       "dense_splitk_fold_{fold}_cg_val_query_len512_poollast",
    "reranker_val": "dense_splitk_fold_{fold}_reranker_val_query_len512_poollast",
    "holdout":      "dense_splitk_holdout_query_len512_poollast",
    "blind":        "dense_blinda_query_len512_poollast",
    "blindb_all":   "dense_blindb_all_query_len512_poollast",
}


def query_dir_name(set_key: str, fold: int | None = None) -> str:
    """Resolve the query-tower sub-dir name for an inference set."""
    if set_key not in _SET_DIRS:
        raise KeyError(f"unknown query set_key {set_key!r}; have {list(_SET_DIRS)}")
    name = _SET_DIRS[set_key]
    if "{fold}" in name:
        if fold is None:
            raise ValueError(f"set_key {set_key!r} requires a fold index")
        name = name.format(fold=fold)
    return name


def load_query_bundle(
    tower_base: str | Path, set_key: str, fold: int | None = None,
) -> pl.DataFrame:
    """Load (session_id, turn_number, query_text, query_emb) for an inference set.

    `query_emb` is a fixed-width `Array(Float32, d)` column row-aligned with the
    L2-normed `query_embeddings.npy`.
    """
    d = Path(tower_base) / query_dir_name(set_key, fold)
    meta = pl.read_parquet(d / "query_meta.parquet")
    emb = np.asarray(np.load(d / "query_embeddings.npy"), dtype=np.float32)
    if emb.shape[0] != meta.height:
        raise ValueError(
            f"query bundle row mismatch in {d}: meta={meta.height} vs emb={emb.shape[0]}"
        )
    return meta.select(["session_id", "turn_number", "query_text"]).with_columns(
        pl.Series("query_emb", emb, dtype=pl.Array(pl.Float32, emb.shape[1]))
    )


def attach_query(sess_info: pl.DataFrame, bundle: pl.DataFrame) -> pl.DataFrame:
    """Left-join `query_text` + `query_emb` onto sess_info per (session, turn).

    `turn_number` may be named `turn` in some sess_info builders; both are
    handled. Missing `query_text` is filled with "" (tfidf/bm25 fall back to
    field assembly); missing `query_emb` stays null (emb signal inactive).
    """
    turn_col = "turn_number" if "turn_number" in sess_info.columns else "turn"
    keyed = bundle.rename({"turn_number": turn_col})
    out = sess_info.join(keyed, on=["session_id", turn_col], how="left")
    return out.with_columns(pl.col("query_text").fill_null(""))
