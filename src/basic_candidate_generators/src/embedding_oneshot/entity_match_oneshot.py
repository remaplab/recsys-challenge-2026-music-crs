"""EntityMatchOneShot — exact entity injection at turn 1 (E3.3).

LLM-generated queries name artists / tracks / albums constantly; dense and
flat-BM25 retrieval blur those exact strings. This CG builds a normalized alias
table (lowercased, punctuation-stripped artist/track/album names → track idx),
scans the turn-1 query for alias hits via token n-gram lookup, and scores each
matched track by its field weight (track-name > album > artist, tunable). It
injects exact-entity candidates the semantic channels miss; as an RRF component
its weight self-prunes if the signal is weak.

Reuses the base query-cache loading, future masking, and seen-removal — only the
index build and scoring are entity-specific.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

from .oneshot_text_base import OneShotTextCG

_NONALNUM = re.compile(r"[^0-9a-z]+")


def _normalize(s: str) -> str:
    return _NONALNUM.sub(" ", s.lower()).strip()


class EntityMatchOneShot(OneShotTextCG):
    RECOMMENDER_NAME = "EntityMatchOneShot"

    # field → weight-attr; tag_list is excluded (not an entity).
    _FIELD_W = {"track_name": "w_track", "album_name": "w_album", "artist_name": "w_artist"}

    def __init__(self, w_artist: float = 1.0, w_track: float = 2.0, w_album: float = 1.0,
                 min_alias_len: int = 3, max_alias_words: int = 6, **kwargs: Any):
        super().__init__(**kwargs)
        self.w_artist = float(w_artist)
        self.w_track = float(w_track)
        self.w_album = float(w_album)
        self.min_alias_len = int(min_alias_len)
        self.max_alias_words = int(max_alias_words)
        self._alias: dict[str, list[tuple[int, float]]] = {}
        self._maxw = 1

    def _build_index(self, docs: list[str]) -> None:
        # `docs` (flat) ignored — aliases come from the per-field docs.
        weights = {f: float(getattr(self, attr)) for f, attr in self._FIELD_W.items()}
        alias: dict[str, list[tuple[int, float]]] = {}
        maxw = 1
        for f, w in weights.items():
            if w == 0.0:
                continue
            for i, doc in enumerate(self.field_docs.get(f, [])):
                norm = _normalize(doc)
                if len(norm) < self.min_alias_len:
                    continue
                nwords = norm.count(" ") + 1
                if nwords > self.max_alias_words:
                    continue
                maxw = max(maxw, nwords)
                alias.setdefault(norm, []).append((i, w))
        self._alias = alias
        self._maxw = maxw

    def _score_batch(self, query_texts: list[str]) -> np.ndarray:
        n_tracks = len(self.track_ids)
        S = np.zeros((len(query_texts), n_tracks), dtype=np.float32)
        for qi, q in enumerate(query_texts):
            toks = _normalize(q).split()
            seen: set[tuple[int, str]] = set()
            for start in range(len(toks)):
                for L in range(1, min(self._maxw, len(toks) - start) + 1):
                    gram = " ".join(toks[start:start + L])
                    hits = self._alias.get(gram)
                    if not hits:
                        continue
                    for ti, w in hits:
                        if (ti, gram) in seen:
                            continue
                        seen.add((ti, gram))
                        S[qi, ti] += w
        # unmatched tracks → -inf so base _topk emits only entity hits (no
        # arbitrary zero-score padding into the candidate list).
        S[S <= 0.0] = -np.inf
        return S

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st.update({"w_artist": self.w_artist, "w_track": self.w_track,
                   "w_album": self.w_album, "min_alias_len": self.min_alias_len,
                   "max_alias_words": self.max_alias_words,
                   "_alias": self._alias, "_maxw": self._maxw})
        return st
