"""Schema-aware helpers for TalkPlayData-Challenge parquets."""
from __future__ import annotations
from typing import Any, Iterable

# ── Goal axis encodings (single source of truth, imported by encoder + model) ──
CATEGORY_TO_IDX: dict[str, int] = {
    "": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5,
    "F": 6, "G": 7, "H": 8, "I": 9, "J": 10, "K": 11,
}
N_CATEGORIES = 12  # 11 real + 1 unknown

SPECIFICITY_TO_IDX: dict[str, int] = {"": 0, "LL": 1, "HL": 2, "LH": 3, "HH": 4}
N_SPECIFICITIES = 5

CATEGORY_NAMES = {
    "A": "Audio-Based Discovery",
    "B": "Lyrical Discovery",
    "C": "Visual-Musical Connections",
    "D": "Contextual & Situational",
    "E": "Interactive Refinement",
    "F": "Metadata-Rich Exploration",
    "G": "Mood & Emotion-Based",
    "H": "Artist & Discography Discovery",
    "I": "Cultural & Geographic",
    "J": "Social & Popularity Context",
    "K": "Temporal & Era Discovery",
}
SPECIFICITY_NAMES = {
    "LL": "low-query low-target",
    "HL": "high-query low-target",
    "LH": "low-query high-target",
    "HH": "high-query high-target",
}


def encode_category(cat: str | None) -> int:
    if not cat:
        return 0
    return CATEGORY_TO_IDX.get(cat, 0)


def encode_specificity(spec: str | None) -> int:
    if not spec:
        return 0
    return SPECIFICITY_TO_IDX.get(spec, 0)


_TAG_BLOCKLIST = {
    "top quality", "memorable", "via pandora", "good", "great", "awesome",
    "favorite", "favourite", "favorites", "favourites", "best", "amazing",
    "love", "loved", "liked", "playlist", "spotify", "youtube",
    "goeiepoep", "cap", "blueish", "welle work", "ion b chill station",
    "3 of 10 stars", "4 of 10 stars", "5 of 10 stars",
    "1 of 10 stars", "2 of 10 stars", "6 of 10 stars",
    "7 of 10 stars", "8 of 10 stars", "9 of 10 stars", "10 of 10 stars",
    "life is easy", "body parts",
}


def _first(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    return str(value)


def clean_tags(tags: Iterable[str] | None, max_tags: int = 10) -> list[str]:
    if not tags:
        return []
    seen, out = set(), []
    for t in tags:
        if not t:
            continue
        key = t.strip().lower()
        if key in _TAG_BLOCKLIST or key in seen or len(key) < 2:
            continue
        seen.add(key)
        out.append(t.strip())
        if len(out) >= max_tags:
            break
    return out


def build_track_text(row: dict) -> str:
    name   = _first(row.get("track_name"))
    artist = _first(row.get("artist_name"))
    album  = _first(row.get("album_name"))
    tags   = clean_tags(row.get("tag_list"))
    year   = (row.get("release_date") or "")[:4]

    parts = []
    if name:   parts.append(f"Track: {name}")
    if artist: parts.append(f"Artist: {artist}")
    if album:  parts.append(f"Album: {album}")
    if year:   parts.append(f"Year: {year}")
    if tags:   parts.append(f"Tags: {', '.join(tags)}")
    return " | ".join(parts) if parts else "Unknown track"


# ── Legacy v1 budgets (kept for build_query_text callers that haven't migrated) ──
_MAX_HISTORY_TURNS = 6
_MAX_THOUGHT_CHARS = 240
_MAX_ASSISTANT_CHARS = 240
_MAX_PROGRESS_CHARS = 200


def _trunc(s: str, n: int) -> str:
    if not s:
        return ""
    s = s.strip()
    if len(s) <= n:
        return s
    return s[:n].rsplit(" ", 1)[0] + "…"


def build_query_text(
    chat_history: list[dict],
    user_query: str,
    user_profile: dict | None = None,
    conversation_goal: dict | None = None,
    prior_progress_assessments: list[str] | None = None,
    use_thoughts: bool = True,
) -> str:
    """LEGACY v1 query-text builder. New code should use build_query_text_v2.

    Kept verbatim so existing artifacts / callers don't change behavior.
    """
    parts = []

    if user_profile:
        prof_bits = []
        if user_profile.get("age") is not None:
            prof_bits.append(f"age {user_profile['age']}")
        elif user_profile.get("age_group"):
            prof_bits.append(f"{user_profile['age_group']}")
        if user_profile.get("gender"):
            prof_bits.append(user_profile["gender"])
        if user_profile.get("country_name"):
            prof_bits.append(f"from {user_profile['country_name']}")
        if user_profile.get("preferred_musical_culture"):
            prof_bits.append(f"likes {user_profile['preferred_musical_culture']}")
        if prof_bits:
            parts.append(f"[USER PROFILE] {', '.join(prof_bits)}")

    if conversation_goal:
        goal_bits = []
        cat = conversation_goal.get("category")
        spec = conversation_goal.get("specificity")
        if cat and cat in CATEGORY_NAMES:
            goal_bits.append(f"category={CATEGORY_NAMES[cat]}")
        if spec and spec in SPECIFICITY_NAMES:
            goal_bits.append(f"specificity={SPECIFICITY_NAMES[spec]}")
        if conversation_goal.get("listener_goal"):
            goal_bits.append(conversation_goal["listener_goal"])
        if goal_bits:
            parts.append("[GOAL] " + " | ".join(goal_bits))

    history = chat_history[-_MAX_HISTORY_TURNS * 3:] if chat_history else []
    for msg in history:
        role = (msg.get("role") or "").lower()
        content = msg.get("content") or ""
        thought = msg.get("thought") or ""
        if role == "user":
            parts.append(f"[USER PRIOR] {content}")
        elif role == "assistant":
            t = _trunc(content, _MAX_ASSISTANT_CHARS)
            if t:
                parts.append(f"[ASSISTANT PRIOR] {t}")
            if use_thoughts:
                tt = _trunc(thought, _MAX_THOUGHT_CHARS)
                if tt:
                    parts.append(f"[ASSISTANT REASONING] {tt}")
        elif role == "music":
            if use_thoughts:
                tt = _trunc(thought, _MAX_THOUGHT_CHARS)
                if tt:
                    parts.append(f"[TRACK REASONING] {tt}")

    if prior_progress_assessments:
        for pa in prior_progress_assessments[-3:]:
            t = _trunc(pa, _MAX_PROGRESS_CHARS)
            if t:
                parts.append(f"[PROGRESS] {t}")

    parts.append(f"[CURRENT USER] {user_query}")
    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# v2: prior-track injection, session-year, profile→bottom for left-truncation,
# no double-encoded fields (category, specificity, progress enums all dropped).
# ──────────────────────────────────────────────────────────────────────────────
_V2_MAX_HISTORY_TURNS = 6           # same as v1: last 6 (user, assistant, music) groups
_V2_MAX_THOUGHT_CHARS = 200         # tightened from 240
_V2_MAX_PRIOR_TRACKS = 8            # cap on resolved prior-track lines
_V2_MAX_TAGS_PER_TRACK = 4          # tags per resolved prior track
_V2_MAX_USER_PRIOR_CHARS = 280      # earlier user messages


def _format_prior_track_line(track_meta_row: dict | None, track_id: str) -> str:
    """Format one resolved prior-track line for the encoder.

    Output: '[PLAYED] Heart-Shaped Box by Nirvana — grunge, 90s, alternative (1993)'
    Falls back to the bare id if metadata is missing.
    """
    if track_meta_row is None:
        return f"[PLAYED] {track_id}"
    name = _first(track_meta_row.get("track_name")) or "Unknown"
    artist = _first(track_meta_row.get("artist_name")) or "Unknown"
    tags = clean_tags(track_meta_row.get("tag_list"), max_tags=_V2_MAX_TAGS_PER_TRACK)
    year = (track_meta_row.get("release_date") or "")[:4]

    line = f"[PLAYED] {name} by {artist}"
    if tags:
        line += f" — {', '.join(tags)}"
    if year:
        line += f" ({year})"
    return line


def build_query_text_v2(
    chat_history: list[dict],
    user_query: str,
    user_profile: dict | None = None,
    conversation_goal: dict | None = None,
    session_date: str | None = None,
    track_lookup: dict[str, dict] | None = None,
    use_thoughts: bool = True,
) -> str:
    """Build the v2 query string for the frozen Qwen3 encoder.

    Layout (top → bottom; tokenizer left-truncates so bottom always survives):

        [HISTORY (oldest first)]               ← truncated first when long
        [USER PRIOR]   ...
        [TRACK REASONING] ...
        [PLAYED] resolved prior tracks
        ----- (always-present block) -----
        [USER PROFILE]    musical_culture only (+ age_group if present)
        [GOAL]            listener_goal only
        [SESSION] year=YYYY
        [CURRENT USER] ...

    Removed vs v1: age (number), gender, country_name, category label,
    specificity label, goal_progress_assessment enums, assistant `content`
    (we keep its `thought` only).
    Added vs v1:   resolved prior-track names+artists+tags from track_lookup,
                   session year.

    Args:
        chat_history: list of {role, content, thought} from build-time. Music
            turns carry the played track id in `content`.
        user_query: the current user message (always rendered last).
        user_profile: organizer dict; we only use age_group + preferred_musical_culture.
        conversation_goal: organizer dict; we only use listener_goal.
        session_date: ISO date string (e.g. '2018-12-16'); only the year is used.
        track_lookup: dict mapping track_id -> track_meta row dict, used to
            resolve [PLAYED] entries. Pass None to skip prior-track resolution.
        use_thoughts: if False, drop assistant/music thoughts (rarely useful).
    """
    history_parts: list[str] = []
    bottom_parts: list[str] = []

    # ── HISTORY block (truncatable from the top) ──────────────────────────
    history = chat_history[-_V2_MAX_HISTORY_TURNS * 3:] if chat_history else []
    prior_track_lines: list[str] = []
    for msg in history:
        role = (msg.get("role") or "").lower()
        content = msg.get("content") or ""
        thought = msg.get("thought") or ""

        if role == "user":
            t = _trunc(content, _V2_MAX_USER_PRIOR_CHARS)
            if t:
                history_parts.append(f"[USER PRIOR] {t}")

        elif role == "assistant":
            # Drop assistant content entirely. Keep thought only.
            if use_thoughts:
                tt = _trunc(thought, _V2_MAX_THOUGHT_CHARS)
                if tt:
                    history_parts.append(f"[ASSISTANT REASONING] {tt}")

        elif role == "music":
            # Resolve the played track to a human-readable line via track_lookup.
            tid = content.strip() if content else ""
            if tid and track_lookup is not None:
                meta_row = track_lookup.get(tid)
                prior_track_lines.append(_format_prior_track_line(meta_row, tid))
            elif tid:
                # No lookup available — fall back to bare id (rare in practice).
                prior_track_lines.append(f"[PLAYED] {tid}")
            # Music thought (the LLM's reason for picking that track) — keep it.
            if use_thoughts:
                tt = _trunc(thought, _V2_MAX_THOUGHT_CHARS)
                if tt:
                    history_parts.append(f"[TRACK REASONING] {tt}")

    # Prior-track lines come after history/thoughts (closer to the bottom →
    # more likely to survive truncation since they carry the strongest signal).
    if prior_track_lines:
        # Cap and keep most recent prior plays (preserves the latest taste signal).
        prior_track_lines = prior_track_lines[-_V2_MAX_PRIOR_TRACKS:]
        history_parts.extend(prior_track_lines)

    # ── BOTTOM block (always-survives-truncation) ────────────────────────
    if user_profile:
        prof_bits = []
        if user_profile.get("preferred_musical_culture"):
            prof_bits.append(f"likes {user_profile['preferred_musical_culture']}")
        # age_group ('20s', '30s', …) is a low-cardinality coarse bucket the
        # encoder can plausibly use; the raw integer age is dropped.
        ag = user_profile.get("age_group")
        if ag:
            prof_bits.append(f"age group {ag}")
        if prof_bits:
            bottom_parts.append(f"[USER PROFILE] {', '.join(prof_bits)}")

    if conversation_goal and conversation_goal.get("listener_goal"):
        bottom_parts.append(f"[GOAL] {conversation_goal['listener_goal']}")

    if session_date:
        year = str(session_date)[:4]
        if year and year.isdigit():
            bottom_parts.append(f"[SESSION] year={year}")

    bottom_parts.append(f"[CURRENT USER] {user_query}")

    return "\n".join(history_parts + bottom_parts)