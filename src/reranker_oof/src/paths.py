"""Single source of truth for filesystem paths used by the reranker pipeline.

All paths are resolved relative to the repository root, which is detected by
walking up from this file four levels (``src/reranker_oof/src/paths.py`` →
``<repo_root>``). If you move this file, update :data:`REPO_ROOT` below.

Layout assumptions
------------------
``<repo_root>/``
├── ``data/talkpl-ai/``                — raw challenge datasets (track meta, user meta, etc.)
├── ``data/splitK/``                   — fold splits produced by ``src/splits/launchers/splitK_crossvalidation.py``
├── ``models/CG_crossvalidation/``     — CG candidate stores, one subdir per CG
├── ``models/reranker_oof/``           — ALL artifacts produced by this package
└── ``src/reranker_oof/``              — this package (code)

The pipeline does NOT write outside ``models/reranker_oof/``.

Notes
-----
- ``CG_FOLDER_MAP`` translates the logical CG names used in configs (e.g.
  ``"bm25"``) into the on-disk subdirectories (e.g. ``"bm25_cg_session"``).
  Add new mappings here as new CGs are introduced.
- ``BLIND_CANDIDATES_FILENAME`` is the filename emitted by
  ``src/basic_candidate_generators/launchers_crossvalidation/retrain_and_export.py``
  for the Blind-A submission set. Verified at planning time across all 16 CGs.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
# This file: <repo_root>/src/reranker_oof/src/paths.py
# parents:    0 -> .../src
#             1 -> .../reranker_oof
#             2 -> .../src
#             3 -> <repo_root>
REPO_ROOT: Path = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# Inputs (read-only at runtime). CG_STORE is mutable via
# :func:`apply_feature_builder_config` so the launchers can switch the CG
# directory at runtime through a YAML config without code edits.
# ---------------------------------------------------------------------------
DATA_DIR: Path = REPO_ROOT / "data" / "talkpl-ai"
SPLITK_DIR: Path = REPO_ROOT / "data" / "splitK"
CG_STORE: Path = REPO_ROOT / "models" / "CG_crossvalidation"

# Raw blind-A source (used by ``holdout_subsets.blind_a_like_subsets`` to
# extract the target-turn distribution).
BLIND_A_RAW: Path = (
    DATA_DIR
    / "TalkPlayData-Challenge-Blind-A"
    / "data"
    / "test-00000-of-00001.parquet"
)

# Raw blind-B source (phase 2). Same schema as blind-A; conversation_goal is
# fully null (withheld). Used as the blind VAL / submission raw.
BLIND_B_RAW: Path = (
    DATA_DIR
    / "TalkPlayData-Challenge-Blind-B"
    / "data"
    / "test-00000-of-00001.parquet"
)

# ---------------------------------------------------------------------------
# Outputs (everything written by this package lives under here)
# ---------------------------------------------------------------------------
OUT_DIR: Path = REPO_ROOT / "models" / "reranker_oof"
DATASETS_DIR: Path = OUT_DIR / "datasets"
SUBSAMPLES_DIR: Path = OUT_DIR / "subsamples"
CALIBRATORS_DIR: Path = OUT_DIR / "calibrators"

# ---------------------------------------------------------------------------
# Active dataset name — set by apply_feature_builder_config or set_active_dataset
# ---------------------------------------------------------------------------
DATASET_NAME: str | None = None


def set_active_dataset(name: str) -> None:
    """Activate a named dataset (writes to DATASET_NAME global)."""
    global DATASET_NAME
    if not name:
        raise ValueError("dataset name must be a non-empty string")
    DATASET_NAME = name


def active_dataset_dir() -> Path:
    """Return datasets/<DATASET_NAME>. Errors if no dataset is active."""
    if DATASET_NAME is None:
        raise RuntimeError(
            "No active dataset. Call set_active_dataset() or "
            "apply_feature_builder_config() before using dataset paths."
        )
    return DATASETS_DIR / DATASET_NAME


def active_subsamples_dir() -> Path:
    """Return subsamples/<DATASET_NAME>. Errors if no dataset is active."""
    if DATASET_NAME is None:
        raise RuntimeError(
            "No active dataset. Call set_active_dataset() before using subsample paths."
        )
    return SUBSAMPLES_DIR / DATASET_NAME
OPTUNA_DIR: Path = OUT_DIR / "optuna"
REPORTS_DIR: Path = OUT_DIR / "reports"
PLOTS_DIR: Path = OUT_DIR / "plots"
SUBMISSIONS_DIR: Path = OUT_DIR / "submissions"

# ---------------------------------------------------------------------------
# CG-name → on-disk folder map (only entries that DIVERGE from identity)
# ---------------------------------------------------------------------------
# Most CGs use the same name as their directory. ``bm25`` and ``tfidf`` ship as
# ``bm25_cg_session`` / ``tfidf_cg_session`` (legacy naming).
CG_FOLDER_MAP: dict[str, str] = {
    "bm25":  "bm25_cg_session",
    "tfidf": "tfidf_cg_session",
}


def cg_folder(cg: str) -> str:
    """Return the on-disk folder name for a logical CG name."""
    return CG_FOLDER_MAP.get(cg, cg)


# ---------------------------------------------------------------------------
# Per-split filename templates inside ``<cg_store>/<cg_folder>/datasets/``.
# Override via :func:`apply_feature_builder_config`.
# ``train`` / ``val`` use ``{fold}`` placeholders interpolated at lookup time.
# ---------------------------------------------------------------------------
FILENAMES_BY_SPLIT: dict[str, str] = {
    "train":   "fold_{fold}_oof_cg_val.parquet",
    "val":     "fold_{fold}_oof_reranker_val.parquet",
    "holdout": "holdout_candidates.parquet",
    "blind_a": "blind_candidates.parquet",
}

# Legacy aliases kept for backwards-compat. Prefer ``FILENAMES_BY_SPLIT``.
HOLDOUT_CANDIDATES_FILENAME: str = FILENAMES_BY_SPLIT["holdout"]
BLIND_CANDIDATES_FILENAME: str = FILENAMES_BY_SPLIT["blind_a"]


# ---------------------------------------------------------------------------
# Runtime overrides via feature-builder YAML config
# ---------------------------------------------------------------------------

def apply_feature_builder_config(cfg: dict) -> None:
    """Mutate the path module from a feature-builder YAML config.

    Recognised top-level keys (all optional except ``name``):

    - ``name`` (str, REQUIRED) — dataset identifier used as the subdirectory
      under ``datasets/`` and ``subsamples/``. Must be unique per feature
      configuration.
    - ``cg_store`` (str) — directory holding per-CG subfolders.
    - ``cg_folder_map`` (dict) — entries merged into :data:`CG_FOLDER_MAP`.
    - ``filenames`` (dict) — entries merged into :data:`FILENAMES_BY_SPLIT`.

    Calling this multiple times accumulates entries; later calls win for
    overlapping keys.
    """
    global CG_STORE
    name = cfg.get("name")
    if not name:
        raise ValueError(
            "Feature-builder config must include a non-empty `name` field "
            "(used as the dataset subdirectory under datasets/ and subsamples/)."
        )
    set_active_dataset(name)
    if "cg_store" in cfg and cfg["cg_store"]:
        p = Path(cfg["cg_store"])
        if not p.is_absolute():
            p = REPO_ROOT / p
        CG_STORE = p
    if isinstance(cfg.get("cg_folder_map"), dict):
        CG_FOLDER_MAP.update(cfg["cg_folder_map"])
    if isinstance(cfg.get("filenames"), dict):
        FILENAMES_BY_SPLIT.update(cfg["filenames"])


def ensure_output_dirs() -> None:
    """Create every artifact directory under ``models/reranker_oof/``.

    Safe to call repeatedly. Launchers call this once at startup.
    """
    for d in (
        OUT_DIR,
        DATASETS_DIR,
        SUBSAMPLES_DIR,
        CALIBRATORS_DIR,
        OPTUNA_DIR,
        REPORTS_DIR,
        PLOTS_DIR,
        SUBMISSIONS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
