"""Abstract base class + shared dataclasses for all reranker backends.

All concrete rerankers (XGBoost, LightGBM, CatBoost, NN, SVGD-NN) must subclass
:class:`BaseReranker`. Doing so guarantees a uniform interface that the
launchers (``tune.py``, ``retrain_and_eval.py``, ``submit_blind_a.py``) can
target without knowing the backend.

The interface is intentionally minimal:

- :meth:`BaseReranker.fit` — train from on-disk parquets. The backend is
  responsible for streaming if necessary.
- :meth:`BaseReranker.predict` — score every candidate of a parquet and
  return a polars DataFrame with one row per candidate plus the score.
- :meth:`BaseReranker.feature_importance` — optional, for GBMs.
- :meth:`BaseReranker.release` — free GPU/CPU memory. Called between trials
  and after submission.

We pass paths-to-parquet rather than in-memory arrays so backends can use
their own streaming primitives (``xgb.DataIter``, LightGBM ``Dataset`` from
files, NN ``IterableDataset``, ...).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import polars as pl


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DatasetSpec:
    """Pointer to the on-disk parquet files used for fit/eval.

    Attributes
    ----------
    train_paths
        One or more parquets to concatenate (streamed) for training.
    val_paths
        One or more parquets to concatenate (streamed) for early-stopping.
        May be empty for a final "no ES" retrain.
    feat_cols
        Ordered list of feature column names. Backends use exactly these
        columns (in this order) when building feature matrices.
    label_col
        Column name carrying the binary label (1 = positive / ground truth).
    group_cols
        Pair of column names identifying a ranking group. Almost always
        ``("session_id", "turn_number")``.
    """
    train_paths: list[Path]
    val_paths: list[Path] = field(default_factory=list)
    feat_cols: list[str] = field(default_factory=list)
    label_col: str = "label"
    group_cols: tuple[str, str] = ("session_id", "turn_number")


# A pruning callback receives (round_idx, metric_value) after every boosting
# round / NN epoch. If it returns True the training loop must stop. Wired by
# the Optuna launcher (``tune.py``) to forward to ``trial.should_prune()``.
PruningCallback = Callable[[int, float], bool]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseReranker(ABC):
    """Common contract for every reranker backend.

    Subclass invariants
    -------------------
    - Subclasses set ``name`` to a short string (``"xgb"``, ``"nn"``, ...).
    - Subclasses set ``supports_gpu`` to indicate whether ``device='cuda'``
      / ``device='gpu'`` is honoured.
    - :meth:`fit` MUST early-stop if ``ds.val_paths`` is non-empty AND
      ``early_stopping_rounds > 0``.
    - :meth:`predict` MUST keep ``(session_id, turn_number, track_id)``
      pass-through so downstream evaluation can join on the original
      ground-truth.
    - :meth:`release` MUST be safe to call multiple times.
    """

    #: Short identifier — also the registry key.
    name: str = "base"

    #: Whether this backend honours ``device`` (XGBoost ``cuda``, LGBM/CatBoost
    #: ``gpu``, NN/SVGD-NN ``cuda``).
    supports_gpu: bool = False

    # =========================================================================
    # Required API
    # =========================================================================

    @abstractmethod
    def fit(
        self,
        ds: DatasetSpec,
        params: dict,
        *,
        device: str = "cpu",
        early_stopping_rounds: int = 30,
        pruning_callback: Optional[PruningCallback] = None,
    ) -> "BaseReranker":
        """Train the reranker.

        Parameters
        ----------
        ds
            The :class:`DatasetSpec` describing where to read training and
            optional validation parquets.
        params
            Backend-specific hyperparameters merged from the YAML config's
            ``static`` section + the Optuna trial's sampled values.
        device
            ``"cpu"`` or ``"cuda"`` (XGB/NN/SVGD-NN) or ``"gpu"``
            (LightGBM/CatBoost). Backends may map this internally.
        early_stopping_rounds
            For GBMs: number of rounds without val-metric improvement before
            stopping. For NN: patience in epochs. ``0`` disables ES.
        pruning_callback
            Optional. Called once per boosting round / epoch with the
            current val metric (nDCG@20 in our setup). Returning ``True``
            stops training immediately. Wired to Optuna's pruner from
            ``tune.py``.

        Returns
        -------
        self
            For chaining.
        """
        ...

    @abstractmethod
    def predict(
        self,
        parquet_path: Path,
        feat_cols: list[str],
    ) -> pl.DataFrame:
        """Score every row of ``parquet_path``.

        Returns
        -------
        polars.DataFrame
            Columns: ``session_id``, ``turn_number``, ``track_id``, ``score``
            (Float64). Higher score = higher rank.
        """
        ...

    # =========================================================================
    # Optional API
    # =========================================================================

    def feature_importance(self) -> dict[str, float] | None:
        """Per-feature importance (gain by default). ``None`` if unsupported.

        The launchers use this for the feature-importance plots in
        ``models/reranker_oof/plots/feature_importance/<reranker>/``.
        """
        return None

    def release(self) -> None:
        """Free GPU/CPU resources owned by the model. Safe to call twice."""
        return None

    # =========================================================================
    # Reusable-handle API used by the Optuna ``tune.py`` launcher.
    #
    # Motivation: per-trial training pays a fixed cost that does not depend
    # on the sampled hyperparameters (parquet decoding, quantile sketching,
    # GPU upload). For XGBoost the ``QuantileDMatrix`` is the heaviest
    # offender — minutes per trial. ``prepare`` lets a backend amortise
    # that cost across the entire study.
    #
    # The default implementations are no-ops that route back to the
    # path-based ``fit`` + ``predict`` so a backend that doesn't override
    # ``prepare`` still works (just at the un-cached cost).
    # =========================================================================

    @classmethod
    def prepare(
        cls,
        ds: DatasetSpec,
        *,
        device: str = "cpu",
        cache_dir=None,
        **kwargs,
    ) -> object:
        """Build expensive per-dataset structures ONCE before the trial loop.

        Returns an opaque handle consumed by :meth:`fit_prepared` and
        :meth:`predict_prepared_val`. Backends override this to cache
        anything that is invariant under hyperparameter changes
        (quantile-binned matrices, on-host feature tensors, ...).

        ``**kwargs`` is forwarded by the launcher from the YAML config's
        ``prepare_kwargs`` section so backends can expose VRAM / RAM
        knobs without changing the launcher signature.

        Default: return ``ds`` unchanged so the path-based code path is used.
        """
        return ds

    def fit_prepared(
        self,
        prepared,
        params: dict,
        *,
        device: str = "cpu",
        early_stopping_rounds: int = 30,
        pruning_callback: Optional[PruningCallback] = None,
    ) -> "BaseReranker":
        """Fit using a handle returned by :meth:`prepare`.

        Default: fall back to the path-based :meth:`fit` (no cross-trial
        caching). Backends that override :meth:`prepare` should override
        this too.
        """
        return self.fit(
            prepared, params,
            device=device,
            early_stopping_rounds=early_stopping_rounds,
            pruning_callback=pruning_callback,
        )

    def predict_prepared_val(self, prepared) -> "pl.DataFrame":
        """Score the val split using cached structures from :meth:`prepare`.

        Default: iterate ``prepared.val_paths`` via :meth:`predict` and
        concatenate. Backends override to reuse a precomputed feature
        matrix or DMatrix.
        """
        import polars as pl                    # local to avoid hard dep at import time
        return pl.concat(
            [self.predict(p, prepared.feat_cols) for p in prepared.val_paths]
        )


# ---------------------------------------------------------------------------
# Helpers for feature-column selection (shared across backends)
# ---------------------------------------------------------------------------

# Columns that are NEVER features (IDs, labels, or kept as side metadata).
ID_AND_LABEL_COLS: frozenset[str] = frozenset({
    "session_id", "turn_number", "user_id", "track_id", "gt_track_id",
    "label", "max_turn",
    # Heavy list / string columns dropped by the FeatureBuilder but listed
    # here in case a future version forgets to drop them.
    "ctx_track_ids", "tag_list", "user_query",
    # Categorical/string side cols that we use for joins/plots but don't
    # feed to the rerankers (encode them explicitly if needed in the future).
    "artist_id_first", "album_id_first",
    "artist_name_first", "album_name_first", "track_name_first",
    "session_dominant_cluster",
    "age_group", "country_code", "gender",
})


def feature_columns(df: pl.DataFrame) -> list[str]:
    """Return the feature columns for a given pool DataFrame.

    Rules:
    - Strip every column in :data:`ID_AND_LABEL_COLS`.
    - Strip any column whose dtype is ``pl.List`` or ``pl.Utf8`` / ``pl.String``
      (lists and strings are not numerical features).
    """
    return [
        c
        for c, dt in zip(df.columns, df.dtypes)
        if c not in ID_AND_LABEL_COLS and dt not in (pl.List, pl.Utf8, pl.String)
    ]
