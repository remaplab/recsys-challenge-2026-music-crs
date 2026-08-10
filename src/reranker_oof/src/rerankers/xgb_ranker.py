"""XGBoost ``rank:ndcg`` reranker — streaming GPU training.

Why streaming?
--------------
Each fold parquet is several GB; concatenating all of them into a single
dense ``DMatrix`` would not fit in 32 GB host RAM, let alone 16 GB VRAM.

This backend solves it in three pieces:

1. **``ParquetRankerIter``** — an ``xgb.DataIter`` that yields one parquet
   per ``next()``. Each chunk is uploaded as a CUDA tensor (via PyTorch
   when ``device='cuda'``) so XGBoost's GPU code path can consume it. An
   optional ``dtype='float16'`` halves the staging tensor footprint.
2. **``QuantileDMatrix`` / ``ExtMemQuantileDMatrix``** — built from the
   iterator. The quantised data is uint8 bin indices, ~8-16× smaller than
   the raw floats. With ``cache_dir`` set, the **external-memory** variant
   is used: bin pages live on disk and the GPU streams them per boosting
   iteration — essential when the full quantised matrix exceeds VRAM.
3. **Booster** — trained on the QuantileDMatrix with ``tree_method='hist'``
   + ``device='cuda'``. Per-trial behaviour during Optuna tuning is
   identical to a one-shot retrain — the QuantileDMatrix is built once via
   :meth:`XGBReranker.build_dmatrix` and passed to :meth:`fit` per trial,
   amortising the ingest cost across the entire study.

The two-pass DataIter contract
------------------------------
XGBoost re-invokes the iterator twice when constructing a QuantileDMatrix
(once for the quantile sketch, once to push the quantised rows). With
``cache_dir`` (ExtMem) a third "extra" pass writes the disk-backed pages.
The iterator MUST be deterministic across passes — we guarantee that by
reading the same parquets in the same order with no on-the-fly sampling.

Unified API at a glance
-----------------------
- ``XGBReranker.build_dmatrix(ds, device, cache_dir=..., **dmat_kwargs)``
  → ``(dtrain, dval, val_meta)``. Heavy step (~minutes); call ONCE per
  Optuna study, reuse across trials.
- ``m.fit(ds=..., params=...)`` — convenience: builds DMatrix internally.
- ``m.fit(dtrain=..., dval=..., params=...)`` — fast path: reuses prebuilt
  matrices.
- ``m.predict(parquet_path, feat_cols)`` — score one parquet (small DMatrix
  built per call).
- ``m.predict_dval(dval, val_meta)`` — score a pre-built DMatrix and attach
  scores back to the cached ``(session_id, turn_number, track_id)``. Used
  by ``s01_tune.py`` to compute the per-trial nDCG@20.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import xgboost as xgb

from .base import BaseReranker, DatasetSpec, PruningCallback


# ---------------------------------------------------------------------------
# CUDA pool helpers (used between trials to drain torch + cupy caches)
# ---------------------------------------------------------------------------

def _free_cuda_pools() -> None:
    """Drain cached blocks of every CUDA allocator the pipeline touches.

    Safe no-op when CUDA / torch / cupy is unavailable. Cheap (~ms).
    XGBoost dispatches torch CUDA tensors through cupy's ``__cuda_array_interface__``
    so both pools accumulate fragments across trials.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                                # noqa: BLE001
        pass
    try:
        import cupy                                                  # type: ignore
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:                                                # noqa: BLE001
        pass


def _vram_summary() -> str:
    """Short human-readable VRAM-usage line, or ``''`` if unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return ""
        free, total = torch.cuda.mem_get_info()
        used = total - free
        return f"VRAM used={used / 1e9:.2f} GB / {total / 1e9:.2f} GB"
    except Exception:                                                # noqa: BLE001
        return ""


# ---------------------------------------------------------------------------
# Streaming DataIter — one chunk per next() call
# ---------------------------------------------------------------------------

class ParquetRankerIter(xgb.DataIter):
    """Yields (X, y, group) per parquet, one chunk per ``next()`` call.

    Parameters
    ----------
    paths
        Ordered list of parquet files. Each call to ``next()`` consumes one.
    feat_cols
        Feature column order (pinned at construction time so every chunk's
        ``X`` has identical schema).
    tag
        Short string used in the progress prints to disambiguate
        train/val/etc. iterators.
    cache_prefix
        When non-None enables XGBoost external-memory mode: bin pages get
        staged to disk under this prefix. Pass ``str(cache_dir / "<name>")``.
    device
        ``"cpu"`` (numpy upload) or ``"cuda"`` (torch CUDA tensor upload).
        CUDA path is REQUIRED for ``ExtMemQuantileDMatrix`` on GPU — XGB
        otherwise errors with "ExtMemQuantileDMatrix initialised with CPU data".
    dtype
        Element type of the staging tensor: ``"float32"`` (default) or
        ``"float16"``. Float16 halves the per-chunk RAM/VRAM footprint
        during ingest at the cost of a slightly noisier quantile sketch
        (bin edges may shift by a fraction of a percent — usually
        immaterial for ranking).
    weights_by_session
        Optional ``{session_id: weight}`` map for importance-weighted
        training (A3). XGBoost learning-to-rank uses ONE weight per group
        (``(session_id, turn_number)``); since every row of a group shares
        one session, the group weight is that session's value. ``None``
        (default) → uniform weighting, identical to the un-weighted path.
        Pass only to the TRAIN iterator (never val / early-stop / eval).
    weight_tau
        Exponent applied to each weight (``w ** weight_tau``). ``1.0``
        (default) = raw weights; ``< 1.0`` softens them toward uniform to
        lift effective sample size. Ignored when ``weights_by_session`` is
        ``None``.
    """

    # Human labels for the (up to) three passes XGBoost makes over the
    # iterator: bin sketch, data load, and (ExtMem only) the disk-page
    # write-out. The progress print uses these for clarity.
    _PASS_LABELS = ("bin-build", "data-load", "extra")

    def __init__(
        self,
        paths: list[Path],
        feat_cols: list[str],
        tag: str = "",
        cache_prefix: Optional[str] = None,
        device: str = "cpu",
        dtype: str = "float32",
        weights_by_session: Optional[dict] = None,
        weight_tau: float = 1.0,
    ) -> None:
        self._paths = list(paths)
        self._feat_cols = feat_cols
        self._tag = tag
        self._i = 0
        self._pass = -1                       # first ``reset()`` advances to 0
        self._device = device
        if dtype not in ("float32", "float16"):
            raise ValueError(f"unsupported dtype {dtype!r}; use float32 or float16")
        self._dtype = dtype
        self._weights_by_session = weights_by_session
        self._weight_tau = float(weight_tau)
        super().__init__(cache_prefix=cache_prefix)

    # ------------------------------------------------------------------ reset
    def reset(self) -> None:
        """Called by XGBoost between passes. Rewinds the file index."""
        self._pass += 1
        self._i = 0
        if self._pass < len(self._PASS_LABELS):
            print(
                f"[xgb] {self._tag}: starting pass "
                f"{self._pass + 1} ({self._PASS_LABELS[self._pass]}) — "
                f"{len(self._paths)} chunks"
            )

    # ------------------------------------------------------------------- next
    def next(self, input_data) -> bool:
        """Push the next file. Returns ``False`` when exhausted."""
        if self._i >= len(self._paths):
            return False
        path = self._paths[self._i]
        self._i += 1
        df = pl.read_parquet(path)
        df = df.sort("session_id", "turn_number", "track_id")
        # One row per group in first-appearance (== sorted) order. Carry the
        # group's session_id alongside its size so we can attach a per-group
        # weight in the SAME order XGBoost expects the ``group=`` vector.
        grp = (
            df.group_by(["session_id", "turn_number"], maintain_order=True)
            .agg(pl.len().alias("len"))
        )
        groups = grp["len"].to_list()
        # Per-group importance weight (A3). One weight per group; missing
        # sessions default to 1.0 (neutral). ``None`` when weighting is off
        # → identical to the legacy un-weighted ``input_data`` call.
        group_weights = None
        if self._weights_by_session is not None:
            wmap = self._weights_by_session
            tau = self._weight_tau
            group_weights = np.asarray(
                [float(wmap.get(s, 1.0)) ** tau for s in grp["session_id"].to_list()],
                dtype=np.float32,
            )
        # Feature matrix as polars → numpy. NaN-fill keeps XGBoost happy and
        # matches its missing-value semantics. Polars does not (yet) carry
        # a Float16 dtype natively, so we always cast to Float32 inside
        # polars and downcast to numpy.float16 afterwards when requested.
        X_np = (
            df.select(self._feat_cols)
              .cast(pl.Float32)
              .fill_null(float("nan"))
              .to_numpy()
        )
        if self._dtype == "float16":
            X_np = X_np.astype(np.float16, copy=False)
        # Defensive: legacy chunks may carry nulls in ``label`` (the bug
        # was in ``pipeline.py``'s label expression — fixed there too, but
        # already-built chunks on disk would still trigger XGBoost's
        # "Label contains NaN" guard at fit time). Coerce nulls to 0.
        y_np = None
        if "label" in df.columns:
            y_np = df["label"].fill_null(0).to_numpy()
            if y_np.dtype.kind == "f":
                y_np = np.nan_to_num(y_np, nan=0.0)
            # Cast to float32 — XGBoost expects float labels.
            y_np = y_np.astype(np.float32, copy=False)

        phase = (
            self._PASS_LABELS[self._pass]
            if 0 <= self._pass < len(self._PASS_LABELS) else "?"
        )
        print(
            f"  [iter {self._tag}/{phase}] "
            f"{self._i:>2d}/{len(self._paths)} {path.name}: "
            f"{df.height:,} rows, {len(groups):,} groups"
        )

        if self._device == "cuda":
            # ExtMemQuantileDMatrix requires GPU-resident input when the
            # booster will train on GPU. Upload the chunk via torch — its
            # ``__cuda_array_interface__`` is what XGB / cupy consume.
            import torch
            X_in = torch.as_tensor(X_np, device="cuda")
            input_data(data=X_in, label=y_np, group=groups, weight=group_weights)
            del X_in
        else:
            input_data(data=X_np, label=y_np, group=groups, weight=group_weights)
        del df, X_np
        gc.collect()
        return True


# ---------------------------------------------------------------------------
# Pruning callback bridge (Optuna ↔ XGBoost)
# ---------------------------------------------------------------------------

class _XGBPruningCallback(xgb.callback.TrainingCallback):
    """Forward the per-iteration val metric to a ``PruningCallback``.

    Stops training early when the callback returns ``True`` (XGBoost
    interprets the return value of ``after_iteration`` as "should abort?").
    """

    def __init__(self, cb: PruningCallback, metric_name: str) -> None:
        super().__init__()
        self._cb = cb
        self._metric = metric_name

    def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        # evals_log layout: ``{"val": {"ndcg@20": [v0, v1, ...]}, ...}``
        if "val" not in evals_log or self._metric not in evals_log["val"]:
            return False
        v = float(evals_log["val"][self._metric][-1])
        return bool(self._cb(epoch, v))


# ---------------------------------------------------------------------------
# Build-time bundle: handed back from ``XGBReranker.build_dmatrix``
# ---------------------------------------------------------------------------

@dataclass
class XGBDMatrixBundle:
    """Bundle returned by :meth:`XGBReranker.build_dmatrix`.

    Holds the (potentially massive) DMatrices that should be built ONCE
    per Optuna study and reused across all trials. The booster-specific
    bits (``learning_rate``, regularisation, etc.) are NOT here — they
    come from each trial's ``params`` dict.

    Fields
    ------
    dtrain, dval
        ``xgb.QuantileDMatrix`` or ``xgb.ExtMemQuantileDMatrix`` over the
        training and validation chunks. ``dval`` may be ``None`` when no
        early-stopping set is provided.
    feat_cols
        Feature column order pinned at construction time.
    val_meta
        Polars DF with ``(session_id, turn_number, track_id)`` in the
        SAME row order as ``dval``. Used by :meth:`predict_dval` to
        attach scores back to the ranking identity without re-reading
        parquets.
    """
    dtrain: "xgb.DMatrix"
    dval: "Optional[xgb.DMatrix]"
    feat_cols: list[str]
    val_meta: "Optional[pl.DataFrame]"


# ---------------------------------------------------------------------------
# XGBReranker — single ``fit`` API
# ---------------------------------------------------------------------------

class XGBReranker(BaseReranker):
    """``rank:ndcg`` GBM with streaming GPU support.

    Two calling patterns for :meth:`fit`:

    1. **Convenience** — pass a :class:`DatasetSpec` and ``fit`` builds the
       DMatrix internally (slow once per call).
    2. **Amortised** — pre-build via :meth:`build_dmatrix` and pass
       ``dtrain`` / ``dval`` to ``fit`` (cheap per call). This is how the
       Optuna tuning launcher avoids paying the DMatrix construction cost
       every trial.
    """

    name = "xgb"
    supports_gpu = True

    #: Metric used for early stopping + Optuna pruning. Kept fixed at the
    #: official challenge cutoff; the YAML's ``eval_metric`` list is
    #: free to add auxiliary signals — they're logged-only.
    _PRIMARY_METRIC = "ndcg@20"

    def __init__(self) -> None:
        self._booster: Optional[xgb.Booster] = None
        self._feat_cols: list[str] = []

    # =========================================================================
    # build_dmatrix — heavy one-time construction
    # =========================================================================

    @classmethod
    def build_dmatrix(
        cls,
        ds: DatasetSpec,
        *,
        device: str = "cpu",
        cache_dir: Optional[Path] = None,
        max_bin: Optional[int] = None,
        max_quantile_batches: Optional[int] = None,
        cache_host_ratio: Optional[float] = None,
        dtype: str = "float32",
        use_cuda_async_pool: bool = True,
        use_rmm: bool = False,
        train_weights: Optional[dict] = None,
        weight_tau: float = 1.0,
    ) -> XGBDMatrixBundle:
        """Build the train + (optional) val DMatrix once.

        Parameters mirror what the YAML's ``prepare_kwargs`` exposes, plus
        ``dtype`` for float16/32 staging selection. See the module docstring
        for the streaming + external-memory rationale.

        ``train_weights`` (A3): optional ``{session_id: weight}`` map applied
        as per-group importance weights to the TRAIN matrix ONLY (val stays
        uniform so early-stop + the DRO evaluator see the true distribution).
        ``None`` (default) reproduces the un-weighted matrix bit-for-bit.
        ``weight_tau`` softens the weights (``w ** weight_tau``).

        Side effects
        ------------
        - Sets ``xgb.set_config(use_cuda_async_pool=True)`` (or ``use_rmm``)
          when ``device='cuda'`` — switches the VRAM allocator to one that
          defragments better when ExtMem ellpack pages cycle in/out.
        - Creates the ``cache_dir`` directory if external memory is used.
        """
        # ---- Global VRAM allocator config (best-effort) -------------------
        if device == "cuda" and (use_cuda_async_pool or use_rmm):
            try:
                cfg = {}
                if use_cuda_async_pool:
                    cfg["use_cuda_async_pool"] = True
                if use_rmm:
                    cfg["use_rmm"] = True
                xgb.set_config(**cfg)
                print(f"[xgb/build_dmatrix] xgb.set_config({cfg})")
            except Exception as e:                                   # noqa: BLE001
                print(f"[xgb/build_dmatrix] WARN: GPU allocator config failed ({e})")

        # ---- Pick concrete DMatrix class + per-CG cache_prefix paths ------
        cache_train = cache_val = None
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_train = str(cache_dir / "train_cache")
            cache_val = str(cache_dir / "val_cache")
            DMatCls = xgb.ExtMemQuantileDMatrix
            print(
                f"[xgb/build_dmatrix] external memory ENABLED — cache_dir={cache_dir}. "
                "Quantised data is staged to disk; GPU streams pages per "
                "boosting iteration. Required when the full matrix exceeds VRAM."
            )
        else:
            DMatCls = xgb.QuantileDMatrix

        # ---- DMatrix kwargs (only forwarded when non-None) ----------------
        dmat_kwargs: dict = {}
        if max_bin is not None:
            # ``max_bin`` MUST match the value passed to ``xgb.train``
            # (otherwise XGBoost raises ``Inconsistent max_bin``). The
            # caller is responsible for keeping yaml's ``static.max_bin``
            # and ``prepare_kwargs.max_bin`` aligned.
            dmat_kwargs["max_bin"] = int(max_bin)
        if DMatCls is xgb.ExtMemQuantileDMatrix:
            if max_quantile_batches is not None:
                # Caps the in-RAM batch buffer used during the quantile
                # sketch. Lower = lower peak RAM at small quantile-accuracy
                # cost. Default ``None`` = buffer all batches → can OOM.
                dmat_kwargs["max_quantile_batches"] = int(max_quantile_batches)
            if cache_host_ratio is not None:
                # Fraction of cache pages kept in pinned host RAM vs disk.
                # 0.0 = fully on disk (lowest host RAM, slightly slower).
                dmat_kwargs["cache_host_ratio"] = float(cache_host_ratio)
        if dmat_kwargs:
            print(f"[xgb/build_dmatrix] DMatrix kwargs: {dmat_kwargs}")

        # ---- Effective sample size of the train weights (A3 diagnostic) ----
        # ESS = (Σw)² / Σw² over per-group weights. Reported once so a
        # collapsed ESS (heavy tail dominates) is caught before a full study.
        if train_weights is not None:
            sess = (
                pl.concat([
                    pl.scan_parquet(p).select("session_id", "turn_number")
                    for p in ds.train_paths
                ])
                .unique()
                .collect()["session_id"]
                .to_list()
            )
            w = np.asarray(
                [float(train_weights.get(s, 1.0)) ** float(weight_tau) for s in sess],
                dtype=np.float64,
            )
            ess = (w.sum() ** 2) / np.maximum((w ** 2).sum(), 1e-12)
            print(
                f"[xgb/build_dmatrix] train weighting ON (tau={weight_tau}): "
                f"{w.size:,} groups, ESS={ess:,.0f} ({100 * ess / max(w.size, 1):.1f}% "
                f"of groups), w[min/mean/max]={w.min():.3f}/{w.mean():.3f}/{w.max():.3f}"
            )

        # ---- Build dtrain --------------------------------------------------
        print(
            f"[xgb/build_dmatrix] building {DMatCls.__name__}(train) over "
            f"{len(ds.train_paths)} chunks (device={device}, dtype={dtype}) ..."
        )
        dtrain = DMatCls(
            ParquetRankerIter(
                ds.train_paths, ds.feat_cols, tag="train",
                cache_prefix=cache_train, device=device, dtype=dtype,
                weights_by_session=train_weights, weight_tau=weight_tau,
            ),
            **dmat_kwargs,
        )
        dtrain.feature_names = ds.feat_cols
        print(f"[xgb/build_dmatrix] dtrain ready: {dtrain.num_row():,} × {dtrain.num_col()}")

        # ---- Build dval (with ref=dtrain to share bins) + val_meta cache --
        dval: Optional[xgb.DMatrix] = None
        val_meta: Optional[pl.DataFrame] = None
        if ds.val_paths:
            print(
                f"[xgb/build_dmatrix] building {DMatCls.__name__}(val) over "
                f"{len(ds.val_paths)} chunks (ref=dtrain) ..."
            )
            dval = DMatCls(
                ParquetRankerIter(
                    ds.val_paths, ds.feat_cols, tag="val",
                    cache_prefix=cache_val, device=device, dtype=dtype,
                ),
                ref=dtrain,
                **dmat_kwargs,
            )
            dval.feature_names = ds.feat_cols
            print(f"[xgb/build_dmatrix] dval ready: {dval.num_row():,} × {dval.num_col()}")

            # Cache (session_id, turn_number, track_id) in the SAME row
            # order as dval so trial-time score attribution skips parquet
            # re-reads. The sort here MUST match the iterator's sort.
            print(f"[xgb/build_dmatrix] caching val_meta ({len(ds.val_paths)} chunks) ...")
            meta_parts = [
                pl.read_parquet(p)
                  .sort("session_id", "turn_number", "track_id")
                  .select("session_id", "turn_number", "track_id")
                for p in ds.val_paths
            ]
            val_meta = pl.concat(meta_parts)
            del meta_parts
            gc.collect()
            assert val_meta.height == dval.num_row(), (
                "val_meta row count must equal dval.num_row() — sort order "
                "drift between iterator and meta cache."
            )
            print(f"[xgb/build_dmatrix] val_meta cached: {val_meta.height:,} rows")

        return XGBDMatrixBundle(
            dtrain=dtrain, dval=dval, feat_cols=list(ds.feat_cols), val_meta=val_meta,
        )

    # =========================================================================
    # fit — unified entry point
    # =========================================================================

    def fit(
        self,
        ds: Optional[DatasetSpec] = None,
        params: Optional[dict] = None,
        *,
        device: str = "cpu",
        dtrain: Optional["xgb.DMatrix"] = None,
        dval: Optional["xgb.DMatrix"] = None,
        feat_cols: Optional[list[str]] = None,
        cache_dir: Optional[Path] = None,
        prepare_kwargs: Optional[dict] = None,
        early_stopping_rounds: int = 30,
        pruning_callback: Optional[PruningCallback] = None,
        train_weights: Optional[dict] = None,
        weight_tau: float = 1.0,
    ) -> "XGBReranker":
        """Train the booster.

        Two calling modes:

        - ``fit(ds=..., params=..., device=...)`` — builds the DMatrix
          internally via :meth:`build_dmatrix`. Use for retrain / submit
          (one-shot).
        - ``fit(dtrain=..., dval=..., feat_cols=..., params=...)`` — reuses
          a prebuilt DMatrix. Use inside Optuna trial loops where the
          DMatrix is identical across trials.

        Parameters
        ----------
        params
            Full XGBoost training params dict (sampled HPs + static fields
            from the YAML). MUST contain ``num_boost_round`` (popped into
            a local).
        cache_dir, prepare_kwargs
            Forwarded to :meth:`build_dmatrix` when ``dtrain`` is built
            inline. Ignored when ``dtrain`` is passed in.
        """
        if params is None:
            raise ValueError("fit() requires params")

        # ---- Resolve dtrain / dval / feat_cols ----------------------------
        if dtrain is None:
            if ds is None:
                raise ValueError("fit() requires either ds=... or dtrain=...")
            bundle = self.build_dmatrix(
                ds, device=device, cache_dir=cache_dir,
                train_weights=train_weights, weight_tau=weight_tau,
                **(prepare_kwargs or {}),
            )
            dtrain = bundle.dtrain
            dval = bundle.dval
            feat_cols = bundle.feat_cols
        else:
            # Caller supplied a prebuilt matrix. Use their feat_cols if
            # given, else fall back to whatever the matrix knows.
            if feat_cols is None:
                feat_cols = list(dtrain.feature_names or [])
        self._feat_cols = list(feat_cols)

        # ---- Resolve booster params (defaults + device override) ----------
        full_params = dict(params)
        full_params.setdefault("objective", "rank:ndcg")
        full_params.setdefault("eval_metric", [self._PRIMARY_METRIC])
        full_params.setdefault("tree_method", "hist")
        if device == "cuda":
            full_params["device"] = "cuda"
        num_boost_round = int(full_params.pop("num_boost_round", 1000))

        # ---- Eval list + Optuna pruning bridge ----------------------------
        evals = [(dtrain, "train")]
        if dval is not None:
            evals.append((dval, "val"))
        callbacks: list = []
        if pruning_callback is not None and dval is not None:
            callbacks.append(
                _XGBPruningCallback(pruning_callback, self._PRIMARY_METRIC)
            )

        # ---- Train --------------------------------------------------------
        # Compact one-liner with the sampled HP values (the heavy "config"
        # banner already printed inside build_dmatrix).
        sampled_keys = ("learning_rate", "max_depth", "min_child_weight",
                        "subsample", "colsample_bytree", "reg_lambda")
        sampled_str = ", ".join(
            f"{k}={full_params[k]:.4g}" for k in sampled_keys if k in full_params
        )
        print(f"[xgb] fit: device={full_params.get('device', 'cpu')}  {sampled_str}")
        self._booster = xgb.train(
            full_params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds if dval is not None else None,
            verbose_eval=25,
            callbacks=callbacks or None,
        )
        best = getattr(self._booster, "best_score", None)
        if best is not None:
            print(
                f"[xgb] best_iter={self._booster.best_iteration}  "
                f"best_val={best:.4f}  {_vram_summary()}"
            )
        return self

    # =========================================================================
    # Predict
    # =========================================================================

    def predict(self, parquet_path: Path, feat_cols: list[str]) -> pl.DataFrame:
        """Score every row of ``parquet_path`` and return
        ``(session_id, turn_number, track_id, score)`` in row order.

        Builds a one-shot ``DMatrix`` internally (cheap for small parquets
        like Blind-A or per-chunk holdout scoring).
        """
        assert self._booster is not None, "must call fit() first"
        df = pl.read_parquet(parquet_path).sort("session_id", "turn_number", "track_id")
        X = df.select(feat_cols).cast(pl.Float32).fill_null(float("nan")).to_numpy()
        scores = self._booster.predict(xgb.DMatrix(X, feature_names=feat_cols))
        keep = ["session_id", "turn_number", "track_id"]
        if "gt_track_id" in df.columns:
            keep.append("gt_track_id")
        return df.select(keep).with_columns(pl.Series("score", scores))

    def predict_dval(
        self, dval: "xgb.DMatrix", val_meta: pl.DataFrame,
    ) -> pl.DataFrame:
        """Score a prebuilt ``dval`` and attach scores to the cached meta.

        Used by ``s01_tune.py`` to compute per-trial nDCG@20 without
        re-reading + re-quantising the val parquets.
        """
        assert self._booster is not None, "must call fit() first"
        scores = self._booster.predict(dval)
        return val_meta.with_columns(pl.Series("score", scores))

    # =========================================================================
    # Feature importance + release
    # =========================================================================

    def feature_importance(self) -> dict[str, float] | None:
        if self._booster is None:
            return None
        return self._booster.get_score(importance_type="gain")

    def release(self) -> None:
        """Free the booster + drain CUDA pools.

        Called by Optuna between every trial; safe to call repeatedly.
        Does NOT free any DMatrix passed via :meth:`fit` — the caller
        (typically ``s01_tune.py``) owns those for the entire study.
        """
        if self._booster is not None:
            del self._booster
            self._booster = None
        _free_cuda_pools()
        gc.collect()
