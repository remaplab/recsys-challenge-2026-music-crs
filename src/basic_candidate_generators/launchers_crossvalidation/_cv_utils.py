"""Shared utilities for cross-validation Optuna tuning.

Fully self-contained — no imports from launchers.tuning.
"""

from __future__ import annotations

import importlib
import math
import sqlite3
import sys
import threading
from pathlib import Path
from typing import Any

_PKG_ROOT  = Path(__file__).resolve().parent.parent          # src/basic_candidate_generators/
_SRC_ROOT  = _PKG_ROOT / "src"                               # src/basic_candidate_generators/src/
_REPO_ROOT = _PKG_ROOT.parent.parent                         # project root

sys.path.insert(0, str(_SRC_ROOT))

import optuna          # noqa: E402
import polars as pl    # noqa: E402


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def repo_path(p: str | Path) -> Path:
    """Resolve relative paths from the repository root (for data/)."""
    path = Path(p)
    return path if path.is_absolute() else _REPO_ROOT / path


def pkg_path(p: str | Path) -> Path:
    """Resolve relative paths from the package root (for configs/, models/)."""
    path = Path(p)
    return path if path.is_absolute() else _PKG_ROOT / path


# ---------------------------------------------------------------------------
# Resource monitor: peak RAM / CPU / GPU / VRAM during a trial
# ---------------------------------------------------------------------------

_CYAN = "\033[96m"
_RESET = "\033[0m"


class ResourceMonitor:
    """Background-thread resource sampler. Use as a context manager.

    Tracks peak values for:
      ram_gb   — process RSS (GB)
      cpu_pct  — process CPU % (may exceed 100 on multi-core)
      gpu_pct  — GPU utilisation % (first device, requires nvidia-ml-py)
      vram_gb  — GPU memory used (GB, first device, requires nvidia-ml-py)

    GPU fields are None when nvidia-ml-py / GPU unavailable.
    Only instantiate when --monitor flag is active (not on cluster).
    """

    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc = None
        self._gpu_handle = None
        self.peak: dict[str, float | None] = {
            "ram_gb": 0.0, "cpu_pct": 0.0, "gpu_pct": None, "vram_gb": None,
        }

    def _init_gpu(self) -> None:
        try:
            import pynvml
            pynvml.nvmlInit()
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.peak["gpu_pct"] = 0.0
            self.peak["vram_gb"] = 0.0
        except Exception:
            pass

    def _sample(self) -> None:
        import psutil
        ram = self._proc.memory_info().rss / 1e9
        cpu = self._proc.cpu_percent()
        self.peak["ram_gb"] = max(self.peak["ram_gb"], ram)        # type: ignore[type-var]
        self.peak["cpu_pct"] = max(self.peak["cpu_pct"], cpu)      # type: ignore[type-var]
        if self._gpu_handle is not None:
            try:
                import pynvml
                util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                mem  = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                self.peak["gpu_pct"] = max(self.peak["gpu_pct"], float(util.gpu))  # type: ignore[type-var]
                self.peak["vram_gb"] = max(self.peak["vram_gb"], mem.used / 1e9)   # type: ignore[type-var]
            except Exception:
                pass

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def start(self) -> "ResourceMonitor":
        import psutil
        self._proc = psutil.Process()
        self._proc.cpu_percent()  # prime — first call always returns 0
        self._init_gpu()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
        self._sample()  # final sample after stop

    def print_summary(self, trial_number: int) -> None:
        p = self.peak
        parts = [f"ram={p['ram_gb']:.1f}GB", f"cpu={p['cpu_pct']:.0f}%"]
        if p["gpu_pct"] is not None:
            parts += [f"gpu={p['gpu_pct']:.0f}%", f"vram={p['vram_gb']:.1f}GB"]
        print(f"{_CYAN}  [trial {trial_number}] peak resources: {', '.join(parts)}{_RESET}",
              flush=True)

    def __enter__(self) -> "ResourceMonitor":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Optuna storage: SQLite with WAL mode + busy timeout for concurrent trials
# ---------------------------------------------------------------------------

def make_storage(db_path: Path) -> "optuna.storages.RDBStorage":
    """Return an RDBStorage backed by SQLite — minimal config for Lustre.

    Lessons learned the hard way on Leonardo Lustre `$HOME`:
      - `PRAGMA journal_mode=WAL` raises `OperationalError: locking protocol`
        (WAL requires mmap shared memory; Lustre doesn't expose it).
      - URI mode with `nolock=1` raises `unable to open database file`
        (SQLite URI handling on Lustre is broken in some configurations).
      - Plain `sqlite:///path` URL + `timeout=30` JUST WORKS for single-
        writer scenarios — this is the config that produced the existing
        tune DBs successfully.

    So: minimal URL, no pragmas, no URI mode. The 30s busy timeout handles
    transient lock contention. Concurrency caveat: NEVER point multiple
    SLURM workers at the same DB — Lustre file locking is unreliable and
    silent corruption is the failure mode. Use 1 SLURM worker per study
    DB, parallelise via multiple DIFFERENT studies.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return optuna.storages.RDBStorage(
        url=f"sqlite:///{db_path}",
        engine_kwargs={"connect_args": {"timeout": 30}},
    )


# ---------------------------------------------------------------------------
# Optuna param suggestion (standalone, no dependency on launchers.tuning)
# ---------------------------------------------------------------------------

def suggest(trial: optuna.Trial, name: str, spec: dict) -> Any:
    """Suggest a hyperparameter value from a spec dict.

    Supported keys: choices, low, high, log, step, type.
    `type` is used for int vs float disambiguation when low/high are both
    defined as floats in YAML (e.g. 1e-4).
    """
    if "choices" in spec:
        return trial.suggest_categorical(name, spec["choices"])
    low, high = spec["low"], spec["high"]
    log  = bool(spec.get("log", False))
    step = spec.get("step")
    # Explicit type declaration takes precedence; otherwise infer from values
    declared = spec.get("type", "")
    is_int = declared == "int" or (
        declared != "float"
        and isinstance(low, int) and isinstance(high, int)
        and (step is None or float(step).is_integer())
    )
    if is_int:
        return trial.suggest_int(name, int(low), int(high),
                                 step=int(step) if step else 1, log=log)
    return trial.suggest_float(name, float(low), float(high), step=step, log=log)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

RECALL_KS: list[int] = [20, 100, 150, 200, 300, 400, 500, 600, 700]
NDCG_KS:   list[int] = [20, 100, 200, 300, 400, 500, 600, 700]


def _per_row_metric(row: dict, metric: str, k: int) -> float | None:
    """Per-(session, target_turn) hit value, or None if no GT.

    NDCG@K is 1/log2(rank+2) if GT appears in the top-k preds, else 0.
    Recall@K is 1 if GT in top-k preds, else 0.
    """
    gt = row["gt_track_id"]
    if gt is None:
        return None
    preds = (row["track_ids"] or [])[:k]
    if metric == "ndcg":
        return next((1.0 / math.log2(i + 2) for i, t in enumerate(preds) if t == gt), 0.0)
    if metric == "recall":
        return 1.0 if gt in preds else 0.0
    raise ValueError(f"unknown metric: {metric!r}")


def _score_macro_by_turn(recs: pl.DataFrame, metric: str, k: int) -> float:
    """Macro-by-turn aggregation, per the metrics spec.

    For each target-turn position t (1..8): mean of per-session hit values
    over sessions whose GT is at turn t. Final = mean over the non-empty
    turn groups. Sessions with null GT are dropped (consistent with the
    old micro implementation).
    """
    per_turn: dict[int, list[float]] = {}
    for row in recs.iter_rows(named=True):
        v = _per_row_metric(row, metric, k)
        if v is None:
            continue
        t = row.get("gt_turn_number")
        if t is None:
            # Backstop: if a recommender forgets to forward gt_turn_number,
            # collapse everything into bucket 0 so the metric is still a
            # well-defined micro mean rather than crashing.
            t = 0
        per_turn.setdefault(int(t), []).append(v)
    if not per_turn:
        return 0.0
    per_turn_means = [sum(vs) / len(vs) for vs in per_turn.values()]
    return sum(per_turn_means) / len(per_turn_means)


def score_fold(recs: pl.DataFrame, recall_ks: list[int], ndcg_ks: list[int]) -> dict[str, float]:
    """Return all metric@k values for a single fold inference result.

    Aggregation is macro-by-turn: per-session metric → mean within each
    last-turn-position group → mean across non-empty groups.
    """
    if "gt_turn_number" not in recs.columns:
        raise ValueError(
            "score_fold requires a 'gt_turn_number' column on recs. "
            "run_inference / _text_run_inference attach it; ensure custom "
            "recommenders do too."
        )
    out: dict[str, float] = {}
    for k in recall_ks:
        out[f"recall@{k}"] = _score_macro_by_turn(recs, "recall", k)
    for k in ndcg_ks:
        out[f"ndcg@{k}"] = _score_macro_by_turn(recs, "ndcg", k)
    return out


def aggregate_mean(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Average metric dicts across folds."""
    keys = fold_metrics[0].keys()
    return {k: sum(m[k] for m in fold_metrics) / len(fold_metrics) for k in keys}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_fold(splitk_dir: Path, fold: int, parts: list[str] | None = None) -> pl.DataFrame:
    """Load train DataFrame for one fold (cg_train split by default)."""
    if parts is None:
        parts = ["cg_train"]
    frames = [pl.read_parquet(splitk_dir / f"fold_{fold}_{p}.parquet") for p in parts]
    return pl.concat(frames)


def load_eval(splitk_dir: Path, fold: int) -> pl.DataFrame:
    """Load cg_val split (used for CG HP tuning and OOF reranker data generation)."""
    return pl.read_parquet(splitk_dir / f"fold_{fold}_cg_val.parquet")


def load_reranker_val(splitk_dir: Path, fold: int) -> pl.DataFrame:
    """Load reranker_val split (used for reranker HP tuning / validation)."""
    return pl.read_parquet(splitk_dir / f"fold_{fold}_reranker_val.parquet")


def load_blind_b_eval(blind_b_path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Explode a raw blind parquet into the splitK long format for ALL turns.

    Returns ``(eval_df, gt_map)``:
      * ``eval_df`` — one row per (session, turn): every visible music turn (its
        played ``track_id``) plus the withheld submission turn (``track_id`` null).
        Same schema family as ``holdout_test.parquet`` (minus ``is_submission``),
        so it feeds ``run_inference_dispatch`` directly (all-turn inference).
      * ``gt_map`` — (session_id, turn, gt_track_id) for the visible turns only
        (the known internal-validation ground truth); submission turns are absent.

    Blind-B withholds ``conversation_goal`` (all null) — kept as-is so goal-aware
    CGs simply see no goal, exactly as on the real submission.
    """
    raw = pl.read_parquet(blind_b_path)
    base = raw.explode("conversations").unnest("conversations")
    sess_cols = ["session_id", "user_id", "session_date", "user_profile",
                 "conversation_goal", "goal_progress_assessments"]
    users = base.filter(pl.col("role") == "user").select(
        *sess_cols, "turn_number",
        pl.col("thought").alias("user_thought"),
        pl.col("content").alias("user_query"),
    )
    music = base.filter(pl.col("role") == "music").select(
        "session_id", "turn_number",
        pl.col("thought").alias("assistant_thought"),
        pl.col("content").alias("track_id"),
    )
    asst = base.filter(pl.col("role") == "assistant").select(
        "session_id", "turn_number",
        pl.col("content").alias("assistant_response"),
    )
    complete = (users.join(music, on=["session_id", "turn_number"], how="inner")
                .join(asst, on=["session_id", "turn_number"], how="left"))
    target = users.join(music.select("session_id", "turn_number"),
                        on=["session_id", "turn_number"], how="anti")
    eval_df = pl.concat([complete, target], how="diagonal").sort("session_id", "turn_number")
    gt_map = complete.select(
        "session_id", pl.col("turn_number").cast(pl.Int64).alias("turn"),
        pl.col("track_id").alias("gt_track_id"),
    )
    return eval_df, gt_map


def build_fold_icm(train_df: pl.DataFrame, track_metadata: pl.DataFrame, urm_mode: str):
    """Build the (HP-independent) ICM for one fold, once, for reuse across trials.

    Mirrors `UserRecommender.fit`'s ICM construction exactly (same deterministic
    id_map), so the result is row-aligned with every trial's per-fit id_map and
    can be passed back as `precomputed_icm` to skip the per-trial rebuild.
    """
    from recommenders.interactions import (
        build_icm, build_id_map, explode_music_turns,
    )
    long = explode_music_turns(train_df)
    extra = track_metadata["track_id"].to_list()
    id_map = build_id_map(long, extra_track_ids=extra, mode=urm_mode)
    return build_icm(track_metadata, id_map, interactions=long)


# ---------------------------------------------------------------------------
# Inference dispatch
# ---------------------------------------------------------------------------

def _text_run_inference(
    rec, eval_df: pl.DataFrame, top_k: int, query_bundle: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build sess_info from eval_df and call rec.recommend_text().

    GT = last turn per session; ctx_tracks = track_ids from prior turns.
    When `query_bundle` is given, the precomputed query_text + query embedding
    are attached per (session, target turn).
    """
    max_t = eval_df.group_by("session_id").agg(
        pl.col("turn_number").max().alias("max_turn")
    )
    df = eval_df.join(max_t, on="session_id")

    gt_df = df.filter(pl.col("turn_number") == pl.col("max_turn"))

    ctx_df = (
        df.filter(pl.col("turn_number") < pl.col("max_turn"))
        .group_by("session_id")
        .agg(pl.col("track_id").drop_nulls().alias("ctx_tracks"))
    )

    sess_info = gt_df.join(ctx_df, on="session_id", how="left").with_columns(
        pl.col("ctx_tracks").fill_null([])
    )
    if query_bundle is not None:
        from embedding_based.query_tower import attach_query
        sess_info = attach_query(sess_info, query_bundle)

    recs = rec.recommend_text(sess_info, top_k=top_k, remove_seen=True)
    # Ensure gt_turn_number is present for macro-by-turn scoring downstream.
    if "gt_turn_number" not in recs.columns:
        gt_turn = gt_df.select(
            pl.col("session_id"),
            pl.col("turn_number").alias("gt_turn_number"),
        )
        recs = recs.join(gt_turn, on="session_id", how="left")
    return recs


def _register_eval_session_extras(rec, eval_df: pl.DataFrame) -> None:
    """Push per-session signals (user_id, conversation_goal) from eval_df into rec.

    Recommender variants that need extra per-session inputs at inference
    (user-aware -> user_idx, goal/film -> cat_idx/spec_idx) expose
    `register_session_user_map` / `register_session_goals`. Fit only sees the
    training data, so without this call the eval sessions are unknown and the
    recommender falls back to the default (<unk> user, (0, 0) goal) silently.
    """
    if hasattr(rec, "register_session_user_map"):
        try:
            from recommenders.feature_bert4rec_user_helpers import build_session_user_map
            rec.register_session_user_map(build_session_user_map(eval_df))
        except ImportError:
            pass
    if hasattr(rec, "register_session_goals"):
        try:
            from recommenders.feature_bert4rec_identity_cosine_rope_goal import build_session_goal_map
            rec.register_session_goals(build_session_goal_map(eval_df))
        except ImportError:
            pass


def _run_inference_all_turns(rec, eval_df: pl.DataFrame, top_k: int) -> pl.DataFrame:
    """Standard-mode multiturn inference.

    For every target turn T present in eval_df:
      - target sessions = those that have a row at turn T;
      - cap target sessions' rows to turn_number <= T (so build_context_df
        picks turn T as GT, turns < T as ctx);
      - other sessions of the same users are passed through untouched in
        user mode so cross-session injection in build_context_df behaves as
        in the original single-pass inference; in session mode they're
        dropped (no injection happens), avoiding wasted inference cost.

    Returns one row per (session, target_turn).
    """
    from recommenders.user_base import run_inference  # local import

    inject = getattr(rec, "urm_mode", "user") == "user"
    parts: list[pl.DataFrame] = []
    for T in sorted(eval_df["turn_number"].unique().to_list()):
        target_ids = eval_df.filter(pl.col("turn_number") == T)["session_id"].unique()
        if target_ids.len() == 0:
            continue
        target_set = set(target_ids.to_list())
        sliced = eval_df.filter(
            ~(pl.col("session_id").is_in(target_set) & (pl.col("turn_number") > T))
        )
        if not inject:
            sliced = sliced.filter(pl.col("session_id").is_in(target_set))
        if sliced.is_empty():
            continue
        recs_T = run_inference(rec, sliced, top_k=top_k, remove_seen=True)
        parts.append(recs_T.filter(pl.col("session_id").is_in(target_set)))
    return pl.concat(parts)


def _text_run_inference_all_turns(
    rec, eval_df: pl.DataFrame, top_k: int, query_bundle: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Text-mode multiturn inference. Same slicing strategy as standard mode.

    Text recommenders don't inject cross-session ctx (sess_info has a single
    ctx_tracks per session), so we always pre-filter to target sessions.
    """
    parts: list[pl.DataFrame] = []
    for T in sorted(eval_df["turn_number"].unique().to_list()):
        target_ids = eval_df.filter(pl.col("turn_number") == T)["session_id"].unique()
        if target_ids.len() == 0:
            continue
        target_set = set(target_ids.to_list())
        sliced = eval_df.filter(
            pl.col("session_id").is_in(target_set)
            & (pl.col("turn_number") <= T)
        )
        if sliced.is_empty():
            continue
        recs_T = _text_run_inference(rec, sliced, top_k=top_k, query_bundle=query_bundle)
        parts.append(recs_T)
    return pl.concat(parts)


def run_inference_dispatch(
    rec,
    eval_df: pl.DataFrame,
    top_k: int,
    inference_mode: str,
    track_meta: pl.DataFrame | None = None,
    query_bundle: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Multiturn inference: predicts every (session, turn) pair in eval_df.

    Returns one row per (session, target_turn). Score with `score_fold`
    downstream (micro-avg across all (session, turn) pairs). `query_bundle`
    (text mode only) supplies precomputed query_text + query embeddings.
    """
    if inference_mode == "text":
        return _text_run_inference_all_turns(rec, eval_df, top_k, query_bundle)
    return _run_inference_all_turns(rec, eval_df, top_k)


# ---------------------------------------------------------------------------
# Recommender instantiation
# ---------------------------------------------------------------------------

def instantiate_rec(class_name: str, module_name: str, params: dict, urm_mode: str):
    """Dynamically import and instantiate a recommender class."""
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    return cls(urm_mode=urm_mode, **params)


_PATH_PARAM_KEYS = ("emb_sim_path", "track_emb_dir", "track_emb_dir_4b",
                    "track_emb_dir_0p6b", "query_cache_root",
                    "img_sim_path", "img_parquet_glob", "icm_sim_path", "cache_dir",
                    "fallback_datasets_dir", "track_parquet_glob",
                    "query_text_root",
                    "rrf_oneshot_datasets_dir", "tower_cf_datasets_dir")


def resolve_param_paths(params: dict) -> dict:
    """Resolve repo-root-relative path-like keys in a params dict.

    Handles `feature_emb_paths` (list[str]) for feature_bert4rec, and the
    embedding-CG path scalars (`emb_sim_path`, `track_emb_dir`,
    `query_cache_root`) for the embedding_based recommenders, which read the
    Qwen tower / similarity caches at fit time. Returns a new dict.
    """
    out = dict(params)
    if "feature_emb_paths" in out:
        out["feature_emb_paths"] = [str(repo_path(p)) for p in out["feature_emb_paths"]]
    for key in _PATH_PARAM_KEYS:
        if out.get(key) is not None:
            out[key] = str(repo_path(out[key]))
    return out


# ---------------------------------------------------------------------------
# Conditional param handling
# ---------------------------------------------------------------------------

def build_params(trial: optuna.Trial, search_space: dict) -> dict:
    """Suggest all params from search_space, skipping conditionals when unmet.

    All params are always suggested (so TPE sees them), but params with
    unmet `conditional_on` are excluded from the returned rec_params dict.
    """
    # First pass: suggest everything (so optuna tracks all params)
    raw: dict[str, Any] = {}
    for name, spec in search_space.items():
        clean = {k: v for k, v in spec.items() if k not in ("conditional_on",)}
        raw[name] = suggest(trial, name, clean)

    # Second pass: filter out unmet conditionals for the recommender
    rec_params: dict[str, Any] = {}
    for name, spec in search_space.items():
        cond = spec.get("conditional_on")
        if cond:
            cond_param, cond_val = next(iter(cond.items()))
            if raw.get(cond_param) != cond_val:
                continue  # condition unmet — param not passed to recommender
        rec_params[name] = raw[name]

    return rec_params


# ---------------------------------------------------------------------------
# Plotting (standalone)
# ---------------------------------------------------------------------------

def plot_study(study: optuna.Study, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    complete = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    if len(complete) < 2:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    maximize = study.direction == optuna.study.StudyDirection.MAXIMIZE

    numbers = [t.number for t in complete]
    values  = [t.value  for t in complete]
    best_so_far: list[float] = []
    running_best = float("-inf") if maximize else float("inf")
    for v in values:
        if (maximize and v > running_best) or (not maximize and v < running_best):
            running_best = v
        best_so_far.append(running_best)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(numbers, values, s=18, alpha=0.6, label="trial")
    ax.plot(numbers, best_so_far, color="red", linewidth=1.5, label="best so far")
    ax.set_xlabel("Trial"); ax.set_ylabel("recall@200")
    ax.set_title(f"{study.study_name} — optimization history"); ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{study.study_name}_history.png", dpi=120)
    plt.close(fig)

    try:
        importances = optuna.importance.get_param_importances(study)
        if importances:
            params  = list(importances.keys())
            weights = [importances[p] for p in params]
            fig2, ax2 = plt.subplots(figsize=(7, max(3, len(params) * 0.5)))
            ax2.barh(params[::-1], weights[::-1])
            ax2.set_title(f"{study.study_name} — param importances")
            fig2.tight_layout()
            fig2.savefig(out_dir / f"{study.study_name}_importance.png", dpi=120)
            plt.close(fig2)
    except Exception:
        pass

    print(f"[cv] plots → {out_dir}/")
