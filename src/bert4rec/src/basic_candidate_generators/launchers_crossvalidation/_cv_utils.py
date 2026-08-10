from __future__ import annotations

import importlib
import math
import sqlite3
import sys
import threading
from pathlib import Path
from typing import Any

_PKG_ROOT  = Path(__file__).resolve().parent.parent
_SRC_ROOT  = _PKG_ROOT / "src"
# Package nests at src/bert4rec/src/basic_candidate_generators → repo root is 4 levels up.
_REPO_ROOT = _PKG_ROOT.parents[3]

sys.path.insert(0, str(_SRC_ROOT))

import optuna
import polars as pl

def repo_path(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else _REPO_ROOT / path

def pkg_path(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else _PKG_ROOT / path

_CYAN = "\033[96m"
_RESET = "\033[0m"

class ResourceMonitor:

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
        self.peak["ram_gb"] = max(self.peak["ram_gb"], ram)
        self.peak["cpu_pct"] = max(self.peak["cpu_pct"], cpu)
        if self._gpu_handle is not None:
            try:
                import pynvml
                util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                mem  = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                self.peak["gpu_pct"] = max(self.peak["gpu_pct"], float(util.gpu))
                self.peak["vram_gb"] = max(self.peak["vram_gb"], mem.used / 1e9)
            except Exception:
                pass

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def start(self) -> "ResourceMonitor":
        import psutil
        self._proc = psutil.Process()
        self._proc.cpu_percent()
        self._init_gpu()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
        self._sample()

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

def make_storage(db_path: Path) -> "optuna.storages.RDBStorage":
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return optuna.storages.RDBStorage(
        url=f"sqlite:///{db_path}",
        engine_kwargs={"connect_args": {"timeout": 30}},
    )

def suggest(trial: optuna.Trial, name: str, spec: dict) -> Any:
    if "choices" in spec:
        return trial.suggest_categorical(name, spec["choices"])
    low, high = spec["low"], spec["high"]
    log  = bool(spec.get("log", False))
    step = spec.get("step")

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

RECALL_KS: list[int] = [20, 100, 150, 200, 300, 400, 500, 600, 700]
NDCG_KS:   list[int] = [20, 100, 200, 300, 400, 500, 600, 700]

def _score_single(recs: pl.DataFrame, metric: str, k: int) -> float:
    total, n = 0.0, 0
    for row in recs.iter_rows(named=True):
        gt = row["gt_track_id"]
        if gt is None:
            continue
        preds = (row["track_ids"] or [])[:k]
        if metric == "ndcg":
            v = next((1.0 / math.log2(i + 2) for i, t in enumerate(preds) if t == gt), 0.0)
        elif metric == "recall":
            v = 1.0 if gt in preds else 0.0
        else:
            raise ValueError(f"unknown metric: {metric!r}")
        total += v
        n += 1
    return total / n if n else 0.0

def _per_row_metric(row: dict, metric: str, k: int) -> float | None:
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
    per_turn: dict[int, list[float]] = {}
    for row in recs.iter_rows(named=True):
        v = _per_row_metric(row, metric, k)
        if v is None:
            continue
        t = row.get("gt_turn_number")
        if t is None:

            t = 0
        per_turn.setdefault(int(t), []).append(v)
    if not per_turn:
        return 0.0
    per_turn_means = [sum(vs) / len(vs) for vs in per_turn.values()]
    return sum(per_turn_means) / len(per_turn_means)

def score_fold(recs: pl.DataFrame, recall_ks: list[int], ndcg_ks: list[int]) -> dict[str, float]:
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

def score_by_turn(
    recs: pl.DataFrame,
    recall_ks: list[int],
    ndcg_ks: list[int],
    turn_col: str = "gt_turn_number",
) -> dict[int, dict[str, float]]:
    if turn_col not in recs.columns:
        return {}
    out: dict[int, dict[str, float]] = {}
    turns = sorted(int(t) for t in recs[turn_col].drop_nulls().unique().to_list())
    for t in turns:
        sub = recs.filter(pl.col(turn_col) == t)
        m: dict[str, float] = {f"recall@{k}": _score_single(sub, "recall", k) for k in recall_ks}
        m.update({f"ndcg@{k}": _score_single(sub, "ndcg", k) for k in ndcg_ks})
        m["n"] = float(sub.height)
        out[t] = m
    return out

def format_by_turn(
    by_turn: dict[int, dict[str, float]],
    cols: tuple[str, ...] = ("ndcg@20", "recall@20", "recall@200"),
) -> str:
    if not by_turn:
        return "  [by-turn] gt_turn_number unavailable — skipped"
    head = "    turn |     n | " + " | ".join(f"{c:>10s}" for c in cols)
    lines = ["  by predicted turn:", head, "    " + "-" * (len(head) - 4)]
    for t, m in by_turn.items():
        cells = " | ".join(f"{m.get(c, float('nan')):>10.4f}" for c in cols)
        lines.append(f"    {t:>4d} | {int(m['n']):>5d} | {cells}")
    return "\n".join(lines)

def aggregate_mean(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    keys = fold_metrics[0].keys()
    return {k: sum(m[k] for m in fold_metrics) / len(fold_metrics) for k in keys}

def load_fold(splitk_dir: Path, fold: int, parts: list[str] | None = None) -> pl.DataFrame:
    if parts is None:
        parts = ["cg_train"]
    frames = [pl.read_parquet(splitk_dir / f"fold_{fold}_{p}.parquet") for p in parts]
    return pl.concat(frames)

def load_eval(splitk_dir: Path, fold: int) -> pl.DataFrame:
    return pl.read_parquet(splitk_dir / f"fold_{fold}_cg_val.parquet")

def load_reranker_val(splitk_dir: Path, fold: int) -> pl.DataFrame:
    return pl.read_parquet(splitk_dir / f"fold_{fold}_reranker_val.parquet")

def load_blind_b_eval(blind_b_path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
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

def _text_run_inference(rec, eval_df: pl.DataFrame, top_k: int) -> pl.DataFrame:
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

    recs = rec.recommend_text(sess_info, top_k=top_k, remove_seen=True)

    if "gt_turn_number" not in recs.columns:
        gt_turn = gt_df.select(
            pl.col("session_id"),
            pl.col("turn_number").alias("gt_turn_number"),
        )
        recs = recs.join(gt_turn, on="session_id", how="left")
    return recs

def _register_eval_session_extras(rec, eval_df: pl.DataFrame) -> None:
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
    from recommenders.user_base import run_inference

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

def _text_run_inference_all_turns(rec, eval_df: pl.DataFrame, top_k: int) -> pl.DataFrame:
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
        recs_T = _text_run_inference(rec, sliced, top_k=top_k)
        parts.append(recs_T)
    return pl.concat(parts)

def run_inference_dispatch(
    rec,
    eval_df: pl.DataFrame,
    top_k: int,
    inference_mode: str,
    track_meta: pl.DataFrame | None = None,
) -> pl.DataFrame:

    _register_eval_session_extras(rec, eval_df)
    if inference_mode == "text":
        return _text_run_inference_all_turns(rec, eval_df, top_k)
    return _run_inference_all_turns(rec, eval_df, top_k)

def instantiate_rec(class_name: str, module_name: str, params: dict, urm_mode: str):
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    return cls(urm_mode=urm_mode, **params)

_PATH_PARAM_KEYS = ("emb_sim_path", "track_emb_dir", "query_cache_root")

def resolve_param_paths(params: dict) -> dict:
    out = dict(params)
    if "feature_emb_paths" in out:
        out["feature_emb_paths"] = [str(repo_path(p)) for p in out["feature_emb_paths"]]
    for key in _PATH_PARAM_KEYS:
        if out.get(key) is not None:
            out[key] = str(repo_path(out[key]))
    return out

def build_params(trial: optuna.Trial, search_space: dict) -> dict:

    raw: dict[str, Any] = {}
    for name, spec in search_space.items():
        clean = {k: v for k, v in spec.items() if k not in ("conditional_on",)}
        raw[name] = suggest(trial, name, clean)

    rec_params: dict[str, Any] = {}
    for name, spec in search_space.items():
        cond = spec.get("conditional_on")
        if cond:
            cond_param, cond_val = next(iter(cond.items()))
            if raw.get(cond_param) != cond_val:
                continue
        rec_params[name] = raw[name]

    return rec_params

def plot_study(study: optuna.Study, out_dir: Path, objective_key: str = "recall@200") -> None:
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
