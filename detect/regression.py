"""Statistical performance regression detection.

For each (workload, mode, batch_size) series of a latency metric, the latest
run is compared against a rolling baseline of prior runs:

1. Primary signal  — robust z-score of the latest value vs the baseline
   window, plus a minimum percent-change guard (so tiny-but-significant
   noise on a flat series doesn't page anyone).
2. Confirmation    — one-sided Mann-Whitney U test of the most recent k runs
   vs the baseline window, when enough history exists. Non-parametric, so it
   makes no normality assumptions about latency distributions.
"""
from __future__ import annotations

import duckdb
import numpy as np
from scipy import stats as sstats

DEFAULTS = {
    "metric": "latency_ms_p50",
    "baseline_window": 10,
    "recent_window": 3,
    "z_threshold": 3.0,
    "min_pct_change": 5.0,
    "mw_alpha": 0.01,
}


def score_series(values: list[float], baseline_window: int = 10, recent_window: int = 3,
                 z_threshold: float = 3.0, min_pct_change: float = 5.0,
                 mw_alpha: float = 0.01) -> dict:
    """Score a chronologically-ordered metric series. Pure function (testable)."""
    out = {"is_regression": False, "z": None, "pct_change": None,
           "mw_pvalue": None, "mw_confirms": None}
    if len(values) < baseline_window + 1:
        out["reason"] = "insufficient history"
        return out

    arr = np.asarray(values, dtype=float)
    latest = arr[-1]
    baseline = arr[-(baseline_window + 1):-1]
    mean, std = float(baseline.mean()), float(baseline.std(ddof=1))
    std = max(std, 1e-9, 0.005 * mean)  # noise floor: protect against ~zero variance

    out["z"] = float((latest - mean) / std)
    out["pct_change"] = float((latest - mean) / mean * 100.0) if mean else 0.0
    step_regression = out["z"] > z_threshold and out["pct_change"] > min_pct_change

    # Sustained-shift path: a multi-run regression contaminates the rolling
    # baseline and dilutes the z-score, so compare the recent window against a
    # clean prior window with a one-sided Mann-Whitney U test.
    sustained_regression = False
    if len(arr) >= baseline_window + recent_window:
        recent = arr[-recent_window:]
        prior = arr[-(baseline_window + recent_window):-recent_window]
        _, p = sstats.mannwhitneyu(recent, prior, alternative="greater")
        sustained_pct = float((recent.mean() - prior.mean()) / prior.mean() * 100.0)
        out["mw_pvalue"] = float(p)
        out["sustained_pct"] = sustained_pct
        out["mw_confirms"] = bool(p < mw_alpha and sustained_pct > min_pct_change)
        sustained_regression = out["mw_confirms"]

    out["is_regression"] = step_regression or sustained_regression
    return out


def series_for(con: duckdb.DuckDBPyConnection, workload: str, mode: str,
               batch_size: int, metric: str) -> list[tuple]:
    """Per-run series, taking the median across distributed workers."""
    return con.execute(
        """
        SELECT run_id, min(ts) AS ts, median(value) AS v
        FROM benchmarks
        WHERE workload = ? AND mode = ? AND batch_size = ? AND metric = ?
        GROUP BY run_id ORDER BY ts
        """,
        [workload, mode, batch_size, metric],
    ).fetchall()


def detect_regressions(db_path: str, **overrides) -> list[dict]:
    cfg = {**DEFAULTS, **overrides}
    con = duckdb.connect(db_path, read_only=True)
    combos = con.execute(
        "SELECT DISTINCT workload, mode, batch_size FROM benchmarks ORDER BY 1, 2, 3"
    ).fetchall()

    findings = []
    for workload, mode, batch_size in combos:
        rows = series_for(con, workload, mode, batch_size, cfg["metric"])
        values = [r[2] for r in rows]
        score = score_series(values, cfg["baseline_window"], cfg["recent_window"],
                             cfg["z_threshold"], cfg["min_pct_change"], cfg["mw_alpha"])
        if score["is_regression"]:
            findings.append({
                "workload": workload, "mode": mode, "batch_size": batch_size,
                "metric": cfg["metric"], "run_id": rows[-1][0],
                "latest_value": round(values[-1], 4), **{k: (round(v, 4) if isinstance(v, float) else v)
                                                          for k, v in score.items()},
            })
    con.close()
    return findings
