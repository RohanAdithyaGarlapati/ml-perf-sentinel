"""Changepoint localization for triage.

When a regression is flagged, engineers' first question is "which run/commit
introduced it?". This module runs a single-changepoint scan (binary
segmentation with a standardized mean-shift score) over the metric series to
identify the most likely first offending run.
"""
from __future__ import annotations

import numpy as np


def find_changepoint(values: list[float], min_segment: int = 3) -> dict:
    """Return the index of the most likely upward mean shift in the series.

    Score at split t = |mean(right) - mean(left)| / pooled_std. The argmax is
    the estimated changepoint (first index of the degraded regime).
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n < 2 * min_segment:
        return {"index": None, "score": None, "reason": "series too short"}

    best_idx, best_score = None, -np.inf
    for t in range(min_segment, n - min_segment + 1):
        left, right = arr[:t], arr[t:]
        pooled = np.sqrt((left.var(ddof=1) * (len(left) - 1) +
                          right.var(ddof=1) * (len(right) - 1)) / (n - 2))
        pooled = max(float(pooled), 1e-9)
        score = (right.mean() - left.mean()) / pooled  # signed: we triage slowdowns
        if score > best_score:
            best_score, best_idx = score, t

    return {
        "index": int(best_idx),
        "score": round(float(best_score), 4),
        "shift_pct": round(float((arr[best_idx:].mean() - arr[:best_idx].mean())
                                 / arr[:best_idx].mean() * 100.0), 2),
    }
