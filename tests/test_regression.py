import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detect.changepoint import find_changepoint
from detect.regression import score_series

RNG = np.random.default_rng(42)


def _flat(n=20, base=5.0, noise=0.02):
    return (base * RNG.normal(1.0, noise, n)).tolist()


def test_flat_series_is_not_flagged():
    score = score_series(_flat())
    assert score["is_regression"] is False


def test_step_regression_is_flagged():
    values = _flat(20)
    values[-1] *= 1.30  # 30% slowdown on the latest run
    score = score_series(values)
    assert score["is_regression"] is True
    assert score["pct_change"] > 20


def test_sustained_regression_confirmed_by_mann_whitney():
    values = _flat(20)
    values[-3:] = [v * 1.25 for v in values[-3:]]
    score = score_series(values)
    assert score["is_regression"] is True
    assert score["mw_confirms"] is True


def test_insufficient_history_is_safe():
    score = score_series(_flat(5))
    assert score["is_regression"] is False
    assert score["reason"] == "insufficient history"


def test_small_noise_below_pct_floor_not_flagged():
    # Ultra-low variance series: z can be large, but pct change is tiny.
    values = [5.0] * 15 + [5.05]
    score = score_series(values)
    assert score["is_regression"] is False


def test_changepoint_localizes_shift():
    values = _flat(15) + [v * 1.3 for v in _flat(5)]
    cp = find_changepoint(values)
    assert cp["index"] is not None
    assert abs(cp["index"] - 15) <= 1
    assert cp["shift_pct"] > 15
