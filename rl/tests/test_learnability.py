"""Doc 38 learnability features: no lookahead, correct target and level arithmetic."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import numpy as np
import pandas as pd
import pytest

import learnability as L


def _path(n=2880, seed=0):
    rng = np.random.default_rng(seed)
    mid = 42_000 * np.exp(np.cumsum(rng.normal(0, 4e-4, n)))
    imb = rng.uniform(-1, 1, n)
    return mid, imb


def test_features_use_no_future_data():
    mid, imb = _path()
    base = L.features(mid, imb, (43_000.0, 41_000.0, 42_100.0))
    for t in L.decision_points(len(mid))[::5]:
        m2, i2 = mid.copy(), imb.copy()
        m2[t + 1:] *= 1.05  # change the future only
        i2[t + 1:] = 0.0
        f = L.features(m2, i2, (43_000.0, 41_000.0, 42_100.0))
        row_a = base[base.step == t][L.ALL].to_numpy(float)
        row_b = f[f.step == t][L.ALL].to_numpy(float)
        np.testing.assert_allclose(row_a, row_b, err_msg=f"feature at step {t} sees the future")


def test_target_is_forward_mean_gain_in_bp():
    mid = np.full(L.WARMUP + L.H + 10, 100.0)
    t = L.WARMUP
    mid[t + 1:t + L.H + 1] = 99.0  # price falls right after t: selling at t gained 1%
    assert L.target(mid, np.array([t]))[0] == pytest.approx(100.0)


def test_round_number_distance_and_side():
    mid, imb = _path()
    mid = mid / mid * 41_700.0 * np.exp(np.random.default_rng(1).normal(0, 1e-4, len(mid)))
    f = L.features(mid, imb, (np.nan, np.nan, np.nan))
    assert (f.side_1000 == -1).all()  # below the $42,000 level
    assert (f.d_1000 < 0).all() and (f.d_500 > 0).all()  # above $41,500
    assert (f.prev_missing == 1).all() and f.d_prev_close.isna().all()


def test_touches_count_separate_visits():
    mid = np.full(L.WARMUP + 1, 100.0)
    level = 101.0
    for start in (50, 51, 52, 200, 400):  # one visit of 3 steps, then two more
        mid[start] = level
    assert L._touches(mid, L.WARMUP, level, 1e-6) == 3


def test_corrected_subtracts_null_mean():
    s = pd.DataFrame({"date": ["d"] * 3, "path": [-1, 0, 1],
                      "ic": [0.3, 0.1, 0.3], "gain": [5.0, 1.0, 3.0]})
    c = L.corrected(s)
    assert c.loc["d", "ic"] == pytest.approx(0.1)
    assert c.loc["d", "gain"] == pytest.approx(3.0)
