"""
Doc 32 method check: on random walks (no predictability by construction), the raw
within-day correlation of drawdown_z with the forward 2 h return is strongly negative
(Stambaugh bias), and the shuffled-returns null removes it. Without the null, doc 27's
"bounce-back" hint (raw corr -0.16, 9/12 days negative) is indistinguishable from this bias.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_EXP = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(_EXP / "belief_precheck"))
_SPEC = importlib.util.spec_from_file_location("bounce_back_test", _EXP / "bounce_back_test.py")
bb = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bb)

KEYS = ("corr", "deep_sum", "deep_n", "all_sum", "all_n")


def _random_walk_days(n_days=30, seed=7):
    rng = np.random.default_rng(seed)
    recs = []
    for _ in range(n_days):
        mid = 60_000 * np.exp(np.cumsum(rng.standard_t(4, 2880) * 2e-4))
        real, null = bb.day_stats(mid, -1.0, rng, n_null=8)
        recs.append({**{f"{k}_real": v for k, v in zip(KEYS, real)},
                     **{f"{k}_null": v for k, v in zip(KEYS, null)}})
    return pd.DataFrame(recs)


def test_raw_statistic_is_biased_negative_on_random_walks():
    df = _random_walk_days()
    assert df.corr_real.mean() < -0.15
    assert (df.corr_real < 0).mean() > 0.7  # "most days negative" arises with no predictability


def test_null_correction_removes_the_bias():
    df = _random_walk_days()
    s = bb.summarise(df)
    d = df.corr_real - df.corr_null
    se = d.std(ddof=1) / np.sqrt(len(d))
    assert abs(s["corr"]) < 3 * se  # consistent with zero
    assert abs(s["corr"]) < abs(s["corr_raw"]) / 4


def test_shuffled_null_keeps_returns_and_endpoints():
    rng = np.random.default_rng(0)
    mid = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, 500)))
    s = bb.shuffled(mid, rng)
    np.testing.assert_allclose(np.sort(np.diff(np.log(s))), np.sort(np.diff(np.log(mid))))
    assert s[0] == mid[0] and np.isclose(s[-1], mid[-1])


def test_rows_align_forward_return_with_its_start():
    mid = np.linspace(100, 200, 1000)
    dz, fwd = bb.rows(mid)
    k = np.arange(bb.WARMUP, len(mid) - bb.H)
    np.testing.assert_allclose(fwd, (mid[k + bb.H] / mid[k] - 1) * 1e4)
