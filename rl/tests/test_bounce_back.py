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


def test_signflip_null_keeps_volatility_clustering_and_removes_bias():
    rng = np.random.default_rng(3)
    mid = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, 500)))
    s = bb.signflip(mid, rng)
    np.testing.assert_allclose(np.abs(np.diff(np.log(s))), np.abs(np.diff(np.log(mid))))  # sizes in order
    recs = []
    for _ in range(30):
        # GARCH-like clustering: the case the shuffled null does not preserve
        vol = 2e-4 * np.exp(np.cumsum(rng.normal(0, 0.05, 2880)))
        m = 60_000 * np.exp(np.cumsum(rng.normal(0, 1, 2880) * vol))
        real, null = bb.day_stats(m, -1.0, rng, n_null=8, null="signflip")
        recs.append({**{f"{k}_real": v for k, v in zip(KEYS, real)},
                     **{f"{k}_null": v for k, v in zip(KEYS, null)}})
    df = pd.DataFrame(recs)
    d = df.corr_real - df.corr_null
    assert abs(d.mean()) < 3 * d.std(ddof=1) / np.sqrt(len(d))


def _s(corr, excess):
    return {"corr": corr, "excess": excess}


def test_two_sided_accepts_momentum_and_bounce():
    """Doc 34: a significant signal of either sign passes; the sign is reported."""
    mom = ((_s(0.05, 8.0), _s(0.02, 3.0), _s(0.08, 12.0)), (_s(0.04, 7.0), _s(0.01, 1.0), _s(0.07, 13.0)))
    assert bb.two_sided(*mom, _s(0.03, 0), 5.0) == (True, 1)
    neg = tuple(tuple(_s(-d["corr"], -d["excess"]) for d in (a, c, b)) for a, b, c in mom)
    assert bb.two_sided(*neg, _s(-0.03, 0), 5.0) == (True, -1)


def test_two_sided_rejects_weak_or_inconsistent_signals():
    good_t = (_s(0.05, 8.0), _s(0.02, 3.0), _s(0.08, 12.0))
    good_v = (_s(0.04, 7.0), _s(0.01, 1.0), _s(0.07, 13.0))
    crash = _s(0.03, 0)
    assert not bb.two_sided((_s(0.05, 8.0), _s(-0.01, 3.0), _s(0.08, 12.0)), good_v, crash, 5.0)[0]  # interval spans 0
    assert not bb.two_sided(good_t, (_s(-0.04, 7.0), _s(-0.07, 1.0), _s(-0.01, 13.0)), crash, 5.0)[0]  # val flips
    assert not bb.two_sided(good_t, good_v, crash, 10.0)[0]  # excess below the scaled floor
    assert not bb.two_sided(good_t, good_v, _s(-0.03, 0), 5.0)[0]  # crash days reverse
