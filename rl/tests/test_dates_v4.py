"""The V4 full-year split keeps held-out dates, their buffer and the future set apart."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.dates import (EXCLUDED_BUFFER_V4, HELD_OUT_DATES_V3, HELD_OUT_FUTURE_V4,
                              SEVERITY_RANK_V2, TRAINING_DATES_V3, TRAINING_DATES_V4,
                              VALIDATION_DATES_V3, VALIDATION_DATES_V4, regime_v4)
from market_data import book_path


def test_splits_are_disjoint_and_exclude_held_out_and_buffer():
    train, val = set(TRAINING_DATES_V4), set(VALIDATION_DATES_V4)
    blocked = set(HELD_OUT_DATES_V3) | set(EXCLUDED_BUFFER_V4) | set(HELD_OUT_FUTURE_V4)
    assert not train & val
    assert not (train | val) & blocked
    assert max(TRAINING_DATES_V4) < min(VALIDATION_DATES_V4)  # out of time


def test_buffer_covers_two_days_each_side_of_every_held_out_date():
    assert "2024-03-03" in EXCLUDED_BUFFER_V4 and "2024-03-07" in EXCLUDED_BUFFER_V4
    assert "2024-08-07" in EXCLUDED_BUFFER_V4 and "2024-12-12" in EXCLUDED_BUFFER_V4
    # V2 training days next to held-out window B keep their place
    assert "2024-07-09" in TRAINING_DATES_V4 and "2024-07-09" not in EXCLUDED_BUFFER_V4


def test_pool_size():
    # 366 days - 9 held-out - 14 new buffer days
    assert len(EXCLUDED_BUFFER_V4) == 14
    assert len(TRAINING_DATES_V4) + len(VALIDATION_DATES_V4) == 343


@pytest.mark.parametrize("drift,label", [(-1.0, "crash"), (-0.99, "normal"),
                                         (0.549, "normal"), (0.55, "calm")])
def test_regime_cut_edges(drift, label):
    assert regime_v4(drift) == label


@pytest.mark.skipif(not all(os.path.exists(book_path(d)) for d in SEVERITY_RANK_V2),
                    reason="requires the 25 V2 book files")
def test_rule_reproduces_14_of_16_v3_labels():
    from evaluation.day_regime import day_stats
    dates = list(TRAINING_DATES_V3) + list(VALIDATION_DATES_V3)
    agree = [day_stats(d)["regime"] == SEVERITY_RANK_V2[d][1] for d in dates]
    assert sum(agree) == 14


@pytest.mark.parametrize("drift,scale,label", [(-1.9, 2.0, "normal"), (-2.0, 2.0, "crash"),
                                               (1.09, 2.0, "normal"), (1.1, 2.0, "calm")])
def test_regime_cuts_scale_with_volatility(drift, scale, label):
    """Doc 34: a coin k times as volatile as BTC needs k times the drift for each label."""
    from evaluation.dates import regime_v4
    assert regime_v4(drift, scale) == label
