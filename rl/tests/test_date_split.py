"""
Executable guard for the Phase 0 train/eval date split (Global Rule 9).

A learned policy evaluated on a date it trained on is partly memorizing
that LOB path -- AC/TWAP/Heuristic can't memorize, so an in-sample
comparison is structurally unfair to them, and any "RL beats AC on the
crash day" claim built on it is unverified. This test exists so that a
future edit to TRAINING_DATES or HELD_OUT_CRASH_DATES in evaluate.py can
never silently reintroduce the overlap the Phase 0 audit found (both
2024-08-05 and 2024-03-05 were in the previously-documented training set).
"""
from __future__ import annotations

from evaluate import HELD_OUT_CRASH_DATES, TRAINING_DATES


def test_no_overlap_between_training_and_held_out_crash_dates():
    overlap = set(TRAINING_DATES) & set(HELD_OUT_CRASH_DATES)
    assert not overlap, (
        f"held-out crash date(s) {overlap} present in TRAINING_DATES -- "
        f"this reintroduces the train/eval overlap Phase 0 found and "
        f"invalidates any crash-day beats-AC claim"
    )


def test_both_mandated_holdout_dates_present():
    # 2024-08-05 (Yen unwind) is explicitly mandated as a holdout.
    # 2024-03-05 was approved as the second holdout (cleanest cascade
    # signature in the stress-day survey).
    assert "2024-08-05" in HELD_OUT_CRASH_DATES
    assert "2024-03-05" in HELD_OUT_CRASH_DATES
    assert len(HELD_OUT_CRASH_DATES) >= 2


def test_training_set_retains_at_least_one_stressed_date():
    # 2024-04-15 was the one remaining date with a genuine (if milder)
    # coincident depth-thinning signature; without it the training set
    # would be entirely crash-blind.
    assert "2024-04-15" in TRAINING_DATES
