"""
Executable guard for the Phase 3 V2 date split (docs/proposals/research-roadmap.md,
"V2 date split").

V1's TRAINING_DATES has zero severe-crash days by construction -- both
crashes are the holdout. V2 exists to answer a different question (does a
model that has seen SOME crash dynamics generalize to a crash it hasn't
seen) and that only holds if the regime balance below actually holds --
this test exists so a future edit to TRAINING_DATES_V2/HELD_OUT_DATES_V2
can't silently regress back to zero crash exposure in training, or
silently drop a date from the 25-date pool.
"""
from __future__ import annotations

from evaluate import (
    SEVERITY_RANK_V2,
    TRAINING_DATES_V2,
    HELD_OUT_DATES_V2,
)


def _regime_counts(dates):
    counts = {"crash": 0, "normal": 0, "calm": 0}
    for d in dates:
        counts[SEVERITY_RANK_V2[d][1]] += 1
    return counts


def test_no_overlap_between_v2_training_and_held_out():
    overlap = set(TRAINING_DATES_V2) & set(HELD_OUT_DATES_V2)
    assert not overlap, f"V2 train/held-out overlap: {overlap}"


def test_v2_covers_all_25_classified_dates_exactly_once():
    combined = set(TRAINING_DATES_V2) | set(HELD_OUT_DATES_V2)
    assert combined == set(SEVERITY_RANK_V2)
    assert len(TRAINING_DATES_V2) + len(HELD_OUT_DATES_V2) == 25


def test_v2_training_has_substantial_crash_exposure():
    # The entire point of V2 over V1: training must NOT be crash-blind.
    counts = _regime_counts(TRAINING_DATES_V2)
    assert counts["crash"] >= 3, (
        f"V2 training set has only {counts['crash']} crash-regime dates -- "
        f"this reintroduces V1's structural bind (0/7 crash exposure in "
        f"training), which V2 exists to fix"
    )


def test_v2_held_out_still_has_crash_dates_for_a_real_generalization_test():
    # Training exposure alone isn't enough -- held-out must still contain
    # genuine unseen crashes, or there's no generalization claim left to make.
    counts = _regime_counts(HELD_OUT_DATES_V2)
    assert counts["crash"] >= 2
    assert "2024-08-05" in HELD_OUT_DATES_V2, "flagship crash holdout must remain held out"
    assert "2024-03-05" in HELD_OUT_DATES_V2, "second crash holdout must remain held out"


def test_every_regime_has_substantial_presence_on_both_sides():
    train_counts = _regime_counts(TRAINING_DATES_V2)
    held_counts = _regime_counts(HELD_OUT_DATES_V2)
    for regime in ("crash", "normal", "calm"):
        assert train_counts[regime] >= 2, f"{regime} underrepresented in training: {train_counts}"
        assert held_counts[regime] >= 2, f"{regime} underrepresented in held-out: {held_counts}"


def test_window_c_entirely_held_out_no_leakage_into_flagship_crash():
    # 2024-08-02/03/04 are the direct run-up to the 08-05 holdout this
    # project's headline comparison depends on most -- none may leak into
    # training.
    window_c = {"2024-08-02", "2024-08-03", "2024-08-04", "2024-08-05"}
    assert window_c.issubset(set(HELD_OUT_DATES_V2))
    assert not (window_c & set(TRAINING_DATES_V2))
