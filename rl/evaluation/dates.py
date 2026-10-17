"""
Date universe, train/held-out splits (V1, V2) and severity ranks.

Also holds the deprecated research-sim point estimates (STATIC_OPTIMAL_IS
etc.), kept only as historical reference values.
"""




TEST_DATES = {
    "2024-04-15": "BTC correction",
    "2024-07-10": "mid-summer sideways",
    "2024-12-10": "December bull run",
}

DATES = {
    "2024-01-15": "calm bull",
    # Was labeled "BTC breakout" -- see HELD_OUT_CRASH_DATES below: this is
    # actually the cleanest liquidation-cascade signature in the dataset.
    "2024-03-05": "crash - flash liquidation cascade",
    "2024-06-10": "quiet consolidation",
    "2024-08-05": "crash - Yen unwind",
    "2024-11-06": "post-election surge",
}

# ── Phase 0 canonical train/eval date split (Global Rule 9) ───────────────
# Declared here so Phase 2's retrain has one source of truth; enforced by
# an executable guard (rl/tests/test_date_split.py) rather than left as
# prose. Do NOT retrain in Phase 0 -- this only declares the split.
#
# Stress-day survey: of the 11 dates with clean BookDepth+AggTrades data,
# only two show a genuine liquidation-cascade signature -- a large adverse
# intraday move AND book depth thinning coincident (within ~1 min) with the
# price drop, not just high volume or volatility:
#   2024-08-05  -15.9% intraday, well-documented Yen-unwind carry-trade
#               crash (mandatory holdout).
#   2024-03-05  -13.5% intraday, the cleanest signature of any date: the
#               bid-depth minimum and the price minimum are 47 seconds
#               apart, and depth drops to 22% of its daily median right at
#               the crash. Previously mislabeled "BTC breakout" above.
# The README's previously-documented --multi-date training invocation
# included BOTH of these dates, meaning any existing "RL beats AC on the
# crash day" claim is unverified until a model is retrained on this split
# (Global Rule 9) -- see README for the corresponding flag on that claim.
#
# 2024-04-15 (-5.1% intraday, moderate but real coincident depth thinning,
# ratio 0.63) is the next most-stressed date and is the only remaining
# candidate that shows genuine stress rather than plain volatility (checked
# and rejected: 2024-02-20, 2024-09-15, 2024-10-15, 2024-12-10 all show
# depth-thinning ratios >= 0.9 at their own price minimum -- volatile but
# not a liquidity event). It is moved from TEST_DATES into TRAINING_DATES
# below (approved 2026-07-11) so the training set retains at least one
# genuine crash-like day without touching either holdout.
HELD_OUT_CRASH_DATES = ["2024-08-05", "2024-03-05"]

TRAINING_DATES = [
    "2024-01-15", "2024-02-20", "2024-04-15", "2024-06-10",
    "2024-09-15", "2024-10-15", "2024-11-06",
]

assert not (set(TRAINING_DATES) & set(HELD_OUT_CRASH_DATES)), (
    "TRAINING_DATES / HELD_OUT_CRASH_DATES overlap -- a held-out crash "
    "date would be trained on, invalidating the crash-day comparison"
)

# ── Phase 3 V2 date split: 25-date pool, regime-balanced ──────────────────
# TRAINING_DATES/HELD_OUT_CRASH_DATES above are UNCHANGED and remain valid
# for reproducing any existing result -- do not repoint old scripts at V2
# silently. This is a new, additive split for any NEW work (retraining,
# eval) going forward, kept alongside V1 the same way Phase 2R kept the
# Task 2/3 simulator provenance boundary explicit rather than overwriting
# history. See docs/proposals/research-roadmap.md, "V2 date split" for the full writeup.
#
# Motivation (Phase 3 Step 0 finding, docs/proposals/research-roadmap.md section A): V1's
# TRAINING_DATES contains ZERO severe-crash days (0/7) -- both severe
# crashes are the holdout, by construction. That's a real, honestly-earned
# result for the λ-calibration question V1 was built to answer (a real desk
# tuning λ also has no future crash to tune on), but it leaves no way to
# ask a DIFFERENT, equally real question: does a model that has SEEN crash
# dynamics (from a different named event) generalize to crash dynamics it
# hasn't seen? V1 structurally cannot answer that. V2 can.
#
# Data: the TradeData pool grew from 11 to 25 dates (14 new dates pulled by
# the Phase 2R severity expansion, in three named-event windows -- see
# docs/experiments/04-crash-severity-expansion.md). All 25 are classified below.
#
# Severity metric: AC's own honest-λ realized cost (ac_mean_IS) on each
# date -- the same outcome-independent severity proxy the severity-
# expansion doc already validated (not a new metric invented for this
# split). Source: rl/results/phase2r_ablation_honest_lambda.csv (11
# original dates) union rl/results/phase2r_severity_expansion_full.csv (14
# new dates). Values frozen here (not recomputed at import time) since
# they're a one-time classification decision, same pattern as
# AC_DEFAULT_LAMBDA-style constants elsewhere in this file.
#
# Regime label: tercile split of all 25 dates by ac_mean_IS (most negative
# = most severe): ranks 1-8 = "crash", 9-17 = "normal", 18-25 = "calm".
# Boundaries are a rank cut on a continuous severity spectrum, not a
# natural category boundary -- treat "normal" vs "calm" at the margin
# (e.g. 2024-09-15 at rank 18 vs 2024-04-14 at rank 17) as a soft line, not
# a claim that those two days are qualitatively different regimes.
SEVERITY_RANK_V2 = {
    # date: (ac_mean_IS, regime, event_window)
    "2024-08-05": (-7.889, "crash", "C (jobs-report -> Aug 5 culmination)"),
    "2024-07-04": (-3.144, "crash", "B (German BTC sales / Mt. Gox fear)"),
    "2024-07-05": (-2.794, "crash", "B"),
    "2024-03-05": (-2.460, "crash", "isolated (flash liquidation cascade)"),
    "2024-08-02": (-2.368, "crash", "C"),
    "2024-07-07": (-1.418, "crash", "B"),
    "2024-08-04": (-1.102, "crash", "C"),
    "2024-04-15": (-1.095, "crash", "A (Iran-Israel escalation)"),
    "2024-04-13": (-0.893, "normal", "A"),
    "2024-10-15": (-0.861, "normal", "isolated"),
    "2024-08-03": (-0.783, "normal", "C"),
    "2024-12-10": (-0.720, "normal", "isolated"),
    "2024-04-12": (-0.649, "normal", "A"),
    "2024-07-12": (-0.577, "normal", "B"),
    "2024-06-10": (-0.569, "normal", "isolated"),
    "2024-02-20": (-0.527, "normal", "isolated"),
    "2024-04-14": (-0.511, "normal", "A"),
    "2024-09-15": (-0.478, "calm", "isolated"),
    "2024-07-08": (-0.257, "calm", "B"),
    "2024-07-06": (-0.209, "calm", "B"),
    "2024-07-11": (-0.063, "calm", "B"),
    "2024-07-10": (0.027, "calm", "isolated"),
    "2024-07-09": (0.466, "calm", "B"),
    "2024-01-15": (0.905, "calm", "isolated"),
    "2024-11-06": (4.977, "calm", "isolated"),
}
assert len(SEVERITY_RANK_V2) == 25

# Split design, applied at the WINDOW level, not the individual-day level,
# for the three multi-day event windows (A, B, C) -- splitting individual
# days out of the same named event across train/held-out risks leaking
# that specific event's dynamics across the boundary (a model trained on
# 2024-04-12 could partly memorize the Iran-Israel episode it would then
# be "tested on generalizing to" via 2024-04-15). Isolated single dates
# have no such adjacency and are assigned individually.
#   Window A (04-12..04-15): kept ENTIRELY in training -- extends the
#     existing 2024-04-15 precedent (already a V1 TRAINING_DATE) rather
#     than splitting a 4-day episode across the boundary.
#   Window B (07-04..07-12): split at ONE temporal boundary, not
#     interleaved -- the early/acute six days (07-04..07-09, includes all
#     three window-B crash days) go to training, the later three
#     (07-10..07-12, calmer/recovery) stay held-out. 2024-07-10 was
#     already a V1 held-out anchor; this preserves it rather than moving
#     it, and gives held-out a short post-event recovery arc instead of
#     scattering days across the same acute window.
#   Window C (08-02..08-05): kept ENTIRELY held-out -- these four days are
#     the direct run-up to the 2024-08-05 flagship crash holdout; training
#     on any of them would leak into the one comparison this project's
#     headline result depends on most.
TRAINING_DATES_V2 = [
    # V1's original 7
    "2024-01-15", "2024-02-20", "2024-04-15", "2024-06-10",
    "2024-09-15", "2024-10-15", "2024-11-06",
    # Window A extras (all normal-regime, 2024-04-15 already above)
    "2024-04-12", "2024-04-13", "2024-04-14",
    # Window B early half (includes 3 of the 4 non-August crash days)
    "2024-07-04", "2024-07-05", "2024-07-06",
    "2024-07-07", "2024-07-08", "2024-07-09",
]

HELD_OUT_DATES_V2 = [
    # V1's original holdouts
    "2024-08-05", "2024-03-05", "2024-07-10", "2024-12-10",
    # Window B late half (post-event recovery)
    "2024-07-11", "2024-07-12",
    # Window C, entire (run-up to the 08-05 holdout)
    "2024-08-02", "2024-08-03", "2024-08-04",
]

assert not (set(TRAINING_DATES_V2) & set(HELD_OUT_DATES_V2)), (
    "TRAINING_DATES_V2 / HELD_OUT_DATES_V2 overlap"
)
assert set(TRAINING_DATES_V2) | set(HELD_OUT_DATES_V2) == set(SEVERITY_RANK_V2), (
    "TRAINING_DATES_V2 + HELD_OUT_DATES_V2 must cover exactly the 25 "
    "classified dates -- a date was dropped or a typo introduced"
)

# Regime composition actually achieved (verified by the test below, not
# just asserted here): crash 4 train / 4 held-out (FIXES the V1 structural
# bind -- 0/7 severe days in training); normal 6 train / 3 held-out; calm
# 6 train / 2 held-out. Every regime has "substantial" (>=2) presence on
# both sides, and crash exposure is now balanced rather than entirely
# absent from training.

# ── DEPRECATED for cross-strategy comparison ──────────────────────────────
# These four dicts are point estimates from the Rust `research-sim` binary's
# Monte Carlo (GBM/GARCH) pipeline with a square-root impact model -- a
# DIFFERENT simulator than the counterfactual LOB replay
# (evaluate_all_strategies_on_date) that produces RL's numbers below. Putting
# these side-by-side with live RL/LOB numbers in one table is the exact
# mixed-simulator bug Phase 0 exists to fix: it is not a valid comparison.
#
# They are retained ONLY as historical reference values, gated behind
# `legacy_mixed_comparison=True` in run_full_analysis (off by default).
# For any real strategy comparison, use evaluate_all_strategies_on_date,
# which runs every strategy through the identical counterfactual LOB path.
STATIC_OPTIMAL_IS = {
    "2024-01-15": -0.9900,
    "2024-03-05": -1.2275,
    "2024-06-10": -0.8684,
    "2024-08-05": -1.0714,
    "2024-11-06": -1.2122,
}
ADAPTIVE_OPTIMAL_IS = {
    "2024-01-15": -0.4935,
    "2024-03-05": -0.5861,
    "2024-06-10": -0.4494,
    "2024-08-05": -0.5854,
    "2024-11-06": -0.5816,
}

TWAP_IS = {
    "2024-01-15": 1.9869,
    "2024-03-05": 2.0544,
    "2024-06-10": 1.9491,
    "2024-08-05": 1.9156,
    "2024-11-06": 1.9261,
}

HEURISTIC_IS = {
    "2024-01-15": 1.8797,
    "2024-03-05": 1.9531,
    "2024-06-10": 1.8417,
    "2024-08-05": 1.8078,
    "2024-11-06": 1.8189,
}
