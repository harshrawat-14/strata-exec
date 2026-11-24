# 29 — How our impact fades, and whether that depends on regime: result

Written 2026-11-24. Predictions were pre-registered in
[29-impact-decay-predictions.md](29-impact-decay-predictions.md) (commit e6d248d).
Held-out dates are untouched.

## Summary

- **The first minutes are wrong in the simulator.** About a quarter of the price response
  to flow reverts within 1–2 minutes: G(30 min)/G(0) = 0.755 [0.598, 0.869]. The simulator
  assumes 0.965. Pre-registered criterion A judged only 2 h and 6 h, so this does not trigger
  recalibration by itself. It is reported as a finding.
- **The long run is not measurable from 12 days.** At 2 h and 6 h the 90% intervals are
  0.52–1.21 and 0.47–1.44. The simulator (0.873, 0.707) sits inside both, and so does
  almost anything.
- **No regime dependence is established** (criterion B fails). High-vol periods show slower
  2 h reversion than low-vol periods (+0.35), and validation agrees in sign, but the
  training interval [−0.34, +0.97] includes 0.

## Result (`rl/experiments/impact_decay_fit.py`, `rl/results/impact_decay_fit.csv`)

**A. Pooled decay, training dates** (90% date-bootstrap interval):

| τ | fitted G(τ)/G(0) | 90% interval | simulator | validation dates |
|---|---:|---|---:|---:|
| 30 min | 0.755 | 0.598–0.869 | 0.965 (outside) | 0.691 |
| 2 h | 0.932 | 0.524–1.206 | 0.873 (inside) | 0.923 |
| 6 h | 0.990 | 0.468–1.437 | 0.707 (inside) | 1.058 |

The kernel per lag, in units of G(0):

- lag 1: −0.21;
- lags 2–3: −0.04 each;
- lags 4–7: −0.01 each;
- lags 8–255: about +0.001 to +0.003 each.

The quick reversion is real and large. The slow positive drift afterwards takes G back
toward 1, which looks like flow that keeps coming in the same direction (order splitting,
informed flow) rather than permanent impact. A market-wide regression cannot separate the
two (stated as a caveat in the design).

**B. By trailing 2 h volatility tercile** (at the time of the flow):

| tercile | G(0) | 30 min | 2 h | 6 h | validation 2 h |
|---|---:|---:|---:|---:|---:|
| low vol | 0.862 | 0.576 | 0.541 | 1.491 | 0.631 |
| mid vol | 1.013 | 0.790 | 1.137 | 1.457 | 1.208 |
| high vol | 1.289 | 0.755 | 0.893 | 0.726 | 0.922 |

Against the three parts of criterion B:

1. High − low at 2 h is +0.352, with interval [−0.340, +0.966], which **includes 0**.
2. Validation high − low is +0.291, the same sign.
3. Validation R²: tercile 0.48247 ≥ shared 0.48138 (pooled 0.47365).

Overall, **B fails** on condition 1. The pattern is not monotone either: mid vol has the
highest 2 h value. The R² gain comes almost entirely from the tercile lag-0 term (shared
over pooled: +0.008), which is doc 28's local-volatility scaling again, not decay.

The hindsight day labels, descriptive only: calm 0.62 / 0.72 / 0.54, normal 0.82 / 1.04 / 1.12,
crash 0.71 / 0.88 / 1.21 (30 min / 2 h / 6 h).

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | Faster short-lag reversion: G(30 min)/G(0) < 0.965 | **Met** (0.755) |
| 2 | Simulator's 6 h value inside a wide interval | **Met** (0.47–1.44) |
| 3 | Simulator outside the interval at 2 h | **Not met** (inside 0.52–1.21) |
| 4 | Criterion B fails | **Met** (interval includes 0) |

Note: the design doc tabulates the simulator's 6 h value as 0.708; the exact value is 0.707.

## What this means

- **Decay is not a demonstrated regime-aware edge** on 12 training days. The point estimates
  hint at slower reversion in high-vol periods, but 12 days cannot tell that from noise.
- **Every pre-check so far is limited by the 16-day sample.** The long-run decay intervals
  here span 0.5–1.4. Doc 27's mean-reversion hint held on 9 of 12 days. Neither can be
  settled without more days. Bybit's free archive (`fetch_bybit.py`, ~110 MB per converted day)
  makes that feasible.
- The fast 1–2 minute reversion is a simulator inaccuracy. It mainly affects aggressive
  slicing within minutes, not the hour-scale pacing the agent controls. Fixing it is
  a candidate simulator change, but not urgent for pacing.
