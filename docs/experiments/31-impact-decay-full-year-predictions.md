# 31 — Impact decay on the full year: design and predictions

Written 2026-11-25, before the fit is run on the V4 pool. No RL is trained.

## Why

Doc 29 asked whether impact decays differently in high and low volatility. With 12 training
days, the high-vol minus low-vol difference in 2 h reversion was +0.35 with a 90% interval of
[−0.34, +0.97]. The validation days agreed in sign, but the interval included zero, so the
test failed for lack of data. Doc 30 grew the pool to 256 training and 87 validation days.
This doc reruns the same test on that pool.

## Method

The model is unchanged from doc 29 (`rl/experiments/impact_decay_fit.py v4`):

- Propagator regression on 30 s buckets, with the same lag bins and the same daily
  normalisation.
- Volatility terciles from trailing 2 h volatility at the flow's time. The cuts are set on
  the V4 training rows.
- A 1,000-resample date bootstrap over the 256 training days, with 90% intervals.
- Hindsight labels (doc 30's drift rule) are reported as a descriptive extra.

The refactor that added the pool argument reproduces doc 29's results file byte for byte.

## Criteria

**A. Is the simulator's decay right?** As in doc 29: recalibrate (a separate, tested
change) if the simulator's G(τ)/G(0) lies outside the training 90% interval at 2 h or 6 h.

**B. Does decay depend on volatility?** Doc 29's three conditions, plus a fourth:

1. The training 90% interval for high − low G(2h)/G(0) excludes 0.
2. The validation point estimate has the same sign.
3. On validation, tercile decay has a pooled R² ≥ shared decay's.
4. **New:** |high − low| ≥ 0.05.

Condition 4 is added now because, with 256 days, a difference too small to change the best
pace could still exclude zero. The 0.05 floor is a judgement call (one seventh of doc 29's
estimate), not derived from a pacing calculation; it is fixed here, before the run.

**If B passes:** the next step is to add volatility-dependent decay to the simulator (as a
tested change), then check that a simple pace rule keyed to trailing volatility beats TWAP
in every regime, before any RL. **If B fails:** decay is ruled out as a regime-aware edge on
this data, and the next test is doc 27's bounce-back hint.

## Predictions

1. **The fast early reversion replicates:** G(30 min)/G(0) < 0.9, and the interval now
   excludes the simulator's 0.965.
2. **The intervals narrow:** the 2 h and 6 h interval widths are at most half of doc 29's
   (0.68 and 0.97).
3. **Criterion A triggers at 6 h:** the simulator's 0.707 lies below the interval. Doc 29's
   point estimates (0.93 at 2 h, 0.99 at 6 h) suggest that real impact barely decays after
   the first minutes, while the simulator reverts 30% of it within 6 h.
4. **Criterion B fails:** the difference is real in sign (positive: less reversion in high
   volatility) but smaller than doc 29's +0.35. Point estimate between 0 and +0.15, with
   condition 1 or 4 failing.
