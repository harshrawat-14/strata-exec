# 29 — How our impact fades, and whether that depends on regime: design and predictions

Written 2026-11-24, before the fit is run. No RL is trained.

## Why

Since doc 21, the simulator's impact has decayed as

    D_t = π · Q_t + (1 − π) · Σ_s q_s · e^(−ρ (t − s)),   I_t = Y · σ_d · √(D_t / V)

with π = 0.3 (permanent share) and ρ = 5/day (half-life ≈ 400 steps, 3.3 h). Both values were
carried over from era 1 and **never fitted**. Docs 26–28 ruled out direction forecasting and
liquidity timing as sources of a regime-aware edge over TWAP. The remaining impact-side
route is decay: how much of the price move from selling persists, and how fast the rest
reverts. If decay differs with an observable condition, the best pace depends on it, and
that is a regime-aware edge that needs no direction forecast.

## Method (`rl/experiments/impact_decay_fit.py`)

The fit is a propagator-style regression on Bybit 30 s buckets, using doc 28's daily
normalisation:

    r_t = Σ_b β_b · Σ_{l ∈ bin b} x_{t−l},   x_t = σ_d · sign(Q_t) · √(|Q_t| / V)

- **Lag bins:** {0}, {1}, 2–3, 4–7, … 256–511, and 512–720 (720 lags = 6 h).
- **Price response:** to one unit of flow, the response τ steps later is
  G(τ) = Σ_l≤τ β(l). The decay curve reported is G(τ)/G(0).
- **Rows:** the first 6 h of each day are dropped, so every lag is defined. There is no
  intercept.
- **Uncertainty:** a date-level bootstrap over the 12 training dates (1,000 resamples);
  90% intervals are reported.
- **Caveat:** taker flow is autocorrelated (order splitting). The regression separates lags
  linearly but cannot remove all of this, so G is a market-wide estimate, not a clean
  metaorder kernel.

**What the simulator implies** for one small sale: G(τ)/G(0) = √(π + (1 − π) e^(−ρτ)).

| τ | 30 min | 2 h | 6 h |
|---|---:|---:|---:|
| simulator (π 0.3, ρ 5/day) | 0.965 | 0.873 | 0.708 |

**Regime split.** The split uses an observable condition: trailing 2 h realised volatility at
the time of the flow, in terciles with thresholds set on the training rows. Two models are
compared:

- **shared decay:** a tercile-specific lag-0 coefficient, with all other lag coefficients
  shared;
- **tercile decay:** every lag coefficient tercile-specific.

As an extra, labelled descriptive, the hindsight day labels (calm / normal / crash) are also
reported.

## Criteria

**A. Is the simulator's decay right?** Recalibrate (a separate, tested change) if the
simulator's G(τ)/G(0) falls outside the training-date 90% bootstrap interval at 2 h or 6 h.

**B. Does decay depend on regime?** This passes only if all of the following hold:

1. The high-vol minus low-vol difference in G(2h)/G(0) has a 90% training-date bootstrap
   interval that excludes 0.
2. The validation dates' point estimate has the same sign.
3. On validation, tercile decay has a pooled R² ≥ shared decay's.

If B passes, the next step is to put regime-dependent decay into the simulator, and to
check that a simple pace rule keyed to trailing volatility beats TWAP in every regime before
any RL. If B fails, decay is not a regime-aware edge on this data.

## Predictions

1. **The real response reverts faster at short lags than the simulator:** G(30 min)/G(0) < 0.965.
2. **The simulator's 6 h value (0.708) lies inside the 90% interval.** The long-run level is
   poorly pinned down by 12 dates, so the interval will be wide.
3. **Criterion A at 2 h: the simulator falls outside the interval** (recalibration needed).
4. **Criterion B fails:** the regime difference does not exclude 0, or does not replicate on
   validation.
