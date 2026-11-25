# 31 — Impact decay on the full year: result

Predictions in [31-impact-decay-full-year-predictions.md](31-impact-decay-full-year-predictions.md).
Run: `python rl/experiments/impact_decay_fit.py v4` → `rl/results/impact_decay_fit_v4.csv`
(256 training and 87 validation days; held-out and future dates not read).

## A. The simulator's decay is wrong: criterion A triggers

| G(τ)/G(0) | training (90% interval) | validation | simulator | |
|---|---|---:|---:|---|
| 30 min | **0.649** [0.621, 0.677] | 0.633 | 0.965 | outside |
| 2 h | **0.692** [0.629, 0.758] | 0.644 | 0.873 | **outside → recalibrate** |
| 6 h | **0.676** [0.587, 0.774] | 0.630 | 0.707 | inside |

**Per-lag kernel (relative to the lag-0 coefficient):**
- lag 1: −0.20
- lags 2–3: −0.05 each
- lags 4–7: −0.013 each
- lags 8–15: −0.003 each
- after that, about zero (slightly positive)

**Shape of real impact:** about 35% of the price response to taker flow reverts within roughly
8 minutes (16 lags of 30 s), then the response is flat out to 6 h. The simulator has the
opposite shape. It has no fast reversion, and it reverts gradually over hours: √(0.3 + 0.7 e^(−ρτ)),
with a 3.3 h half-life, falling to 0.55 in the long run. The two agree at about 6 h only
because the simulator's slow decay crosses the real plateau there.

**Caveat:** taker flow is autocorrelated, so this is a market-wide response, not a clean
metaorder kernel. That applies equally to doc 29.

## B. Decay does not depend on volatility: criterion B fails

| | value | condition |
|---|---|---|
| high − low G(2h)/G(0), training | −0.055 [−0.170, +0.068] | 1. excludes 0: **no** |
| validation high − low | −0.022 | 2. same sign: yes |
| validation R²: pooled / shared / tercile | 0.4317 / 0.4355 / 0.4361 | 3. tercile ≥ shared: yes |
| magnitude | 0.055 | 4. ≥ 0.05: yes |

Doc 29's +0.35 was noise: with 21 times as many days, the estimate is slightly negative and
its interval spans zero. The high-volatility tercile has a larger instantaneous response
(lag-0 coefficient 1.18 vs 0.89 in low volatility, relative units), but the shape of its decay
is the same.

**Descriptive (hindsight labels, not a criterion):**

| label | days | 2 h | 6 h |
|---|---:|---:|---:|
| calm | 109 | 0.68 | 0.64 |
| normal | 180 | 0.71 | 0.70 |
| crash | 54 | 0.63 | 0.63 |

## Predictions

| # | prediction | result | |
|---|---|---|---|
| 1 | G(30 min)/G(0) < 0.9; interval excludes 0.965 | 0.649 [0.621, 0.677] | met |
| 2 | 2 h and 6 h interval widths at most half of doc 29's | 0.13 (was 0.68); 0.19 (was 0.97) | met |
| 3 | criterion A triggers at 6 h | it triggers at 2 h; 6 h is inside | not met |
| 4 | B fails; difference positive, 0 to +0.15 | B fails, but the difference is −0.055 | half met (sign wrong) |

## What follows

1. **Decay is ruled out as a regime-aware edge.** The route joins direction forecasting
   (docs 26, 27) and liquidity timing (doc 28).
2. **Criterion A was pre-registered to trigger recalibration.** The simulator's decay
   (π, ρ, and probably its functional form, given the fast-then-flat shape) needs refitting as a
   separate, tested change with its own pre-registration. This changes the costs every
   baseline and model is scored on.
3. **The remaining regime-aware route** is doc 27's bounce-back hint, rerun on the V4 pool.
   It uses only prices, so it does not depend on the recalibration.
