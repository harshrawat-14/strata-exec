# 28 — Does impact scale with local liquidity? Result — no; the daily model stays

Written 2026-11-24. Predictions were pre-registered in
[28-local-liquidity-impact-predictions.md](28-local-liquidity-impact-predictions.md) (commit 24e9f24).
Held-out dates are untouched.

## Summary

Scaling impact by the trailing hour's volume **fits real price moves worse** than the
daily model: validation R² 0.365 against 0.441. Scaling by both the trailing hour's volume
and its volatility ties with the daily model (+0.002). The reason is measured: within a
day, busy hours are also volatile hours. The impact cost per unit of order, σ/√V, barely
moves, so on this data **a busy hour is not cheaper to sell into**. The simulator's daily
model is supported, and the pass criterion fails.

## Result (`rl/experiments/impact_fit_local_liquidity.py`, `rl/results/impact_fit_local_liquidity.csv`)

Main setting: 30 s buckets, 60 min trailing window, Y frozen at the training-date median.

| model | Y | spread of Y across dates (CV) | pooled validation R² | beats day: validation / training dates |
|---|---:|---:|---:|---:|
| **day** (current) | 1.070 | **0.074** | 0.4409 | — |
| local V | 0.876 | 0.104 | 0.3654 (−0.076) | 0/4, 0/12 |
| local σ | 0.848 | 0.127 | 0.4313 (−0.010) | 1/4, 0/12 |
| local both | 1.096 | 0.107 | 0.4428 (+0.002) | 3/4, 6/12 |

The sensitivity checks agree: 15 and 120 min windows, and 5 min buckets. local V is always
worst, by −0.065 to −0.136. local both is always within ±0.011 of day. day always has the
most stable Y.

**Why** (follow-up check, not pre-registered). Across the 16 dates, the 60 min trailing
values give:

| quantity, within a day | sd of log |
|---|---:|
| volume V | 0.69 |
| volatility σ | 0.39 |
| impact per unit order, σ/√V | **0.16** |
| correlation of log σ with log V | 0.92 (0.79–0.97) |

When volume doubles, volatility rises with it, so the price cost of selling a given
quantity stays nearly the same. The leftover ±16% variation in σ/√V does not improve the
fit either (local both ≈ day).

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | local V beats day on criteria 1–2 (+0.01 to +0.05) | **Not met** (−0.076; 0/4, 0/12) |
| 2 | local V lowers the spread of Y; crash dates no longer highest | **Not met** (CV 0.104 vs 0.074) |
| 3 | local σ helps less than local V | **Not met**: neither helps, and local σ hurts less |
| 4 | local both best of the four on pooled validation R² | **Met**, by +0.002 |
| 5 | Same ranking at 5 min | **Not met**: day and local both swap (by 0.0014) |

**Pass criterion: fail.** No local model gains 0.01 R² or lowers the spread of Y.

## What this means

- **The simulator's impact model is right in the respect tested.** Using the day's σ and V
  is not a simplification that hides a liquidity-timing edge. There is little such edge
  in this data.
- With docs 26 and 27, three routes to a regime-aware edge over TWAP have now been checked.
  Forecasting direction fails on two independent pre-checks, and timing liquidity fails here.
  Still untested: whether the *decay* side of impact (the 30/70 permanent split and ρ, carried
  over from era 1, never fitted; doc 21) differs by regime. Also untested: whether the
  16-day sample is simply too small for a weak signal, such as the mean reversion seen in
  doc 27.
