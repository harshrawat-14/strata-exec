# 38 — Learnability: result — fails; price levels do not say when to sell faster

Run 2026-11-26 with `rl/experiments/learnability.py`, as pre-registered in
`38-learnability-predictions.md`. Held-out and future dates were not read. Per-day scores:
`rl/results/learnability_days.csv`. Tests: `rl/tests/test_learnability.py` (no lookahead,
mutation-checked against a centred rolling high; target; round-number levels; touches; null
correction).

## Data

256 training days (9,216 decisions) and 87 validation days (3,132 decisions, plus 20
sign-flipped null paths each). The target y, the gain from selling now instead of over the
next 2 h, has an sd of 45.8 bp (training) and 44.6 bp (validation).

## Result (corrected = real − null; 90% intervals over days)

| | validation IC, corrected | timing gain, corrected (bp) | crash | calm | verdict |
|---|---:|---:|---:|---:|---|
| **H1 all inputs, ridge** | +0.035 [−0.003, +0.076] | −0.64 [−2.37, +1.26] | −3.13 | −1.29 | **FAIL** (0 of 3) |
| H1 all inputs, boosting | +0.010 [−0.029, +0.054] | −0.62 [−2.21, +1.01] | −5.92 | −1.13 | reported only |
| **H2 levels only, ridge** | +0.020 [−0.019, +0.062] | −0.65 [−2.22, +0.94] | −4.71 | −1.65 | **FAIL** (0 of 3) |
| H2 levels only, boosting | +0.016 [−0.026, +0.060] | −0.78 [−2.64, +0.92] | −2.59 | −2.12 | reported only |

The pass rule needed a corrected gain of at least 3 bp. Every model is below zero, and every
model loses on crash days. Crash days are where selling faster should pay.

## Predictions

| # | prediction | outcome | |
|---|---|---|---|
| 1 | H1 fails; corrected IC within ±0.02 | fails; IC +0.035 | half: failed, but IC above the band |
| 2 | H2 fails | fails; IC +0.020 | met |
| 3 | a level coefficient looks strong on training; the null reproduces most of it | the largest level coefficient (distance to the 4 h high, +1.8 bp/sd) has the opposite sign on the null (−1.6) | not met as stated; significance was not computed |
| 4 | boosting fits training (IC > 0.10) and scores below ridge on validation | training 0.29, validation 0.010 vs 0.035 | met |
| 5 | round numbers have the smallest importance of the new groups | round numbers −0.006 / +0.011 vs support/resistance +0.035 / +0.044 | met |

## Why the IC is positive but the gain is negative (post hoc)

The tilt is not one-sided: 52% of validation decisions tilt toward faster. Ranking every
day's decisions against that day's own median gives a gain of +0.58 bp before the null is
subtracted. That is still a fifth of the floor.

The positive rank correlation comes from small moves. The gain is set by the large ones, which
the model gets wrong, most of all on crash days. Support/resistance carries what little
ranking there is (most of the permutation importance). It does not tell a seller anything worth
a basis point.

## Conclusion

On 87 unseen days, the levels a trader watches do not say when selling faster pays, once the
random-walk artefact is removed:
- round numbers;
- today's, the last 4 h's and yesterday's highs and lows, and touches of them;
- longer past returns, volatility and book imbalance.

This adds to docs 26 (agent state), 27 and 32 (bounce-back) and 34 (SOL, DOGE). BTC price history
has no real-time regime signal that a seller can act on.

Following the pre-registered branch, this closes "regime awareness from BTC price history". The
doc 37 retrain is not run, because no input would give it anything regime-aware to learn. Next:
equities, together with the live-feed platform.
