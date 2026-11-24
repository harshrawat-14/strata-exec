# 27 — Real-time regime features from the full Bybit book: result — fails the pass criteria

Written 2026-11-24. Predictions were pre-registered in
[27-book-regime-features-predictions.md](27-book-regime-features-predictions.md) (commit 4eec34a).
The models are fit offline. Held-out dates are untouched.

## Summary

The full-book, trade and price features **do not predict the next 1–2 h out of sample**,
and the fall-event signal is not stable across days. **No belief head is trained on them.**
Measured from Bybit's book, "is this a falling market?" is not knowable in real time on
these dates, beyond what volatility already says.

## Result (`book_predictability.py`, `rl/results/book_regime_features_precheck.csv`)

The data is 12 training dates (31,680 snapshots) and 4 validation dates (10,560 snapshots).
"LODO" is leave-one-date-out over the 12 training dates.

**2 h return (`fwd_240`).** R² is measured against predicting zero; the per-date figure is a
correlation.

| features | model | validation R² | validation dates > 0 | LODO dates > 0 |
|---|---|---:|---:|---:|
| reference (doc-26 equivalents) | ridge | −0.006 | 2/4 | 2/12 |
| all book + trade + price | ridge | −0.039 | 3/4 | 2/12 |
| all book + trade + price | boosting | −0.911 | 1/4 | 9/12 |

The 1 h return gives the same picture: validation R² ridge −0.033, boosting −0.991.

**Fall > 0.5% in the next 2 h (`fall_240`; base rate 0.20 train, 0.24 validation).**

| features | model | validation AUC | LODO dates ≥ 0.55 | LODO mean AUC |
|---|---|---:|---:|---:|
| reference | logistic | 0.526 | 7/12 | 0.517 |
| all | logistic | 0.620 | 2/12 | 0.441 |
| all | boosting | 0.622 | 4/12 | 0.483 |

On validation, the full feature set clears 0.60. Within the training dates it is *below*
0.5 on average, so the validation figure does not hold across days. Four dates are too few
to tell skill from chance.

**Strongest single features** (correlation with the 2 h return, averaged over training
dates):

| feature | mean corr | dates with that sign |
|---|---:|---:|
| drawdown from day high | −0.16 | 9/12 |
| 6 h return | −0.14 | 9/12 |
| 30 min trade flow | −0.10 | 7/12 |

These are weak signs of **mean reversion**: after a fall, the price tends to recover. This
is exploratory, since the list was not pre-registered. It would favour *slower* selling
after drops, the opposite of M0-e2b's crash behaviour. The multivariate fits do not
carry it out of sample.

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | 2 h return fails criterion 1 (validation R² ≤ 0) | **Met** (−0.039) |
| 2 | Fall-event validation AUC 0.55–0.65 | **Met** (0.62), but it does not hold in LODO |
| 3 | Book features add ≤ ±0.05 validation correlation over the reference (return) | **Met** (mean +0.003 vs −0.012) |
| 4 | Boosting worse than ridge out of sample | **Partly**: worse on validation, better in LODO for the 2 h return (9/12 vs 2/12 dates) |

**Pass criteria:** criterion 1 fails (validation R² < 0). Criterion 2 fails: the validation
AUC passes, but LODO is 2/12 and 4/12 against the required 9/12. **Overall: fail.**

## Mistake caught before reporting

In the first run, the `fall_240` label was among the features, because the feature filter
excluded only `fwd_*`. That gave AUC 1.00 and a validation R² of +0.38 for the 2 h return.
The run was discarded. The filter now excludes every target, and an assertion guards it.
See the timeline.

## What this means

- Combined with doc 26: on 16 days of free Bybit data, neither the state nor a broad set of
  real-time book, trade and price features forecasts direction at execution horizons. A
  belief head that predicts drift or crash would learn noise.
- A regime-aware edge over TWAP therefore has to come from something other than forecasting
  direction. See the options in the conversation and timeline.
