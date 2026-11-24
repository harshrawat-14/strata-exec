# 27 — Real-time regime features from the full Bybit book: pre-check design and predictions

Written 2026-11-24, before the check is run. No RL is trained.

## Why

Doc 26 found that the 16-dim state cannot predict regime or drift on unseen days. Part
of the reason is that the state reads almost none of the Bybit book, which is the reason
era 2 moved to it. Dim 4 was saturated by a unit bug (fixed in 09beaf7), and the
spread dims are nearly constant. This check asks whether features built from the full
500-level book and the trades carry real-time information about where the price goes
next. A belief head is only worth training if they do.

## Features (`rl/experiments/belief_precheck/book_features.py`)

There is one row per 30 s snapshot. Each feature uses only the same day's data up to that
snapshot, and each is a ratio, so no feature can identify a day by its price or volume
level.

- **Book:**
  - spread (bp);
  - microprice deviation;
  - bid/ask imbalance and depth relative to its trailing 2 h mean, within 2, 5 and 10 bp
    and over the whole visible book (the free book spans only ~13–20 bp per side);
  - bid thinning: change in log bid depth over 10 min.
- **Trades:**
  - taker buy/sell imbalance over 1, 5, 30 and 120 min;
  - trade intensity relative to the trailing level.
- **Price:**
  - returns over 5 min, 30 min, 2 h and 6 h, in units of trailing volatility;
  - volatility ratios;
  - drawdown from the day's running high.

## Targets

These are step-level: thousands of examples, not one label per day.

- `fwd_120` and `fwd_240`: return over the next 1 h and 2 h (pp).
- `fall_240`: the next 2 h fall by more than 0.5%.

## Test

- **Main:** fit on the 12 training dates, test on the 4 validation dates (as in doc 26).
- **Robustness:** leave-one-date-out over the 12 training dates.
- **Models:** ridge and gradient boosting.
- **Comparison:** the doc-26 state-equivalent features (volatility, 30 s flow) as the
  reference.

**Pass**, meaning a belief head is worth training, requires both of:

1. **2 h return:** on validation, positive ridge R² against predicting zero, *and* a positive
   per-date correlation on at least 3 of the 4 validation dates and at least 9 of 12 in
   leave-one-date-out.
2. **Or the fall event:** a validation AUC of at least 0.60, and at least 0.55 on at least
   9 of 12 leave-one-date-out dates.

If neither holds, the features are not used as a regime signal and the negative result is
written up.

## Predictions

1. **The 2 h return fails criterion 1.** Short-horizon BTC returns are close to
   unpredictable from public data. Validation R² ≤ 0.
2. **The fall event fares better than the return but is borderline:** validation AUC
   0.55–0.65. Volatility and flow features should carry some information about large moves,
   mostly their size, not their sign.
3. **The book features add little over the doc-26 features for the return** (validation
   correlation within ±0.05 of the reference).
4. **Gradient boosting does worse than ridge out of sample** (overfitting days, as in doc 26).
