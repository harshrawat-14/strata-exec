# 32 — Does the price bounce back after a drawdown? Design and predictions

Written 2026-11-25, before the test is run on real data. No RL is trained.

## Why

Docs 26–31 ruled out three routes to a regime-aware edge over TWAP: forecasting direction from
the state or book, timing liquidity, and volatility-dependent impact decay. One hint remains.
In doc 27 (12 days, exploratory), drawdown from the day's high had a mean within-day
correlation of −0.16 with the next 2 h return, negative on 9 of 12 days: falls tended to be
followed by recoveries. If this is real, the regime-aware action is to **sell more slowly
after a drawdown**. The agent can observe a drawdown, and the action needs no direction
forecast beyond this conditional one.

## A flaw in the hint, found while designing this test

A drawdown from the running high moves with the price, so its within-day predictive
correlation with forward returns is **biased negative even when prices are a pure random
walk** (Stambaugh bias in predictive regressions). The bias has the same sign every day.

On 60 synthetic random-walk days (`rl/tests/test_bounce_back.py`):
- the raw statistic averaged **−0.29**;
- **87% of days were negative**.

Doc 27's −0.16 on 75% of days is weaker than pure noise produces. The raw statistic cannot
tell bounce-back from no predictability.

## Method (`rl/experiments/bounce_back_test.py`)

- **Feature:** drawdown_z, exactly as doc 27: log(mid / running high) ÷ (trailing 2 h volatility
  × √240). It is now one tested function (`book_features.drawdown_z`).
- **Target:** the forward 2 h mid return, in bp.
- **Rows:** from 2 h into the day (volatility window full) to 2 h before the end.
- **Null:** for every day, the same 30 s returns shuffled 20 times. This keeps the return
  distribution and removes any predictability. Every statistic is reported raw, null and
  **corrected (raw − null)**. On random walks the corrected statistic is about zero (tested).
- **Statistics:**
  - the mean per-day correlation of drawdown_z with the 2 h return;
  - the deep-drawdown excess: the mean 2 h return in rows with drawdown_z at or below the
    training rows' 10th percentile, minus the mean over all rows, in bp.
- **Uncertainty:** a 1,000-resample date bootstrap, with 90% intervals.
- **Days:** 256 training (Jan–Sep), 87 validation (Oct–Dec). Held-out dates are not read.
- **Limitation:** shuffling also removes volatility clustering, which the drawdown_z
  normaliser responds to. The null is therefore exact for "no predictability with i.i.d.
  returns", not for every form of no predictability.

## Pass criteria (all four)

1. **Training** corrected correlation < 0, with a 90% interval excluding 0.
2. **Validation** corrected correlation < 0, with a 90% interval excluding 0.
3. Corrected deep-drawdown excess ≥ +5 bp in training and in validation, with the training
   interval above 0. The 5 bp floor is a judgement call: smaller than that, the gain from
   slowing down is within TWAP's day-to-day noise.
4. **Crash days** (all 54, by doc 30's rule): corrected correlation ≤ 0. Slowing down after a
   fall is most dangerous on days that keep falling, so the effect must not reverse there.

**If it passes:** add nothing to the simulator (bounce-back is in the replayed prices
already). Next, check that a simple rule (slow down when drawdown_z is deep) beats TWAP in
every regime on V4 validation, before any RL. **If it fails:** all four tested routes to a
regime-aware edge on risk-neutral cost are closed on this data. That is reported as a
finding, and the next step is discussed with the user before any change of direction.

## Predictions

1. **The raw training correlation is negative** (−0.35 to −0.10), as in doc 27.
2. **The null correlation is close to the raw one, and the corrected correlation is within
   ±0.05.** Criterion 1 fails.
3. **The corrected deep-drawdown excess is below +5 bp.** Criterion 3 fails.
4. **Overall: FAIL.** Doc 27's bounce-back hint is mostly Stambaugh bias.
