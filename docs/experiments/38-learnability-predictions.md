# 38 — Learnability: can history say when selling faster pays? Predictions

Written 2026-11-26, before any feature is computed. No RL is trained. Held-out and future dates
are not read.

## Why

Doc 37 found that the training recipe cannot reward regime awareness. Before changing the
recipe, this doc checks whether there is anything to learn. The user's idea: learn from history
when and why selling faster or slower was right, using what a trader can see at that moment,
including support/resistance and round-number price levels. Those levels have never been tested
here.

If a model predicts it on unseen days, its output becomes StrataExec's regime signal and the
retrain in doc 37's plan goes ahead with it. If not, no state built from these inputs can make
the agent regime-aware, and the next step is a different market (equities), not more training.

Doc 26 asked a narrower version (forward drift from the agent's 16 inputs, 12 training days) and
found nothing. Docs 27, 32 and 34 ruled out a bounce-back after drawdowns. This doc widens the
inputs and uses the full V4 pool (256 training, 87 validation days).

## Target

At each decision point t, the seller chooses the pace for the next 2 h (H = 240 steps of 30 s).
Doc 37 showed own impact is nearly independent of pace, so the hindsight-best pace is set by the
price path alone: selling a slice now instead of spreading it over the next 2 h gains

  **y_t = −(mean mid over (t, t+H] / mid_t − 1)**, in bp.

y > 0 means selling faster paid; y < 0 means slower paid. The hindsight-best pace is the sign
of y (beyond the extra spread, about 0.3 bp, ignored here).

Decision points: every 60 steps (30 min), from step 480 (4 h of history) to the last step that
leaves a full 2 h ahead. About 36 per day: roughly 9,200 training and 3,100 validation rows.

## Inputs (visible at t; `rl/experiments/learnability.py`)

σ is the trailing 2 h standard deviation of 30 s log returns, scaled to 2 h (× √240). Every
distance and return is in σ units, so days with different volatility are comparable.

| group | features |
|---|---|
| **round numbers** (new) | signed distance from mid to the nearest $1,000 and $500 multiple; whether mid is within 0.5 σ of one; which side of it mid is on |
| **support/resistance** (new) | distance to today's high and low so far, to the 4 h high and low, and to the previous day's high, low and close; touches of the 4 h high and of the 4 h low in the last 4 h (separate visits within 0.25 σ, at least 10 steps apart); position in the 4 h range (0 = low, 1 = high) |
| **trend** | past returns over 10, 60, 240 and 480 steps (this fixes dim 6's scaling) |
| **volatility** | log σ; ratio of 1 h to 4 h trailing σ |
| **order book** | top-5-level depth imbalance, mean over the last 10 steps |
| **time** | sine and cosine of time of day |

The previous day's levels are missing when that day is not on disk (buffer days). Boosting
handles missing values. Ridge gets 0 plus a missing flag.

## Models

- **Ridge** (primary, pre-selected) on standardised inputs. Its coefficients are the
  explanation.
- **Gradient boosting** (sklearn HistGradientBoosting, depth 3, 200 iterations, learning
  rate 0.05) for non-linear effects, with permutation importance on validation.

Both are fit once on the 256 training days and scored on the 87 validation days. Nothing is tuned
on validation.

## The bias this must survive

Level features are built from the same path as the target. On a pure random walk, distance from
the running high still correlates with forward returns (doc 32's Stambaugh bias). A model can
learn that artefact and look predictive. Every statistic is therefore also computed on **null
paths**: the same day's 30 s returns with random signs (doc 34's sign-flip null). That keeps
volatility clustering but removes any predictability. Round-number levels are recomputed on the
null path's prices, so their artefacts appear in the null too. The fitted models are applied
unchanged to 20 null paths per validation day.

**Corrected = real − null mean.** Intervals are 90% bootstraps over days.

## Statistics

1. **IC:** the per-day Spearman correlation between prediction and y, averaged over days.
2. **Timing gain:** tilt toward selling faster when the prediction is above the training
   median, and slower when it is below. Gain = mean over decisions of sign(tilt) × y, in bp.

## Pass rule (fixed now)

**H1 (all inputs, ridge):**
1. corrected validation IC > 0, with an interval that excludes 0;
2. corrected validation timing gain ≥ 3 bp, with an interval that excludes 0;
3. corrected timing gain > 0 on validation crash days **and** on validation calm days, measured
   separately. A signal that only says "sell fast" would pass on crash days and fail on calm
   ones, which is what M0 already does.

**H2 (round-number and support/resistance inputs only, ridge):** the same three conditions. This
is the user's hypothesis on its own.

Boosting is reported next to ridge. It cannot pass on its own, because choosing the better model
after seeing validation would spend validation twice.

3 bp is about one-seventh of TWAP's impact cost at 5% ADV (about 20 bp, doc 26). It is enough
to show up in the simulator.

## Predictions

1. **H1 fails.** Corrected validation IC is between −0.02 and +0.02. Probability of a pass: 10%.
2. **H2 fails.** Probability of a pass: 7%.
3. Before correction, at least one level feature (distance to the 4 h high or to today's high)
   has a ridge coefficient that looks significant on training data. The null reproduces most of
   it.
4. Boosting fits training (IC > 0.10) and scores below ridge on validation (memorisation, as in
   doc 26).
5. Round-number distance has the smallest permutation importance of the new groups. BTC moves
   through $1,000 levels too often in a day for them to act as barriers.

## If it passes

Write the passing model's output into the state as one input, then run doc 37's retrain plan.
First, pre-register a check in the simulator: a rule that sells at 2× pace when the signal says
faster and 0.5× when it says slower beats TWAP in every regime on validation days.

## If it fails

Together with docs 26, 27, 32 and 34, this closes "regime awareness from BTC price history".
Next is the equities data and the live-feed platform (memory: equities first).
