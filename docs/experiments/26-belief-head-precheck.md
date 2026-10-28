# 26 — Regime belief head: pre-check — negative, not trained

Written 2026-10-28, before any training. This applies the lesson from doc 25: check that a
signal is learnable from existing data before spending a run on it. Validation dates are
used as the test set for the offline fits; held-out dates are untouched.

## Question

The belief head is an auxiliary supervised loss that predicts the episode's regime (calm /
normal / crash) or the coming price trend from the state. It can only help if the state
carries that information on **unseen days**. The cheapest upper bound is to fit the same
targets offline, with the network replaced by ridge regression and gradient boosting.

## The state in era 2 (16 dims, unchanged since era 1)

Sampled on 01-15 (calm), 04-15 (crash) and 02-20 (normal), 3 starts each, 5% ADV
(`state_dims.py`):

| # | dim | in era 2 |
|---|---|---|
| 0, 1 | inventory left, time left | live (progress) |
| 2, 10, 13 | volatility, three encodings | live, but one signal |
| 3, 11 | half-spread, liquidity regime | nearly constant (sd ≤ 0.09) |
| 4 | depth within 1% | **dead**: saturated at 1.000 |
| 5 | order-flow imbalance | live, noisy (sd ~0.5) |
| 6 | 10-step price drift | **dead**: sd ≤ 0.004, scaled by an annualised σ |
| 7 | last slippage | live (own fills) |
| 8, 9 | permanent / temporary impact | live (own fills) |
| 12, 14, 15 | time of day | live, identical on every date |

The only price-trend input (6) is dead. The state also keeps just 16 snapshots (8 min) of
price history.

## Result (`collect_states.py`, `predictability.py`)

The data is 16 dates × 5 starts × 1,920 steps, with action 0, so the book carries none of
our own impact. The models are fit on the 12 training dates and scored on the 4 validation
dates.

**Forward drift**, defined as the mean future mid over the rest of the episode divided by
the current mid, minus 1 (sd 1.1 pp). This is what decides between selling now and selling
later.

| features | ridge R², val (train) | boosting R², val (train) |
|---|---:|---:|
| market dims (2–6, 10, 11, 13) | −0.21 (+0.02) | −0.56 (+0.38) |
| + time of day | −0.21 (+0.03) | −0.67 (+0.64) |
| + past returns over 10 / 60 / 240 steps | −0.21 (+0.02) | −0.65 (+0.52) |
| predicting 0 | −0.11 | |

On validation, every model does worse than predicting no drift. The ridge predictions
correlate −0.26 with the outcome; by date: 04-15 +0.14, 02-20 −0.34, 01-15 +0.09,
09-15 +0.10.

**Regime class**, scored by balanced accuracy (chance is 0.33):

| features | validation | train |
|---|---:|---:|
| market dims | 0.15 | 0.84 |
| + time of day | 0.13 | 0.95 |
| + past returns | 0.16 | 0.90 |

The classifiers memorise the 12 training days and score below chance on new days.
Volatility alone does not separate crashes. A logistic fit of P(crash) on the
volatility-regime dim gives 0.20–0.28 on every date, crash or not.

## What this means

- **A belief head on this state would learn to recognise training days, not regimes.** The
  regime labels are per date, so the head has 12 effective labels, and it fits them through
  time-of-day and day-specific level features. That is the memorisation risk doc 25 flagged,
  measured. It is not being trained.
- **"Regime" here is a hindsight label.** A crash is a day whose price fell. Telling it apart
  in real time means forecasting direction from the path so far. No feature tried here,
  including longer past returns, does that out of sample.
- **What TWAP leaves to beat at 5% ADV.** TWAP's impact cost is about 0.2 pp, and its
  execution cost on the book is only 0.003 pp. Timing noise is 1.1 pp. So beating TWAP in every
  regime is mostly a direction bet, which is the one thing the state cannot support.
  M0-e2b's crash wins come from selling early, and its calm losses are the same choice
  seen from the other side.

## Options

1. **Fix the dead inputs first** (a state rung): rescale dim 6 by per-step σ, use longer
   lookbacks (e.g. 1 h / 4 h returns), and replace the saturated depth dim with a usable
   scale. This is cheap to pre-check with the same scripts. The past-returns row above
   already suggests trend features do not forecast drift on these dates, so expect little.
2. **Change the target from regime to risk.** Beat TWAP on a risk-adjusted IS (mean −
   λ·variance), the textbook execution objective. Here front-loading is right in every
   regime, and the variance it removes is real. This changes the goal, so it is the
   user's decision.
3. **Larger orders** (e.g. 10–25% ADV). Impact grows as √size, so schedule and book-reading
   skill matter more relative to timing noise. TWAP IS goes from −0.23% at 5% ADV to
   −0.38% at 10%.
4. **Train M0-e2b longer.** It was still improving at 1.5M steps. Some of the calm gap may
   be undertraining, but it cannot fix a pace that ignores regime.
