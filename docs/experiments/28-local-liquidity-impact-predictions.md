# 28 — Does impact scale with local liquidity? Design and pre-registered predictions

Written 2026-11-24, before the fit is run. No RL is trained.

## Why

The era-2 impact model (doc 21) is I = Y · σ_d · √(D/V). Here σ_d is the date's daily
volatility and V is the date's total volume, both constant for the whole day. A sale
therefore costs the same in a busy, deep hour as in a thin one, so the simulator gives no
reward for timing liquidity. Docs 26 and 27 showed that price direction is not
forecastable, which leaves no regime-aware edge over TWAP in this simulator. If real impact
depends on the liquidity at the time, the simulator is missing the mechanism through which
regime-aware execution earns its edge.

## Method (`rl/experiments/impact_fit_local_liquidity.py`)

The fit follows doc 21: regress each bucket's mid log-return r on net taker flow Q, on
Bybit BTCUSDT. One coefficient Y per model:

    r = Y · σ · sign(Q) · √(|Q| / V)

| model | σ | V |
|---|---|---|
| **day** (current) | date's daily vol | date's total volume |
| **local V** | date's daily vol | trailing 60 min volume, scaled to a daily rate |
| **local σ** | trailing 60 min realised vol, scaled to daily | date's total volume |
| **local both** | trailing 60 min, daily-scaled | trailing 60 min, daily-scaled |

- **Causal inputs only.** The trailing windows end *before* the bucket, so Q never appears in
  its own normaliser and the simulator could use the same values.
- **Same rows for every model.** The first 60 min of each day are dropped for all models.
- **Dates:** the 12 training dates and 4 validation dates. Held-out dates are not used
  (doc 21 fitted on all 25; this test does not need them).
- **Main bucket:** 30 s (the simulator step). The 5 min bucket and 15 / 120 min windows are
  reported as sensitivity checks only.

**How each model is scored**, the way the simulator uses it:

1. Y is frozen at the training-date median.
2. Out-of-sample R² on each validation date and pooled.
3. Per-date R² on the training dates with that frozen Y.
4. Spread of per-date fitted Y across the 16 dates (coefficient of variation). A
   normalisation that captures liquidity should make Y more constant across regimes.

## Pass criterion: local liquidity belongs in the simulator

The **local V** or **local both** model must beat **day** on all of:

1. Pooled validation R² higher by at least 0.01 (absolute).
2. Higher R² on at least 3 of 4 validation dates and at least 10 of 12 training dates.
3. Lower coefficient of variation of Y across the 16 dates.

If it passes, the next step is to put the winning normalisation into `counterfactual_lob`.
Then, before any RL, check that a simple volume-following rule beats TWAP under it in
every regime. If it fails, the daily model stays, and the result is reported as a finding:
liquidity timing has no measurable impact edge on this data.

## Predictions

1. **local V beats day on criteria 1–2.** Busier hours absorb flow with less price move.
   Expected pooled validation R² gain: +0.01 to +0.05.
2. **local V lowers the spread of Y across dates** (criterion 3), and crash dates no longer
   have the highest Y.
3. **local σ alone helps less than local V.** Vol scaling mostly re-labels moves the flow
   already explains.
4. **local both is the best of the four** on pooled validation R².
5. **At 5 min, the ranking of models is the same as at 30 s.**
