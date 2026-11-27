# 39 — Equities pre-check: is there a real-time regime here? Predictions

Written 2026-11-27, before any equity data is downloaded. No RL is trained. Each test below is
fixed now and run once.

## Why

On BTC, SOL and DOGE, regime meant direction: a crash day is a day whose price fell. Docs 26,
27, 32, 34 and 38 found nothing visible in real time that forecasts it. A seller therefore cannot
act on that kind of regime, and every trained model fell back to a constant fast pace (doc 36).

Equities are probably no more predictable in direction. What they have that crypto lacks is
structure a seller can see in advance:
- a trading session with an open and a close;
- a strong intraday volume and volatility profile;
- scheduled events;
- large ticks, which make the queue at each price level meaningful.

This doc tests whether any of that is visible in real time **and** moves execution cost on
unseen days. Only a test that passes leads to equities in the simulator and to RL.

## Thesis, restated (needs the user's approval)

The aim stays the same: a regime-aware agent beats TWAP in every regime. What changes is what
"regime" means. On equities it is the **liquidity and volatility state** a seller faces: a thin or
deep market, an event or no event, the open, the middle of the day or the close. It is no longer
"the price is falling".

This is still regime-aware execution, not a switch to a textbook objective. The agent must
still learn to change its pace with the state. The difference is that this state is observable,
and a price trend was not.

Regimes for scoring are fixed by the realised liquidity and volatility of the day: terciles of
the day's realised volatility divided by the square root of its volume, a proxy for √-law impact
cost. The terciles are set on training days. The old direction label (drift) is reported
alongside, so the result can be compared with the crypto work.

## Data and split

**India (main):** Upstox historical 1-minute candles for the 50 stocks of NIFTY 50 as of
2023-01-01, plus NIFTY futures (near month, rolled on expiry). Fixing the list at the start date
avoids survivorship bias from today's list.

**US (book test only):** IEX HIST DEEP files (free, trailing 12 months), for 10 large-tick names
fixed now: AAPL, MSFT, AMZN, GOOGL, META, NVDA, JPM, XOM, KO, PFE. IEX is a small slice of US
volume, which limits this test (see below).

| set | India | US |
|---|---|---|
| training | 2023-01-02 to 2024-06-28 | first 6 months available |
| validation | 2024-07-01 to 2024-12-31 | next 3 months |
| held-out | 2025-01-01 onward, not downloaded until a final test | last 3 months, not read |

Days with a half session, a special session (Muhurat) or a market-wide halt are dropped. The
list is written to `rl/evaluation/dates_equities.py` before the data is read.

## Tests

### L — Is the day's liquidity forecastable from its first 30 minutes?

At 09:45 IST, forecast the rest of the day's volume and realised volatility for each stock.
- **Inputs:**
  - the first 30 minutes' volume and volatility, relative to their 20-day averages;
  - the absolute overnight gap;
  - the previous day's realised volatility;
  - the stock's 20-day average volume profile;
  - day of week;
  - expiry day.
- **Model:** ridge, fit on training days and scored on validation.
- **Baseline:** the static forecast, i.e. the 20-day average for the rest of the day.

**The cost test** sells 5% of the rest of the day's volume from 09:45 to 15:15 under four
schedules:
- TWAP;
- static VWAP (the 20-day average profile);
- **adaptive VWAP**: the profile rescaled by the 09:45 forecast and updated every 30 minutes
  with the volume seen so far;
- the hindsight VWAP (the realised profile; an upper bound, not a strategy).

Cost is the √-law impact on each minute's realised volume and volatility, `c = Y σ_min √(q/V_min)`,
with Y = 0.5. The value of Y does not change rankings, only the scale.

**What makes this non-circular.** The √ law makes any volume-following schedule beat TWAP by
construction, so TWAP is reported but is not the bar. **The bar is adaptive vs static VWAP.**
Adaptive only wins if liquidity is forecastable in real time beyond the average profile, which
is the empirical question.

**Pass (all of):**
1. validation R² of log rest-of-day volume improves on the static forecast by ≥ 0.10;
2. the adaptive schedule's validation cost beats static VWAP by ≥ 1 bp, with a 90% day-bootstrap
   interval excluding 0;
3. adaptive beats static on average in **each** of the three liquidity regimes.

### E — Do scheduled events move volatility at a known time?

**Events:**
- RBI policy announcements (10:00 IST);
- the Union Budget speech day;
- US CPI release days (08:30 ET, so NSE is affected the next morning);
- each stock's own results day (dates from NSE's corporate announcements archive).

All dates are fixed before the price data is read.

**Measure:** the ratio of realised volatility in the 60 minutes after the scheduled time to the
same window on matched non-event days (same weekday, nearest 4 weeks).

**Pass (all of):**
1. across validation events, the mean ratio is ≥ 1.5, with a 90% interval above 1;
2. a schedule that pauses during the event window and catches up afterwards beats static VWAP on
   event days by ≥ 1 bp under the same cost model, without losing on matched non-event days.

### B — Does the book's imbalance predict the next price move, on a horizon that matters?

US, IEX HIST DEEP. Top-of-book imbalance `(Qbid − Qask) / (Qbid + Qask)`, sampled every second.
Target: the sign of the mid move over the next 30 seconds, ignoring zero moves.
- **Null:** the same imbalance series lined up with mid moves from a different day of the same
  stock.
- **Pass:** validation accuracy ≥ 0.55, with a 90% interval above the null, **and** the implied
  saving from delaying or advancing a child order by 30 seconds is ≥ a quarter of the half-spread.

**Caveat, stated now:** an IEX-only book shows a small share of the US market. A failure here
does not rule imbalance out on the consolidated book. A pass would be worth confirming with the
$125 Databento credit.

## Predictions

1. **L passes on R²** (≥ 0.10 improvement). The first 30 minutes' volume is a known strong
   predictor of the rest of the day. **L passes on cost against static VWAP** with probability
   about 45%. The gain will be small (1–3 bp). It is most likely to fail condition 3, in the
   calm-liquidity regime.
2. **E passes condition 1** for RBI and results days (ratio 1.5–3) and **fails** for US CPI on
   NSE (ratio below 1.3). Condition 2 passes with probability about 35%.
3. **B fails on IEX data** (accuracy 0.51–0.54). Large-tick imbalance is known to predict the
   next tick, but 30 seconds is long, and IEX sees only part of the queue.
4. The direction label (drift) is no more predictable than on BTC: the doc 38 timing-gain test
   run on NIFTY futures gives under 1 bp, inside the null.

## What happens next

- **L or E passes:** add an equities dataset to the simulator, with a session, a volume profile
  and an event calendar. Then pre-register a rule-based check (does a liquidity-aware rule beat
  TWAP in every liquidity regime in the simulator?) before any RL.
- **B passes:** confirm it on Databento consolidated data before building it in.
- **All fail:** the equities route closes the way crypto did. The thesis then needs a different
  source of state, such as other assets, news or order-flow toxicity. That decision is the
  user's.

## What the user provides

- **An Upstox account with API access,** with the token in the environment variable
  `UPSTOX_ACCESS_TOKEN`, never in chat or in the repo. Ask Upstox in writing whether storing
  historical candles locally for private research is allowed. Until they answer, files stay
  local and are never committed or published.
- **Permission to download the IEX HIST DEEP files.** Each day is a large compressed capture file
  (sizes listed before downloading). Only the converted 1-second top-of-book is kept.
