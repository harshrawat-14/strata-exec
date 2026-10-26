# Era 1 — Binance bookDepth simulator (2026-05-22 → 2026-10-23)

The record of everything run on the first counterfactual simulator, written when it was
retired. The raw data was deleted on 2026-10-26 to free space; it can be re-downloaded
(see the end of this page).

## The data

Binance **BTC-USDT perpetual futures**, 2024, from the free archive `data.binance.vision`
(`futures/um/daily/`):

- **bookDepth** — every ~30 s, the cumulative quantity resting within ±1%, ±2%, ±3%, ±4%
  and ±5% of the mid price. Five bands per side: **not an order book.**
- **aggTrades** — every trade (1–6 million per day), used for volume, order flow and the
  impact calibration.
- **daily candles** — daily volume, used for the 20-day ADV (`rl/results/adv20.csv`).

## The 25 dates

Regime = AC's own honest-λ cost on the day (an outcome-independent severity proxy), split in
terciles. Events: **A** Iran–Israel escalation (Apr 12–15), **B** German government BTC
sales / Mt. Gox fear (Jul 4–12), **C** US jobs report → yen-carry unwind (Aug 2–5).

| date | regime | split | event / significance | AC cost on day | real volume (BTC) | 20-day ADV | 50k-BTC order share |
|---|---|---|---|---:|---:|---:|---:|
| 2024-01-15 | calm | validation | isolated | +0.91% | 240,059 | 351,404 | 20.8% |
| 2024-02-20 | normal | validation | isolated | −0.53% | 388,913 | 268,738 | 12.9% |
| 2024-03-05 | crash | held-out | isolated (flash liquidation cascade) | −2.46% | 1,053,023 | 340,418 | 4.7% |
| 2024-04-12 | normal | train | A | −0.65% | 415,528 | 273,453 | 12.0% |
| 2024-04-13 | normal | train | A | −0.89% | 515,005 | 282,273 | 9.7% |
| 2024-04-14 | normal | train | A | −0.51% | 395,770 | 295,814 | 12.6% |
| 2024-04-15 | crash | validation | A (Iran–Israel escalation) | −1.09% | 381,553 | 295,845 | 13.1% |
| 2024-06-10 | normal | train | isolated | −0.57% | 127,382 | 172,653 | 39.3% |
| 2024-07-04 | crash | train | B (German BTC sales / Mt. Gox fear) | −3.14% | 445,656 | 189,991 | 11.2% |
| 2024-07-05 | crash | train | B (has a 13.5-min data gap) | −2.79% | 577,194 | 200,303 | 8.7% |
| 2024-07-06 | calm | train | B | −0.21% | 178,580 | 225,541 | 28.0% |
| 2024-07-07 | crash | train | B | −1.42% | 210,362 | 230,452 | 23.8% |
| 2024-07-08 | calm | train | B | −0.26% | 464,107 | 228,549 | 10.8% |
| 2024-07-09 | calm | train | B | +0.47% | 257,237 | 234,977 | 19.4% |
| 2024-07-10 | calm | held-out | isolated | +0.03% | 254,252 | 240,337 | 19.7% |
| 2024-07-11 | calm | held-out | B | −0.06% | 293,113 | 242,002 | 17.1% |
| 2024-07-12 | normal | held-out | B | −0.58% | 238,111 | 246,152 | 21.0% |
| 2024-08-02 | crash | held-out | C | −2.37% | 421,628 | 257,399 | 11.9% |
| 2024-08-03 | normal | held-out | C | −0.78% | 290,470 | 270,586 | 17.2% |
| 2024-08-04 | crash | held-out | C | −1.10% | 339,512 | 274,707 | 14.7% |
| 2024-08-05 | crash | held-out | C (jobs report → yen-carry unwind; flagship) | −7.89% | 1,281,982 | 274,071 | 3.9% |
| 2024-09-15 | calm | validation | isolated (Sunday) | −0.48% | 130,524 | 249,301 | 38.3% |
| 2024-10-15 | normal | train | isolated | −0.86% | 422,006 | 224,935 | 11.8% |
| 2024-11-06 | calm | train | isolated (US election result; ~+8% rally) | +4.98% | 677,440 | 220,120 | 7.4% |
| 2024-12-10 | normal | held-out | isolated | −0.72% | 291,490 | 243,261 | 17.2% |

Splits: V2 = 16 train / 9 held-out (event C wholly held out); V3 carved 4 validation dates
from the 16 (12 / 4 / 9). The held-out dates were never used for the v3 models.

## Best results of the era

| stage | result | where | caveat |
|---|---|---|---|
| Legacy headline (Step A) | RL beat honest-λ AC on **4/4 held-out crash dates** (+0.92, +0.92, +0.37, +0.26 pp), lost **0/5** calm/normal (−0.31 to −0.80) | doc 10 | 3 of 4 crash dates are one event; legacy protocol (all starts at 00:00, argmax decoding); RL trained on one replayed morning per date (seed bug) |
| Best fixed schedule | sqrt-impact baseline beat AC on **9/9** held-out dates (+0.01 to +0.38 pp); RL beat it on the 4 crash dates | docs 08, 12 | fitted on the same date it is scored on |
| Protocol v3, M0 | validation: crash **+0.45** vs AC, normal −0.07, calm −0.12 / −0.35; mean −0.465% vs AC −0.4425% | doc 13 | crash win came from selling early (timing), execution worse than AC on every date |
| Best validation number | M4 (rate actions) **−0.311%** | doc 19 | a constant 1.5× pace scored −0.245%; selection + falling validation dates, not learned skill |

Negative variants: M1 γ 0.999 (doc 14), M2 residual-around-AC (doc 15), M3 longer credit
horizon to 6M (doc 16), M4 impact shaping stopped by its pre-check (doc 17).

## The dead end

**The book itself.** The simulator treated the ±1% band as the best price level, priced at
the band's average: a simulated spread of **86–103 bp** against a real spread of about
**0.02 bp**, so every sale paid ~43–51 bp whatever its size (doc 20). Consequences:

- every strategy's "execution cost" (~0.46%) was mostly this floor;
- cost was identical for orders from 500 to 50,000 BTC — order size, the core variable of
  institutional execution, barely mattered;
- the "2.7× cost above 0.3% of the order" (doc 18) was crossing into the next band;
- strategy differences reduced mostly to timing (selling before or after price moves).

So the era could not answer its own question: how to execute a large order at lower cost.

## What carries forward

Valid regardless of the book: the evaluation protocol (v3: 16-h episodes, spread starts,
validation split, held-out once), mean-fraction decoding and offline checkpoint selection,
pre-registration with stopping rules, ADV-relative sizing, the golden regression guard, and
the lessons in [lessons.md](lessons.md) — including the RL training pathologies (seed
replay, argmax decoding, drift into deferral). **Not carried forward:** any cost magnitude
or execution-quality comparison.

## Why it changed

Bybit publishes **free real order-book data** (500 levels per side, full snapshot plus
tick-by-tick updates) for these dates — verified 2024-01-15: spread $0.10, 18.6 BTC within
1 bp, 257 BTC within 10 bp. Era 2 runs the simulator on Bybit's book and Bybit's own trades,
one venue end to end.

## Re-creating the deleted data

```
https://data.binance.vision/data/futures/um/daily/bookDepth/BTCUSDT/BTCUSDT-bookDepth-<date>.zip
https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-<date>.zip
git show era-1-binance-bookdepth:rl/experiments/fetch_daily_klines.py  # candles + ADV
```
Unzip into `TradeData/BookDepth/` and `TradeData/AggTrades/`. Era-1 results also need the
era-1 binary: build at the commit tagged `era-1-binance-bookdepth`.
