# 20 — The simulated order book has a ~45 bp cost floor; order size barely matters

Written 2026-10-23. Found while re-baselining with orders sized as a share of ADV
(protocol v3.1). No RL trained.

## What was done

- **Relative sizing** (`evaluation/liquidity.py`): order = share of the date's 20-day
  trailing ADV, from Binance daily candles (validated against our trade sums to 0.005%;
  strictly prior days, no look-ahead). Available via `order_share_adv` /
  `--order-share-adv`. At 1% of ADV the order is ~1,700–3,500 BTC ($100–200M).
- **Checked first:** the impact model's fixed 240,000 BTC volume setting does *not*
  distort impact — the per-date calibration rescales the impact coefficient to absorb it
  exactly (identical impact at 240k and 1.28M). An earlier claim that it overstated
  impact on busy days by up to 5× was wrong.

## The finding

`rl/results/v31_baselines.csv`: AC and constant paces give **identical results at 0.5%,
1% and 2% of ADV**, and match the 50,000 BTC runs to the fourth decimal. Direct test, same
AC episode: IS 0.319% at 500, 5,000 and 50,000 BTC; 1.64% only at 500,000 BTC (about two
days' volume). The first fill costs the same 40 bp at every size.

Cause: Binance's free `bookDepth` files are not an order book. They give cumulative depth
within ±1%, ±2%, … ±5% of mid — five coarse bands per side. The simulator treats the 1%
band as the best level, priced at the band's average:

| date | simulated "best bid" below mid | simulated spread | real BTCUSDT-perp spread |
|---|---:|---:|---:|
| 2024-01-15 | 43 bp | 86 bp | ~0.02 bp ($0.10) |
| 2024-07-05 | 44 bp | 88 bp | |
| 2024-08-05 | 50 bp | 100 bp | |
| 2024-09-15 | 51 bp | 103 bp | |

So every sale pays ~43–51 bp regardless of size — this is the ~0.46% "execution" component
seen for every strategy in docs 13–19. Size only matters once an order spills past the
first band.

## What it means

- **Execution-cost comparisons in this simulator are dominated by an artificial floor.**
  At realistic institutional sizes, strategies differ almost only through timing (price
  drift while holding) and through occasionally crossing into the next 1% band (the
  "2.7× cost above 0.3% of the order" in doc 18 is that band crossing, not real convexity).
- **The research question (institutional execution) cannot be answered well on this book.**
  The project already knew `half_spread()` was a weak proxy for this format (doc 06 s11
  diagnosis); its consequence for every cost number had not been drawn.
- Published relative findings (RL vs AC on crash days, etc.) were measured on this book and
  inherit the caveat.

## Data available to fix it

Binance's free archive has real best bid/ask with sizes (`futures/um/daily/bookTicker`) only
through March 2024: **3 of our 25 dates** (01-15, 02-20, 03-05). Full historical L2 needs a
paid source (e.g. Tardis.dev, scoped earlier at ~$599/month).
