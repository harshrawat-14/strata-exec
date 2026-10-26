# 21 — Era 2 impact model: our selling moves the price

Date: 2026-10-26. Data: Bybit BTCUSDT perpetual, 25 research dates, real 500-level book
(30 s) and every trade.

## What was wrong

On the real book, AC's mid-price path was identical whether it sold 100 or 20,000 BTC.
The impact model never moved the price:

1. **It cut depth instead.** The impact fraction thinned the resting bids, so our sales
   walked a little further down the book while the price stayed put. Era 1's 1%-wide bands
   hid this. A real book with $0.10 between levels does not.
2. **The fitted Y measured volatility, not impact.** Each trade's Y was the whole 30 s mid
   move divided by that one trade's √(size/volume), averaged over trades, so it gave Y = 7–11.
3. **Every slice added its own square root.** Summing per-slice square-root impacts makes
   1,920 small slices far costlier than one block, the opposite of the square-root law for
   a metaorder.

## The fit

`rl/experiments/impact_fit_signed_flow.py` regresses each bucket's mid log-return on the
net taker flow Q (buys − sells): r = Y · σ_d · sign(Q) · √(|Q|/V), with σ_d the date's daily
volatility and V its daily volume. Results are in `rl/results/impact_fit_signed_flow.csv`.

| bucket | Y, training-date median | Y range, all 25 dates | R² |
|---|---:|---:|---:|
| 30 s | **1.07** | 0.91–1.33 | 0.38–0.51 |
| 5 min | 1.57 | 1.22–2.04 | 0.43–0.64 |

Taker flow explains about half of every 30 s price move. Y is stable across dates and close
to published values (~0.5–1.5). Crash dates sit at the top: 03-05 1.22, 08-05 1.33. The fit
is concurrent flow vs return, so it is an upper-side estimate: some of the flow is itself a
reaction to the price move.

## The model now

    I_t = Y · σ_d · √(D_t / V)          applied as a shift of every price in the book
    D_t = 0.3 · Q_t + 0.7 · Σ q_s · e^(−ρ (t − s))     ρ = 5/day (half-life ≈ 400 steps)

- Y = 1.0715 is frozen at the training-date median, like λ.
- σ_d and V come from each date's own book and trades. These are environment physics, not
  agent inputs.
- Depth stays as recorded, so the book walk still prices spread and depth.
- The 30% permanent / 70% decaying split and ρ are carried over unchanged from era 1. They
  are not fitted yet.
- Unit tests cover square-root scaling, that slicing does not multiply impact, the price
  shift with unchanged depth, decay to the permanent level, and reset.
- State dims 8 and 9 now read price impact, with 1% of price ≈ 0.76.

## First result (AC, λ = 1e-6, 16 h from the middle of 2024-01-15)

| order | share of day volume | IS | spread / depth | mid move incl. our impact |
|---|---:|---:|---:|---:|
| 100 BTC | 0.1% | +0.165% | −0.0002% | +0.165% |
| 1,000 BTC | 0.9% | +0.079% | −0.0008% | +0.080% |
| 5,000 BTC | 4.5% | −0.076% | −0.0025% | −0.074% |
| 20,000 BTC | 17.9% | −0.360% | −0.0076% | −0.353% |

The 100-BTC run is essentially pure market drift. Our impact costs −0.09, −0.24 and
−0.52 pp at 1k, 5k and 20k BTC, roughly square-root in size. Crossing the spread costs under
1 bp, so the problem is now the classical one: pace against impact and risk.

## Not yet redone for era 2

- AC's λ selection.
- Constant-pace baselines.
- ADV-relative sizing from Bybit volume.
- The golden fixtures.
