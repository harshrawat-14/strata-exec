# 34 — Do the BTC findings hold on SOL and DOGE? Design and predictions

Written 2026-11-25, before any SOL or DOGE data is analysed. No RL is trained.

## Why

Docs 26–32 closed every tested route to a regime-aware edge over TWAP on BTCUSDT:

- forecasting direction from the state or the book (docs 26–27);
- timing liquidity (doc 28);
- impact that fades at a different speed in volatile hours (docs 29, 31);
- a bounce-back after drawdowns (doc 32).

BTC is the most liquid and most arbitraged crypto market. If a regime-dependent effect
exists anywhere in crypto, it is more likely in thinner markets with more retail flow. This
doc runs the same tests on two Bybit perpetuals:

- **SOLUSDT**, large but less liquid than BTC;
- **DOGEUSDT**, retail-driven.

The data is the same V4 pool of dates as doc 30, downloaded the same way: 343 of 343 days
for each coin. Training is 256 days (Jan–Sep 2024) and validation is 87 days (Oct–Dec 2024).
Held-out dates are not read.

## What changes from the BTC runs

Each script gets a `--dataset NAME` option and writes results with a per-coin suffix. With
no option, every BTC output stays the same. Three changes are needed so that the tests mean
the same thing on a more volatile coin. Each is fixed here, before any alt-coin data is seen:

1. **Regime cuts scale with volatility.** Doc 30's drift cuts (−1.0%, +0.55%) were set on BTC.
   For another coin, both cuts are multiplied by k = (the coin's median daily volatility over
   its training days) ÷ (BTC's, 2.485%). A coin twice as volatile needs twice the drift to
   count as a crash. The labels are used only for descriptive tables and for doc 32's
   crash-day criterion.
2. **Doc 32's 5 bp floor is scaled by the same k.** A bounce of 5 bp on a coin twice as
   volatile is half as large relative to its noise.
3. **Doc 32's test counts both directions.** Doc 32 tested only for a bounce-back. Momentum
   after drawdowns, the opposite sign, would also be a usable regime signal: sell faster after
   a fall. On SOL and DOGE, either sign passes if all four criteria hold with that sign. The
   **sign-flip null is the main null**. Doc 32 showed that the shuffled null leaves a false
   momentum signal of about +0.07, because it also removes volatility clustering. The
   shuffled null is still reported.

Doc 28's test also moves to the V4 pool. It was run on the 16 V3 dates for BTC; no V3 dates
exist for these coins. Its win-count criterion becomes a share: at least 75% of validation
dates and 83% of training dates (doc 28's 3 of 4 and 10 of 12).

Doc 27's machine-learning pre-check on book features is not repeated. It was the most
expensive test, its one hint (drawdown) is tested directly by doc 32's method, and doc 26
and doc 27 agreed that direction is not forecastable on BTC.

## Tests, per coin

| # | Test | Script | Pass criterion (unchanged unless noted above) |
|---|---|---|---|
| 0 | Data completeness | `v4_data_check.py` | ≥ 99% of days with 2,879–2,880 snapshots |
| 1 | Local liquidity (doc 28) | `impact_fit_local_liquidity.py v4` | local V or local both beats day: pooled validation R² +0.01; ≥75% validation and ≥83% training dates; lower CV of Y |
| 2 | Decay by volatility (doc 31, B) | `impact_decay_fit.py v4` | high − low G(2h)/G(0): training 90% interval excludes 0; validation same sign; tercile R² ≥ shared; magnitude ≥ 0.05 |
| 3 | Drawdown signal (doc 32) | `bounce_back_test.py signflip` | training and validation corrected correlation intervals exclude 0, same sign; deep-drawdown excess ≥ 5k bp in both with that sign, training interval excluding 0; crash-day correlation with that sign or 0 |

Test 2 also reports the coin's pooled decay curve (doc 31, A) for the record. It is not a
criterion: the simulator is not calibrated to these coins yet.

## Decision rule

- **Any test passes on a coin:** that coin becomes the candidate market. Next steps, in
  order: calibrate the simulator to it (doc 33's method); check that a simple rule using the
  passing signal beats TWAP in every regime on validation; only then RL.
- **All tests fail on both coins:** the BTC result generalises across three crypto perpetuals
  of very different liquidity. That is reported as a finding. The next step goes to the user:
  re-scoring M0-e2b on the recalibrated simulator is due either way, and another asset class
  (equities, which the user raised) is the main open option.

## Correction before the main run (2026-11-25)

Test 3's wording "with that sign" was ambiguous, and the code (`two_sided`) read it wrongly.
One effect moves the two statistics in **opposite** directions. drawdown_z is negative in a
drawdown, so a bounce-back is correlation < 0 with deep-drawdown excess > 0 (doc 32), and
momentum is correlation > 0 with excess < 0. The code required both to share a sign, so it
could never pass a real effect. Corrected criterion: the excess has the sign opposite to the
correlation. Everything else is unchanged.

Found when the secondary shuffled-null run printed SOL training corr +0.055 with excess
−16 bp. A broken shell loop meant that this run was the only one made. The main
sign-flip run and tests 1–2 had not run yet; they are run after this correction is committed.
Shuffled-null figures already seen (training / validation corrected corr): SOL +0.055 / +0.066,
DOGE +0.065 / +0.015. On BTC this null gave +0.071 where the sign-flip null gave +0.013.

## Predictions

1. **Data is complete** (≥ 99% full days) for both coins.
2. **Volatility:** k is 1.5–2.5 for SOL and 1.7–3.0 for DOGE.
3. **Test 1 fails on both coins.** Within a day, volume and volatility move together on
   altcoins at least as much as on BTC, so σ/√V stays flat.
4. **Test 2 fails on both coins.** The early reversion is fast (G(30 min)/G(0) < 0.9 on both),
   as on BTC, and does not differ by volatility tercile by 0.05 or more.
5. **Test 3 fails on both coins**, with |corrected training correlation| < 0.03.
6. **Overall: no test passes on either coin.** Confidence is lower for DOGE: its retail flow is
   the likeliest place for a drawdown effect (prediction 5).
