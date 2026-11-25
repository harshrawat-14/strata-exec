# 34 — Do the BTC findings hold on SOL and DOGE? Result — yes; every route stays closed

Written 2026-11-25. Predictions were pre-registered in
[34-alt-coins-predictions.md](34-alt-coins-predictions.md) (commit 648c138). The drawdown
check's sign convention was corrected before the main run (commit 971f595; see that doc).
Held-out dates are untouched.

## Summary

All three tests fail on **both** coins, as on BTC:

- **Local liquidity:** a busy hour is not cheaper to sell into.
- **Impact decay:** it does not depend on volatility in a way that replicates.
- **Drawdowns:** no bounce-back and no momentum after a drawdown.

The BTC result now holds on three crypto perpetuals of very different liquidity: BTC, SOL and
retail-driven DOGE. **Within crypto perpetuals, no
regime signal observable in real time is left to exploit.**

## Data (test 0)

| | SOL | DOGE |
|---|---:|---:|
| full days (2,879–2,880 snapshots) | 343/343 | 343/343 |
| median trades per day (BTC: 1,456k) | 859k | 601k |
| volatility scale k (training-median daily vol ÷ BTC's) | 1.741 | 1.654 |
| regime cuts (crash / calm drift) | −1.74% / +0.96% | −1.65% / +0.91% |
| days crash / normal / calm | 53 / 173 / 117 | 73 / 149 / 121 |

Results: `rl/results/v4_day_stats_{solusdt,dogeusdt}.csv`.

## Test 1 — local liquidity (`impact_fit_local_liquidity_v4_*.csv`)

Main setting: 30 s buckets, 60 min trailing window.

| | model | pooled validation R² vs day | beats day: validation / training dates | CV of Y (day's) |
|---|---|---:|---:|---:|
| SOL | local V | −0.038 | 1/87, 0/256 | 0.107 (0.099) |
| SOL | local both | −0.001 | 23/87, 82/256 | 0.113 |
| DOGE | local V | −0.048 | 0/87, 8/256 | 0.182 (0.169) |
| DOGE | local both | −0.001 | 33/87, 99/256 | 0.191 |

This is BTC's pattern exactly. Scaling by the trailing hour's volume fits worse. Scaling by
volume and volatility together ties with the daily model. The 15 and 120 min windows and 5 min
buckets agree: local V is always worst, and local both is always within ±0.01 of day.
**FAIL on both.**

## Test 2 — impact decay by volatility (`impact_decay_fit_v4_*.csv`)

Pooled response G(τ)/G(0), training days (for the record, not a criterion):

| | 30 min | 2 h | 6 h |
|---|---:|---:|---:|
| BTC (doc 31) | 0.649 | 0.692 | 0.676 |
| SOL | 0.583 [0.561, 0.605] | 0.600 [0.567, 0.635] | 0.620 [0.568, 0.672] |
| DOGE | 0.649 [0.602, 0.700] | 0.587 [0.540, 0.635] | 0.607 [0.538, 0.676] |

Both coins have the same shape as BTC: about 40% of the impact fades within minutes, and the
rest is permanent. The permanent share is slightly lower than BTC's.

Decay by trailing 2 h volatility tercile, high − low at 2 h:

| | training | validation | tercile R² ≥ shared | verdict |
|---|---:|---:|---|---|
| SOL | +0.083 [−0.008, +0.170] | −0.012 (opposite sign) | yes | **FAIL** |
| DOGE | −0.092 [−0.276, +0.063] | −0.150 (same sign) | yes | **FAIL** (training interval includes 0) |

The two coins point in opposite directions, and neither is significant in training.

## Test 3 — drawdown signal (`bounce_back_test_*_signflip.csv`, main null)

Corrected correlation of drawdown_z with the next 2 h return (sign-flip null), with 90% intervals:

| | training | validation | deep-drawdown excess, training (bp) | verdict |
|---|---:|---:|---:|---|
| BTC (doc 32) | +0.013 [−0.012, +0.038] | −0.002 | — | fail |
| SOL | +0.012 [−0.013, +0.040] | +0.003 [−0.037, +0.048] | −3.0 [−13.4, +6.7] | **FAIL** |
| DOGE | +0.019 [−0.006, +0.044] | −0.048 [−0.087, −0.009] | −4.9 [−13.9, +3.6] | **FAIL** |

The training correlation is zero on both coins. DOGE's validation value is significant but has
the *opposite* sign to its training value: noise, not a signal that holds across periods.

**The secondary shuffled null "passes" for SOL** (momentum: +0.055 training, +0.066
validation, excess −16 / −17 bp). This is the artefact that doc 32 found on BTC. Shuffling also
removes volatility clustering, which by itself creates false momentum of about +0.04 to +0.06:

| corrected training corr | shuffled null | sign-flip null | artefact |
|---|---:|---:|---:|
| BTC | +0.071 | +0.013 | +0.058 |
| SOL | +0.055 | +0.012 | +0.043 |
| DOGE | +0.065 | +0.019 | +0.046 |

The pre-registration named the sign-flip null as the main one for this reason. The SOL
shuffled result is not a finding.

**Crash days** show a corrected deep-drawdown excess of +29 bp on both coins (BTC: +16 bp).
As in doc 32, this is a hindsight subgroup. A day labelled crash by its own drift selects
for paths whose average is low, so its conditional returns cannot be used in real time.
It is reported, not used.

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | Data complete (≥ 99%) for both | **Met** (100%, 100%) |
| 2 | k: SOL 1.5–2.5, DOGE 1.7–3.0 | **Half met**: SOL 1.741; DOGE 1.654, just below the range |
| 3 | Test 1 fails on both | **Met** |
| 4 | Test 2 fails on both; G(30 min)/G(0) < 0.9; terciles do not differ by ≥ 0.05 | **Mostly met**: fails on both and G(30 min) is 0.583 / 0.649. But the point differences, +0.083 and −0.092, exceed 0.05; they are not significant and disagree in sign |
| 5 | Test 3 fails on both, \|corrected training corr\| < 0.03 | **Met** (0.012, 0.019) |
| 6 | No test passes on either coin | **Met** |

## What this means

- **Decision rule, applied:** all tests fail on both coins. The BTC result generalises
  across three crypto perpetuals. Every route tested from docs 26–32 is closed, and less liquid
  markets did not reopen any of them.
- **Data quality is not the reason.** The same tests run on complete full-year data, on three
  markets, give the same answers.
- **The open options go to the user** (decision rule):
  1. Re-score and retrain M0-e2b on the recalibrated simulator (doc 33). This is due either
     way. M0-e2b's crash gain was measured on the old simulator and must be confirmed or
     withdrawn.
  2. Another asset class, such as equities, where intraday patterns (open and close auctions,
     the U-shaped volume curve, scheduled news) are structural rather than forecasts.
- Side note: the simulator's decay constants were fitted to BTC (doc 33). SOL's and DOGE's
  response curves are close to BTC's but not identical. A per-coin calibration would be needed
  before training on them.
