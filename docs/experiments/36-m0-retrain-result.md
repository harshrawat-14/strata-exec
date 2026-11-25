# 36 — M0 recipe retrained on the corrected simulator. Result — a constant 10× pace

Written 2026-11-25. Predictions were pre-registered in
[36-m0-retrain-predictions.md](36-m0-retrain-predictions.md) (commit 9613d0a). Held-out dates
are untouched.

## Summary

Retrained on the corrected simulator, the M0 recipe (M0-e2c) learned to **sell at one fast
constant pace: about 10× TWAP**. It finishes in a median of 173 of 1,920 steps. Out of sample,
its per-regime IS is within 0.02 pp of a fixed 10× pace in every regime. Cheaper fast selling
(doc 33) moved it from about 4× (M0-e2b) to about 10×. It still does not condition on the
regime. The project goal is not met.

## Training and selection

- Same command, seeds and dates as doc 23. 1.5M steps took 33 min, with no errors.
- Selected checkpoint: `final` (mean V3 validation IS −0.200, against M0-e2b's −0.207).
- The validation curve was still improving at the end, as in doc 23.
- No forced liquidation at any checkpoint.

## Result (`m0_rescore.py --model rl/models/m0_e2c_final --name M0-e2c`)

Files: `rl/results/m0_e2c_rescore_doc36.csv`; 10× and 20× paces in
`m0_e2c_fast_paces_doc36.csv`.

### oos: 85 V4 validation days

| regime | days | M0-e2c | TWAP | pace 5 | pace 10 | pace 20 | M0-e2c − TWAP [90%] | days M0-e2c wins |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| crash | 10 | −0.416 | −1.251 | −0.521 | −0.411 | −0.361 | **+0.835** [+0.620, +1.067] | 10/10 |
| normal | 46 | −0.287 | −0.421 | −0.269 | −0.285 | −0.300 | +0.134 [+0.040, +0.233] | 32/46 |
| calm | 29 | −0.197 | +0.717 | −0.077 | −0.180 | −0.245 | **−0.915** [−1.095, −0.727] | 2/29 |
| **all** | 85 | −0.272 | **−0.130** | −0.233 | −0.264 | −0.288 | −0.141 | |

M0-e2c minus a constant pace, by regime:

| | crash | normal | calm |
|---|---:|---:|---:|
| pace 5 | +0.105 | −0.017 | −0.121 |
| **pace 10** | **−0.005** | **−0.002** | **−0.017** |
| pace 20 | −0.055 | +0.013 | +0.048 |

Median steps to finish: M0-e2c 173, pace 10 192, pace 20 96.

**Regime-awareness check:** paces 10 and 20 bracket the model's overall IS (−0.264 and −0.288
against −0.272). M0-e2c beats neither in crash *and* calm. It beats pace 20 only on calm days
and pace 10 nowhere. **FAIL:** it is a constant pace.

### v3: the 16 doc 23 dates

| | crash (3) | normal (8) | calm (5) | all |
|---|---:|---:|---:|---:|
| M0-e2c − TWAP | +0.556 [−0.143, +1.255] | +0.122 | −0.849 | −0.100 |
| M0-e2b − TWAP (doc 35) | +0.533 | +0.136 | −0.770 | −0.072 |

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | No forced liquidation | **Met** |
| 2 | Sells at least as fast as M0-e2b (oos median steps ≤ 676) | **Met** (173) |
| 3 | Again one constant pace (regime-awareness check fails) | **Met**: within 0.02 pp of pace 10 in every regime |
| 4 | Goal criterion fails: calm interval below 0 | **Met** (−0.915 [−1.095, −0.727]) |
| 5 | Overall oos mean below TWAP's | **Met** (−0.272 vs −0.130) |

## What this means

- **The pace the recipe learns is set by impact costs, not by the regime.** Cheaper fast
  selling moved it from about 4× to about 10× TWAP. Both models then win crashes and lose calm
  days by the same mechanism. Neither conditions its pace on what it observes. This matches
  docs 26–34: BTC, SOL and DOGE offer no real-time regime signal to condition on.
- **Why fast and not TWAP?** TWAP is the best constant pace overall (−0.130). The recipe
  learns speed because its reward gives waiting a small penalty and charges our own impact,
  but not the market's drift. On its training and selection dates, fast selling also scored
  well: the 4 V3 validation dates include a crash and falling days (doc 23's caveat). The
  learned pace reflects this setup, not a forecast.
- **The crypto chapter of the ladder is complete, and negative.** On this data no tested
  policy beats TWAP, which equals honest AC, in every regime. The model "wins" only on
  hindsight-labelled crash days, by selling fast. Both era-2 models are now explained.
- **Next (agreed with the user):** equities, where intraday structure exists (open and close
  auctions, the U-shaped volume curve, scheduled news). That builds on the live-feed platform
  the user wants designed next. Before any RL there, run the same pre-checks (docs 26–34) to see
  whether a real-time regime signal exists.
