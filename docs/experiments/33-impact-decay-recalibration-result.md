# 33 — Recalibrating the simulator's impact decay: result

Predictions in [33-impact-decay-recalibration-predictions.md](33-impact-decay-recalibration-predictions.md).

**Runs and data:**
- Fit: `rl/experiments/impact_decay_calibrate.py` → `rl/results/impact_decay_calibrate*.csv`.
- Simulator change: commit c72c1ce.
- Baselines rerun: `python rl/experiments/era2_baselines.py rl/results/era2_baselines_doc33.csv`
  (protocol v3, the same 16 V3 dates and sizes as doc 22).

## Fit: accepted

| | fitted (90% interval) | old |
|---|---|---|
| permanent share π | **0.435** [0.392, 0.486] | 0.3 |
| decay rate ρ | **2,284/day** [1,771, 3,050] | 5/day |
| transient half-life | **26 s** [20, 34] | 200 min |

**A.** Inside the training band at 30 min (0.660 vs 0.647 [0.621, 0.677]), 2 h (0.660 vs 0.687
[0.627, 0.751]) and 6 h (0.660 vs 0.676 [0.590, 0.781]): **pass**.

**B.** Validation RMSE 0.034 vs 0.307 for the old constants (11%): **pass**.

**Limitation:** only 62% of the 34 grid lags fall inside the band. Real reversion has a sharp
part (lag 1: −0.20 of the instant response) and a smaller tail out to about 8 minutes. A
single exponential compresses both into a 26 s half-life. A two-speed transient would match
better. It is not needed for the decisions here: the agent acts every 30 s, and both
versions put nearly all lasting impact in the permanent share.

## Simulator change and tests

`CounterfactualLobConfig::default()` now uses `CALIBRATED_PERMANENT_FRACTION` = 0.435 and
`CALIBRATED_DECAY_RHO` = 2,284.

**Tests:**
- a single sale's response follows the kernel exactly, for any parameters;
- the defaults lie inside doc 31's measured bands (this fails with the old constants);
- `slicing_an_order_does_not_multiply_its_impact` keeps its purpose, but its lower bound is now the
  permanent level. The old 0.95 × one sale only held for a 3.3 h half-life.

**Golden fixtures,** regenerated deliberately:
- only the counterfactual fixtures changed;
- the permanent-impact readout rose by exactly √(0.435/0.3).

## Baselines on the recalibrated simulator (5% of ADV)

**Mean IS (%), training dates:**

| pace | 0.5× | 1× (TWAP) | 1.5× | 2× | 3× |
|---|---:|---:|---:|---:|---:|
| doc 22 (old decay) | −0.496 | −0.205 | −0.252 | −0.299 | −0.347 |
| **doc 33** | −0.439 | **−0.145** | −0.171 | −0.204 | −0.236 |

**By regime (doc 33):**

| split / regime | 0.5× | 1× | 1.5× | 2× | 3× |
|---|---:|---:|---:|---:|---:|
| train calm | +1.09 | +0.97 | +0.72 | +0.59 | +0.39 |
| train normal | −1.18 | −0.37 | −0.16 | −0.16 | −0.19 |
| train crash | −1.24 | −1.26 | −1.37 | −1.33 | −1.15 |
| validation calm | −0.65 | −0.13 | −0.12 | −0.10 | −0.11 |
| validation normal | −0.32 | −0.02 | +0.06 | +0.03 | −0.12 |
| validation crash | −2.32 | −0.70 | +0.11 | +0.25 | +0.08 |

**Findings:**
- **TWAP:** 0.059 pp cheaper at 5% of ADV. At 1% and 10% of ADV it is 0.027 and 0.084 pp
  cheaper.
- **Honest AC is still TWAP.** λ = 3e-9 is selected again at every size, and AC − TWAP is
  0.0001 pp.
- **Selling faster costs about 40% less extra impact.** On training days, 3× pace vs TWAP goes
  from −0.143 to −0.091 pp, 2× from −0.094 to −0.059, and 1.5× from −0.048 to −0.026. Market
  drift is identical between the two runs, so this difference is purely the impact model. The
  remaining cost is book walking and the brief transient. (This hypothesis was stated after
  the fit, before the rerun.)
- **Hindsight headroom** (the best constant pace per date vs TWAP) is 0.389 pp, up from 0.369:
  calm 0.16, normal 0.35, crash 0.79.
- 0.5× still cannot finish and dumps the remainder at the end.

## Predictions

| # | prediction | result | |
|---|---|---|---|
| 1 | π in [0.38, 0.52] | 0.435 | met |
| 2 | half-life 0.5–5 min | 0.44 min | not met (faster) |
| 3 | both criteria pass; validation RMSE ≤ ⅓ of old | pass; 11% | met |
| 4 | TWAP 5% training IS changes by +0.005 to +0.05 pp | +0.059 pp | not met (slightly larger) |
| 5 | honest AC still selects λ → 0 | λ = 3e-9; AC = TWAP to 0.0001 pp | met |

## Consequences

- **Every era-2 result scored before c72c1ce used the old decay.** That covers M0-e2b, M1 and
  doc 22's baselines. Doc 22 is kept for the record. Model comparisons must be re-scored, and
  models retrained, on the recalibrated simulator.
- **Deviating from TWAP is cheaper than before,** but there is still no signal that tells the
  agent which way to deviate (docs 26–32). The calibration lowers the price of being wrong,
  not the odds.
