# 33 — Recalibrating the simulator's impact decay: design and predictions

Written 2026-11-25, before the fit is run. Triggered by doc 31's pre-registered criterion A.
No RL is trained.

## Why

The simulator's impact is I = Y · σ_d · √(D/V), with

    D_t = π · Q_t + (1 − π) · Σ_s q_s · e^(−ρ (t − s))

and π = 0.3, ρ = 5/day (3.3 h half-life). Both values came from era 1 and were never fitted.

**What doc 31 measured, on 256 training days:**
- about 35% of the response to taker flow reverts within about 8 minutes;
- after that the response is flat out to 6 h: G(τ)/G(0) is 0.65 at 30 min, 0.69 at 2 h and
  0.68 at 6 h.

The simulator gives 0.965, 0.873 and 0.707, which is outside the interval at 2 h. The model's
form (a permanent part plus an exponentially decaying part) can produce the measured shape.
Only its two constants are wrong. This doc refits π and ρ and keeps everything else (Y, σ_d,
V) unchanged.

## Method (`rl/experiments/impact_decay_calibrate.py`)

- **Real curve:** G(τ)/G(0) from doc 31's pooled propagator kernel on the 256 V4 training days.
- **Simulator curve** for one small sale: √(π + (1 − π) e^(−ρ τ dt)), with dt = 1/2880 day.
- **Fit:** least squares over 40 log-spaced lags from 1 to 720. Each part of the curve is
  weighted alike, so the ~8-minute reversion is not swamped by the long plateau.
- **Uncertainty:** a 300-resample date bootstrap for intervals on π, ρ and the curve.
- **Validation:** the 87 V4 validation days' curve, scored by RMSE over the same lags.

Tests (`rl/tests/test_impact_decay_calibrate.py`) check three things: the curve code matches
doc 29/31's; the old parameters reproduce the documented 0.965 / 0.873 / 0.707; and the fit
recovers known parameters.

## Acceptance (both must hold to change the simulator)

**A.** The fitted curve lies inside the training 90% band at 30 min, 2 h and 6 h.

**B.** On validation, the fitted curve's RMSE is at most half the old parameters' RMSE.

If A fails, the two-parameter form cannot match the data. A different form would then need
its own design doc. If both hold, the new constants go into `CounterfactualLobConfig::default()`
as a tested change:
- a unit test that a single sale's simulated response follows the fitted curve;
- the decay tests' comments updated;
- golden fixtures regenerated deliberately, with every changed value inspected.

Then the doc 22 baselines are re-run on the V3 dates (same protocol) into a new results file,
with honest-λ AC selected again on training dates only.

## Predictions

1. **π between 0.38 and 0.52** (the plateau G∞/G(0) = √π is 0.62–0.72).
2. **Transient half-life between 0.5 and 5 minutes** (ρ ≈ 200–2,000 per day).
3. **Both acceptance criteria pass.** Validation RMSE falls to at most a third of the old.
4. **TWAP's mean IS at 5% of ADV on V3 training days changes by between +0.005 and +0.05
   pp** (slightly cheaper). Old: D at mid-horizon ≈ 0.32 Q, at the end ≈ 0.50 Q. New: about
   0.23 Q and 0.46 Q.
5. **Honest-λ AC still selects λ → 0, so AC = TWAP.** A larger permanent share does not
   reward selling early under risk-neutral cost.
