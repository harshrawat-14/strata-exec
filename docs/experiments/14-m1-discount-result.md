# 14 — M1 (γ 0.99 → 0.999): result — negative, and the design was flawed

Written 2026-10-19. Predictions in [14-m1-discount-predictions.md](14-m1-discount-predictions.md).
Validation dates only.

## Result

Same recipe as M0 except `--gamma 0.999`; 1.5M steps, 44 min, no errors. Offline
selection (`rl/results/m1_v3_gamma999_checkpoint_selection.csv`) picks the **final**
checkpoint at **−1.27%** mean validation IS (M0: −0.47%). The learning curve is still
rising at 1.5M: flat around −2.19% to −2.02% until 800k, then −1.93 → −1.27. M0 rose
steeply from ~500k and peaked at 1.2M.

`rl/results/m1_v3_is_decomposition.csv` (M1 − AC, pp):

| date | regime | IS | execution | timing | avg sell time M1 / AC |
|---|---|---:|---:|---:|---|
| 01-15 | calm | −1.14 | −0.88 | −0.25 | 0.05 / 0.45 |
| 02-20 | normal | −1.03 | −0.76 | −0.26 | 0.05 / 0.45 |
| 04-15 | crash | −0.58 | −0.92 | +0.34 | 0.06 / 0.45 |
| 09-15 | calm | −0.57 | −0.70 | +0.14 | 0.05 / 0.45 |

M1 sells almost the whole order in the first ~5% of the episode, the behaviour of a barely
trained policy. Training reward reached −0.0122 per episode (M0: −0.0085).

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | Execution cost moves toward AC on ≥3/4 dates | **Not met** (worse than M0 on 4/4) |
| 2 | Average sell time later than M0 on ≥3/4 | **Not met** (earlier on 4/4) |
| 3 | Mean validation IS beats M0's −0.465% | **Not met** (−1.27%) |
| 4 | Crash-date timing gain smaller than M0's +0.63 | Met (+0.34), but only because M1 is undertrained |
| 5 | Training reward ≥ AC's on ≥1 date | **Not met** |

## Why: the design did not test its own hypothesis

The hypothesis was "γ = 0.99 limits planning to ~100 steps." PPO's advantage estimate
(GAE) assigns credit over an effective horizon of about 1 / (1 − γλ). With λ = 0.95 that
is ~17 steps at γ = 0.99 and only ~20 steps at γ = 0.999 — so raising γ alone barely
lengthens credit assignment. What it does change is the value function's target: a
discounted sum over ~1,000 steps instead of ~100, which is higher-variance and slower to
fit. That predicts exactly what was observed: slow learning, a policy still near its
starting behaviour at 1.5M. **This result says "γ alone is the wrong knob at this
budget", not "the planning horizon doesn't matter."**

## Status

M1 is not adopted; M0 remains the best model. Logged in docs/history (L6/L9 pattern:
a mechanism claim acted on without checking the actual algorithm's horizon).
