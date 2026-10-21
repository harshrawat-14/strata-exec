# 16 — M3 (γ 0.999, λ 0.99): interim result at 3M steps — not converged

Written 2026-10-21. Predictions in [16-m3-credit-horizon-predictions.md](16-m3-credit-horizon-predictions.md).
Validation dates only.

## Result

M0's recipe with `--gamma 0.999 --gae-lambda 0.99` (credit horizon ~91 steps), 3M steps,
~80 min, no errors, no sleep gaps. Offline selection
(`rl/results/m3_v3_credit91_checkpoint_selection.csv`) picks the **3M** checkpoint at
**−0.925%** mean validation IS. The curve improves monotonically and is steepest at the
end: −2.23% (100k) → −1.51% (1.5M) → −1.49% (2.0M) → −1.26% (2.7M) → −0.92% (3.0M).
No forced dumps at any checkpoint. In-training validation reward rose smoothly across all
60 rounds — unlike M0 (oscillating) and M2 (degrading).

Selected checkpoint vs AC (`rl/results/m3_v3_is_decomposition.csv`, pp):

| date | regime | IS | execution | timing | M3 avg sell time (AC 0.45) | largest trade |
|---|---|---:|---:|---:|---:|---:|
| 01-15 | calm | −0.72 | −0.54 | −0.16 | 0.08 | 6.5% |
| 02-20 | normal | −0.66 | −0.41 | −0.22 | 0.08 | 8.6% |
| 04-15 | crash | −0.19 | −0.57 | +0.39 | 0.09 | 2.9% |
| 09-15 | calm | −0.36 | −0.49 | +0.14 | 0.07 | 8.5% |

At 3M, M3 still sells almost the whole order in the first ~8% of the episode in large
chunks — the behaviour M0 showed at ~500–600k steps. Training episodes lengthen slowly
(172 → 464 steps; M0 reached ~1,400).

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | Validation curve above −1.0% before 1.5M | **Not met** (−1.51% at 1.5M) |
| 2 | Selected checkpoint beats M0 (−0.4651%) | **Not met** (−0.925%) |
| 3 | Execution closer to AC than M0 on ≥3/4 | **Not met** (worse on 4/4) |
| 4 | Largest trade < 1% of order on every date | **Not met** (2.9–8.6%) |
| 5 | Beats AC's mean (−0.4425%) | **Not met** |

## Reading

Unlike M1 (γ alone: barely learning) and M2 (learned the wrong thing), M3 is learning
steadily in the right direction but has not converged. The longer credit horizon costs
far more training than budgeted: by sample count M3 at 3M is roughly where M0 was at
0.5M. 3M steps cannot say whether the ~91-step horizon ends up better than M0.

## Status

Interim. M0 remains the best RL model; AC leads the validation mean.

## Continuation to 6M (decided 2026-10-21, before resuming)

Resumed from the 3M checkpoint with no setting changed, to 6M total steps. **Stopping
rule, fixed now:** if no checkpoint up to 6M beats M0's −0.4651% mean validation IS
(selected the same way), the credit-horizon line is closed and the next experiment is
potential-based shaping that charges the cost of falling behind immediately.
