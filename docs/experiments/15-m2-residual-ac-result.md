# 15 — M2 (residual around AC): result — negative; the credit horizon is the common cause

Written 2026-10-20. Predictions in [15-m2-residual-ac-predictions.md](15-m2-residual-ac-predictions.md).
Validation dates only.

## Result

Same recipe as M0 plus `--action-space residual_ac`; 1.5M steps, 28 min, no errors.
Offline selection (`rl/results/m2_v3_residual_ac_checkpoint_selection.csv`) picks the
**400k** checkpoint at **−0.686%** mean validation IS. Validation IS gets *worse* with
training (−0.70 at 100–400k → −1.05 to −1.65 later), and from 500k on 60–80% of episodes
end in a (small, 1–2.5% of the order) forced dump. Training reward meanwhile improved
(−0.0199 → −0.0148).

Selected checkpoint vs AC (`rl/results/m2_v3_is_decomposition.csv`, pp):

| date | regime | IS | execution | timing |
|---|---|---:|---:|---:|
| 01-15 | calm | −0.29 | −0.29 | −0.01 |
| 02-20 | normal | −0.25 | −0.27 | +0.01 |
| 04-15 | crash | −0.24 | −0.23 | −0.02 |
| 09-15 | calm | −0.20 | −0.23 | +0.01 |

## What M2 actually does

| policy (01-15) | offsets chosen | switches / episode | trades / episode | largest trade |
|---|---|---|---|---|
| AC (δ = 0) | always 0 | 0 | 1,758 | 0.2% of order |
| M2 @ 400k | −0.02 (74%), 0 (24%) | 116 | 1,464 | 2.9% |
| M2 final | −0.05/−0.10/−0.02/−0.20 | 336 | 558 | 4.9% (~10 trades > 1% per episode) |

M2 learns to **fall behind** AC's schedule and catch up in **bursts** whenever its offset
rises; each burst walks deep into the book. More training makes it lag and burst more.
Not a reward loophole: M2's reward is also worse than AC's (−0.017 vs −0.0044 at the end).

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | Mean validation IS > AC's −0.4425% | **Not met** (−0.686%) |
| 2 | Execution within 0.05 pp of AC on ≥3/4 | **Not met** (−0.23 to −0.29) |
| 3 | Beats AC on 04-15 | **Not met** (−0.24) |
| 4 | Within 0.10 pp of AC on both calm dates | **Not met** (−0.29, −0.20) |
| 5 | Higher average offset on 04-15 than calm dates | Not meaningfully met (marginal; offsets all near −0.02/0) |
| 6 | Beats M0's −0.4651% | **Not met** |

## The common cause: PPO's ~17-step credit horizon

Falling behind the schedule costs nothing on the step it happens; the cost arrives at the
later catch-up step. PPO's GAE (γ 0.99, λ 0.95) assigns credit over ~1/(1−γλ) ≈ 17 steps,
so lagging looks free and the catch-up looks expensive — the agent learns to lag more.
The same blind spot fits the earlier results: M0's fixed early selling (it avoids
deferred costs it cannot see), and M1, where raising γ alone left the ~17-step credit
horizon unchanged. **Pacing costs arrive hundreds of steps after the decision; the
learner cannot connect them.** The next experiment should change the credit horizon
itself (GAE λ), not the action space or γ alone.

## Status

M2 not adopted; M0 remains the best RL model, and AC still leads on the validation mean.
