# 15 — M2: residual action space around AC (design and pre-registered predictions)

Written 2026-10-20, before training.

## Why

M0 loses to AC on **execution cost** on every validation date and scores worse than AC on
its own training reward; its crash-day win is timing (doc 14). M1 (γ 0.999) failed to
fix execution (doc 14 result). M2 hands RL AC's execution and asks it only to learn
timing.

## Design

`evaluation/residual.py`: each step the agent picks an offset δ ∈ `DELTAS` =
{−0.2, −0.1, −0.05, −0.02, 0, +0.02, +0.05, +0.1, +0.2, +0.3, +0.5} (fraction of the
episode). The target inventory after the step is AC's scheduled inventory at
(elapsed + δ·horizon); the trade is executed with AC's closed-loop tracking. **δ = 0 is AC
action for action** (tested), so RL starts from AC's execution quality. Training wrapper
and evaluation agent share one mapper (tested equal).

Everything else identical to M0: protocol v3, 12 training dates, γ 0.99, 1.5M steps,
`ent_coef` 0.01, constant LR, CPU, offline selection (`select_checkpoint_v3.py
--action-space residual_ac`), decoding by expected δ (`MeanValueDecoder`).

## What constant offsets can do (no learning)

`rl/results/residual_constant_offsets.csv`: the best *constant* δ beats AC by only
0.04–0.06 pp per date, and running far ahead is costly (δ = +0.5 loses ~1 pp). Yet M0's
04-15 result (−0.20%) beat every constant offset (best −0.61%): adaptive timing has real
value. M2's gain must come from choosing δ by state, within the episode.

The bar: AC's mean validation IS is **−0.4425%**, already slightly better than M0's
−0.4651%.

## Predictions (validation dates)

1. **Beats AC on average:** M2's selected-checkpoint mean validation IS > −0.4425%.
2. **Execution stays AC-like:** |M2 − AC| execution component ≤ 0.05 pp on ≥ 3 of 4 dates.
3. **Crash date:** M2 beats AC on 04-15.
4. **Calm dates:** M2 is within 0.10 pp of AC on both 01-15 and 09-15.
5. **Regime-dependent timing:** M2's average chosen δ is higher on 04-15 than on both
   calm dates.
6. **Beats M0** on mean validation IS (> −0.4651%).

Each will be marked met / not met, including misses.
