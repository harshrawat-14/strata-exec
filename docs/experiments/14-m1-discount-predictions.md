# 14 — M1: discount factor 0.99 → 0.999 (design and pre-registered predictions)

Written 2026-10-19, before training.

## Why this is M1 (the ladder was reordered)

Splitting M0's validation shortfall into execution and timing
(`rl/experiments/decompose_is_v3.py` → `rl/results/m0_v3_is_decomposition.csv`):

| date | regime | IS (M0 − AC) | execution | timing (drift) | training reward (M0 − AC) |
|---|---|---:|---:|---:|---:|
| 01-15 | calm | −0.35 | −0.24 | −0.09 | −0.0024 |
| 02-20 | normal | −0.07 | −0.13 | +0.08 | −0.0011 |
| 04-15 | crash | +0.45 | −0.16 | +0.63 | −0.0015 |
| 09-15 | calm | −0.12 | −0.18 | +0.08 | −0.0018 |

M0 pays more execution cost than AC on every date and scores worse than AC on **its own
training reward** on every date. It sells earlier than AC on every date regardless of
regime (inventory-weighted average sell time 0.28–0.31 of the episode vs AC's 0.45); its
crash-day win is that early selling meeting a falling price, not better execution.

The originally planned M1 (AC-relative reward) only subtracts an action-independent
baseline: it cannot change the optimal policy, and the noise it would remove is small
because the reward is already measured against the contemporaneous mid. A more direct
suspect is the discount: **γ = 0.99 gives an effective planning horizon of ~100 steps
(~50 minutes) inside a 1,920-step (16-hour) episode.** An agent that cannot see beyond
50 minutes cannot learn a day-long schedule and plausibly falls back on "sell fairly
fast". Reordered ladder: **M1 γ**, M2 AC-relative reward, M3 regime-belief head, M4
belief-conditioned pace.

## Design

Identical to M0 in every respect except `--gamma 0.999` (effective horizon ~1,000 steps).
Same protocol v3, 12 training dates, 1.5M steps, `ent_coef` 0.01, constant LR, CPU,
checkpoints every 100k, offline selection with `select_checkpoint_v3.py`
(MeanFractionDecoder, best mean validation IS), scoring with `m0_v3_validation_eval.py`-
style comparison and `decompose_is_v3.py`. Validation dates only.

Known risk: a longer horizon raises the variance of the return the value function must
fit, so 1.5M steps may not be enough. If the learning curve is still rising at 1.5M, that
is reported, not fixed silently by training longer.

## Predictions (M1 vs M0, validation dates)

1. **Execution cost moves toward AC:** M1's execution component (M1 − AC) is less negative
   than M0's on at least 3 of 4 dates.
2. **Selling moves later:** M1's average sell time is later than M0's on at least 3 of 4
   dates (toward AC's 0.45).
3. **Better overall:** M1's selected-checkpoint mean validation IS beats M0's −0.465%.
4. **The crash-date timing gain shrinks:** M1's drift advantage over AC on 04-15 is smaller
   than M0's +0.63 pp (a direct consequence of 2).
5. **Training reward:** M1's reward reaches AC's (M1 − AC ≥ 0) on at least 1 of 4 dates.

Each will be marked met / not met, including misses.
