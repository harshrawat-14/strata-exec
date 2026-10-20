# 16 — M3: longer credit horizon (GAE λ 0.95 → 0.99, γ 0.99 → 0.999) — pre-registered

Written 2026-10-21, before training.

## Why

M0, M1 and M2 all fit one blind spot (docs 14–15): PPO assigns credit over roughly
1/(1 − γλ) ≈ **17 steps** at γ 0.99, λ 0.95, but the cost of a pacing decision in a
1,920-step episode arrives hundreds of steps later (at a catch-up or the final dump).
M1 raised γ alone, which left that reach at ~20 steps. M3 changes the reach itself:
γ 0.999, λ 0.99 → 1/(1 − 0.98901) ≈ **91 steps**, about 5× further. The two settings jointly
define one quantity (the credit horizon), so this is a single conceptual change.

## Design

Identical to M0 (raw action space, protocol v3, 12 training dates, `ent_coef` 0.01,
constant LR, CPU) except `--gamma 0.999 --gae-lambda 0.99` and **3M steps** instead of
1.5M, because M1 showed a longer horizon learns more slowly. Checkpoints every 100k;
offline selection with `select_checkpoint_v3.py` (mean-fraction decoding, best mean
validation IS). Selecting from 30 checkpoints instead of 16 adds some selection
optimism; validation numbers are for ranking only.

Comparison points (validation mean IS): AC −0.4425%, M0 −0.4651%, M2 −0.686%, M1 −1.27%.

## Predictions (validation dates)

1. **Learns past M1:** the validation learning curve rises above −1.0% before 1.5M steps
   (M1 was still at −1.30% at 1.5M).
2. **Beats M0:** the selected checkpoint's mean validation IS > −0.4651%.
3. **Execution toward AC:** M3's execution component (M3 − AC) is less negative than
   M0's on at least 3 of 4 dates.
4. **No burst trading:** the largest single trade stays below 1% of the order on every
   validation date (M2 reached 4.9%).
5. **Beats AC on average:** mean validation IS > −0.4425%. (Stated as the stretch goal;
   expected to be the hardest to meet.)

Each will be marked met / not met, including misses. If the curve is still rising at 3M,
that is reported as such.
