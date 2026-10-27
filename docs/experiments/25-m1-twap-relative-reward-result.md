# 25 — M1 (TWAP-relative reward): result — negative

Written 2026-10-28. Predictions were pre-registered in
[25-m1-twap-relative-reward-predictions.md](25-m1-twap-relative-reward-predictions.md).
Validation dates are used for selection only; held-out dates are untouched.

## Summary

Putting market moves into the reward made M1 **worse than M0-e2b on every summary
number**. It sells *faster*, not slower: a median of ~281 steps to finish, against 647.
Its calm-day loss widens to −0.81 pp. Training stalls: validation IS stops improving
around −0.43% from 400k steps, while M0-e2b improves steadily to −0.21%. The cause is
measured: market moves add noise that swamps the reward for pacing well.

## Result (mean IS %, 5% ADV, 20 starts per date)

Selected checkpoint: `final` (−0.379% validation), offline, by `select_checkpoint_v3.py`.

| | M1 | M0-e2b | TWAP |
|---|---:|---:|---:|
| validation mean | −0.379 | −0.207 | −0.299 |
| training mean | −0.484 | −0.404 | −0.205 |
| calm (6), vs TWAP | −0.806 | −0.669 | — |
| crash (4), vs TWAP | +0.442 | +0.489 | — |
| normal (6), vs TWAP | −0.100 | +0.007 | — |
| median steps to finish, validation mean | 281 | 647 | 1,918 |

Validation per date (M1 vs TWAP): 01-15 calm −0.407 · 09-15 calm −0.008 · 04-15 crash
+0.380 · 02-20 normal −0.284. No forced liquidation on any date.

Validation mean IS by checkpoint:

| steps | 100k | 300k | 500k | 700k | 900k | 1.1M | 1.3M | 1.5M | final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0-e2b | −0.714 | −0.495 | −0.439 | −0.398 | −0.359 | −0.313 | −0.256 | −0.207 | −0.245 |
| M1 | −0.752 | −0.514 | −0.471 | −0.445 | −0.431 | −0.432 | −0.431 | −0.425 | −0.379 |

## Why: signal-to-noise (measured, `rl/results/era2_baselines.csv`)

Per-episode difference between a fast constant pace and TWAP on the training dates
(240 episodes):

| | mean | sd | mean / sd |
|---|---:|---:|---:|
| pace 2× − TWAP, full IS (M1's reward) | −0.094 pp | 0.733 pp | 0.13 |
| pace 3× − TWAP, full IS (M1's reward) | −0.143 pp | 0.923 pp | 0.15 |
| pace 3× − TWAP, execution only | −0.0030 pp | 0.0012 pp | 2.5 |

Selling at TWAP's pace instead of 3× is worth +0.14 pp per episode on average. Under
M1's reward, though, each episode's outcome swings by ±0.9 pp depending on when the
price happened to move. Subtracting TWAP's gain (B) removes the day's overall direction,
but not the noise from timing: an agent that sells ahead of TWAP is exposed to every move
TWAP is not. The policy starts out selling fast (a random policy finishes in ~50 steps).
With a signal this weak, PPO does not learn its way back toward TWAP's pace in 1.5M
steps. Under M0's reward, the same choice is measured almost without noise, so M0 keeps
improving.

The "final" checkpoint scoring 0.05 pp better than the 1.5M checkpoint, with nothing
between them but noise, fits the same picture.

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | Training mean better than M0-e2b (> −0.404) | **Not met** (−0.484) |
| 2 | Smaller calm gap (> −0.669 pp) | **Not met** (−0.806) |
| 3 | Slower selling (validation median steps > 647) | **Not met** (281) |
| 4 | Smaller crash advantage (< +0.489 pp) | Met (+0.442) |
| 5 | Does not beat M0-e2b on validation (≤ −0.207) | Met (−0.379) |
| 6 | Below TWAP in at least one regime | Met (calm, normal) |
| 7 | No forced liquidation | Met |

The three predictions about what the reward change would *do* (1–3) all failed. The
predictions that held are the ones that would also hold for a model that learned less
(4–7). The mechanism the predictions assumed, more incentive to pace by regime, was right
in expectation but drowned in variance.

## What this means

- **M0-e2b remains the best era-2 model.** The mid baseline (`--reward-baseline mid`)
  stays the default.
- The hypothesis behind M1 still stands: with market moves left out, the reward cannot
  teach regime-aware pacing. Adding them back raw does not work. A regime signal has to
  reach the policy with less noise.
- Options for the next rung, in order of the cost of testing them:
  1. **Regime belief head (the next planned rung)** on M0-e2b's low-noise reward. An
     auxiliary supervised loss predicts the episode's regime / forward drift from the
     state, so the information is in the features even though the reward is drift-free.
     This is low-noise, because supervised labels carry no timing variance.
  2. **Lower-variance drift term:** add an *expected* drift (from the belief head's
     prediction) to the reward instead of the realised one. This depends on (1).
  3. **Many more samples:** the pacing signal's mean/sd is ~0.15 under M1's reward, against
     ~2.5 for the execution part alone. Equal precision would take roughly (2.5/0.15)² ≈ 300×
     more episodes. That is an order-of-magnitude figure, since the execution-only row leaves
     out our own impact. Not recommended.

## Artifacts

- Training: as M0-e2b plus `--reward-baseline twap --name m1_e2_twap`.
- `rl/results/m1_e2_twap_checkpoint_selection.csv`, `rl/results/m1_e2_twap_era2_eval.csv`
