# 12 — Forced liquidation at data exhaustion: a cheap dump that flattered back-loaded baselines

Written 2026-10-18. Simulator bug found while building the golden regression guard for the
2026-10 refactor, measured first on **TRAINING_DATES_V2** (16 dates × eval seeds 0–19), then
fixed. Step A was then re-run once on the 9 held-out dates to check the published table
(baselines only, no model selection — see "Held-out re-run").

## Summary

- **The headline RL-vs-AC comparison is unaffected.** AC-Optimal (honest λ=1e-6) and the
  Step 4 RL checkpoint never reach the end of the data: their numbers are bit-identical
  before and after the fix.
- **Back-loaded baselines were flattered.** Adaptive-AC by 0.14 pp mean IS (up to 0.45 pp
  per date); the numerically-optimized sqrt-impact baseline by **0.53 pp**, because its
  optimizer learned to exploit the bug.
- **Training was never exposed.** Training uses random horizons (200–1,440 steps from a
  start in the first third of the day), which cannot reach the end of the data.

## The bug

In counterfactual mode with `--fixed-steps`, the horizon defaults to `n`, the number of
order-book snapshots in the day (≈2,880), and the start is `seed % (n/3)`. An episode with
start offset `k` therefore runs out of data at step `n − k`, before its clock ends — and
even `k = 0` ends exactly at the end of the data.

Once the replay passes its last snapshot, the impact-adjusted book is empty. The forced
end-of-episode dump walked that empty book, filled **nothing**, and priced 100% of the
remainder by the punitive fallback (`mid − 5 × spread_ref`). On BTC that fallback is cheap:
a policy that **never traded all day** and dumped everything at the end scored **−0.26% IS**
on 2024-01-15 — better than AC's −0.42% training-date mean. That violates the simulator's
own invariant that never-trading must not beat competent execution.

## Fix

At data exhaustion the dump walks the **last real book** (this step's adjusted book, after
our own fill) instead of the empty one. Everything else is unchanged.

Verified against the pre-fix binary on all 42 golden episodes: exactly 6 differ, all
fixed-horizon counterfactual "never trade" episodes that ended on an empty book; every step
before the final dump is byte-identical. The never-trade episode above now scores −2.38%.
Regression test: `forced_liquidation_at_data_exhaustion_walks_the_last_real_book`.

## Measured impact (16 training dates × seeds 0–19)

| Policy | ended on empty book | mean remainder dumped | mean IS before | after | change |
|---|---:|---:|---:|---:|---:|
| AC-Optimal | 0% | 0% | −0.4240 | −0.4240 | 0 |
| RL (Step 4) | 0% | 0% | −0.8943 | −0.8943 | 0 |
| Heuristic | 2% | 0.002% | −0.4467 | −0.4466 | ~0 |
| TWAP | 95% | 0.42% | −0.2651 | −0.2638 | +0.001 |
| Adaptive-AC | 67% | 7.4% | −0.7606 | −0.9006 | **−0.140** |

**Sqrt-impact baseline** (refit per date under each binary, Step A settings: 3 fit seeds,
25 Nelder-Mead iterations):

| | before fix | after fix |
|---|---:|---:|
| mean IS | +0.31% | −0.22% (**−0.53 pp**) |
| edge over AC | +0.74 pp | +0.21 pp |
| remainder left for the end dump | 7.8% | 4.4% |
| dates with back-loaded fit (a ≥ 2.8) | 7 / 16 | 2 / 16 |

Before the fix the optimizer chose heavily back-loaded schedules (a = 6.8–27) on 7 dates,
leaving 6–58% of the order for the cheap dump. After it, those shapes disappear except on
2024-07-06 and 2024-11-06 — strongly rising days, where waiting genuinely pays in a
single-day replay (the separate, still-open hindsight trend-following issue).

## What this changes

- **docs/experiments/08 (Step 3) stands.** The held-out re-run below shows the sqrt-impact
  baseline was *not* flattered on held-out dates: it still beats AC on 9/9, and RL still
  beats it on the 4 crash dates. The back-loading exploit appeared only on training dates
  with strong rallies (2024-11-06, 2024-07-06, …), which the held-out set does not contain.
- **Adaptive-AC numbers in docs 05–11** are optimistic by up to ~0.5 pp, on held-out dates
  as well as training dates.

## Held-out re-run (2026-10-18)

`phase3_stepA_full_holdout_eval.py` re-run exactly as published (Step 4 checkpoint,
n_episodes=20, sqrt fit 3 seeds × 25 iterations) on the fixed simulator →
`rl/results/issue1_full_holdout_rerun.csv`, compared with `phase3_stepA_full_holdout.csv`.

| Strategy | largest change on any held-out date |
|---|---:|
| AC-Optimal, RL, Heuristic | **0** (bit-identical on all 9 dates) |
| TWAP | 0.005 pp |
| Sqrt-impact | 0 on 7 dates; **+0.05 / +0.08 pp better** on 08-03 / 12-10 (refit found a better shape) |
| Adaptive-AC | up to **−0.46 pp** (e.g. 12-10 −0.84 → −1.30, 07-11 −0.63 → −0.99) |

Sqrt-impact beats AC on 9/9 held-out dates before and after; RL beats sqrt-impact on 4/9
(the crash dates) before and after. The published headline (RL vs AC: 4/4 crash, 0/5
calm/normal) is unchanged to the last decimal.
- **Anything new built on "AC on the same path"** (the planned M1 reward) needs this fix
  in place; it is.

## Artifacts

- Scripts: `rl/experiments/issue1_measure_forced_liq.py`, `rl/experiments/issue1_measure_sqrt.py`
- Results: `rl/results/issue1_episode_endings_{prefix,fixed}.csv`,
  `rl/results/issue1_sqrt_impact_{prefix,fixed}.csv`, `rl/results/issue1_full_holdout_rerun.csv`
