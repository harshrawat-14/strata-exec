# 12 — Forced liquidation at data exhaustion: a cheap dump that flattered back-loaded baselines

Written 2026-10-18. Simulator bug found while building the golden regression guard for the
2026-10 refactor, measured on **TRAINING_DATES_V2 only** (16 dates × eval seeds 0–19),
then fixed. No held-out date was evaluated.

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

- **docs/experiments/08 (Step 3):** "RL beats the best static schedule on crashes" is
  *strengthened* — that baseline was flattered. "Sqrt-impact beats AC on all four held-out
  dates" (margins +0.05 to +0.38 pp) is **now doubtful**: the flattering measured here is
  larger than those margins. Not re-run, to avoid another look at the held-out dates.
- **Adaptive-AC numbers in docs 05–11** are optimistic by roughly 0.1–0.5 pp.
- **Anything new built on "AC on the same path"** (the planned M1 reward) needs this fix
  in place; it is.

## Artifacts

- Scripts: `rl/experiments/issue1_measure_forced_liq.py`, `rl/experiments/issue1_measure_sqrt.py`
- Results: `rl/results/issue1_episode_endings_{prefix,fixed}.csv`,
  `rl/results/issue1_sqrt_impact_{prefix,fixed}.csv`
