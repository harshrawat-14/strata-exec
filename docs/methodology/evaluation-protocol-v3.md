# Evaluation protocol v3

Adopted 2026-10-18. Applies to every model from M0 onward (the incremental ladder in the
v2 plan). The legacy protocol stays selectable, so every earlier published number still
reproduces byte-for-byte (golden regression tests).

## Why

BTC trades 24/7, so a "day" here is one UTC day of 30-second order-book snapshots
(≈2,880). The legacy protocol had three defects:

1. **Near-identical eval episodes.** Starts were `seed % (day/3)` and evals used seeds
   0–19, so all 20 episodes began within 9.5 minutes of midnight (audit finding 5).
2. **Episodes ran off the end of the data.** The horizon was the whole day, so any start
   after 00:00 exhausted the replay before the clock ended
   (docs/experiments/12-forced-liquidation-at-data-exhaustion.md).
3. **Training replayed one episode per date.** Seedless resets reused the constructor
   seed, so every training episode started at 00:21 UTC, lasted 853 steps and used a
   58.85% order — on every date, for every model since 2026-05-22.

## The protocol

| | v3 |
|---|---|
| Episode length | **1,920 steps (16 h)**, training and eval alike (`--episode-steps 1920`) |
| Start window | any start leaving 1,920 further snapshots: `[0, n − 1921]`, ≈8 h (933 on 2024-07-05, which has a data gap) |
| Eval starts | 20 per date, evenly spaced across the window (≈25 min apart): `eval_start_fracs(20)` |
| Training starts | fresh uniform draw every episode, distinct seed stream per worker; order size random 10–100% |
| Order size in eval | fixed (100%) |
| Model selection | `VALIDATION_DATES_V3` (04-15 crash, 02-20 normal, 01-15 calm, 09-15 calm), 5 evenly spaced starts per date, identical every round |
| Training dates | `TRAINING_DATES_V3` (12: 3 crash, 5 normal, 4 calm) |
| Test dates | `HELD_OUT_DATES_V3` = the 9 V2 held-out dates, evaluated **once**, at the final comparison |
| Baseline horizon | every schedule baseline adopts the horizon the environment reports (`set_horizon`) |
| Baseline execution | AC, Adaptive-AC and sqrt-impact follow their schedule **closed-loop** (`schedule_target(track=True)`) |

## Closed-loop baselines

Under the legacy horizon, AC-Optimal at the honest λ=1e-6 did not follow its own schedule:
its per-step target was smaller than the smallest action bucket, so it really traded at
"0.1% of remaining per step" and finished only because 2,880 steps was long enough. At
1,920 steps the same open-loop code leaves ~5% unsold and has to dump it. Tracking the
schedule (sell whatever returns inventory to the scheduled path) fixes this: 0% forced
liquidation, and better IS on 8 of 9 test episodes. Leaving the open-loop version in
place would handicap AC, which is exactly the kind of tilted comparison
docs/history/lessons.md L3 warns about.

## Still to do before results under v3 mean anything

- **Re-select the honest λ** on `TRAINING_DATES_V3` under v3. The λ=1e-6 choice was made
  under the legacy protocol.
- **Retrain M0** (the current recipe) under v3. Comparing a v3-trained M1 with the
  legacy-trained Step 4 checkpoint would confound the reward change with the sampler fix.

## Using it

```
python rl/train.py --mode counterfactual --episode-steps 1920 \
  --multi-date <TRAINING_DATES_V3> --eval-dates <VALIDATION_DATES_V3> ...
evaluate_all_strategies_on_date(..., episode_steps=1920)   # eval starts spread automatically
```
