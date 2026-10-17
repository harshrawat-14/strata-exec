# Phase 3 Step B — pre-registered predictions (RL fine-tune from BC warm start)

Written 2026-09-06, BEFORE launching the fine-tune. Same discipline as
`docs/experiments/10-full-holdout-evaluation-predictions.md`: this project has twice built a comparison that looked like a
clean finding and turned out to be a strawman, so the expected outcomes are locked in before the
numbers exist.

## What is being run

Stage 1 (DONE, `docs/experiments/10-full-holdout-evaluation-result.md` sibling work): behavior-cloned honest-λ AC-Optimal into
a fresh RecurrentPPO policy. Floor verified — the clone tracks AC-Optimal within **0.09pp on all 9
`HELD_OUT_DATES_V2`** (`rl/results/phase3_stepB_floor_check.csv`).

Stage 2 (THIS RUN): RL fine-tune starting from that BC checkpoint, via `train.py --resume-from`
(BC checkpoint has `num_timesteps=0` / `progress_remaining=1.0`, so the LR schedule starts fresh —
this is a warm start, not a mid-schedule resume). Settings: 1.5M timesteps and `ent_coef=0.01`
(Step 4's validated non-underfit configuration), but **`base_lr` reduced 3e-4 → 1e-4** specifically
to reduce the chance that the first PPO updates destroy the verified BC floor.

## The central tension this experiment tests

The BC expert (honest-λ AC) is **regime-blind by construction** — measured responsiveness
`t50(calm) − t50(crash)` = **+0.0000** exactly, vs RL Step 4's −0.0892
(`rl/results/phase3_step2_urgency_step4.csv`). AC is also far more urgent in absolute terms
(t50 ≈ 0.2611 vs RL's 0.5145–0.6037).

So stage 1 deliberately cloned a uniformly front-loaded, regime-blind policy. Two consequences,
both of which make this a genuine test rather than an expected improvement:

1. The warm start begins **below** the current RL model on crash dates. AC's crash-day cost is
   exactly what RL Step 4 beats by +0.26 to +0.92pp. Fine-tuning must climb back up, not merely
   improve on a neutral starting point.
2. RL's crash edge is believed to come from per-fill reactivity to the realized order book
   (`docs/experiments/08-sqrt-impact-baseline.md` — pacing and impact-model shape were both tested and ruled out).
   BC overwrote that with "follow the AC curve." The LR reduction that protects the calm-day floor is
   the same force that may prevent rediscovering the reactive behavior.

**The anchor that protects the floor is the anchor that may block the edge.** This is not a
guaranteed win and must not be written up as one.

## Predictions (fine-tuned model vs. AC-Optimal, on all 9 HELD_OUT_DATES_V2)

Baseline to beat — Step 4's current deltas (`docs/experiments/10-full-holdout-evaluation-result.md`):
crash +0.92 / +0.92 / +0.37 / +0.26; calm/normal −0.31 / −0.46 / −0.58 / −0.59 / −0.80.

- **Calm/normal dates (5):** predict the fine-tuned model **loses by less** than Step 4 did — the
  whole point of the floor. Specifically, predict mean calm/normal deficit improves from Step 4's
  ≈ −0.55pp to better than −0.30pp. If it does not improve here, the warm start bought nothing.
- **Crash dates (4):** predict the crash-day edge **shrinks** relative to Step 4's +0.92/+0.92/+0.37/
  +0.26, because the policy was initialized toward regime-blind pacing. Predict it stays **positive
  on at least 2 of 4** crash dates but with reduced margins.
- **Overall shape:** predict a flattening — less regime-conditional than Step 4, better on calm,
  worse on crash. That is the expected trade, not a failure.

## Outcomes and what each would mean

- **Floor holds AND crash edge survives (4/4 still positive, calm improved):** the best case, and
  the one I consider least likely. Would mean warm-starting is strictly better and should become the
  default recipe.
- **Floor holds, crash edge shrinks but stays positive on ≥2:** the predicted case. A real trade-off,
  honestly reportable either way depending on which regime is valued.
- **Floor holds, crash edge disappears (0/4 or 1/4 positive):** the warm start bought a boring model.
  **Report plainly as a negative result** — the interesting finding (regime-conditional edge) was
  traded away for uniform mediocrity. Do not bury this behind the improved calm numbers.
- **Floor does NOT hold (calm no better than Step 4):** the fine-tune destroyed the BC initialization;
  LR was still too high or PPO's first updates overwhelmed it. Diagnose before re-running — do not
  simply retry at a lower LR without understanding why.

## Guardrails

- The expert cloned in stage 1 was honest-λ (1e-6), never per-date hindsight-λ. Verified: the
  collection loop in `rl/experiments/phase3_stepB_bc_pretrain.py` restricts to `TRAINING_DATES_V2` and uses one
  fixed `HONEST_LAMBDA`.
- Evaluation must reuse `rl/experiments/phase3_stepA_full_holdout_eval.py` unchanged (same 9 dates, same
  n_episodes, same policy set) so the comparison against Step 4 is single-variable.
- Audit finding 5 (near-duplicate eval episodes) remains open and is NOT addressed here; per-date
  significance claims stay uncredible regardless of this run's outcome.
