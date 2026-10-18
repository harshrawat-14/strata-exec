# 13 — M0 under protocol v3: result

Written 2026-10-18. Predictions were pre-registered in
[13-m0-protocol-v3-predictions.md](13-m0-protocol-v3-predictions.md) before training
finished. **Validation dates only; the held-out test dates were not used.**

## Summary

Retraining the Step 4 recipe on varied episodes (protocol v3) made RL far better and
closed most of the calm-day gap: **M0 beats AC on the crash date by +0.45 pp, is within
0.07 pp on the normal date, and trails AC by 0.12–0.35 pp on the two calm dates.** The
legacy-trained Step 4 checkpoint, scored the same way, loses all four dates by 0.64–1.32 pp.

Getting there required one finding that changes how RL is scored: **SB3's default
`deterministic=True` (argmax) decoding misrepresents this policy.** It is replaced by
mean-fraction decoding for every RL model from M0 onward.

## The decoding finding

The 20 actions are ordered sell fractions. M0's policy puts ~30% probability on "wait"
(action 0) and ~70% spread across the sell actions. Argmax therefore plays "wait" on
60–84% of steps, the order piles up, and 10–45% of it is dumped at the end. Sampled
actions, or the action nearest the policy's *expected* fraction, sell steadily with zero
forced dumps:

| date (final checkpoint) | argmax: IS / dumped | sampled: IS / dumped | mean-fraction: IS / dumped |
|---|---|---|---|
| 02-20 normal | −1.64 / 24% | −1.01 / 0% | −1.31 / 0% |
| 09-15 calm | −2.91 / 45% | −1.10 / 0% | −1.42 / 0% |
| 06-10 normal (train) | −1.78 / 29% | −1.14 / 0% | −1.36 / 0% |
| 07-05 crash (train) | −0.25 / 10% | −1.63 / 0% | −1.37 / 0% |

In-training model selection (SB3 `EvalCallback`) also decodes by argmax, so it chose the
near-random **50k-step** checkpoint and never replaced it. With checkpoints re-scored by
mean-fraction decoding (`rl/results/m0_v3_checkpoint_selection.csv`), training is a
normal learning curve: validation IS −2.19% at 100k → **−0.47% at 1.2M** → −1.24% at
1.5M (late overfitting), with zero forced dumps at every checkpoint.

**Decision:** RL models are decoded with `evaluation.decoding.MeanFractionDecoder`
(deterministic, reproducible; it beat seeded sampling on every well-trained checkpoint,
e.g. −0.47 vs −0.73 at the selected points). Model selection under v3 is offline, by
`rl/experiments/select_checkpoint_v3.py`: best mean validation IS over saved checkpoints.
**M0 = checkpoint at 1,199,952 steps.**

## Result (validation dates, protocol v3, mean IS %)

| date | regime | AC (λ=1e-6) | Adaptive-AC (λ=1e-9) | TWAP | Heuristic | Sqrt-impact | **M0** | Step 4 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2024-01-15 | calm | −0.093 | −0.098 | −0.071 | −0.128 | +0.054 | **−0.443** | −1.410 |
| 2024-02-20 | normal | −0.288 | −0.276 | −0.299 | −0.317 | −0.065 | **−0.358** | −1.416 |
| 2024-04-15 | crash | −0.652 | −0.827 | −0.678 | −0.551 | +0.306 | **−0.204** | −1.321 |
| 2024-09-15 | calm | −0.737 | −0.779 | −0.710 | −0.730 | −0.716 | **−0.855** | −1.381 |

| | 01-15 | 02-20 | 04-15 | 09-15 |
|---|---:|---:|---:|---:|
| M0 − AC | −0.350 | −0.071 | **+0.449** | −0.118 |
| Step 4 − AC | −1.316 | −1.129 | −0.668 | −0.644 |
| M0 − Step 4 | +0.966 | +1.058 | +1.117 | +0.526 |

No strategy force-liquidated on any date.

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | M0 beats Step 4 on ≥3 of 4 validation dates | **Met** (4/4, by 0.53–1.12 pp) |
| 2 | M0's calm-date gap to AC is smaller than Step 4's | **Met** (−0.35 vs −1.32; −0.12 vs −0.64) |
| 3 | On 04-15 (crash) M0 is within 0.3 pp of AC or better | **Met** (+0.45, better) |
| 4 | M0 does not beat AC on both calm dates | **Met** (loses both) |
| 5 | No strategy force-liquidates under v3 | **Met** |

**Deviations from the pre-registration, stated plainly:** the predictions assumed M0 would
be the in-training best model and that RL would be decoded by argmax. Both changed after
training, for the reasons above. Step 4 is also decoded with mean-fraction here (and was
trained on these dates' first 7 hours, which favours it).

## Caveats

- **M0's validation numbers are optimistic.** M0 was *selected* by validation IS from 16
  checkpoints, so these numbers carry selection bias. They are for comparing variants;
  the honest estimate comes from the single held-out evaluation at the end of the ladder.
- **n is small:** 4 dates, one per regime (two calm). 20 starts per date are spread
  across the day, so they are no longer near-duplicates, but they are still one day each.
- **Sqrt-impact is fit on the same date** it is scored on, which flatters it; it is an
  upper bound for fixed schedules, not a deployable baseline.
- **The argmax finding may bear on legacy results.** Every published RL number used
  argmax decoding. Whether legacy checkpoints had the same wait-heavy argmax behaviour
  (and whether the "RL paces slower than AC" urgency findings are partly a decoding
  artifact) is not established here; it is logged as an open question.

## What this means for the ladder

M0 is the baseline for M1–M3: same protocol, same decoding, same offline selection. Its
remaining gap is on calm days (−0.12 to −0.35 pp); the crash edge is intact. M1
(AC-relative reward) targets exactly that gap.

## Artifacts

- Training: `train.py --episode-steps 1920 --multi-date <TRAINING_DATES_V3> --eval-dates
  <VALIDATION_DATES_V3> --timesteps 1500000 --ent-coef 0.01 --lr-schedule constant
  --device cpu --name m0_v3`; checkpoints `rl/models/checkpoints/m0_v3/`.
- `rl/experiments/select_checkpoint_v3.py` → `rl/results/m0_v3_checkpoint_selection.csv`
- `rl/experiments/m0_v3_validation_eval.py` → `rl/results/m0_v3_validation.csv`
- `rl/evaluation/decoding.py`
