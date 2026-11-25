# 35 — M0-e2b on the recalibrated simulator, and out of sample. Design and predictions

Written 2026-11-25, before the re-score runs. No RL is trained.

## Why

M0-e2b (doc 23) is the only era-2 model with a positive result: +0.49 pp over TWAP on crash
days. It was scored on the old impact model. Doc 33 refitted the simulator to the measured
price response, which makes fast selling about 40% cheaper in impact. M0-e2b front-loads
(median about 500 of 1,920 steps), so its scores move more than TWAP's. Two questions:

1. **Does the doc 23 result stand on the corrected simulator?** Same 16 V3 dates, same starts.
2. **Does it hold out of sample?** Doc 23 had 4 validation dates, which were used to select the
   checkpoint. The V4 pool adds 85 validation days (Oct–Dec 2024) that M0-e2b never trained on
   or was selected on. This is the first test of the project's goal on enough days: **beat
   TWAP (= honest AC, doc 33) in every regime.**

## Method (`rl/experiments/m0_rescore.py`)

- Model: `m0_e2b/ckpt_1499940_steps`, doc 23's selected checkpoint, decoded as in doc 23.
- Baselines on the same 20 starts per date, 5% of ADV: TWAP (pace 1.0) and constant paces
  1.5, 2, 3 and 5× TWAP.
- Simulator binary rebuilt from the current tree. Checked: TWAP on 2024-04-15 reproduces doc
  33's 20 episodes exactly (max difference 4e-16).
- Date sets:
  - **v3:** the 16 dates of doc 23.
  - **oos:** the 87 V4 validation days minus 2024-10-15 and 2024-11-06 (V3 dates): 85 days.
- Regimes are doc 30's drift labels (`v4_day_stats.csv`) for both sets. These differ from doc
  23's labels on a few V3 dates, so the v3 regime means are not directly comparable to doc 23's table.
  The per-date numbers are.
- Uncertainty: 2,000-resample day bootstrap of the per-day (M0-e2b − TWAP) mean, 90% intervals.

**Already seen:** a smoke test on one date before this doc was written. On 2024-04-15
(crash), M0-e2b scored −0.106% against −0.251% in doc 23. TWAP on that date moved from −0.774%
(doc 23) to −0.703% (doc 33).

## Goal criterion (out of sample)

M0-e2b beats TWAP in **every** regime on the oos days. The per-regime mean of
(M0-e2b − TWAP) must be > 0 for crash, normal and calm, with the crash and calm 90% intervals
above 0.

## Decision rule

- **Passes:** M0-e2b becomes the reference model, and a retrain on the corrected simulator
  checks whether it still learns this.
- **Fails:** retrain the M0 recipe on the corrected simulator (doc 36, one change: the
  simulator). It learned under impact costs that were wrong, so its current pace is not what
  the recipe would learn now. Then move to equities, as agreed with the user.

## Predictions

1. **v3: M0-e2b gains more than TWAP from the correction.** The mean over 16 dates improves by
   0.08–0.20 pp, against TWAP's 0.06.
2. **v3 crash dates: the gap over TWAP stays positive and grows**, to +0.50 to +0.70 pp.
3. **v3 calm dates: M0-e2b still loses to TWAP.**
4. **oos: M0-e2b beats TWAP on crash days** (interval above 0) **and loses on calm days**
   (interval below 0). **The goal criterion fails.**
5. **oos: M0-e2b behaves like a fast constant pace.** In every regime its mean IS lies between
   pace 3 and pace 5 (± 0.05 pp), and its median steps to finish are 450–650.
6. **No forced liquidation** on any date.
