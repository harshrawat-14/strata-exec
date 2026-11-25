# 36 — M0 recipe retrained on the corrected simulator. Design and predictions

Written 2026-11-25, before training. One change from M0-e2b (doc 23): the simulator's impact
decay (doc 33).

## Why

M0-e2b learned on impact costs that made fast selling about 40% too expensive (doc 33). Doc 35
found that it behaves as one fast constant pace, between 3× and 5× TWAP, and that this pace
wins crashes and loses calm days out of sample. The recipe has to be retrained on the corrected
simulator before any statement about what it learns. The question: **does the recipe now learn
a regime-dependent pace, or again one constant pace?**

## Method

- Training, identical to doc 23 except the simulator binary and the run name:
  `train.py --mode counterfactual --episode-steps 1920 --multi-date <TRAINING_DATES_V3>
  --eval-dates <VALIDATION_DATES_V3> --timesteps 1500000 --ent-coef 0.01 --lr-schedule constant
  --device cpu --order-share-adv 0.05 --name m0_e2c`. Seeds are fixed in `train.py` (42 + worker).
- Checkpoint selection, as in doc 23: the best mean IS on the V3 validation dates over the 16
  saved checkpoints (`select_checkpoint_v3.py --run m0_e2c --order-share-adv 0.05`).
- Scoring, as in doc 35 (`m0_rescore.py --model <selected> --name M0-e2c`): the 16 V3 dates and
  the 85 out-of-sample V4 validation days, against TWAP and constant paces on the same starts.

## Goal criterion (out of sample, unchanged from doc 35)

Beats TWAP in every regime: the per-regime mean of (M0-e2c − TWAP) > 0 for crash, normal and calm,
with the crash and calm 90% intervals above 0.

**Regime awareness (descriptive check).** Take the two constant paces that bracket the model's
overall oos IS. The model must beat both of them in both crash and calm. A single constant pace
cannot do that.

## Predictions

1. **No forced liquidation** at the selected checkpoint.
2. **It sells at least as fast as M0-e2b.** Fast selling is now cheaper in the reward, so its oos
   median steps to finish are ≤ 676.
3. **It is again one constant pace:** the regime-awareness check fails.
4. **The goal criterion fails:** the calm-day oos interval is below 0.
5. **Overall oos mean IS is below TWAP's** (TWAP −0.130).
