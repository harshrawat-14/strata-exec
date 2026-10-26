# 23 — M0 retrained in era 2: result

Written 2026-10-27. Predictions were pre-registered in
[23-m0-era2-predictions.md](23-m0-era2-predictions.md). This run is `m0_e2b`, which
uses the reward fixed in doc 24. The first run (`m0_e2`) is invalid. Validation dates are
used for selection only; held-out dates are untouched.

## Summary

With the reward fixed, M0's recipe learns normally. Validation IS improves steadily from
−0.71% (100k steps) to **−0.207% (1.5M, selected)**, with no forced dumps. It was still
improving when training stopped: episode length rose from ~47 to ~840 steps.

- It **beats TWAP on validation** (+0.09 pp) but **loses on the training dates**
  (−0.20 pp).
- **By regime** (all 16 dates): crash **+0.49 pp** vs TWAP, normal +0.01, calm **−0.67**.
- Era 1 had the same shape: RL wins crashes and loses calm days.
- It is a **fast seller, not a regime-aware one**. It finishes in 109–540 of 1,920 steps
  (1–4.5 h instead of 16 h), and its pace does not track regime.

## Result (mean IS %, 20 starts per date, 5% of ADV)

| split | M0-e2b | TWAP | pace 1.5× | pace 2× | M0-e2b − TWAP |
|---|---:|---:|---:|---:|---:|
| train (12) | −0.404 | −0.205 | −0.252 | −0.299 | −0.199 |
| validation (4) | −0.207 | −0.299 | −0.091 | −0.071 | **+0.092** |

| validation date | regime | M0-e2b | TWAP | best pace | − TWAP |
|---|---|---:|---:|---:|---:|
| 2024-01-15 | calm | −0.137 | +0.109 | +0.109 (1×) | −0.246 |
| 2024-02-20 | normal | −0.072 | −0.060 | +0.001 (1.5×) | −0.012 |
| 2024-04-15 | crash | −0.251 | −0.774 | +0.133 (2×) | +0.523 |
| 2024-09-15 | calm | −0.367 | −0.471 | −0.395 (3×) | +0.104 |

| regime (all dates) | M0-e2b | TWAP | − TWAP |
|---|---:|---:|---:|
| calm (6) | −0.123 | +0.546 | −0.669 |
| crash (4) | −0.687 | −1.176 | **+0.489** |
| normal (6) | −0.365 | −0.372 | +0.007 |

On two training crash dates (07-04, 07-07), M0-e2b beats every constant pace. On 09-15 it
beats the best constant pace by 0.03 pp.

## How it sells (5 starts per date)

| date | regime | steps to finish (of 1,920) | largest slice (share of order) | execution % |
|---|---|---:|---:|---:|
| 2024-11-06 | calm | 109 | 4.5% | −0.054 |
| 2024-07-04 | crash | 141 | 6.5% | −0.046 |
| 2024-04-15 | crash | 312 | 3.9% | −0.039 |
| 2024-07-07 | crash | 346 | 3.2% | −0.019 |
| 2024-07-06 | calm | 382 | 3.4% | −0.024 |
| 2024-01-15 | calm | 470 | 1.3% | −0.014 |
| 2024-09-15 | calm | 506 | 1.0% | −0.006 |
| 2024-02-20 | normal | 540 | 1.4% | −0.008 |

TWAP takes 1,918 steps, with 0.15% slices and about −0.003% execution. M0-e2b front-loads
on every date. This wins when prices fall, i.e. crashes, and loses when they rise, i.e.
calm days.

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | Beats TWAP on mean validation IS (> −0.299) | **Met** (−0.207) |
| 2 | Does not beat the best constant pace on validation (≤ −0.071) | **Met** (−0.207) |
| 3 | Beats TWAP on 04-15 (> −0.774) | **Met** (−0.251) |
| 4 | Loses to TWAP on 01-15 (< +0.109) | **Met** (−0.137) |
| 5 | Does not beat TWAP on the training mean (≤ −0.205) | **Met** (−0.404) |
| 6 | No forced liquidation at the selected checkpoint | **Met** (0 on every date) |

**Deviation:** the predictions were registered for `m0_e2`. They were scored on `m0_e2b`,
the same recipe with the reward bug fixed (doc 24), unchanged and before `m0_e2b` ran.

## Caveats

- Selection is by validation IS over 16 checkpoints, so the validation numbers are
  optimistic. The training-date score is not selection-biased, but those dates were
  trained on.
- **Not converged.** The best checkpoint is the last saved one, and the curve was still
  improving. Some of the calm-day gap may be undertraining rather than a structural
  limit; a longer run would test that. Any change to the recipe (steps) counts as a
  ladder rung and needs its own predictions.
- Validation favours fast selling (doc 22), so the validation win is largely the
  fast-selling bias that doc 19 found for M4.

## What this means for the ladder

The era-2 starting point repeats the era-1 headline: RL wins crashes and loses calm days,
because its pace does not follow the regime. Next rung (M1, one change): a **TWAP-relative
reward**. Each step is scored against what TWAP would have earned on the same step, so
the agent is paid only for beating the even pace. Selling fast stops being a free bet on
falling prices.

## Artifacts

- Training: `train.py --mode counterfactual --episode-steps 1920 --multi-date
  <TRAINING_DATES_V3> --eval-dates <VALIDATION_DATES_V3> --timesteps 1500000 --ent-coef
  0.01 --lr-schedule constant --device cpu --order-share-adv 0.05 --name m0_e2b`
- `rl/results/m0_e2b_checkpoint_selection.csv` (`select_checkpoint_v3.py --order-share-adv 0.05`)
- `rl/results/m0_e2b_era2_eval.csv` (`rl/experiments/era2_model_eval.py`)
