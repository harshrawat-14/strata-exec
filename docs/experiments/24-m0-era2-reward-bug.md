# 24 — M0-e2 first run: invalid, reward did not charge our own impact

Written 2026-10-27. Predictions were pre-registered in
[23-m0-era2-predictions.md](23-m0-era2-predictions.md). Validation dates only for selection.

## What happened

M0's recipe was trained on era-2 data at 5% of ADV (`--name m0_e2`, 1.5M steps). Every
checkpoint lost to TWAP on mean validation IS. The best was 900k at −0.746%, against TWAP's
−0.299%. Scored on all 16 training and validation dates (`rl/results/m0_e2_era2_eval.csv`),
IS was −0.45% to −1.17% on every date, whatever the regime.

| regime | M0-e2 IS | TWAP IS | M0-e2 − TWAP |
|---|---:|---:|---:|
| calm (6 dates) | −0.722 | +0.546 | −1.269 |
| crash (4) | −0.759 | −1.176 | +0.416 |
| normal (6) | −0.774 | −0.372 | −0.403 |

## Cause

Cost split on 5 starts of each validation date:

| | IS % | execution % | average sell time (episode fraction) | steps to finish |
|---|---:|---:|---:|---:|
| M0-e2 | −0.65 to −1.00 | −0.05 to −0.07 | 0.19–0.28 | 57–78 |
| TWAP | −0.25 to +0.72 | −0.001 to −0.004 | 0.50 | 1,918 |

M0-e2 sells the whole order in about 30 minutes. In the env, the reward of a fill was
`(fill price − book mid) × qty`, where the book mid is that of the **impact-shifted** book.
Since doc 21, our impact moves the mid, so the price drop we cause was no longer in the
reward. Waiting carries a small penalty and selling looked free, so dumping fast was the
learned optimum. Forced liquidation and dust settlement used the same impacted mid.

## Fix

Rewards are charged against the **undisturbed** mid, the real snapshot's mid without our
impact (`CounterfactualLob::current_undisturbed_mid`). Our impact is now a cost the agent
sees, and market drift still stays out of the reward (the original design). Evaluation
metrics were never affected: on regenerated golden fixtures, IS, fills and prices are
bit-identical, and only counterfactual rewards changed.

## Predictions

The doc 23 predictions are **not scored on this run**, because the run did not test the
recipe. They carry over unchanged to the retrain with the fixed reward (`m0_e2b`),
registered before that run.
