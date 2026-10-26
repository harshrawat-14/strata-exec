# 23 — M0 retrained in era 2: predictions (pre-registered)

Written 2026-10-27, before training starts.

## What is being measured

**M0-e2** is M0's recipe (doc 13) retrained on era-2 data: the Bybit L2 book and the
price-moving impact model (doc 21). The recipe is unchanged:
- raw sell-fraction actions and the 16-dim state;
- protocol v3: 1,920-step episodes with random starts;
- 12 `TRAINING_DATES_V3`, 1.5M steps, `ent_coef` 0.01, constant LR, CPU.

The only change is order size, set with `--order-share-adv 0.05`. During training the
order is 10–100% of 5% of the date's ADV (the env's usual random size); evaluation uses
exactly 5%.

Selection is offline, by `select_checkpoint_v3.py --order-share-adv 0.05`: the checkpoint
with the best mean validation IS, decoded with MeanFractionDecoder.

The bar is TWAP, because honest AC is TWAP in era 2 (doc 22). Doc 22 gives the
reference numbers on the same 20 starts per date, at 5% ADV (mean IS %):

| | 01-15 calm | 02-20 normal | 04-15 crash | 09-15 calm | val mean | train mean |
|---|---:|---:|---:|---:|---:|---:|
| TWAP (pace 1×) | +0.109 | −0.060 | −0.774 | −0.471 | −0.299 | −0.205 |
| pace 1.5× | +0.027 | +0.001 | +0.015 | −0.406 | −0.091 | −0.252 |
| pace 2× | +0.033 | −0.046 | +0.133 | −0.402 | −0.071 | −0.299 |

The validation dates reward selling faster than TWAP (mostly 04-15). The training dates
do not. A model picked on validation IS can therefore look good just by selling fast,
the trap M4 fell into (doc 19).

## Predictions

1. **Beats TWAP on mean validation IS** (> −0.299%) at the selected checkpoint.
2. **Does not beat the best constant pace on validation** (≤ −0.071%, pace 2×). Any
   gain over TWAP is explained by pace, not by adapting to state.
3. **Beats TWAP on the crash date 04-15** (> −0.774%).
4. **Loses to TWAP on the calm date 01-15** (< +0.109%).
5. **Does not beat TWAP on the training-date mean** (≤ −0.205%) when the selected
   checkpoint is scored on the 12 training dates at 5% ADV with 20 starts.
6. **No forced liquidation** at the selected checkpoint on any validation date.

If 1 holds but 2 and 5 do too, M0-e2 is a fast seller, not a regime-aware one. That is
the expected starting point for the ladder: the AC-relative reward, then the regime
belief head.

Each prediction will be marked met / not met in the result document, including misses.
