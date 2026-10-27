# 25 — M1: TWAP-relative reward (design and pre-registered predictions)

Written 2026-10-28, before training.

## Why

M0-e2b (doc 23) beats TWAP on crash days (+0.49 pp) and loses on calm days (−0.67 pp). Its
pace is the same in every regime: it finishes in 3.8–7.4 h of 16. The cause is
structural. Its reward charges each fill against the mid without our impact, so **market
moves are left out of the reward entirely**. Selling early on a falling day earns nothing
in training, and selling late on a rising day costs nothing. A regime-aware pace cannot
be learned from a reward that does not see what the regime does to the price. M0's crash
wins are a side effect of selling fast.

## The one change

`--reward-baseline twap` (in the Rust env, counterfactual mode only):

- Each fill is charged against the **arrival price**, so the episode's fill rewards sum
  to −IS exactly, including market moves.
- **TWAP's market-move gain** on the same price path is subtracted once, on the first
  step: B = mean over the episode's 1,920 snapshots of (real mid / arrival mid − 1).
  This is the user's "same path" comparison (v2 plan): the return is about IS_RL − IS_TWAP,
  apart from TWAP's own small execution cost.
- B does not depend on the agent's actions, so it does not change the optimal policy; it
  removes the day's direction from the return and lowers variance. The real change is
  that market moves are now in the reward.
- The episode identity, return = −IS − B, is tested
  (`twap_baseline::episode_reward_is_minus_is_minus_twap_drift_gain`, mutation-checked).

Everything else is M0-e2b's recipe: raw actions, 16-dim state, protocol v3,
12 training dates, 5% ADV (random 10–100% sizes in training), 1.5M steps, `ent_coef` 0.01,
γ 0.99, λ 0.95, constant LR, CPU. Selection is offline on validation IS with
MeanFractionDecoder (`select_checkpoint_v3.py --order-share-adv 0.05`).

**Known risk:** with market moves in the reward, the agent can be paid for memorising
when each training date rose or fell, using time-of-day features, rather than for
reading the regime. Validation dates are unseen days, which tests this. Doc 22 also showed
that no constant pace helps reliably on the training crash days.

## Reference numbers (mean IS %, 5% ADV, 20 starts per date)

| | M0-e2b | TWAP |
|---|---:|---:|
| validation mean | −0.207 | −0.299 |
| training mean | −0.404 | −0.205 |
| calm (6 dates), vs TWAP | −0.669 | — |
| crash (4 dates), vs TWAP | +0.489 | — |
| normal (6 dates), vs TWAP | +0.007 | — |
| median steps to finish, validation dates | 454–850 (mean 647) | 1,918 |

## Predictions

1. **Better on the training dates than M0-e2b**: training mean IS > −0.404%.
2. **Smaller calm-day gap**: M1 − TWAP on calm dates > −0.669 pp.
3. **Slower selling**: median steps to finish, averaged over the 4 validation dates, above
   M0-e2b's 647.
4. **Smaller crash advantage than M0-e2b**: M1 − TWAP on crash dates < +0.489 pp. With
   one market-move-blind pace, slowing down for calm days gives up some crash gain.
5. **Does not beat M0-e2b's validation mean** (≤ −0.207%). The validation dates favour fast
   selling (doc 22).
6. **Not yet the goal**: in at least one regime, M1's mean is below TWAP's.
7. **No forced liquidation** at the selected checkpoint.

If 1–3 hold, the reward fix is doing its job, and the next rung is the regime belief head.
If M1 instead becomes *more* regime-dependent (a larger crash gain *and* a smaller calm
loss), predictions 4 and 6 fail in the good direction, and that will be reported as such.

Each prediction will be marked met / not met in the result document, including misses.
