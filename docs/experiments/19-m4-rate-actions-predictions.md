# 19 — M4: rate-based action space (design and pre-registered predictions)

Written 2026-10-22, before training.

## Why

Doc 18: AC-quality pacing in the raw "fraction of remaining" table needs ~837 action
switches per episode among the smallest buckets, while sampled large actions hit a
strongly convex cost (trades > 0.3% of the order cost 2.7× per unit). A constant rule
already beat M0.

## Design

`evaluation/rate.py`: the agent picks a pace k ∈ `RATES` = {0, 0.25, 0.5, 1, 1.5, 2, 3,
5, 10, 20} × the steady rate (order / horizon). `RateTracker` keeps the "should have
sold by now" inventory implied by the agent's own paces and corrects rounding next step,
so the fine bucket alternation is mechanical. The largest pace sells ~1% of the order per
step. No external schedule, so no catch-up bursts. Training wrapper and evaluation agent
share one tracker (tested). Everything else identical to M0 (`--action-space rate`;
protocol v3, 12 training dates, γ 0.99, λ 0.95, 1.5M steps, `ent_coef` 0.01, constant
LR, CPU; offline selection with `MeanValueDecoder` over `RATES`).

## No-learning bound (`rl/results/rate_constant_paces.csv`, training dates)

| constant pace | IS % | execution % |
|---|---:|---:|
| 1× | −0.340 | −0.4606 |
| 1.5× | −0.345 | −0.4714 |
| 2× | −0.441 | −0.5089 |
| AC (doc 18) | −0.331 | −0.4605 |
| M0 (doc 18) | −0.540 | −0.6372 |

A constant 1× pace already matches AC. M4's value must come from adaptive pacing.

## Predictions (validation dates)

1. **Beats M0:** selected mean validation IS > −0.4651%.
2. **AC-like execution:** |M4 − AC| execution ≤ 0.05 pp on ≥ 3 of 4 dates.
3. **At least matches AC:** mean validation IS ≥ −0.4425%.
4. **Crash date:** M4 beats AC on 04-15.
5. **Bounded trades:** largest single trade < 1.1% of the order on every date.
6. **Regime-dependent pace:** average chosen pace on 04-15 is higher than on both calm dates.

Each will be marked met / not met, including misses.
