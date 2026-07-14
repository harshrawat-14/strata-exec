# Phase 2R — Wait-Collapse Is a Training Failure, Not an Objective Bias (Corrected)

> **CORRECTION NOTICE (supersedes the prior version of this document).** An earlier version of
> this file asserted that a naive execution-RL *objective* silently rewards directional
> speculation, and that a drift-neutral VWAP benchmark was the fix, culminating in a
> "state-observability limit" finding. **Both of those conclusions were wrong.** They were
> artifacts of analyzing **implementation shortfall vs. a static arrival price (IS-vs-arrival)** —
> a *reporting* metric — rather than the quantity the agent actually receives as reward. Once the
> real training reward was read from the code and measured directly, the picture changed
> completely. The corrected finding, and the methodological lesson behind the error, are below.
> The single result that survives unchanged is the forced-liquidation loophole fix (`5630e5d`).

---

## What the training reward actually is (verified by reading the code)

The per-step reward the agent receives (`src/bin/rl_env.rs`, `step()` / `step_counterfactual()`):

```
reward (on a fill) = (fill_price - book_mid) * fill_qty / (arrival_price * total)
reward (on a wait) = -(var_penalty + inv_pressure) * 0.0001        # tiny
terminal forced-liquidation adj = -( (book_mid_at_end - liquidated_at) * qty / (arrival*total)
                                     + gamma * (qty/total) )
```

`fill_price` and `book_mid` are **contemporaneous** (same step); the terminal term compares the
dump price to the **end** mid. **There is no drift term anywhere in the reward.** It is pure
execution-quality (slippage vs. the prevailing mid), and it is already drift-neutral. Selling 1
unit at mid=108 on a rally with a 10bp spread yields the same reward as selling at mid=100 with a
10bp spread — the agent gets **no** credit for the price *level*, only for beating the current mid.

## What the reward prefers, measured directly

Cumulative RL reward (the actual training signal, summed over the episode), blanket-WAIT vs. a
trivial "sell 10% of remaining per step" execution, 8 seeds/date:

| Date | WAIT cum. reward | EXEC (10%/step) cum. reward | reward prefers |
|---|---:|---:|---|
| 2024-11-06 (rally) | −0.0089 | −0.0048 | **EXEC** |
| 2024-01-15 (rally) | −0.0092 | −0.0046 | **EXEC** |
| 2024-03-05 (crash) | −0.0798 | −0.0046 | **EXEC** (17× better) |
| 2024-08-05 (crash) | −0.0106 | −0.0046 | **EXEC** |
| 2024-06-10 | −0.0094 | −0.0048 | **EXEC** |

**Execution beats waiting on the actual reward on every date, rally days included.** The reward
does not reward directional speculation, and it does not prefer waiting anywhere.

## The trained agent is suboptimal on its own reward

Trained `ppo_lstm_v6r_time_features_best` cumulative reward vs. the trivial 10%/step policy, same
metric, 8 seeds/date:

| Date | trained agent | trivial EXEC (10%/step) | |
|---|---:|---:|---|
| 2024-11-06 | −0.0089 | −0.0048 | agent worse |
| 2024-01-15 | −0.0092 | −0.0046 | agent worse |
| 2024-03-05 | −0.0798 | −0.0046 | agent worse (17×) |
| 2024-08-05 | −0.0106 | −0.0046 | agent worse |
| 2024-06-10 | −0.0046 | −0.0048 | agent ~tie |
| 2024-09-15 | −0.0053 | −0.0052 | agent worse |

**The trained agent scores worse than a five-line constant-fraction policy on the very reward it
was trained to maximize, on 5 of 6 dates.** It is leaving reward on the table. This is an
unambiguous **training / optimization failure** — the agent failed to learn the execution policy
its own reward prefers. It is *not* a policy correctly optimizing a mis-specified objective.

## What this overturns

- **"The objective rewards directional speculation" — WRONG.** That was measured on IS-vs-arrival,
  which the agent never receives as reward. The training reward is drift-neutral (above).
- **The VWAP-benchmark fix — NOT APPLICABLE.** It would change the *reporting* benchmark, not the
  training reward. The training reward is already drift-neutral and already prefers execution;
  there is no objective-side drift bias for VWAP to remove. (The VWAP gate scripts and the
  `interval_volume`/`book_mid` telemetry remain in the tree as sound instrumentation, but the
  conclusion they were built to support does not hold.)
- **The "state-observability limit" finding (03-05 vs 08-05 demand opposite actions) — WRONG.** It
  was an artifact of benchmark-relative IS. Under the *actual reward*, execution beats waiting on
  **both** 03-05 and 08-05 — they do not demand opposite actions. There is no evidence here of a
  regime the state cannot distinguish; there is just an agent that failed to learn to execute.

## What survives

- **The forced-liquidation loophole fix (`5630e5d`) is correct and necessary.** Pre-fix, terminal
  liquidation was a flat `gamma·(qty/total) ≈ 0.005` penalty, which made *waiting genuinely
  reward-optimal* (cheaper than the cumulative slippage of executing). Post-fix, terminal
  liquidation is priced by real book-walking slippage, and execution is reward-optimal (measured
  above). That fix changed the reward's true optimum from wait to execute. It stands, with its
  regression test (`forced_liquidation_is_not_reward_superior_to_trading`).
- **The wait-collapse persisting *after* that fix is the real open problem** — and it is a training
  failure, diagnosed next.

## Leading hypothesis for the training failure (NOT yet verified — do not treat as fact)

Episodes are ~500 steps. Trading incurs **immediate, per-fill** small negative rewards; the penalty
for waiting is a **single, delayed** terminal forced-liquidation cost ~500 steps later. This is the
classic long-horizon **credit-assignment** setup: the agent learns to avoid the visible immediate
cost of trading and never properly propagates the large delayed penalty back to the decision to
wait, collapsing onto action 0 (the trivial do-nothing default). Two independent 3M-step runs
collapsing identically is consistent with a *systematic* optimization failure — but it is also
consistent with other systematic causes. **This is a hypothesis to test (shorter horizons, reward
shaping to surface the terminal cost earlier, or exploration/entropy changes), not an established
conclusion.** Verify with a cheap smoke test before adopting it.

---

## Meta-lesson (the most transferable result of this phase)

**Twice in this project we were misled by evaluating on a quantity that is not the training
objective:**

1. The **forced-liquidation eval bug** (`price_factor` pinning + flat penalty) hid the loophole's
   true cost, making a reward-exploiting policy look costless in the reported IS.
2. **IS-vs-arrival** manufactured a phantom "objective rewards speculation" finding — and an entire
   downstream VWAP fix and "observability limit" narrative — none of which described what the agent
   actually optimizes.

This is a systemic hazard in RL research, not two flukes: **the reporting metric and the training
reward are different objects, and reasoning about "what the agent is optimizing" from the reporting
metric is invalid.** The rule, stated for the record and for anyone building on this work:

> **Before theorizing about what an RL agent is optimizing, read the reward code and measure the
> cumulative reward the agent actually receives. Do not reason about the agent's incentives from an
> evaluation/reporting metric it never sees.**

A reviewer will trust this project more for stating that plainly — learned the hard way, twice —
than for either wrong conclusion it replaces.

## Status

- Forced-liquidation loophole + `price_factor` fix (`5630e5d`): **correct, stands, tested.**
- Objective-bias / VWAP / observability findings: **retracted** (artifacts of IS-vs-arrival).
- Wait-collapse: **re-diagnosed as a training/optimization failure** (agent suboptimal on its own
  reward). Leading hypothesis = long-horizon credit assignment — **to verify, not assumed.**
- `interval_volume` / `book_mid` telemetry: retained as sound instrumentation.
- No VWAP implementation. No retrain until the training-failure diagnosis is verified and a fix is
  planned. Nothing pushed; demo untouched.
