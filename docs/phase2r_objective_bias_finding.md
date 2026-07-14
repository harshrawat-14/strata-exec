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

## The training failure is LATE-TRAINING POLICY COLLAPSE (measured, decisive)

The credit-assignment-via-long-horizon hypothesis was tested and is **not** the cause. A controlled
two-arm smoke experiment (`rl/phase2r_horizon_experiment.py`, 80k steps each, identical but for
horizon: SHORT=60 vs LONG=800 fixed steps, gamma=0.99) had **both** arms learn to execute
(wait-fraction 0.000) — the LONG arm did not collapse. That experiment was, however, too short to
reach the collapse regime (see below), so it is *inconclusive on horizon* rather than a clean
falsification. What is decisive is the **checkpoint trajectory of the actual 3M-step run** (every
saved checkpoint, wait-fraction on a crash date and a rally date):

| checkpoint | 2024-03-05 wait-frac | 2024-11-06 wait-frac |
|---|---:|---:|
| 100k – 900k | **0.000** | **0.000** |
| 1.1M | 0.741 | 0.802 |
| 1.3M – 1.7M | 0.56 – 0.80 | 0.70 – 1.00 |
| 2.0M – 3.0M | **1.000** | **1.000** |

**The agent learns a good execution policy by 100k, holds it cleanly through ~900k, then collapses
to blanket-wait between 1.1M and 2.0M and locks there.** The collapse onset (~1.1M) is well before
the mid-run OOM crash/resume (~1.7M), so it is not a resume artifact.

The 900k checkpoint is not merely "not collapsed" — it is **reward-optimal**, matching or beating a
trivial 10%/step baseline on its own reward on every date tested:

| date | ckpt 900k | ckpt 3M (final) | trivial 10%/step |
|---|---:|---:|---:|
| 2024-03-05 | **−0.00445** | −0.07981 | −0.00459 |
| 2024-11-06 | −0.00498 | −0.01007 | −0.00481 |
| 2024-08-05 | **−0.00459** | −0.01061 | −0.00462 |
| 2024-01-15 | **−0.00435** | −0.00954 | −0.00458 |

**So the run contained a genuinely good execution policy and then destroyed it.** This reframes the
problem entirely: it is not "the agent can't learn execution" (it did, by 100k) — it is **late-training
policy collapse / catastrophic forgetting of a reward-optimal policy**, a known PPO failure mode, plus
a **model-selection failure** (the good 900k-era policy was not kept; both `best_model` and `_final`
are the collapsed policy). The good policy is sitting recoverable in the checkpoint directory
(`ckpt_900000_steps`).

## Collapse mechanism: a gradual entropy-decay slide, NOT a sharp destabilizing update (measured)

Parsed the logged training metrics across the collapse window (`/tmp/train_phase2r_full.log` +
`_resume.log`). The two candidate signatures were: a **sharp event** (a KL/clip spike at one update
batch → fix = KL early-stopping) vs. a **gradual slide** (smooth entropy decay + monotonic drift to
action 0 → fix = entropy floor + reward shaping). The data is unambiguous: **gradual slide.**

- **Entropy decays smoothly and monotonically** (`entropy_loss`, i.e. −entropy): −2.13 (900k) →
  −2.19 (1.1M) → −1.49 (1.2M) → −1.72 (1.5M) → −1.03 (1.7M) → −0.67 (2.0M) → −0.38 (2.1M). A steady
  ~1M-step decline toward determinism, tracking the checkpoint action-drift (wait-frac 0.0 → 0.74 →
  1.0) rather than jumping.
- **No sharp KL event.** The largest `approx_kl` in the *entire 3M-step run* is 0.055; at the collapse
  onset (~1.02M) it is 0.037 — a modest ~3× bump over the ~0.01 baseline, nowhere near a
  destabilizing update (clip_range=0.2 absorbs it). There is no single batch to "stop before."
- **Reward degrades gradually** alongside: `ep_rew_mean` drifts from −0.0050 (1.0M) worse over the
  window, confirming the slide moves the policy *off* its good optimum, not a discontinuous break.

**Self-reinforcing mechanism (consistent with the data, stated as the working explanation):** as
entropy decays the agent samples the execute actions less and less, so it receives progressively less
gradient signal about how good executing is, while the immediate per-fill cost of trading stays
visible whenever it does trade. A policy that trades less sees mostly "trading costs me now" and
rarely samples the counterfactual "executing avoids the terminal dump" → it drifts deeper into the
do-nothing (action 0) corner, which lowers entropy further, which reduces execution sampling further.
A self-reinforcing entropy-collapse drift into the action-space corner that avoids the *immediate*
cost.

## The action-corner unifying hypothesis: PARTIALLY supported, NOT the strong version

Hypothesis tested: v5 collapsed to action 19 (dump-all) under adversarial LOB; v6 collapsed to action
0 (do-nothing) with a delayed forced-liq penalty — two absorbing corners at opposite ends of the
action space, one mechanism (drift into whichever corner avoids the immediate cost; which corner
depends on whether trading or waiting is the immediately-costly one).

What the data supports: **the gradual entropy-decay drift mechanism is common to both runs.** v5's
TensorBoard telemetry shows the same smooth entropy decline (−2.98 → −1.56 over 1.54M steps) with no
sharp KL event (max 0.067, at step 49k — early, not a collapse). So the *mechanism* (gradual
entropy-driven drift, not a destabilizing update) is shared.

What the data does **not** support: the strong "same pathology" claim. **v5's `ep_rew_mean` is flat
and stable (−0.0050 to −0.0053) all the way to its 1.54M endpoint — it shows no late reward-degrading
collapse at all**, unlike v6. v5's saved checkpoints stop at 800k, so its endpoint action distribution
could not be inspected, and the "v5 → action 19" characterization is from prior project history, not
re-measured here; v5's stable reward telemetry actively argues against v5 having suffered v6's
reward-degrading collapse (if v5 did drift toward dump-all under adversarial liquidity withdrawal,
that may have been reward-neutral or even near-optimal there, i.e. not a pathology). **Honest verdict:
common drift mechanism, yes; "two instances of the same failure," not established — and partly
contradicted by v5's stable reward.** Do not write the strong unification into any external record.

## Implied fix (for the collapse specifically)

Because it is a gradual entropy-decay slide, not a sharp event, the fix is **not** KL early-stopping.
The indicated levers, in order:

1. **Entropy floor / higher `ent_coef` late in training** (or a target-entropy schedule that does not
   anneal to near-zero) — directly opposes the self-reinforcing entropy collapse.
2. **Best-by-reward checkpoint selection + early stopping** — a reward-optimal policy exists at ~900k;
   retaining it (and/or stopping once eval reward plateaus and entropy starts collapsing) recovers a
   good model immediately, independent of preventing the drift. Requires fixing the `EvalCallback`
   single-date selection first.
3. **Reward shaping to surface the delayed terminal cost earlier** (e.g. per-step inventory-holding
   penalty scaled to remaining exposure) — attacks the immediate-vs-delayed asymmetry that biases the
   drift toward the do-nothing corner. Secondary; test only if 1+2 are insufficient.

Do not retrain until a fix is chosen from these; the diagnosis now points at (1)+(2) first.

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
