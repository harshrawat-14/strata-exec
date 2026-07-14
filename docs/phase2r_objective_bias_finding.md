# Phase 2R — The Execution RL Objective Silently Rewards Directional Speculation

**This is the headline finding of Phase 2, and it is a negative result that is more valuable than
the positive result it replaces.** The short version: a naively-specified execution-RL objective
(minimize implementation shortfall vs. a static arrival-price benchmark, evaluated on single
historical replay paths) does not train an execution agent. It trains an agent that discovers it
can *speculate on price direction* instead of executing well, because on realized historical paths
"hold inventory and sell late" is a directional bet that pays better than competent execution. The
agent found that bet and took it. The old headline ("RL beats optimal execution") was a mirage
produced partly by a reward loophole (now fixed) and partly by this deeper objective bias (not
fixable by tuning). This finding survives scrutiny; the old one did not.

Reproduce the analyses below with the scripts noted per-section. Model under analysis:
`ppo_lstm_v6r_time_features` (16-dim state, corrected split, full 3M-step schedule, trained on the
Phase 2R-corrected environment — reward loophole and price_factor bug both fixed, see
`git log 5630e5d`).

---

## Background: what Phase 2R fixed first (and why that wasn't enough)

Two environment bugs were found and fixed in commit `5630e5d` before this analysis (full detail in
that commit message and the persistent project memory):

1. **Forced-liquidation reward loophole.** Terminal forced liquidation was priced at a flat
   `gamma * (forced_liq_qty/total)` penalty (≈0.005 for dumping 100%), decoupled from the real cost
   of dumping that size into the book. Since real execution slippage runs ~0.4–0.6%, "never trade,
   eat the flat penalty" was reward-*superior* to executing. The original v6 collapsed onto this.
   Fixed: forced liquidation now walks the real book (`fill_market_sell`) and is priced by genuinely
   consumed depth.
2. **`price_factor` rescaling bug.** Historical mode recomputed the price rescale factor every
   snapshot, pinning the book mid to ≈arrival at every step and erasing real intraday drift — which
   made forced-liquidation IS evaluate to ≈0% regardless of market movement. Fixed: factor is fixed
   once per episode.

Both fixes are correct and verified (3 new Rust regression tests, one confirmed to fail on the old
logic and pass on the new). **But fixing them did not fix the trained policy.** The retrained
`ppo_lstm_v6r` model still collapses toward "wait" on most dates and *catastrophically* on every
crash date. The reason is not a third bug and not a tuning problem — it is the objective itself.

---

## The evidence, in three steps

### Step 1 — The trained policy waits, and it is worst exactly where waiting is worst

Deterministic policy, 20 seeds/date, on all 7 training dates + 2 held-out crash dates
(`ppo_lstm_v6r_time_features_best`):

| Date | Split | action=0 (wait) frac | forced-liq rate | mean IS |
|---|---|---:|---:|---:|
| 2024-01-15 (trend up) | TRAIN | 1.000 | 20/20 | +1.28% |
| 2024-02-20 | TRAIN | 0.982 | 20/20 | −0.04% |
| 2024-04-15 (crash-like) | TRAIN | 1.000 | 20/20 | **−3.58%** |
| 2024-06-10 | TRAIN | 0.915 | 0/20 | −0.83% |
| 2024-09-15 | TRAIN | 0.912 | 0/20 | −0.46% |
| 2024-10-15 (trend up) | TRAIN | 0.985 | 20/20 | −0.46% |
| 2024-11-06 (trend up) | TRAIN | 0.997 | 20/20 | +7.99% |
| 2024-08-05 (crash) | HELD-OUT | 1.000 | 20/20 | **−7.66%** |
| 2024-03-05 (crash) | HELD-OUT | 1.000 | 20/20 | **−7.11%** |

The policy is **not frozen** — on 2024-06-10 and 2024-09-15 it trades ~9% of steps with zero forced
liquidation. It acts *selectively*, and it specifically declines to act on crash days, where acting
is most urgent. That is learned behavior, not a collapse.

### Step 2 — H1 (reward still mis-specified) is killed: execution beats waiting by 6–7pp on crash days

The obvious worry: maybe on a thin, crashing book, walking the inventory through the depth genuinely
costs more than the (now correctly-priced) terminal dump — i.e. waiting is rational and the reward
is still wrong. **Tested directly** — best achievable constant-fraction execution vs. pure-wait,
10 seeds/date:

| Date | best executable IS | wait / forced-liq IS | executable advantage |
|---|---:|---:|---:|
| 2024-03-05 | −0.47% (sell 10%/step) | −7.03% | **+6.55pp** |
| 2024-08-05 | −0.53% (sell 30%/step) | −7.71% | **+7.18pp** |

A competent execution path exists and is 6–7 percentage points better than waiting on both crash
days — achievable by a trivial constant-fraction policy. The inequality is firmly closed. The agent
is leaving 6–7pp on the table by waiting. This is **not** a still-misspecified reward.

### Step 3 — H2 confirmed, sharpened: blanket-waiting maximizes *expected* reward over the training date mix

The decisive test. Expected terminal IS across the 7 training dates, blanket-wait vs. best-constant-
execute (10 seeds/date):

| Date | pure-WAIT IS | best-execute IS | winner |
|---|---:|---:|---|
| 2024-01-15 (trend up) | **+1.38%** | +0.21% | WAIT |
| 2024-02-20 | **+0.43%** | −0.51% | WAIT |
| 2024-04-15 (crash) | −3.62% | −0.47% | EXEC |
| 2024-06-10 | −0.62% | −0.59% | EXEC (barely) |
| 2024-09-15 | −2.01% | −0.55% | EXEC |
| 2024-10-15 (trend up) | **+0.98%** | −0.59% | WAIT |
| 2024-11-06 (trend up) | **+8.43%** | +0.43% | WAIT |
| **Expected (equal-weight)** | **+0.71%** | **−0.30%** | **WAIT by ~1pp** |

**The agent is correctly optimizing.** Blanket-waiting yields +0.71% expected IS across the training
mix; competent execution yields −0.30%. Waiting wins the average — driven overwhelmingly by the
trending-up days, above all **2024-11-06 at +8.43%**, a single date doing most of the work. On a
single realized historical price path, "hold and sell into the realized trend" is a directional bet,
and over these seven particular 2024 days that bet has positive expected value. The RL agent exploits
it precisely because it *has* a wait action; TWAP and AC cannot, which is the only reason those
baselines "look" better on crash days.

---

## What this means

Minimizing expected implementation shortfall **against a static arrival-price benchmark, on single
historical replay paths, is not an execution objective — it is a market-timing objective.** The
arrival-price benchmark is fixed at t=0, so any post-arrival drift flows straight into the score;
selling late on a day that rose beats the benchmark by (roughly) the full realized drift. The RL
agent, given a "wait" action and a reward that pays for favorable drift, correctly learns to
speculate rather than execute. No `ent_coef` or learning-rate change touches this — it is not a
convergence artifact; it is the optimum of a directionally-biased objective.

This is the **same** hindsight-trend-following issue logged as an open limitation during Phase 2R's
validation gate — but that entry framed it as a generalizability *caveat*. This analysis shows it is
not a caveat; it is **fatal to training**: the naive objective's optimal policy is speculation, not
execution.

### Why the obvious fix ("just sample more paths") is wrong

Sampling more real historical BTC days does **not** fix this. 2024 was a year in which BTC roughly
tripled; the aggregate drift across any sample of real 2024 days is still upward, so waiting still
wins on average. That would be averaging over a biased sample and calling it unbiased. The problem is
not path count — it is that a static arrival-price benchmark leaves directional drift inside the
objective at all.

### The actual fix (scoped, not yet implemented — gated on analytic approval)

In priority order:

1. **Drift-neutral benchmark (the real fix).** Replace/augment static-arrival IS with an
   **interval-VWAP** benchmark over the execution window. If the market rallies, the benchmark
   rallies too, so "wait and sell late" stops earning a free +8.43%. This is what institutional
   execution desks are actually graded on ("did you beat the market's own average price over your
   window," not "did BTC go up today"), and it encodes execution quality rather than a directional
   bet directly into the objective. **Must be validated by the same cheap analytic gate first**:
   compute blanket-wait vs. blanket-execute under a VWAP benchmark on all 7 dates before any retrain.
   (Note: it is *not* a priori obvious VWAP fully neutralizes the asymmetry — a late dump on a rally
   still beats a rising VWAP by roughly half the drift; the gate must confirm the sign/magnitude, not
   assume it.)
2. **De-drifted / bootstrapped paths** as a complement to (1), not a replacement.
3. **Inventory-schedule constraint** (must execute ≥X% by time T) as a blunt sanity rail that
   structurally forbids the pure-speculation policy.

---

## Two implementation bugs found alongside (fix before any retrain)

1. **`EvalCallback` selects `best_model` on a single trending-up date.** `rl/train.py` evaluates on
   `multi_dates[0]` = 2024-01-15 only — a date where waiting is *optimal* (+1.28%). So the callback
   actively selects the **most wait-collapsed** checkpoint as "best," which is exactly backwards. (It
   did not change the conclusion here, since `best` and `_final` were near-identical, but it would
   silently corrupt any future model selection.)
2. **No per-date mid-training telemetry.** The single-date eval means crash-day behavior can never be
   observed evolving over training, even with Phase 3's MLflow metric logging now live. This blinds us
   to exactly this class of failure during a run. Fix before retraining or we repeat it.

---

## VWAP analytic gate results (drift-neutral benchmark)

Ran `rl/phase2r_vwap_gate.py`: blanket-WAIT vs a genuine equal-quantity TWAP execution, scored
against three benchmarks — static-arrival (current objective), interval-TWAP, and true interval-VWAP
(real aggTrades volume, exposed via the new `interval_volume` telemetry). 10 seeds/date.

**Q1 — does a drift-neutral benchmark remove waiting's positive expected value?** Yes, under both
drift-neutral benchmarks:

| Benchmark | E[WAIT] over 7 training dates | verdict |
|---|---:|---|
| static-arrival (current) | **+0.71%** | positive → speculation pays → objective broken |
| interval-TWAP | −0.46% | negative → speculation no longer pays |
| interval-VWAP (real volume) | **−0.39%** | negative → speculation no longer pays |

The core mechanism is addressed: switching to a drift-neutral benchmark flips waiting from +EV to
−EV, so an expected-reward maximizer no longer prefers blanket-waiting. **This validates the VWAP
objective fix on its primary purpose.**

**Q2 — does competent execution then beat waiting, and does volume weighting widen the margin?**
Execution wins, but the margin is **thin and essentially unchanged by volume weighting**:
E[EXEC]−E[WAIT] = +0.056pp (arrival) → +0.075pp (TWAP) → +0.076pp (VWAP). Volume weighting did
**not** widen the aggregate margin.

Where volume weighting *did* matter is the crash-day split — and it **sharpened** it rather than
resolving it toward execution (volume clusters during crashes, as predicted):

| Crash date | vwap WAIT | vwap EXEC | verdict |
|---|---:|---:|---|
| 2024-03-05 (intra-window crash) | −2.82% | +0.22% | **EXEC wins by +3.03pp** |
| 2024-08-05 (already-crashed, flat thin book) | +1.17% | −0.83% | **WAIT wins by −2.00pp** |

On 2024-03-05 the crash happens *during* the window, so slicing in ahead of it clearly beats a
terminal dump. On 2024-08-05 the market gapped down *before/at* the window and stayed flat, so a
single terminal dump crosses the wide/thin spread once instead of many times — waiting genuinely
wins. **This is a true statement about markets, not a bug**, and any inventory rail forcing execution
on 2024-08-05 would be overriding a correct answer.

## The conditional-learnability finding (the most important result)

The decisive question for whether the fixed objective is *sufficient*: is the WAIT-vs-EXECUTE
preference **distinguishable from the state vector the agent observes**? If yes, the correct policy is
conditional and learnable ("slice into forming crashes, be patient in post-crash flat books") — no
rail needed. If no, it's a state-representation limit. Ran `rl/phase2r_conditional_signal.py`:
early-window (60-step) undisturbed signature of the observable dims (realized vol s2, spread s3,
depth s4, OFI s5, short drift s6), grouped by VWAP verdict.

**Result: the EXEC-wins and WAIT-wins dates are NOT cleanly separable in the observable state.** No
dim separates the two groups by even one pooled standard deviation (best: spread at 0.55σ; vol 0.49σ;
OFI 0.41σ; depth 0.39σ; drift 0.19σ). The decisive contrast makes it concrete — the two crash dates
demand *opposite* actions yet look alike in the observables:

| | vol(s2) | spread(s3) | depth(s4) | OFI(s5) | drift(s6) |
|---|---:|---:|---:|---:|---:|
| 2024-03-05 (EXEC) | +0.236 | +0.782 | +0.777 | −0.028 | −0.004 |
| 2024-08-05 (WAIT) | +0.379 | +0.773 | +0.771 | −0.032 | −0.011 |

They differ mainly in realized vol — and in the **counterintuitive** direction: 2024-08-05, where
*waiting* is correct, has the **highest** realized vol of all nine dates. The signal that actually
separates "slice into a forming crash" from "be patient in a post-crash flat book" is essentially
*the future price path* (does it keep falling?), which is not — and cannot be — in the current state.

**Interpretation.** The VWAP objective correctly removes the speculation exploit (Q1), but (a) the
residual execution-over-waiting advantage is thin (+0.08pp), and (b) the per-date optimal choice on
stressed days is **not reliably predictable from the observable state**. So we should *not* expect a
cleanly conditional policy to emerge, and there is a real risk the retrained agent settles on a
wait-leaning compromise. This bounds what *any* execution agent can achieve on this data — it is a
state-observability limit, not a tuning or representation-richness problem that more dims would fix
(the missing signal is future direction). This is a genuinely more interesting and defensible result
than a "16-dim beats 12-dim" ablation.

## Recommended next step (for approval)

1. **Implement the interval-VWAP benchmark as the reward/eval objective** — validated by Q1 above.
2. **Fix the two bugs** (EvalCallback single-date `best_model` selection; add per-date mid-training
   telemetry) — required before retrain regardless.
3. **Retrain and measure the per-date action distribution.** Do *not* add the inventory rail
   preemptively — it would force execution on 2024-08-05 where waiting is correct and would
   permanently contaminate any later 12-vs-16-dim ablation. The rail is a **fallback**, justified only
   if the agent still collapses to blanket-wait under the VWAP objective — and if it is needed, this
   analysis shows it would be compensating for a genuine state-observability limit, not papering over
   a learnable policy.

## Status

- Reward loophole + price_factor bug: **fixed** (`5630e5d`), verified.
- Objective bias: **diagnosed.** VWAP-benchmark fix **validated by the analytic gate (Q1 passes on
  real volume weighting).** Not yet implemented in the reward/eval path.
- Conditional-learnability: **the WAIT/EXEC split is not cleanly separable from the observable state**
  — a state-observability finding; expect the rail may genuinely be needed, but only after measuring.
- `EvalCallback` single-date selection + missing per-date telemetry: **to fix before any retrain.**
- `interval_volume` + `book_mid` telemetry added to the env (commit with this doc).
- No retrain until the objective-fix implementation plan is approved.
</content>
