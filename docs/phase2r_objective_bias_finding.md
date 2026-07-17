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

## Task 2 — Obizhaeva-Wang / impact-model audit (gates the ablation)

**Verdict: the model is a mislabeled hybrid, not a broken or fabricated one.** Read
`total_residual_impact()` line-by-line before concluding anything: each trade's temporary
contribution is tracked with its own timestamp and decayed individually via
`impact * exp(-rho*(now-t))`, summed across all outstanding trades, with a separate
never-decaying permanent accumulator. **This is genuine Obizhaeva-Wang-style transient-impact
dynamics** — the earlier fear (from Task 3's framing, before this audit ran) that dims 8/9
"might be measuring something that isn't there" does not hold; they measure something real.

What's actually wrong is narrower: the **magnitude** term (`sqrt(participation) * 0.1`) is a
square-root law, not OW's linear form — a different, separately well-supported empirical
literature, incorrectly given the OW name — and that magnitude was **hardcoded and uncalibrated**
(`0.1` constant, `sigma` term silently absent from the formula entirely), even though a
`CalibrationEngine` implementing exactly this fitting (`Y * sigma * sqrt(Q/V)`, fit to real
trade slippage) already existed in the codebase. It was wired into a *different* code path
(`research_sim.rs`'s separate `SquareRootImpact` model, via an opt-in `--calibrate` flag) but
never connected to `CounterfactualLob`, the environment actually used for RL training and for
`evaluate_all_strategies_on_date`'s counterfactual-mode comparisons.

**Consistent-application check (positive):** confirmed `evaluate_all_strategies_on_date` steps
TWAP, Heuristic, AC-Optimal, Adaptive-AC, and the RL agent all through the *same*
`StrataExecEnv`/`CounterfactualLob` instance per date — identical impact model and parameters
applied uniformly across strategies. Being uncalibrated did not mean being inconsistently
applied; the ruler was bent the same way for everyone, which matters for the ablation's internal
validity (relative comparisons) even before any fix — but the *absolute* magnitude of every
reported cost was still an assumed constant, not fit to this venue's data.

**Fixed, per direction ("relabel, don't rewrite" + "wire in calibration, don't skip it"):**
1. Relabeled every "Obizhaeva-Wang [whole model]" claim in code comments, README, and design
   docs to correctly scope OW to the decay structure only.
2. Wired the calibration engine into `CounterfactualLob`: `build_calibration_trades()` bridges
   Binance aggTrades (single price per trade) to the engine's expected before/after-bracketed
   format using nearest LOB-snapshot mids; `realized_volatility()` computes the previously-missing
   sigma term from the same snapshot series; `CounterfactualLobConfig::calibrated()` fits both and
   is now used automatically in `reset_counterfactual` whenever aggTrades data is available for a
   date, falling back to the prior hardcoded constants only when it isn't.
3. Added the requested sanity tests (direction, monotonic decay to the permanent level, temporary
   → 0 as decay time → ∞, and a no-free-lunch check) — one design fix needed along the way: a
   literal round-trip (sell then buy back) isn't constructible, since this simulator's impact model
   is sell-only (no buy-side/ask mechanism exists), so "no free lunch" is tested as "a larger
   self-trade must never leave a subsequent sell better off," the sell-side analogue.

All 15 `counterfactual_lob` tests + 205 lib tests pass; default behavior is byte-for-byte
unchanged when no aggTrades are available (verified: `impact_coefficient=0.1, sigma=1.0` remain
the fallback). Gate cleared — proceeding to Task 3.

## Task 3 — drift restoration + train/eval state consistency

**Root cause confirmed, same bug pattern as the historical-mode fix in `5630e5d`, never applied
here:** `CounterfactualLob::rescale_book()` recomputed `price_factor = simulation_mid_price /
raw_book.mid_price()` fresh from the CURRENT book on every call — renormalizing every snapshot's
rescaled mid back to ≈100 (the arrival value) regardless of real price movement. This is exactly
why dim 6 (short-term drift) was dead during counterfactual (training) mode: the rescaled mid
never actually moved.

**Fix:** `price_factor` is now a field on `CounterfactualLob`, computed once from the episode's
own arrival snapshot in `new()`/`reset()`/`seek()` (each episode's own start_offset gets its own
arrival-anchored factor, matching the historical-mode fix's per-episode semantics) and reused for
every snapshot within that episode. `rescale_book()` uses the fixed factor instead of
recomputing. This separates SCALE normalization (still expressing prices relative to arrival=100,
comparable across dates) from PATH flattening (real relative movement is now preserved) — the two
were conflated before, as flagged. 6 new tests (3 for the fix itself: `price_factor` fixed across
an episode / rescaled mid tracks a synthetic drifting fixture's real movement / `seek`/`reset`
each re-anchor to their own arrival). All 232 tests (208 lib + 27 rl_env, cross-referenced) pass.

**Confirmed: dim 6 is now live in training (counterfactual) mode.** Measured directly:
`std=0.055` (was exactly `0.0`) on a real date/seed with meaningful mid-movement observed
(min −0.23, max +0.19). Dims 8/9 remain live as before. The train/eval state-support mismatch
logged as an open issue is dissolved for dim 6; the flagged issue can be closed once Task 4's
ablation confirms no other mode-dependent dims remain.

**The critical safety check — did restoring observable drift reopen the speculation exploit?
No, confirmed directly, not assumed.** Re-ran the exact method that caught the original exploit
(measure the agent's actual cumulative RL reward, not IS-vs-arrival, for a blanket-wait policy vs.
a trivial 10%/step execute policy) on the fixed environment, at both the long default eval horizon
(2880 steps) and a training-representative horizon (800 steps, inside the 200–1440 random range
training actually samples):

| date | wait reward (2880-step) | exec reward | wait reward (800-step) | exec reward |
|---|---:|---:|---:|---:|
| 2024-11-06 | −1.0948 | −0.0235 | −0.9621 | −0.0235 |
| 2024-01-15 | −1.0235 | −0.0199 | −0.8418 | −0.0199 |
| 2024-03-05 | −1.0096 | −0.0214 | −0.8857 | −0.0214 |
| 2024-08-05 | −0.9336 | −0.0179 | −0.7825 | −0.0179 |
| 2024-06-10 | −1.0030 | −0.0179 | −0.8418 | −0.0179 |

**Execute beats wait on 9/9 dates tested at both horizons — the required invariant holds, robustly
and by a wide margin.** Drift did not reopen the exploit; if anything, it made the (already-fixed)
forced-liquidation loophole even more strongly, correctly punished than before.

**Honest nuance on the magnitude (worth recording, not glossing over):** the wait-penalty jumped
roughly 50× versus the pre-drift-restoration numbers (~−0.01 → ~−1.0). Traced this to its source
rather than accepting it at face value: on 2024-11-06 (a real rally day), the terminal book_mid
after 800 steps of real drift was 108.02 (vs. arrival 100.0) — dumping the full untouched
1,000,000-unit inventory in one shot at that point produced a terminal reward of −0.964, accounting
for essentially the entire episode's wait-penalty. This traces to the earlier forced-liquidation fix
(`5630e5d`): a full-inventory one-shot dump against a real, depth-limited counterfactual book very
plausibly exhausts available bid depth, triggering that fix's punitive-price extrapolation for the
unfillable residual (`worst_price − 5×spread_ref`). **This is directionally correct** (waiting is
genuinely catastrophic, now realistically so, not an artifact) **but the exact magnitude is
dominated by an extrapolation heuristic for depth beyond what's actually in the book, not a
literal market-clearing price** — worth a future refinement (e.g. a more principled deep-liquidity
model) but not a blocker: it doesn't threaten the required invariant, and if anything strengthens
the signal that never-trading is bad.

## Task 3 follow-up — quantifying and fixing the ~50× wait-penalty jump (pre-ablation gate)

Investigated before committing to any retrain, per explicit instruction not to let an unvalidated
heuristic dominate the incentive at the dump-all corner where both prior collapses happened.

**Instrumented the forced-liquidation path** (`forced_liq_filled_qty/notional` vs.
`forced_liq_residual_qty`/`punitive_notional` now in `step_counterfactual`'s info dict) and
measured the split directly. First measurement showed the punitive-price extrapolation
contributing ~0% of notional — which looked reassuring but was actually a red flag pointing at a
**different, real bug**: `s3` (spread state dim) was `tanh(1) = 0.7616` at t=0 and collapsing to
near 0 after step 1 in every date tested.

**Root cause found:** `reset_counterfactual`'s `spread_ref` was computed from the **raw,
unrescaled** BTC-dollar-scale snapshot (`first.half_spread()`), while `historical` mode's
`reset_historical` already correctly uses the rescaled book (`scaled_first.half_spread()`) for the
identical purpose — an isolated, pre-existing scale-mismatch bug specific to the counterfactual
path, not something Task 3 introduced (Task 3's drift restoration just made its effect visible for
the first time, since previously the near-static price meant this rarely mattered). Consequence:
`spread_ref` was ~$20-50 (raw) being compared against a ~0.03 (simulation-scale) half-spread every
step, and the same wildly-oversized `spread_ref` fed the punitive-price formula
(`worst_price − 5×spread_ref`), driving it deeply negative and flooring at `1e-6` — pricing ~80–90%
of any forced liquidation at an **effectively-zero** price, not a merely-punitive one.

**Fixed:** `spread_ref` is now computed on the same rescaled book historical mode already used
correctly. Verified: `s3` no longer collapses (stays ~0.68–0.78 across the episode instead of
dropping toward 0 after step 1).

**Re-quantified after the fix:** the punitive-extrapolation share of forced-liquidation notional is
still large (80–92% across the 9 dates tested) — but this is now understood to be **realistic, not
a remaining bug**: `total_inventory` defaults to 1,000,000 simulation units = 50,000 BTC (given
`btc_target=50000.0`), a large institutional-sized order that no real L2 top-of-book snapshot shows
enough visible depth to absorb in one shot. Dumping the full order in one step being
mostly-unfillable-against-real-depth is exactly the premise gradual execution research is built on,
not a defect. The *magnitude* of the extrapolation (`worst_price − 5×spread_ref`, a simple linear
offset) remains a simplification worth refining later, but it is no longer flooring at an
effectively-zero price.

**Re-ran the safety gate after the fix:** execute still beats wait on 9/9 dates, and the magnitude
is now far more defensible — wait-penalty dropped from the buggy ~−0.96 to ~−0.045 to −0.05 (a
~2–2.5× gap vs. execute's ~−0.02, not a 50× runaway):

| date | wait reward (fixed) | exec reward |
|---|---:|---:|
| 2024-11-06 | −0.0459 | −0.0235 |
| 2024-01-15 | −0.0498 | −0.0199 |
| 2024-03-05 | −0.0467 | −0.0214 |
| 2024-08-05 | −0.0462 | −0.0179 |
| 2024-06-10 | −0.0472 | −0.0179 |

**Verdict: fix confirmed necessary and applied before the ablation, per instruction.** Gate 1
cleared.

## Gate 2 — integrated smoke test (entropy floor + CPU + calibrated impact + drift-restored env, together for the first time)

Ran 400k steps (16-dim, real training config, CPU) as a first integration check before committing
to two full 3M ablation runs. Result: **clean.**

- Ran to completion at ~612 fps, no crash, no NaN, no error.
- Entropy declined gently 2.99 → 1.80 over 400k steps — normal early-training annealing, not a
  collapse slope. The floor never needed to engage (as expected this early).
- Action distribution per date (8 seeds, deterministic): `a0_frac` 24–50% with **5–7 unique actions
  used per date** — healthy exploration, not the collapse signature (`a0_frac≈1.0, n_unique=1`) seen
  in both prior collapsed models. **Zero drift to action 19** on any date.

**Caveat, stated plainly:** 400k steps is 13% of a full 3M schedule, and both prior collapses (v6r,
and the original v6) only manifested much later (~1.1M onset). This smoke cannot and does not claim
immunity from a late collapse — that is what the full 3M runs will test, with the entropy floor as
the (still-unvalidated-in-practice) safety net and CPU as the operationally-adopted device. What it
does confirm: the four pieces (entropy floor, CPU, calibrated impact, drift-restored env) run
together without fighting each other structurally. Gate 2 cleared for its intended narrow purpose.

## Gate 3 — train/eval state consistency: dim 6 resolved, dims 8/9 remain (architectural, not a bug)

Confirmed the Task 4 ablation harness (`evaluate_all_strategies_on_date`) uses `mode="historical"`
throughout. Dim 6 (fixed in Task 3) is now consistent between train (counterfactual) and eval
(historical). **Dims 8/9 (own-impact fractions) are not** — measured directly: `std=0.0` in
historical mode, `std≈0.07` in counterfactual mode, unchanged from before Task 3.

This is architecturally different from dim 6's fix and was flagged as a decision, not something
fixed unilaterally: historical mode is intentionally "pure replay, no causal feedback" (no `cf_lob`
exists in that mode at all), so dims 8/9 cannot be non-zero there without a materially bigger
architectural change (giving historical mode its own impact model, or switching the eval harness to
counterfactual mode — which would break comparability with every prior historical-mode-based report
back to Phase 0/1). **Awaiting explicit direction on which path Task 4 takes; not resolved as of this
entry.**

## Pre-fix code verifications (requested before the collapse fix; each with a test)

Three quick code checks run before touching training, given this project's sign-bug history:

1. **Reward sign convention — verified, tested.** The fill reward is
   `(fill_price - book_mid)·fill_qty / (arrival·total)`. A sell fill closer to mid earns a higher
   (less negative) per-unit reward than one that walks deeper into the book. Rust test
   `reward_higher_for_sell_fill_closer_to_mid` (per-unit, to isolate slippage sign from qty scaling).

2. **Start-offset bounding — verified, tested, with one documented caveat.** `start_offset = seed %
   (len/3)`; training horizon ≤ 1440. Real days have ~2880 snapshots, so worst-case training episode
   (offset ~959 + 1440 = 2399) fits — **training episodes never trade against a stale book, so the
   collapse is not a data-bounding artifact.** Rust test
   `training_horizon_episode_fits_within_day_no_stale_tail`. Caveat (documented, negligible in
   practice): the *fixed eval* horizon `HISTORICAL_DEFAULT_STEPS=2880` does *not* satisfy the bound
   for `start_offset > 0` — a wait-heavy eval episode at large start_offset trades its tail against a
   frozen book (up to ~33% of steps at max offset). All evals in this project use small seeds
   (start_offset 0–19, <1% of the episode), so results are unaffected, but bounding
   `start_offset ≤ len − total_steps` would remove it.

3. **State dims — dim 15 is fine; but dim 6 is a DEAD feature during training.** Dim 15 (a time
   feature) is non-constant (std ≈ 0.013). However, checking all 16 dims across real episodes surfaced
   a **train/eval state discrepancy** and a **dead dimension**:
   - **Dim 6 (`s6`, short-term price drift) is constant 0.0 in counterfactual mode — the actual
     training regime — across every date and seed tested.** The counterfactual book's mid stays pinned
     at ~100 (the impact model operates in a normalized frame; `price_history` never drifts), so the
     drift feature carries no information during training. The agent trains with an effectively dead
     momentum input.
   - Dims 8, 9 (`s8`/`s9`, counterfactual permanent/temporary impact) are live in training
     (counterfactual) but constant 0 in historical-mode eval by construction (no `cf_lob`). Dim 11
     (liq regime) was constant on some historical dates.
   - **Consequence:** the agent is trained (counterfactual) and analyzed (historical) on partially
     different state supports — dim 6 dead in train / live in eval; dims 8,9 live in train / dead in
     eval. This did not affect the reward/collapse findings (those measure behavior, not dim
     attribution) but it is a real issue.

   **For the writeup (not action now):** a proper ablation over the 16 state dims — which actually earn
   their place — should be done before any state-design claim goes in a paper. Dim 6's training-time
   deadness and the train/eval support mismatch are the first things such an ablation must confront.

## Collapse-fix smoke test (condition C) — INCONCLUSIVE on the floor, due to an LR confound

Ran the fixed config (entropy floor + multi-date reward selection + resume-safe best) to 1.3M steps
on CPU, past the historical ~1.1M collapse onset. Results:

- **The policy did not collapse.** Entropy held at ~2.26 across 1.0M–1.3M; the final and best models
  execute cleanly (wait-fraction 0.000 on all dates including both crash dates).
- **But this does NOT validate the entropy floor**, for two reasons:
  1. **The floor never engaged** — `ent_coef` stayed at its base 0.05 the entire run, because entropy
     never fell below the 1.2 floor. The safety net was present but untriggered, so its active
     behavior is unproven here.
  2. **LR-schedule confound.** The linear LR schedule anneals over `total_timesteps`, so the 1.3M
     smoke had LR ≈ 4.7e-5 at 1.1M vs the original 3M run's ≈ 1.9e-4 at the same step (~4× lower).
     The original collapse coincided with sustained ~1.9e-4 LR eroding entropy (2.19 → 1.49 over
     1.1M–1.2M); the smoke's much smaller updates plausibly prevented that erosion **independent of
     the floor**. Matched-step comparison confirms it: at 1.1M the smoke and original had nearly
     identical entropy (2.20 vs 2.19) but 4× different LR, and only the original then declined.

**Honest verdict (per condition C):** the smoke shows the fixed config produces a good executing
policy at 1.3M, but it **cannot attribute the non-collapse to the entropy floor** — the floor was
never exercised, and the compressed-LR confound could independently explain it. A clean validation
needs either (a) a discriminating run at constant LR ≈ 1.9e-4 (reproducing the collapse-region LR, to
see whether entropy then collapses and whether the floor catches it), or (b) the full 3M retrain with
the real LR schedule, where the floor engages if entropy drops. This smoke does not, on its own,
prove the fix.

## Task 1 — LR-vs-entropy isolation: the collapse does NOT reproduce on CPU (outcome 3)

Ran constant LR = 1.9e-4 (the original's collapse-onset LR) on CPU to 1.3M, fixed config, floor
armed. **The collapse did not reproduce:** entropy held ~2.0 across the whole run (lowest ~1.77 at
1.27M), never approached the 1.2 floor, the floor never engaged (ent_coef stayed 0.05), and the final
policy executes (wait-frac 0.000 on all dates). At the matched step/LR where the original collapsed
(1.1M–1.2M: original entropy 2.19 → 1.49), this run stayed ~2.0.

Cross-run summary (all on the same env/dates):

| run | device | LR profile | entropy @1.1M–1.2M | collapsed? |
|---|---|---|---|---|
| original v6r (3M) | **MPS** | linear 3e-4→0 over 3M (1.9e-4 at 1.1M) | 2.19 → 1.49 | **YES** |
| entropy-floor smoke | CPU | linear 3e-4→0 over 1.3M (4.7e-5 at 1.1M) | ~2.26 | no |
| constant-LR isolation | CPU | constant 1.9e-4 | ~2.0 | no |

**What this rules out:**
- **"Sustained collapse-region LR (1.9e-4) erodes entropy" — FALSIFIED.** Same LR on CPU, no collapse.
- **"High early LR (3e-4) drives the collapse" — also not supported.** The smoke's early LR (0–300k)
  was 3e-4 declining, identical to the original's early phase, on CPU — no collapse.
- **The fixed-config changes did NOT prevent it, and cannot be credited.** The entropy floor never
  engaged (gradient-identical to no floor); the multi-date eval and resume-safe callbacks don't touch
  training gradients. Attributing the non-collapse to "the fix" would be wrong.

**What remains (the confound):** the only uncontrolled variable between the collapsing original and
both healthy reproductions is **device — MPS (collapsed) vs CPU (both fine)** (plus the original's
full 3M-schedule trajectory, but both LR regimes it passed through were separately tested on CPU
without collapse). Given this project's documented MPS instability history (the `_patch_mps_lstm` LSTM
workaround; the MPS-optimizer-device resume bug), **MPS numerical instability is now the leading
hypothesis for the collapse** — not entropy dynamics, and not LR.

**Consequences (honest):**
- **The entropy floor is unvalidated as a fix** — it never had a collapse to catch, in any CPU run.
  It remains a cheap, correct safety net, but it is NOT demonstrated to be the fix.
- **The most-likely fix is simply "train on CPU"** — both CPU runs produce healthy executing policies,
  and CPU is also faster here (~600–1800 fps vs MPS's degrading 12–545). But this is *not yet
  confirmed as the cause*: the clean confirmation is an MPS run to 1.3M (does it collapse on MPS with
  everything else matched?), which has not been run.
- Do not claim the collapse is "fixed." State: the collapse does not reproduce on CPU; the leading
  (unconfirmed) cause is MPS; the entropy floor is an untriggered safety net.

**Follow-up: resume de-confounded cheaply instead of the 6h MPS run.** An MPS-to-1.3M confirmation was
started but stopped after ~13 min when its fps (280→41, still degrading) put the decisive answer
~6+ hours away with real OOM exposure — not worth it when "train on CPU" is the fallback either way.
Instead, the remaining confound (device vs. the original's mid-run OOM resume) was resolved directly
and cheaply: took the constant-LR isolation run's `ckpt_700000` (healthy, mid-collapse-window), forced
a real resume through it on CPU (`--resume-from` + `--resume-mlflow-run-id`), and ran 600k further
steps (past 1.3M total, well past the original's ~1.1M collapse onset).

**Result: the resume is exonerated.** `Restored num_timesteps=700000` and
`[resume-safe-eval] loaded prior best_mean_reward=-0.004660` confirmed clean resume mechanics (fix 3
working as designed). Entropy stayed healthy across and after the resume boundary (2.28 at 708k →
2.16 at 1.30M, never dropped below ~2.04), `best_mean_reward.json` correctly remained unchanged
post-run (nothing worse overwrote it), and the final policy executes cleanly (wait-frac 0.000 on all
four dates checked, including both crash-adjacent ones). **Resume mechanics do not cause the
collapse.**

**Final Task 1 verdict:** by elimination — LR falsified, resume exonerated, CPU healthy in three
separate runs (smoke, constant-LR isolation, and now CPU-with-resume) — **device (MPS) is the
remaining, now much more confidently implicated cause**, though the direct MPS-to-1.3M confirmation
was not completed (stopped for cost/risk reasons, not because it contradicted the hypothesis; it ran
135k steps healthy before being stopped, consistent with but not proof of the hypothesis).
**Operational decision: adopt CPU as the training recipe.** Three independent CPU runs (smoke,
constant-LR, CPU-resume) are all healthy; device is the best-supported explanation for why MPS
differs, but is recorded as "MPS strongly implicated by elimination, not directly confirmed" — not
"cause is MPS," full stop. The entropy floor remains the honest, unvalidated safety net (present,
never engaged across any of these four CPU runs, not demonstrated to be doing the work) — it stays in
because it's cheap and correct, not because it's been shown to be load-bearing.

**Proceeding to Task 2 with confidence per the outcome-1-equivalent path** (CPU recipe validated
operationally across three independent runs, resume mechanics separately verified safe for Phase 4's
longer runs).

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

## Task 4 — the ablation deliverable (final, corrected)

Tasks 1-3 all closed (device/resume isolation → entropy floor + CPU recipe adopted; impact-model
audit → relabeled + calibration wired; drift restoration → dim 6 live, safety gate re-confirmed at
two horizons; a `spread_ref` scale bug found and fixed while quantifying the wait-penalty
magnitude). Two matched 3M-step RL arms trained (16-dim, 12-dim; identical config otherwise) and
verified healthy via `best_model` (reward-selected, resume-safe, all-dates — the artifact that fix
exists to produce; `_final` showed partial, non-corner drift in the 16-dim arm specifically that
the floor slowed but did not fully prevent).

**A second real bug was found and fixed during the ablation itself, before any result was treated
as final:** the classical-baseline policies (TWAP/Heuristic/AC-Optimal/Adaptive-AC) hand-maintained
two mirror copies of the action→fraction table (`V5_FRACTIONS`/`V6_FRACTIONS`) that had silently
diverged from the real Rust environment's `ACTION_FRACTIONS` — the `n_state_dims=12` mirror matched
the real table at only 3/20 indices. Found by the ablation script's own built-in consistency check
(AC/TWAP results must be identical whether computed alongside the 12-dim or 16-dim RL call, since
neither baseline depends on the RL model — the check fired on every date). Root-fixed: Python now
queries the real table from the Rust binary (`--print-action-fractions`) instead of copying it, with
a cross-language test asserting identity. This is the second instance in this project of a
hand-maintained mirror of a canonical value causing a silent bug (cf. the CVaR sign inversion) —
logged as a meta-lesson in `docs/phase2r_standing_issues.md`. **RL-vs-RL numbers were never affected**
(RL picks raw action indices from a trained policy, never routed through these tables) — only the
classical-baseline comparisons needed redoing.

**An earlier draft of this ablation's "sharpest finding" — that the 16-dim arm flips the 12-dim
arm's loss-to-AC into a win on 2024-08-05 — does NOT survive the fraction-table fix and is
RETRACTED.** It was computed against the corrupted 12-dim-derived AC baseline. The corrected
picture on that date: **both RL arms lose to the real AC-Optimal** (RL-12dim −8.63% vs AC's −4.96%,
a −3.67pp gap; RL-16dim −6.48% vs AC's −4.96%, a −1.52pp gap). Not a flip — but the 16-dim state
representation does narrow the loss margin substantially (from −3.67pp to −1.52pp) even in a
losing scenario, which is itself a real, smaller finding worth keeping.

**Final ablation, all 11 clean dates** (7 training + 2 held-out crash + 2 previously-unused
held-out calm dates added specifically to enlarge the out-of-sample evidence beyond crash-only
n=2), 5-way comparison (RL-12dim, RL-16dim, AC-Optimal, Adaptive-AC, TWAP), paired bootstrap
significance vs. every baseline:

| split | RL-16dim beats AC | RL-12dim beats AC | note |
|---|---|---|---|
| in-sample training (7) | 7/7 | 7/7 | both RL arms beat AC comfortably in-sample |
| held-out crash (2) | 1/2 | 1/2 | wins 2024-03-05, **loses 2024-08-05** (both arms) |
| held-out calm (2) | 2/2 | 2/2 | RL beats AC even on calm dates — see caveat below |

**The representation ablation itself (16-dim vs 12-dim) is unaffected by the fraction-table bug and
reconfirms the earlier finding exactly:** on both held-out crash dates, 16-dim beats 12-dim
substantially (2024-03-05: +0.87pp; 2024-08-05: +2.15pp mean IS) — unchanged from the pre-fix
numbers, as expected since RL was never affected. **This remains the credible part of the result.**

**Honest reframing of the "calm is a wash" thesis:** the corrected baseline numbers show RL beating
AC-Optimal and Adaptive-AC even on the held-out calm dates (mean delta +1.36 to +1.96pp), not a
clean wash as hypothesized going in. Against TWAP specifically the calm-date picture is mixed
(16-dim loses 0/2, 12-dim wins 2/2) — TWAP's naive uniform slicing is a genuinely tough target on
quiet, low-drift days. Not investigated further at this session's depth (why RL beats a *supposedly
near-optimal* AC baseline even in AC's own designed regime is a real question — possible
explanations include AC's fixed λ not matching every calm day's true risk-aversion tradeoff, or a
residual calibration/scale mismatch between the AC/Adaptive-AC baseline and the RL agent's
observation space — flagged for future investigation, not resolved here).

**Data quality:** all 11 dates checked directly (consistent BookDepth row counts 27811-28801, no
truncation; AggTrades volumes 600K-6M rows, no near-empty files) before inclusion — no dates were
excluded.

**Clean ablation:** `forced_liquidation_rate = 0.0` for both RL arms on all 11 dates — neither arm
collapsed during evaluation.

## Status

- Forced-liquidation loophole + `price_factor` fix (`5630e5d`): **correct, stands, tested.**
- Objective-bias / VWAP / observability findings: **retracted** (artifacts of IS-vs-arrival).
- Wait-collapse: re-diagnosed as **late-training policy collapse**, isolated by elimination to
  device (MPS); CPU adopted as the training recipe; entropy floor + resume-safe/multi-date
  best-model selection implemented and verified.
- Impact model: relabeled honestly (OW-style decay structure + square-root magnitude), calibration
  engine wired in.
- Drift restoration (Task 3): dim 6 live in training, safety gate re-confirmed at two horizons.
- `spread_ref` scale bug: found and fixed while quantifying the wait-penalty magnitude.
- Classical-baseline fraction-table bug: found (via the ablation's own consistency check),
  root-fixed (single source of truth, cross-language test), and the affected "sharpest finding"
  explicitly retracted above.
- **Task 4 ablation: complete.** 16-dim beats 12-dim on both held-out crash dates (unaffected by
  the fraction-table bug); both RL arms beat AC/Adaptive-AC in-sample and on held-out calm dates;
  both RL arms lose to AC on the more severe held-out crash date (2024-08-05), though 16-dim loses
  by less than 12-dim.
- `interval_volume` / `book_mid` telemetry: retained as sound instrumentation.
- Nothing pushed; demo untouched throughout.
