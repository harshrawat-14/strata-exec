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

## Task 4 close-out gate — three investigations before Phase 2R can close

The corrected 5-way ablation above (RL beats AC/AAC in-sample and on both held-out calm dates,
loses only on the harder held-out crash date) was flagged as suspicious rather than accepted: it is
close to the *opposite* of the mechanistic story this phase had been building toward (tie on calm,
win under stress), and the AC baseline in that table ran at a single hardcoded λ, never swept. Three
things were required to report before considering the phase closeable: an AC λ-grid check, an
action-granularity/bucketing investigation, and a full implementation-consistency audit. All three
are reported below. **None of them are closure — they are the requested report.**

### Gate B — AC λ-grid sweep: the fixed-λ comparison was a strawman baseline (CONFIRMED)

`AC_DEFAULT_LAMBDA = 0.001` is a single hardcoded constant, used identically for every date in the
ablation with no per-regime tuning — confirmed by reading every call site (`ACOptimalPolicy(...,
lam=ac_lambda)` and `AdaptiveACPolicy(..., lam=lam)`, always fed the same default). A pre-existing
but never-run-post-correction `evaluate_ac_lambda_grid` function (Phase 1-era) was parameterized for
counterfactual mode and run across a wide grid (`1e-6` to `1.0`) on the 4 anchor dates — both held-out
crash dates and both held-out calm dates — comparing best-tuned AC/Adaptive-AC per date against the
already-trained RL-16dim arm (`rl/phase2r_lambda_grid_check.py`, local only):

| date | regime | RL-16dim mean_IS | AC @ default λ=1e-3 | **AC @ best λ** | best λ | result at best λ |
|---|---|---|---|---|---|---|
| 2024-03-05 | crash (milder) | -1.186% | -4.191% | **-1.682%** | 1e-4 | RL16 still wins, +0.50pp |
| 2024-08-05 | crash (severe) | -6.482% | -4.960% | **-4.960%** | 1e-3 (=default) | RL16 **loses**, -1.52pp |
| 2024-07-10 | calm | -0.223% | -1.885% | **+0.027%** | 1e-6 | RL16 **loses**, -0.25pp |
| 2024-12-10 | calm | -0.802% | -1.868% | **-0.720%** | 1e-6 | RL16 **loses**, -0.08pp |

(Adaptive-AC's best-λ numbers move in the same direction on every date, usually a touch further
against RL16 — full grid and per-λ table in the script's saved output.)

**This overturns the close-out table's headline claim.** At AC's own best λ per date: AC/AAC ties or
beats RL-16dim on **three of the four anchor dates** (both calm dates, and the *harder* crash date),
and RL only clearly wins on the milder crash date (2024-03-05) — where even there the margin over
best-tuned Adaptive-AC (+0.07pp) is thin enough to be noise at n=50 episodes. The "RL beats AC
in-sample and on calm dates" finding in the table above is a direct artifact of AC running at a
single, un-swept λ that happens to only be near-optimal on the one date (2024-08-05) where RL was
already losing. On every other date, that fixed λ was leaving free performance on the table for AC.
**The honest picture, once AC is fairly tuned, is closer to "AC/AAC generally ties-or-beats RL,
except possibly on milder stress" — nearly inverted from the un-swept table.** This is exactly the
strawman-baseline failure mode flagged as a precondition for closing this phase; it is now confirmed,
not hypothetical.

Caveat: the grid's most competitive λ was the smallest tested (1e-6) on 3 of 4 dates, meaning AC's
true unconstrained optimum could be even more patient (lower λ) than anything measured here — this
grid is a lower bound on how well AC can do, not necessarily its ceiling. Extending the grid further
downward is cheap (no retraining involved) and worth doing before any final claim is written, but is
not done here to avoid re-litigating the number a third time before reporting back.

### Gate A — action-granularity/bucketing: NOT the mechanism (investigated, mostly ruled out)

Two sub-hypotheses were tested:

1. **"Baselines execute continuous fractions, RL is bucketed" (simple version) — ruled out.**
   Every classical baseline (`TWAPPolicy`, `HeuristicPolicy`, `ACOptimalPolicy`, `AdaptiveACPolicy`)
   computes a continuous target fraction and then calls the identical `nearest_action(target,
   fractions)` used everywhere else — confirmed by reading `predict()` on all four classes. The Rust
   environment's `step()` only ever accepts a discrete action index; no strategy, RL included, can
   execute an exact continuous fraction. There is no asymmetric handicap of this form.

2. **"Coarse buckets disproportionately distort AC's schedule, hurting it more than RL" (subtler
   version) — investigated quantitatively, does not explain the anomaly.** Simulated AC-Optimal's
   full 2880-step schedule against the real 20-bucket `ACTION_FRACTIONS` ladder across the same λ
   grid as Gate B, tracking cumulative bucketing-induced quantity error and end-of-horizon residual
   (any inventory AC's schedule wanted to hold back past the horizon gets forced-liquidated, a
   punitive tail event):

   | λ | cumulative \|bucketing error\| (frac of orig. inventory) | residual left at horizon end |
   |---|---|---|
   | 1e-6 | 0.229 | 0.00009 (negligible) |
   | 1e-5 | 0.375 | 0.249 (large — bucket floor forces early rounding to zero) |
   | 1e-4 | 0.134 | 0.072 |
   | 1e-3 (old default) | 0.007 (lowest) | 0.0001 |
   | 1e-2 – 1.0 | 0.03 – 0.08 | ~0 (schedule urgent enough to clear the bucket floor easily) |

   Bucketing distortion is real and **non-monotonic in λ.**

   **Correction (from the item-1 full 11-date/13-λ grid, run after this section was first written):**
   the original characterization above — "second-order, small precisely where AC's optimum sits" —
   understated it. Extending the grid down to 1e-9 and rerunning the same schedule-vs-bucket
   simulation across the full range shows a **hard floor, not a small effect**: at every λ ≤ 3e-7,
   on every one of the 11 dates, **100% of the intended schedule rounds down to the zero bucket for
   the entire horizon** — the per-step target implied by that λ never clears the smallest nonzero
   action (0.1%), so AC never voluntarily trades at all and the *entire* position gets dumped via
   punitive forced liquidation at the end. This is not a minor distortion; it is total execution
   failure, and it is why the `phase2r_lambda_grid_check.py` script's 4-date grid (which only tested
   down to 1e-6) never surfaced it — 1e-6 happens to sit just above the floor for most dates.

   | λ | residual still unexecuted at horizon end (2024-01-15, representative) |
   |---|---|
   | 1e-9 – 3e-7 | **1.0000 — total failure, 100% dumped via punitive liquidation** |
   | 1e-6 | 0.0001 — essentially fully executed |
   | 1e-5 | 0.2492 — a quarter of inventory dumped |
   | 1e-4 | 0.0722 |
   | 1e-3 (old default) | 0.0001 |

   **Revised conclusion:** granularity is not "ruled out, second-order" — it is the mechanism that
   determines the *floor* of AC's usable λ range on every date. Below that floor, AC doesn't have a
   worse schedule, it has no schedule at all. AC's actual optimum (λ≈1e-6 on most dates) sits
   immediately above this floor, which is now a mechanistic explanation, not just an empirical
   observation: it's the most patient schedule that still clears the bucket floor reliably. Above the
   floor, once AC can actually execute, granularity's effect on schedule *quality* really is small
   (matching the original table's 1e-3-to-1e-2 region) — that part of the original claim stands. Only
   the "ruled out at the low end" framing was wrong, and is corrected here rather than quietly
   updated.

### Gate A follow-up (greenlit): the specific chattering mechanism — tested, NOT confirmed as framed; a clearer mechanism found instead

The hypothesis to test: AC's continuous intent survives 20-bucket rounding because per-step errors
partly average out over 2880 steps, while RL — learning natively in bucket space with no schedule
prior — cannot express a smooth slow liquidation once the smallest nonzero bucket (0.1%, confirmed
by printing the full table below) exceeds what a good per-step schedule wants, and is forced to
chatter between the zero bucket and an overshoot bucket. Predicted consequence: this should cost
least on calm days and most on crash days (thin book, precision matters most) — matching where RL
loses worst.

**Full `ACTION_FRACTIONS` table** (queried live from the Rust binary, not assumed):
`[0.000, 0.001, 0.003, 0.005, 0.008, 0.010, 0.015, 0.020, 0.030, 0.050, 0.070, 0.100, 0.150, 0.200,
0.300, 0.400, 0.500, 0.700, 0.900, 1.000]`. Smallest nonzero bucket: 0.001 (0.1%). Largest gaps:
0.5→0.7 and 0.7→0.9, both 0.200 — confirmed, nothing between 70% and 90% except 90% itself.

**Method:** ran one fixed-seed (seed=0) trajectory each for RL-16dim, AC-Optimal at its per-date best
λ (from the item-1 grid), and a trivial constant 10%/step policy, on the two held-out crash dates and
two held-out calm dates, in the real counterfactual simulator (not a toy replica) —
`rl/phase2r_granularity_investigation.py`, trajectories plotted at `rl/results/plots/
granularity_<date>.png`, summary in `rl/results/phase2r_granularity_investigation.csv`.

**The specific chattering mechanism is NOT what the trajectories show.** RL does not alternate
between zero and an overshoot bucket — `frac_isolated_bursts` (a nonzero action immediately
surrounded by zero actions on both sides, the signature of literal chatter) is **0.2%–1.3% across
all four dates**, i.e. essentially absent. RL trades on 93–99% of its steps, continuously, in small
increments (`mean_slippage_bps` per fill: 50–69, actually *lower* than AC's 43–95 on the same dates).
RL's trajectory is smooth, not jagged. The severity prediction built on the chattering hypothesis
also fails directly: `mean_slippage_bps` is 53.6 on calm dates vs 59.7 on crash dates (a small
difference in the predicted direction, not the large one hypothesized), and `frac_isolated_bursts` is
*higher* on calm dates (0.8%) than crash dates (0.2%) — the opposite of the prediction. **This specific
mechanism is not confirmed; recorded as tested and not supported, not quietly dropped.**

**What the trajectories show instead — a clearer, decisive mechanism: RL's execution pace does not
adapt to regime the way tuned-AC's does.** Plotting cumulative-fraction-executed against time
(`rl/results/plots/granularity_*.png`) shows a consistent pattern across all four dates:
- **RL finishes in roughly the same wall-clock window regardless of regime** — ~900–1050 of 2880
  steps (~33–36% of the horizon) on every one of the four dates tested, calm or crash alike.
- **AC's *tuned* pace varies enormously by regime**, because that's exactly what a properly-set λ is
  supposed to produce: on calm days (λ=1e-6) it spreads execution across nearly the *entire*
  2880-step horizon in tiny slices (2,650 trades, still only 63–64% done by step 950, where RL is
  already ~85% done); on the milder crash (2024-03-05, λ=1e-4) it front-loads faster but still uses
  most of the horizon (93% by step ~700, dust dumped at the end); on the severe crash (2024-08-05,
  λ=1e-3) it front-loads *hard* — 73% executed by step ~250, then goes silent (hits the granularity
  floor documented above) until a terminal forced-liquidation dump for the remaining 27%.
- **RL's fixed pace happens to roughly match what 2024-03-05 needed (its one win) and is badly
  mismatched everywhere else:** too fast for the calm days (accepting avoidable impact cost when
  patience was free — RL's `exec_pct` -0.63 vs AC's -0.45 on 2024-07-10, -0.72 vs -0.43 on
  2024-12-10), and far too slow for the severe crash (accepting large avoidable drift exposure when
  urgency was needed — see the decomposition immediately below).

**Crash-date IS decomposition (the other requested analysis), 2024-08-05:**

| policy | drift_pct ("adverse holding") | execution_pct ("spread+impact") | dust_pct | total (= reported mean_IS) |
|---|---|---|---|---|
| RL-16dim | **-5.29** | -1.19 | -0.10 | **-6.58** |
| AC (best λ=1e-3) | -3.11 | -1.67 | 0.00 | -4.79 |
| TWAP 10%/step (trivial) | **-1.52** | -1.80 | -0.10 | -3.43 |

(Decomposition reconstructs the environment's own `total_IS_pct` exactly for every row — verified,
not assumed; see `decompose_is()`.) **RL's execution quality is actually fine — its `execution_pct`
(-1.19) is the best of the three, better than AC's (-1.67).** RL loses almost entirely on the
`drift_pct` term: -5.29, nearly double AC's -3.11 and more than 3× the trivial policy's -1.52. **Yes,
RL is back-loading into the falling market** — by continuing to trade gradually across ~950 steps
while the price keeps falling, RL holds meaningful inventory deep into the crash's continued decline.
The *naive* 10%/step policy, by finishing in ~160 steps, minimizes drift exposure better than either
tuned AC or RL, despite having no adaptivity at all — on this date, raw urgency dominates execution
sophistication. This is a clean, mechanistic, and complete answer to the "is RL back-loading" question:
yes, specifically through under-urgent, regime-insensitive pacing, not through poor per-trade execution.

**Net revision to Gate A:** the earlier "ruled out" verdict is revised twice now — once (above) to
"granularity sets a hard floor on AC's usable λ," and now again to identify what *does* explain RL's
regime-specific losses: not bucket-forced chatter, but **a fixed, regime-blind execution pace that
happens to only suit the one regime (milder crash) where RL wins.** This directly supports the
reframe below (a diagnosable RL-setup gap, not a ceiling) and directly motivates the residual/delta
action-space design sketch (also below): a design that inherits AC's regime-appropriate pace as a
prior would remove exactly this specific failure mode.

### "RL learned execution but not urgency" — the two follow-up checks before an item 5/6 decision

The decomposition above supports a specific, stronger headline: **RL learned execution but not
urgency.** RL's execution quality is the best of the three policies compared (`execution_pct`
-1.19 vs AC's -1.67 vs the trivial policy's -1.80 on 2024-08-05; mean per-fill slippage 69bps vs AC's
95bps). Trade-frequency makes the mechanism concrete: AC's urgency swings 1.00 → 0.285 → 0.088 across
calm → milder-crash → severe-crash (κ=√(λσ²/η) recomputing from live vol, for free), while RL sits at
0.93–0.99 on every regime tested. Two cheap, no-training checks were run before deciding what to do
about it (`rl/phase2r_urgency_diagnosis.py`).

**Task A — does AC's pacing combined with RL's execution style beat tuned AC, with no training?**
Two estimates were compared against a real simulated hybrid: a *naive* analytical one
(`AC's drift_pct + RL's execution_pct`, added directly) and an *actual* hybrid policy run through the
real simulator — AC's cumulative schedule sets how much should have been sold by each step, but the
catch-up is capped at RL's own median per-step chunk size on that date (0.5% of remaining on 2024-08-05,
0.3% on 2024-03-05) and sliced across as many steps as needed, rather than executed in whatever
(possibly much larger) single bucket AC-Optimal itself would pick:

| date | RL (pure) | AC (pure, best λ) | **hybrid (simulated)** | naive estimate |
|---|---|---|---|---|
| 2024-08-05 (severe crash) | -6.58 | -4.79 | **-5.15 (does not beat AC, -0.36pp; beats RL by +1.43pp)** | -4.30 (would have beaten AC) |
| 2024-03-05 (milder crash) | -1.11 | -1.64 | **-1.16 (beats AC by +0.48pp; slightly behind pure RL)** | -1.30 (also beats AC) |

**The naive analytical shortcut is unreliable — it disagrees with the real simulated result in both
magnitude and, on 2024-08-05, direction.** The real hybrid's own decomposition on 2024-08-05 shows why:
`execution_pct` improves dramatically (-1.67 → -0.81, more than halved — RL-style fine slicing is
confirmed to be a real, substantial, transferable execution-quality gain), but `drift_pct` gets *worse*
than pure AC (-3.11 → -4.24) rather than inheriting AC's own drift number, because capping every
catch-up step at RL's small chunk size imposes a speed limit AC's own (occasionally much larger,
front-loaded) chunks aren't subject to — on the day that most rewards moving fast, the cap itself
becomes the bottleneck. **The quantified prize is real but partial and conditional, not a guaranteed
win**: substantial and unambiguous relative to pure RL (+1.43pp / +0.48pp on the two crash dates), but
only sometimes enough to beat tuned AC outright, specifically falling short on the date where speed
matters most. This directly informs the residual-action-space design: its multiplier range/resolution
must not recreate this same speed-limit bottleneck, or it risks capping exactly the upside this
experiment just measured as achievable.

**Task B — is regime information present and discriminative early in the episode (dims s2, s10, s11,
s13), or does it arrive late?** Collected RL's full state vector at every decision point on both crash
dates and both calm dates (seed=0), compared early-episode windows:

| window | s2 (vol ratio) crash−calm gap | s13 (intraday vol ratio) crash−calm gap | s10 (vol_regime) | s11 (liq_regime) |
|---|---|---|---|---|
| step 0 | 0.0000 | 0.0000 | identical (0.500, cold-start default) | identical (0.000) |
| steps 0–9 | +0.040 | +0.020 | **identical across all 4 dates** | **identical across all 4 dates** |
| steps 0–49 | +0.167 | +0.103 | **identical across all 4 dates** | **identical across all 4 dates** |
| steps 0–99 | +0.231 | +0.152 | **identical across all 4 dates** | **identical across all 4 dates** |

**Mixed, precise answer — not a clean "present" or "absent":** the two *continuous* volatility dims
(s2, s13) carry real, correctly-directioned, and monotonically strengthening signal starting within
the first 10–50 steps (5–25 minutes) — the severe-crash date is measurably, increasingly distinguishable
from calm dates on exactly the information channel that should matter. **But the two *discrete*
regime-bucket dims (s10 vol_regime, s11 liq_regime) — purpose-built as regime indicators — are
completely flat and bit-identical across all four dates for the entire first 100 steps (50 minutes)
tested.** Reading `regime_dims()`: the discrete buckets require the rolling realized-vol estimate to
cross a fixed threshold (0.7×/1.3× `sigma_ref`) before reclassifying away from "normal" — with a return
buffer that only fills to 50 samples and a `sigma_ref` calibrated at the whole-day level, the *early*
window's short-sample vol estimate stays inside the "normal" band on every date tested here, even
though the *continuous* signal is already separating. **This means part of the urgency-learning
failure is plausibly a credit-assignment problem (continuous info was present and usable, evidence
below in Task A shows it's exploitable) and part is a genuine, shared state-estimation limitation** —
the same rolling-vol estimator feeds both RL's state *and* Adaptive-AC's own live-schedule
recomputation, so a residual-action-space design using an Adaptive-AC-style reference schedule would
not automatically escape this early-window lag; it would need to be checked, not assumed away.
(Single-seed, seed=0, on 4 dates — a directional finding, not a statistically exhaustive one; same
caveat as the granularity investigation above.)

### Follow-up (greenlit): full 11-date λ-grid, extended to bracket the true optimum

Gate B's 4-date grid left the optimum at the grid edge (1e-6) on 3/4 dates — not yet bracketed.
Extended the grid down to 1e-9 and ran all 11 dates (`rl/phase2r_lambda_grid_full.py`,
`rl/results/phase2r_lambda_grid_full.csv`, 143 rows, ~24 min wall-clock):

| date | regime | RL16 | best AC | @λ | best AAC | @λ | result |
|---|---|---|---|---|---|---|---|
| 2024-01-15 | calm bull | +0.37% | +0.91% | 1e-6 | +0.88% | 1e-6 | RL16 loses |
| 2024-02-20 | moderate vol | -0.80% | -0.53% | 1e-6 | -0.41% | 1e-6 | RL16 loses |
| 2024-03-05 | crash (flash cascade) | -1.19% | -1.68% | 1e-4 | -1.26% | 1e-5 | **RL16 wins** |
| 2024-04-15 | correction | -1.06% | -1.09% | 1e-6 | -0.97% | 1e-5 | RL16 loses |
| 2024-06-10 | quiet consolidation | -0.57% | -0.57% | 1e-6 | -0.55% | 1e-5 | RL16 loses |
| 2024-07-10 | calm | -0.22% | +0.03% | 1e-6 | +0.04% | 1e-6 | RL16 loses |
| 2024-08-05 | crash (Yen unwind) | -6.48% | -4.96% | 1e-3 | -4.63% | 1e-4 | RL16 loses |
| 2024-09-15 | moderate vol | -0.50% | -0.48% | 1e-6 | -0.48% | 1e-6 | RL16 loses |
| 2024-10-15 | moderate vol | -1.09% | -0.86% | 1e-6 | -0.86% | 1e-6 | RL16 loses |
| 2024-11-06 | post-election surge | +2.94% | +6.17% | 1e-9 | +6.17% | 1e-9 | RL16 loses |
| 2024-12-10 | bull run | -0.80% | -0.72% | 1e-6 | -0.64% | 1e-5 | RL16 loses |

**RL-16dim vs best-tuned AC/Adaptive-AC across all 11 dates: 1 win, 10 losses.** The optimum is now
genuinely bracketed on 9/11 dates (interior at 1e-6, confirmed not still climbing at the grid edge —
see the granularity correction above for why 1e-6 specifically). Two dates (03-05, 08-05 — the two
held-out crash dates) have their optimum at a materially higher λ (1e-4/1e-3), meaning some real
urgency, not patience, is what best serves AC there — and RL only wins on the milder of the two.
2024-11-06's best λ (1e-9) sits inside what the granularity correction above calls the
"total-failure zone" for most dates — worth a flag: this date's AC/AAC numbers at 1e-9 need a
sanity check (does 100%-forced-liquidation coincidentally produce a good price on a strongly
trending-up day, i.e. is this itself a hindsight-drift artifact analogous to open issue #1?) before
being treated as a real optimum rather than a fluke; not resolved here.

**This is the decisive answer to the close-out gate's original question.** Once AC/Adaptive-AC are
run at their own best λ, they beat RL-16dim on 10 of the 11 dates this phase has ever evaluated on,
across every regime bucket (calm, moderate, crash, trending) except the milder of the two crash days.
The earlier fixed-λ table's "RL beats AC on calm dates" was not just an artifact on the 4 anchor
dates — it doesn't hold anywhere in the full date set once AC is fairly tuned.

**Reframing (explicit correction to how this result should be recorded):** this is not a settled
negative result about RL's ceiling — it is a diagnosable gap in this project's RL setup. AC-Optimal
at its best λ is *linear*-impact-misspecified for a simulator that is calibrated to *square-root*
impact (see the design note below) — it is provably not optimal here, only locally optimal within
the wrong model family. A correctly-functioning learner should, at minimum, be able to match a
misspecified closed-form heuristic on the heuristic's own home ground, since the learner faces the
true (sqrt) dynamics directly and isn't constrained to any particular functional form. That it
instead loses on 10/11 dates to a baseline operating on the *wrong* impact model means the shortfall
is in how this project's RL is set up (state, reward-to-schedule credit assignment, action space —
see the granularity investigation and the residual-action-space design sketch below), not evidence
that classical execution beats learned execution in general on this problem. Do not write this up as
"RL loses to AC, full stop" — write it up as "RL currently fails to close a gap that its own more
expressive setup should in principle be able to close, and here is the specific mechanism (bucketed
action space colliding with delayed-reward credit assignment) that plausibly explains why."

### Item 4 (greenlit, audit only): is Adaptive-AC's update frequency a hidden hardcoded parameter?

Read `AdaptiveACPolicy` in full (`rl/evaluate.py:426-483`). Findings:
- `predict()` is called once per environment step, and **every single call** re-derives
  `current_sigma` from the live state observation and recomputes+renormalizes the *entire* schedule
  from scratch (`_get_schedule`, cached only by `round(sigma, 2)`). There is no separate "update
  frequency" or "re-estimation window" parameter anywhere in the class — the update cadence is
  implicitly "every step" (up to 2880×/episode) purely as a byproduct of how often `predict()` is
  called, with no mechanism to decouple re-estimation cadence from the environment's step cadence.
  This is itself another instance of a parameter that was never exposed to be tuned — it isn't even
  hardcoded to a specific value, it's structurally absent.
- The input it reacts to (`current_sigma`, Rust `rl_env.rs:1337-1348`) is not raw per-tick noise — it
  is already a 50-sample rolling-window realized-volatility estimate, annualized. So Adaptive-AC is
  not literally chasing single-tick noise; it's reactively reshaping its entire multi-hour schedule
  every single step in response to a ~25-minute rolling vol estimate that itself moves continuously.
- This directly supports the pre-existing README finding (Phase 1-era, different/synthetic
  GBM+GARCH simulator, not this phase's calibrated Binance pipeline): "AdaptiveOptimal over-trades
  during GARCH volatility clustering (+1.22% IS vs +0.76%)." The mechanism this audit surfaces —
  full schedule reshaping on every step in response to a continuously-moving rolling vol estimate —
  is a plausible, structurally-supported explanation for that older finding, and the two are
  consistent even though they come from different simulators. **Not independently re-verified on the
  current calibrated pipeline here** (that would mean comparing Adaptive-AC's realized cost against a
  version with slower/smoothed re-estimation, which is new code — out of this audit's scope).
- **Answer to "should update frequency be a swept parameter rather than hardcoded": yes.** It is
  currently not a parameter at all (not hardcoded — simply absent), so there is nothing to sweep yet;
  adding an explicit re-estimation-interval or smoothing knob and sweeping it is a well-scoped,
  bounded follow-up (same cost order as item 1's grid) if this is prioritized. Logged as a new open
  item, not actioned.

### Item 2 (greenlit, audit only): mapping calibrated impact onto AC's (σ, η) — design note, no code

Full design note: `docs/phase2r_ac_calibration_design_note.md`. Summary:
- **σ: no modeling ambiguity.** The simulator already computes properly-annualized realized
  volatility per date (`realized_volatility` in `impact_fit.rs`, same units as the hardcoded
  `sigma=0.20`). Exposing and using it is a straightforward, low-risk fix whenever plumbing work is
  approved — does not need to wait on anything else.
- **η: genuinely ambiguous, not just unimplemented.** The simulator's calibrated impact law is
  square-root in participation; AC's `η` is the slope of a *linear* law. There is no single value of
  `η` that makes the linear model agree with the sqrt model except at one arbitrarily-chosen
  participation rate — three candidate ways to pick that rate were worked through (linearize at a
  reference participation, match total cost at TWAP, match the low-participation limit) and none is
  more "correct" than the others; each is a different modeling commitment with different blind spots,
  and picking one silently would be a subtler version of this phase's recurring bug class (a value
  that looks calibrated while still encoding an arbitrary, unstated structural assumption).
- **Recommendation:** calibrate σ now if/when plumbing is approved; treat η calibration as blocked on
  item 5's go/no-go rather than adopting a stopgap. The currently-running λ-grid (above) does not
  depend on η being fixed — it holds the linear-vs-sqrt mismatch constant and only sweeps risk
  aversion, so it answers "how good can AC get within its own currently-mismatched model," which is
  still the right question for deciding whether to invest in item 5 at all.

### Gate C — implementation-consistency audit (grep for mirrors, verify reward/config wiring)

- **Reward formula: verified identical in code to what the docs describe, and identical between
  `step()` (historical) and `step_counterfactual()`** (`src/bin/rl_env.rs:939-949` vs
  `:1124-1141` — byte-identical formula, no drift term, matches the "What the training reward
  actually is" section above). `step_batch` delegates to `step()` per instance — no third
  reimplementation. **No drift found here.**
- **Calibrated impact wiring: verified correctly threaded, not silently dropped.**
  `reset_counterfactual` (`rl_env.rs:834-853`) computes `calibrated` from real aggTrades/LOB data and
  rebinds it into the `cf_config` used to construct `CounterfactualLob` — confirmed via careful
  reading of the shadowing (`let cf_config = if ... { calibrated } else { cf_config };`), not an
  obviously-similar bug where the calibrated value gets computed and discarded. **No drift found.**
- **Third hand-maintained `ACTION_FRACTIONS` mirror — FOUND and FIXED.** `rl/environment.py` carried
  its own hardcoded copy of the fraction table (currently byte-identical to the real Rust table by
  luck, unprotected by any test), still imported by the retracted-but-tree-resident
  `rl/phase2r_vwap_gate.py`. Third instance of this exact bug class in the project. Root-fixed:
  `environment.py` is now the one canonical Python-side query point (queries the Rust binary at
  import time); `evaluate.py` imports `ACTION_FRACTIONS` from `environment.py` instead of querying
  independently. New test asserts identity (`is`, not `==`) so a fourth copy can't silently
  reappear. Committed locally (`2f129f5`), full 57-test suite green.
- **AC-Optimal / Adaptive-AC's own market-model parameters (`sigma=0.20`, `eta=0.10`) are hardcoded
  constants, never calibrated per date — a genuine, previously-undocumented drift between intent and
  implementation.** `ACOptimalPolicy.__init__` and `AdaptiveACPolicy.__init__` default `sigma=0.20,
  eta=0.10` and **no call site in `rl/evaluate.py` ever overrides either** — every construction only
  varies `lam`. Meanwhile the simulator's *own* per-date calibrated impact coefficient and realized
  volatility (`CounterfactualLobConfig::calibrated`, `realized_volatility` in
  `src/calibration/impact_fit.rs`, wired in during Task 2) are computed in Rust and never exposed to
  Python — there is no CLI flag analogous to `--print-action-fractions` for them. This means AC's
  closed-form schedule shape is derived from a generic textbook volatility/impact assumption that has
  no connection whatsoever to what the simulator actually calibrated for that specific date, on top
  of the un-swept λ from Gate B. This is a second, independent way the AC baseline in the close-out
  table was not fairly parametrized — not yet fixed here (would require a new
  `--print-calibrated-params` CLI flag plus re-running Gate B's grid with per-date sigma/eta, which
  is new scope, not part of this report). **Logged as a named open issue, not actioned.**
- No other hand-maintained numeric mirrors found. Checked: SB3 hyperparameters (`gamma=0.99` in
  `train.py` is PPO's discount factor, an unrelated concept from the impact model's `gamma` in Rust —
  confirmed by reading both, not a mirror of the same value); `AC_LAMBDA_GRID` (a grid definition, not
  a mirror of a canonical value); state-dimension wiring between `train.py` and `evaluate.py` (both
  key `n_state_dims`/`no_time_features` off the same convention, no divergence found).

### Net effect on Task 4's claims — corrected framing

**The "RL beats classical baselines" claim is not weakened by this gate — it was never supported in
the first place, and this gate is what establishes that, not what puts it in doubt.** The comparison
rested on three baseline parameters (λ, σ, η) that were each hardcoded rather than sourced or fitted,
and each of those three specific hardcoded values happened to handicap the comparator, not the RL
arm. A comparison built on an unfairly-parametrized opponent was never evidence of anything about RL
— the earlier table reporting "RL beats AC" was never a finding to begin with; it was a measurement
of how bad an under-tuned AC is, mislabeled as a measurement of how good RL is. Do not describe this
as the claim "needing more work" or being "weakened" — it is retracted, on the same footing as the
earlier objective-bias retraction elsewhere in this document, for the same underlying reason
(reporting a quantity that looked like the thing being claimed, but wasn't).

- The **16-dim vs 12-dim ablation itself is untouched by any of this** — it is RL-vs-RL, same action
  space, never routed through AC or λ at any point, and the numbers stand exactly as reported
  (16-dim beats 12-dim substantially on both held-out crash dates). This is the one surviving,
  supportable claim from Task 4.
- Two open items carried into `docs/phase2r_standing_issues.md`: (1) AC/Adaptive-AC's hardcoded
  `sigma=0.20`/`eta=0.10` never calibrated per date, and (2) the AC λ-grid's tested edge (1e-6) was
  still the best point on 3/4 dates, meaning even the "corrected" λ numbers in Gate B likely
  understate how strong a properly-tuned AC is.

### Methodological finding for the writeup (worth more than any single baseline number)

Four instances of the same bug class surfaced this phase, and every one biased the comparison toward
the project's own method:

1. **CVaR sign inversion** — several call sites reimplemented "CVaR-95" locally instead of using one
   sourced implementation; one variant had an outright sign inversion, averaging the *best* tail.
2. **`ACTION_FRACTIONS` mirror #1** (`rl/evaluate.py`'s `V5_FRACTIONS`/`V6_FRACTIONS`) — hand-copied
   instead of queried from the real environment; diverged silently, causing every classical baseline
   evaluated at `n_state_dims=12` to pick actions under a fraction table the environment never
   honored.
3. **`ACTION_FRACTIONS` mirror #2** (`rl/environment.py`) — a second independent hand-copy of the
   same table, found only by an explicit audit request, not by observed divergence.
4. **AC/Adaptive-AC's `sigma`/`eta`/`lambda`** — hardcoded instead of sourced (from the simulator's
   own calibration) or fitted (via a grid search that was actually run). Structurally the same
   failure as 2 and 3 — a value that should have been read from a canonical source or tuned, and was
   instead assumed — just applied to a baseline's own parameters rather than a shared constant.

**The common thread: every one of these four was a value duplicated, assumed, or left at a default
rather than sourced from its canonical origin or properly fitted — and every one, without exception,
biased the result in the same direction (toward the project's own method looking better).** None of
the four were caught by a passing test or a clean-looking result; all four were caught by treating a
suspiciously favorable or suspiciously inverted result as a reason to re-derive the number from first
principles rather than accept it. That is the transferable lesson for comparative ML research this
phase produced, independent of whatever the final RL-vs-AC number turns out to be: an unfavorable
baseline is not a nuisance to tune away, it is the default failure mode of any comparison the
researcher is incentivized to win, and it should be assumed present until specifically ruled out —
symmetrically, on the researcher's own method's assumptions too (see the square-root-vs-linear impact
gap noted as the next open thread — it applies to AC's optimality proof, not to RL, and hasn't yet
been checked for an equivalent bias against AC's *own* claimed baseline optimality).

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
- **Task 4 ablation (16-dim vs 12-dim RL): complete and stands.** 16-dim beats 12-dim on both
  held-out crash dates, unaffected by anything in the close-out gate below.
- **Task 4 close-out gate (AC λ-grid, granularity, implementation audit): reported, NOT closing the
  phase.** The "RL beats AC/Adaptive-AC" claim in the corrected 5-way table is **retracted, not
  merely weakened** — it rested on three hardcoded baseline parameters (λ, σ, η) that each
  handicapped the comparator, so it was never supported to begin with (Gate B, confirmed and then
  extended to all 11 dates: **best-tuned AC/Adaptive-AC beats RL-16dim on 10 of 11 dates**, RL
  winning only on the milder held-out crash date). Granularity/bucketing (Gate A) was first reported
  as second-order and later **corrected** once the grid was extended: it is a hard floor causing
  total execution failure at very low λ on every date, not a small effect — see the correction inside
  Gate A. A third hand-maintained `ACTION_FRACTIONS` mirror found and root-fixed (Gate C); AC's
  sigma/eta/lambda hardcoding and Adaptive-AC's structurally-absent update-frequency parameter are a
  fourth and fifth instance of the same duplicated-or-assumed-or-unexposed-value bug class — σ is
  cheap to fix (design note, `docs/phase2r_ac_calibration_design_note.md`), η has no principled fix
  short of item 5. See "Task 4 close-out gate" and "Methodological finding" sections above for the
  full report and the transferable lesson. **The 16-dim vs 12-dim RL ablation is the only surviving
  Task 4 claim about RL** — it never routed through AC or λ. **A properly-tuned classical baseline
  beating RL almost everywhere is itself the phase's headline empirical finding now, not an absence
  of one.**
- `interval_volume` / `book_mid` telemetry: retained as sound instrumentation.
- Nothing pushed; demo untouched throughout.
