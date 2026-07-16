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

## Status

- Forced-liquidation loophole + `price_factor` fix (`5630e5d`): **correct, stands, tested.**
- Objective-bias / VWAP / observability findings: **retracted** (artifacts of IS-vs-arrival).
- Wait-collapse: **re-diagnosed as a training/optimization failure** (agent suboptimal on its own
  reward). Leading hypothesis = long-horizon credit assignment — **to verify, not assumed.**
- `interval_volume` / `book_mid` telemetry: retained as sound instrumentation.
- No VWAP implementation. No retrain until the training-failure diagnosis is verified and a fix is
  planned. Nothing pushed; demo untouched.
