# Timeline

Entries in date order. Tags: **BUG**, **METHOD**, **CLAIM**, **PROCESS**, **DECISION**
(see [README.md](README.md)). Patterns are cross-referenced as **[L#]** from
[lessons.md](lessons.md).

---

## Before July 2026 — build-out (Feb–Jun)

The engine, research simulator, counterfactual LOB, v4/v5 RL models, web stack and
deployment were built in this period. Most of what follows was found later, when the
July audit (Phase −1/0) went back over it.

### 2026-06-11 · BUG · Dead "regime impact scaling" scaffolding broke training
- **What:** the "ui/ux integration" commit added a `no_regime_impact` parameter, a log
  line and an MLflow param for a crash=1.8× / quiet=0.7× feature that was never
  implemented. `StrataExecEnv` never accepted the parameter, so `train.py` crashed.
- **Found by:** getting `train.py` to run for Phase 2 (2026-07-11).
- **Impact:** together with the next two entries, `train.py` could not run at all in its
  committed form.
- **Fix / status:** removed. **Refs:** `bd54dec`. **[L4]**

---

## July 2026 — Phase 0 audit and Phase 1–2R

### 2026-07-10 · BUG · `train.py` was unrunnable three ways over
- **What:** besides the dead parameter above, `get_lr_schedule` was called but never
  defined, and import-time asserts on `cosine_with_restarts` failed (test inputs sat on
  the wrong side of a cycle boundary).
- **Found by:** first noted while adding MLflow (`40a9436`), fixed in Phase 2.
- **Fix / status:** all three fixed, 13 tests added. **Refs:** `bd54dec`. **[L5]**

### 2026-07-11 · BUG · CVaR sign inversion, in Rust and Python independently
- **What:** `DistributionStats::cvar_95` averaged the *best* 5% of outcomes (IS is
  negative-is-bad). `parallel_sim.py` had the same bug independently, with a test whose
  fixture encoded the wrong assumption. `evaluate.py` labelled VaR-5 as "cvar95".
- **Found by:** Phase 0 audit; `mean_is_pct=-0.73` sat next to `cvar95=+7.91` in one record.
- **Impact:** every reported tail number until then. `demo_results/*.json` still serves
  the old values.
- **Fix / status:** fixed, one shared helper. **Refs:** `b304b2c`. **[L1]**

### 2026-07-11 · METHOD · RL compared against baselines from a different simulator
- **What:** the default comparison table put Monte-Carlo (GBM/GARCH) TWAP/AC numbers next
  to LOB-replay RL numbers.
- **Fix / status:** the unified same-environment harness became the default; the mixed
  table is opt-in and labelled invalid. **Refs:** `04b0691`. **[L3]**

### 2026-07-11 · METHOD · The headline crash days were in the training set
- **What:** the documented `--multi-date` training run included 2024-08-05 and
  2024-03-05, the two crash days the "RL beats AC" claim was built on. 03-05 was also
  mislabelled "BTC breakout".
- **Impact:** the original crash-day results were partly memorisation.
- **Fix / status:** first holdout split declared, with an executable guard.
  **Refs:** `96170aa`. **[L3]**

### 2026-07-11 · BUG · AC risk aversion hard-coded and never reconciled
- **What:** Python AC used λ=0.001 with no override; Rust research-sim used 1e-4.
- **Fix / status:** λ became a real parameter. It later turned out to be one of three
  handicapped AC parameters (see 07-18). **Refs:** `04b0691`. **[L3] [L4]**

### 2026-07-11 · BUG · AC schedule produced NaN at high λ
- **What:** `sinh(κT)` overflowed to inf, and inf/inf gave NaN. The trajectory-level test
  still passed because the average direction held.
- **Fix / status:** `_stable_sinh_ratio`. **Refs:** `0c9a024`. **[L5]**

### 2026-07-11 · BUG · Counterfactual training silently discarded the time features
- **What:** `n_state_dims` was hard-coded to 12 for counterfactual mode and
  `no_time_features` was never threaded through, so every counterfactual run trained on
  12 of the 16 dims the env emitted.
- **Fix / status:** threaded through; verified the model's observation space is 16.
  **Refs:** `bd54dec`. **[L4]**

### 2026-07-11 · BUG · MLflow never recorded a metric
- **What:** `start_run`/`log_params` existed, but there were zero `log_metric` calls
  anywhere, so `get_best_run()` always returned `{}`.
- **Fix / status:** callbacks plus eval logging. **Refs:** `07627d5`. **[L4]**

### 2026-07-13 · BUG · `n_envs` hard-coded to 8
- **What:** `train()` ignored its own `n_envs` parameter; this surfaced during an
  out-of-memory incident.
- **Fix / status:** respected and exposed; resume support added. **Refs:** `c37ad4a`. **[L4]**

### 2026-07-13 · BUG · Forced-liquidation loophole made "never trade" optimal
- **What:** the end-of-episode dump cost a flat `gamma × fraction` (0.5% for 100%),
  cheaper than real slippage. The v6 policy converged to action 0 everywhere. A second
  bug, `price_factor` recomputed each snapshot, erased real price drift in historical
  mode.
- **Impact:** the v6 training collapse. v5 was verified unaffected (it never triggered
  either path).
- **Fix / status:** the dump walks the real book; invariant test "never trading must not
  beat execution". **Refs:** `34276ae`. **[L2] [L7]**

### 2026-07-14 · CLAIM · "The objective rewards directional speculation" — retracted
- **What:** analysis concluded the reward made waiting optimal on trending days, and a
  VWAP-benchmark fix was proposed.
- **Found by:** reading the reward code. The per-step reward is slippage against the
  *contemporaneous* mid, so it is already drift-neutral; the analysis had measured
  IS-vs-arrival, a reporting metric.
- **Impact:** the VWAP fix and the "state-observability limit" were withdrawn the same day.
- **Refs:** `a8850eb` → `c62ab68`. **[L2]**

### 2026-07-14…16 · METHOD · Wait-collapse root cause chased through several wrong suspects
- **What:** credit assignment, then learning rate, then resume mechanics were each
  proposed and tested. The collapse was a late-training slide of a good policy (the
  900k checkpoint was reward-optimal). It did not reproduce on CPU in any of 3 runs.
  MPS (Apple GPU) numerics remain the leading, unconfirmed cause.
- **Also:** best-model selection scored only one trending-up date, and a resume reset
  `best_mean_reward` to −inf, overwriting the good model.
- **Fix / status:** **DECISION:** train on CPU. Multi-date reward selection and
  resume-safe best model added; entropy floor added but never engaged (unvalidated).
  **Refs:** `8d2224b`, `013ae18`, `8c354a5`, `aee11f7`. **[L6]**

### 2026-07-15 · METHOD · The eval data-exhaustion problem was spotted and dismissed
- **What:** a code check noted that a 2,880-step eval horizon does not fit the day for
  start offset > 0, and concluded "evals use small seeds (offset 0–19, <1%), so
  unaffected". That reasoning only held for strategies that finish on time.
- **Impact:** see 2026-10-18; Adaptive-AC and the sqrt-impact baseline were flattered for
  three months.
- **Refs:** `e0057e3`. **[L9]**

### 2026-07-16 · CLAIM · Impact model mislabelled "Obizhaeva-Wang"
- **What:** the decay structure is OW-style, but the magnitude is a square-root law with
  an uncalibrated 0.1 constant and no σ term.
- **Fix / status:** relabelled everywhere; per-date calibration wired in.
  **Refs:** `319ec16`, `aceb3e6`. **[L4]**

### 2026-07-16 · BUG · Counterfactual mode flattened price drift (mirror of the 07-13 fix)
- **What:** `CounterfactualLob::rescale_book` recomputed `price_factor` every snapshot,
  the same bug fixed for historical mode three days earlier. State dim 6 (drift) was
  therefore constant in training.
- **Fix / status:** fixed; dim 6 now varies. **Refs:** `8f270ee`. **[L1]**

### 2026-07-16 · BUG · `spread_ref` computed at the wrong scale in counterfactual mode
- **What:** raw ~$40k-scale spread against a ~100-scale book. This collapsed state dim 3
  and floored the punitive dump price at about $0.
- **Found by:** a "0% punitive share" measurement that looked reassuring but was a red flag.
- **Fix / status:** fixed, with a regression test. **Refs:** `92615ee`. **[L1]**

### 2026-07-17 · BUG + CLAIM · Hand-copied action tables diverged; "16-dim flips the loss" retracted
- **What:** `V5_FRACTIONS`/`V6_FRACTIONS` in `evaluate.py` mirrored Rust's
  `ACTION_FRACTIONS` and had drifted (V6 matched at only 3 of 20 indices), so every
  baseline executed a different schedule than intended. A third copy in
  `environment.py` was found the next day.
- **Impact:** the "16-dim turns RL's loss to AC into a win on 08-05" finding was
  retracted. The RL-vs-RL ablation was unaffected.
- **Fix / status:** Python now queries the table from the binary
  (`--print-action-fractions`). **Refs:** `568ca74`, `df388ba`, `68cf9aa`. **[L1]**

### 2026-07-18…19 · CLAIM · "RL beats classical baselines" — retracted, not "unresolved"
- **What:** AC's λ, σ and η were all hard-coded, and each one handicapped AC. Tuning λ
  alone made AC tie or beat RL on 3 of 4 anchor dates.
- **Fix / status:** reworded as a retraction at the user's correction. **Refs:** `3b16823`, `6aaf1be`. **[L3]**

### 2026-07-20 · CLAIM · "Granularity is second-order" — wrong at the low end
- **What:** at λ ≤ 3e-7, AC's whole schedule rounds into the zero bucket, which is total
  execution failure; AC's optimum sits just above that floor.
- **Fix / status:** corrected in place. **Refs:** `fb37f24`.

### 2026-07-20 · CLAIM · "AC beats RL on 10/11 dates" — reversed; the λ choice was hindsight
- **What:** AC's λ had been picked per date using that date's own outcome, which RL never
  gets.
- **Fix / status:** **DECISION:** the honest-λ protocol (one λ = 1e-6 from training dates
  only). Result: RL wins both held-out crash dates and loses the rest, which became the
  phase headline. **Refs:** `ef3a24f`, `f20d135`. **[L3]**

### 2026-07-20 · PROCESS · Capped hybrid reported as a win while omitting it lost to RL
- **What:** "beats AC on 03-05" left out that it was second-best behind pure RL there.
- **Fix / status:** reframed. **Refs:** `ef3a24f`. **[L8]**

### 2026-07-30 · CLAIM · Event-window "2/2 crash sweep" did not replicate; narrowed to severity scaling
- **What:** over 17 event-window days, RL beat AC on only 8 of 17. Ranking days by
  severity showed the top 7 were 7/7 wins.
- **Also:** 2024-04-15 was folded into its window up front so it couldn't be counted as
  an independent event.
- **Refs:** `f7a4699`, `aba7896`. **[L8]**

---

## August 2026 — Phase 3

### 2026-08-13 · BUG · Regime dims dead for two compounding reasons; recorded mechanism was wrong
- **What:** `sigma_ref` was a hard-coded 0.20 and volatility annualisation was ~19× off,
  so `vol_regime` was pinned LOW on every date. Fixing either bug alone just moved the
  dim to the other dead rail. The standing-issues doc had attributed this to a different
  cause.
- **Fix / status:** calibrated `sigma_ref`, fixed annualisation, made the dims
  continuous. s11 diagnosed as a genuine null result (`763d90e`). **Refs:** `ad23467`. **[L4]**

### 2026-08-13 · DECISION · V2 date split (25 dates, 16 train / 9 held-out)
- **Why:** V1's training set had 0 of 7 severe crash days. V2 balances crash exposure
  (4/4), keeping event windows whole or split at one boundary, and holds event C out
  entirely.
- **Refs:** `7861d1e`.

### 2026-08-13 · BUG · Self-inflicted mirror bug right after writing the mirror-bug lesson
- **What:** Step 0 recalibrated `sigma_ref` in Rust but left two Python copies, one live
  and one dead. Adaptive-AC also inverted the wrong formula (`1 + …`), a pre-existing bug
  that had been cancelling the annualisation bug, so fixing one made AC pathologically
  patient (−9.08% on a crash day). An earlier ±0.06 pp impact estimate from a 400-step
  probe was retracted.
- **Fix / status:** `--print-env-constants`; AST test for literal `sigma_ref` defaults.
  Adaptive-AC comparisons blocked until a common binary existed.
  **Refs:** `a3dfa2f`, `2f89d18`. **[L1] [L8]**

### 2026-08-13 · METHOD · Old checkpoints on a new binary silently change results
- **What:** RL16 (trained before Step 0) evaluated on the post-Step-0 binary moved
  2024-08-05 from −6.48% to −4.49%, with no error.
- **Fix / status:** PROVENANCE boundary; AC/TWAP used as a canary; binary selected via
  `STRATAEXEC_RL_ENV_BIN`. **Refs:** `ae90ac1`, `3cabcdd`. **[L7]**

### 2026-08-13 · METHOD · Best-model selection used the held-out crash dates
- **What:** checkpoint selection scored on the held-out crash dates, so RL16's "held-out"
  wins were partly selected for.
- **Fix / status:** training-date-only selection, with a loud fallback.
  **Refs:** `23ef4dd`. **[L3]**

### 2026-08-13 · METHOD · Small sample overstated a result ~4×
- **What:** a 1-seed, 2-date smoke put RL16's regime responsiveness at +0.10; at 9 dates
  it was +0.028. **Refs:** `2ed3881`. **[L8]**

### 2026-08-16 · BUG (critical) · Training saw 3 of 16 dates and zero crash dates
- **What:** `random.choice(multi_dates)` ran once per SubprocVecEnv worker, pinning each
  worker to one random date. This affected every multi-date run in the project,
  including RL16.
- **Impact:** the Step 2 negative result was retracted, because it measured a model that
  had never seen a crash.
- **Fix / status:** round-robin date ownership, with a per-episode redraw fallback.
  **Refs:** `969ea4a`, `201b238`. **[L4] [L6]**

### 2026-08-16 · METHOD (critical) · Eval episodes are near-duplicates
- **What:** seeds 0–19 start within 9.5 minutes of each other out of 8 hours of possible
  starts, so 20 "episodes" replay almost the same path. Crash-day variance was
  understated ~30×, and p-values are not credible (effective n ≈ 1).
- **Why missed:** standing issue #5 framed start offsets purely as a bounds question.
- **Fix / status:** OPEN (fixing needs a shorter horizon or multi-day data).
  **Refs:** `3ce7164`. **[L8] [L9]**

### 2026-08-16 · BUG · Two defects recorded, not fixed
- **What:** (a) `CALIBRATED_SIGMA_REF` was fitted on V1 but used with V2 (~22% low);
  (b) training horizons of 200–1,440 steps vs a 2,880-step eval horizon.
- **Fix / status:** OPEN. **Refs:** `201b238`.

### 2026-08-16 · BUG · Policy underfit, at 98% of uniform entropy
- **What:** at 500k steps with `ent_coef=0.05` the policy barely sharpened.
- **Fix / status:** Step 4 (1.5M steps, 0.01). Timesteps and `ent_coef` were changed
  together, which is a confound. **Refs:** `50c4edf`, `63f47d3`. **[L6]**

### 2026-08-22 · METHOD · Distant fit seeds produced a baseline worse than TWAP
- **What:** one date's mean IS varied −0.7% to −9.7% depending on which seeds were
  sampled; fit seeds must sit adjacent to the eval range.
- **Refs:** `168e4ce`. **[L8]**

---

## September 2026 — Steps A and B

### 2026-09-01 · METHOD · The headline table covered only 4 of the 9 held-out dates
- **What:** two classified held-out crash dates had never been scored for IS.
- **Fix / status:** the full 9-date eval became the headline: 4/4 crash wins, 0/5
  calm/normal. **Refs:** `b4e45ac`.

### 2026-09-07 · CLAIM · Imitation warm-start from AC — negative result
- **What:** cloning AC removed the calm-day loss and the crash-day edge together.
  Whether that was regime-blind imitation or too low a learning rate was left
  unresolved. Held-out dates had now been looked at 3 times (test-set erosion).
- **Refs:** `b157144`. **[L6] [L8]**

---

## October 2026 — refactor and simulator audit

### 2026-10-17 · BUG · CI's zero-warnings gate could never fail
- **What:** it re-ran a build that was already cached, which printed nothing. The Python
  smoke job also imported a removed name and never had the binary.
- **Fix / status:** `RUSTFLAGS=-D warnings`, `cargo fmt --check`, a working smoke job.
  First real CI run passed on 2026-10-18. **Refs:** `dc6c4ad`. **[L5]**

### 2026-10-17 · BUG · Every run logged the wrong LSTM size
- **What:** MLflow recorded 128×2 from unused parameters; the model was built as 64×1.
- **Fix / status:** one constant feeds both; historical `lstm_*` params are wrong (read
  them as 64×1). **Refs:** `d941550`. **[L4]**

### 2026-10-17 · BUG · `--jump-diffusion` was a silent no-op; the parser accepted anything
- **What:** the binary ignored unknown flags, mapped unknown `--rl-mode` values to
  synthetic, and swapped bad numbers for defaults. Training printed "jump events ACTIVE"
  while nothing happened.
- **Fix / status:** strict parser (exits 2); flag removed. **Refs:** `364f205`. **[L4]**

### 2026-10-17 · BUG · Tests and sweeps overwrote the tracked `results/comparison.csv`
- **What:** every runner exported by default, including thousands of Monte Carlo sweep
  paths. **Refs:** `7034fa8`; flaky fixture race in `2855072`. **[L4]**

### 2026-10-17 · BUG · `step()` and `step_counterfactual()` had drifted apart
- **What:** ~120 duplicated lines with different info fields; the adversary existed only
  in one copy, so `--adversarial` did nothing in counterfactual mode.
- **Fix / status:** shared helpers (`7c28794`, `8fef77e`). The flag is rejected in
  counterfactual mode as of 2026-10-18. **[L1]**

### 2026-10-17 · BUG · A crate-wide `allow(dead_code)` hid dead code
- **Refs:** `25fd2c6`; the binary also compiled the library twice (`d1d348d`). **[L5]**

### 2026-10-18 · BUG · Forced dump at data exhaustion priced against an empty book
- **What:** fixed-horizon eval episodes ran past the day's last snapshot, so the leftover
  inventory was sold at the cheap punitive fallback against an empty book. A never-trade
  policy scored −0.26% IS, better than AC's −0.42%.
- **Impact:** AC and RL were unaffected (bit-identical). Adaptive-AC was flattered by
  0.14 pp, and the sqrt-impact baseline by 0.53 pp, because its fit learned to exploit
  the bug on training-date rallies. This sibling of the 07-13 loophole had been noted on
  07-15 and dismissed.
- **Held-out check:** a single re-run of Step A confirmed AC/RL bit-identical on all 9
  held-out dates and Step 3's "sqrt-impact beats AC 9/9" intact; only Adaptive-AC moved
  (up to −0.46 pp). An initial "Step 3 is now doubtful" warning was too pessimistic.
- **Fix / status:** dump into the last real book. **Refs:** `32fa384`, `11199af`,
  `docs/experiments/12-…`. **[L2] [L9]**

### 2026-10-18 · PROCESS · Mistakes made during the refactor session itself
- A commit swept in 20 doc renames that were already staged; undone before anything
  built on it.
- `--help` passed to a script with no argument parser ran the whole experiment and
  rewrote a results CSV; restored from git.
- A before/after diff used the wrong baseline binary (two commits old), so every episode
  "changed"; re-run against the true HEAD.
- **Lesson:** stage only named paths, check a script takes arguments before passing any,
  and build the comparison baseline from the exact parent commit. **[L10]**

### 2026-10-18 · DECISION · Commits carry only the author's identity
- Claude co-author and session trailers were removed from all 116 unpushed commits
  before the first push; the code tree was verified identical.

### 2026-10-18 · PROCESS · README told users to train adversarially in counterfactual mode
- **Fix / status:** corrected, and the combination is now rejected. `ppo_adversarial_v1`
  has no MLflow record, so whether it was ever adversarial is unknown. **Refs:** `50d6004`.

### 2026-10-18 · BUG (critical) · Every training episode replayed the same seed
- **What:** `StrataExecEnv.reset(seed=None)` reused the constructor seed (42 for every
  worker), and SB3 calls `reset()` without a seed after each worker's first episode. The
  simulator derives start, horizon and order size from the seed, so every training
  episode began at 00:21 UTC, lasted 853 steps (to ~07:27 UTC) and used a 58.85% order,
  on every date. The documented "random 200–1,440 horizon, random size" never happened.
- **Found by:** tracing how start points flow through the code while implementing
  protocol v3; confirmed with a real SubprocVecEnv seeded like `RecurrentPPO(seed=42)`
  (55 episodes, one distinct configuration after the first).
- **Impact:** every model trained since 2026-05-22, including the Step 4 headline
  checkpoint, saw 16 fixed morning paths and was evaluated on full days. AC numbers are
  unaffected; RL has been training on far less variety than assumed.
- **Fix / status:** per-env seed stream (gymnasium semantics), distinct seed per worker,
  regression test `test_seedless_resets_draw_new_episodes`. M0 must be retrained.
  **[L4] [L5]**

### 2026-10-18 · METHOD · Open-loop AC baseline traded at the action table's smallest bucket
- **What:** at honest λ=1e-6, AC's per-step target was below the smallest non-zero action
  (0.1% of remaining), so its real pace came from rounding, not its schedule. It finished
  under the legacy 2,880-step horizon by luck; at 1,920 steps it left ~5% to dump.
- **Found by:** a new protocol-v3 test asserting AC never force-liquidates.
- **Impact:** legacy AC numbers describe this quantized policy, not textbook AC.
- **Fix / status:** v3 baselines track their schedule closed-loop (`schedule_target`);
  legacy path unchanged. The honest λ must be re-selected under v3. **[L3] [L5]**

### 2026-10-18 · DECISION · Evaluation protocol v3
- Fixed 16-hour episodes for training and eval, starts spread across the valid window,
  4 validation dates carved from training, held-out dates used once at the end.
- **Refs:** `docs/methodology/evaluation-protocol-v3.md`.

### 2026-10-18 · METHOD · Argmax decoding misrepresented the RL policy
- **What:** SB3 `deterministic=True` takes the most likely action. M0's policy puts ~30% on
  "wait" and ~70% across the sell actions, so argmax waited on 60–84% of steps and dumped
  10–45% of the order at the end. In-training selection used the same decoding and chose
  the near-random 50k checkpoint.
- **Found by:** M0's validation reward never beating its 50k checkpoint, then a
  checkpoint sweep showing the trained policy force-liquidating only under argmax.
- **Impact:** every published RL number used argmax; whether legacy checkpoints were
  affected is OPEN.
- **Fix / status:** `MeanFractionDecoder` (nearest action to the expected fraction) and
  offline checkpoint selection on validation IS for all v3 models.
  **Refs:** `docs/experiments/13-m0-protocol-v3-result.md`. **[L2] [L5]**

### 2026-10-18 · PROCESS · M0's first run stopped at 2 minutes; long runs slept
- Selection rounds compared different episode mixes; the run was stopped and relaunched
  with the fix. The relaunch then lost ~4 h 20 min to the laptop sleeping. **Lesson:** run
  long jobs under `caffeinate -i -w <pid>` on mains power. **[L10]**

### 2026-10-19 · CLAIM · M1 (γ 0.999) negative; its design missed GAE's horizon
- **What:** M1 raised γ to lengthen the planning horizon, but PPO's credit horizon is set
  by GAE, ~1/(1−γλ): with λ = 0.95 it moved only from ~17 to ~20 steps, while the value
  target became a noisier 1,000-step sum. M1 learned slowly and at 1.5M still dumped the
  order in the first 5% of the episode (validation IS −1.27% vs M0's −0.47%); 1 of 5
  predictions met, and that one for the wrong reason.
- **Found by:** the pre-registered comparison, then checking the actual algorithm.
- **Fix / status:** M1 not adopted; M0 remains best. **Lesson:** check a mechanism claim
  against the algorithm's real parameters before spending a training run on it. **[L9]**
- **Refs:** `docs/experiments/14-m1-discount-result.md`.

### 2026-10-20 · CLAIM · M2 (residual around AC) negative: learned to lag and burst
- **What:** with AC's execution underneath, RL only had to pick a schedule offset. It
  learned to fall behind and catch up in bursts (up to 4.9% of the order per trade,
  ~300 offset switches per episode), losing 0.20–0.29 pp to AC on every validation date
  (0 of 6 predictions met). Validation got worse with training while training reward
  improved.
- **Cause:** lagging is free on the step it happens and costs only at the later catch-up,
  beyond PPO's ~17-step credit horizon (GAE λ 0.95). The same blind spot explains M0's
  fixed early selling and M1's failure. **[L2] [L6]**
- **Refs:** `docs/experiments/15-m2-residual-ac-result.md`.

### 2026-10-21 · CLAIM · M3 (γ 0.999, λ 0.99) not converged at 3M; 0 of 5 predictions met
- **What:** the ~91-step credit horizon learns steadily (validation IS −2.23% → −0.92%,
  steepest at the end) but at 3M still sells the order in the first ~8% of the episode,
  like M0 at ~0.5M. Predictions assumed it would pass M0 within the budget.
- **Lesson:** the training cost of a longer credit horizon was underestimated twice (M1,
  M3); budget runs from the observed learning curve, not from the base recipe. **[L8]**
- **Refs:** `docs/experiments/16-m3-credit-horizon-result.md`.

### 2026-10-21 · DECISION · Credit-horizon line closed at 6M
- M3 continued to 6M under a stopping rule fixed before resuming. Best −0.634% (M0
  −0.465%): rule triggered, line closed. At 6M M3 matches M0's profile (crash win from
  early selling, worse execution than AC everywhere). **Refs:** doc 16.

### 2026-10-22 · CLAIM · "RL can't see its own delayed impact cost" — withdrawn by M4's pre-check
- **What:** after M3, RL's execution gap was explained as delayed impact cost beyond PPO's
  credit reach, and presented to the user with confidence before being measured. M4's
  pre-check (fixed rule: impact must predict cost-to-go at |t| > 5) found t = −0.27 over
  36,449 training-date observations; remaining inventory alone explains R² 0.79.
- **Fix / status:** M4 not trained; explanation withdrawn; the execution gap is open.
- **Lesson:** a plausible mechanism is a hypothesis until a cheap test supports it — say
  so when explaining it. The pre-check rule worked as intended. **[L9]**
- **Refs:** `docs/experiments/17-m4-impact-shaping-precheck.md`.

### 2026-10-22 · METHOD · Value-function diagnostic compared the critic with the wrong policy
- **What:** diagnostic B compared M0's critic (trained on the sampling policy) with
  returns of the decoded deterministic policy and read corr 0.13 as "bad critic". PPO's
  own explained variance (0.70–0.93 throughout training) shows the critic was fine.
- **Fix / status:** caught before acting on it; recorded in doc 18. **[L2]**

### 2026-10-22 · METHOD · M4's best validation score is mostly selection, not skill
- **What:** M4's selected 200k checkpoint (−0.311%) beat AC and M0, but a constant 1.5×
  pace scores −0.245% on the same episodes. The selected checkpoint is near its random
  start (fast seller); picking the best of 16 checkpoints on four falling validation
  dates favours speed. Later checkpoints degrade into pausing and forced dumps.
- **Lesson:** always compare a learned policy with the simple fixed baselines its action
  space makes available, before reading a gain as learned. **[L3] [L8]**
- **Refs:** `docs/experiments/19-m4-rate-actions-result.md`.

### 2026-10-23 · BUG (critical) · The simulated order book has a ~45 bp cost floor
- **What:** Binance `bookDepth` gives cumulative depth in ±1..5% bands, not an order book.
  The simulator treats the 1% band as the best level, so every sale pays ~43–51 bp below
  mid (simulated spread 86–103 bp vs ~0.02 bp real), and IS is identical for orders of
  500 to 50,000 BTC. Every strategy's ~0.46% "execution" cost is this floor.
- **Found by:** v3.1 baselines giving identical results at 0.5%, 1% and 2% of ADV.
- **Impact:** all execution-cost comparisons since the counterfactual simulator was built;
  strategy differences reduce mostly to timing. Doc 06 had flagged `half_spread()` as a
  weak proxy for this format without drawing the consequence. **[L9]**
- **Also:** an earlier claim that the fixed 240k ADV overstated impact up to 5× was wrong
  (calibration absorbs it exactly) — caught by testing before acting.
- **Status:** OPEN. **Refs:** `docs/experiments/20-order-book-floor.md`.

### 2026-10-26 · DECISION · Era 1 closed; simulator moves to Bybit's real order book
- Binance `bookDepth` bands could not represent institutional execution (doc 20). The era is
  summarised in [era-1-binance-bookdepth.md](era-1-binance-bookdepth.md) and tagged
  `era-1-binance-bookdepth`; its raw data was deleted to free space (re-downloadable).
  Era 2 uses Bybit's free 500-level L2 book and Bybit trades, one venue end to end.

### 2026-10-26 · BUG · Final sliver of every order was never sold
- **What:** an episode ended as "done" once ≤0.1% of the order remained, but that remainder
  was neither sold nor charged — IS counted it as received at zero.
- **Found by:** first AC run on the real Bybit book: a 100-BTC order showed 0.078% of IS
  that was neither execution nor timing.
- **Impact:** every era-1 result for strategies that finished early (AC, TWAP, fixed
  schedules, most RL episodes) was too pessimistic by up to 0.1 pp; episodes ending by
  the time limit were force-liquidated and unaffected. Small against era 1's ~45 bp floor
  and mostly common to all strategies; era-1 numbers are not re-run (data retired).
- **Fix / status:** FIXED — the sliver is sold into the current book and reported as
  `dust_liquidated_qty`.

### 2026-10-26 · METHOD · Forced dump beyond the visible book priced at a flat penalty
- **What:** quantity a forced liquidation could not fill against the visible book was
  priced at a fixed 5 × spread below the average fill — with a 0.02 bp real spread, deep
  sales would have looked nearly free.
- **Fix / status:** the book is now assumed to continue below its last visible level at
  the visible book's own depth per unit of price, so larger residuals sell progressively
  deeper (unit tests in `rl_env` `book_continuation`). The flat penalty remains only for
  a book with no usable depth.

### 2026-10-26 · BUG · Our own selling never moved the price
- **What:** the counterfactual impact model reduced bid depth rather than shifting prices; its Y was
  fitted by attributing every 30 s move to every trade (Y = 7–11, a volatility measure); and per-slice
  square roots were summed.
- **Found by:** AC's mid path was identical at 100 and 20,000 BTC on the real Bybit book.
- **Impact:** in era 1 the effect of our own impact on the price was missing. Pace-vs-impact trade-offs
  (the core of AC) could not show, which adds to the timing-only findings of doc 20.
- **Fix / status:** FIXED. Square-root metaorder impact on the decayed accumulated quantity, applied as
  a price shift; Y = 1.07 fitted on signed flow (doc 21).

### 2026-10-26 · DECISION · Era-2 baseline: AC = TWAP; ladder sized at 5% of ADV
- Under price-moving impact, honest-λ AC selects λ → 0 on training dates, which is TWAP (doc 22).
  The target becomes "beat TWAP in every regime". The ladder runs at 5% of Bybit ADV, with 1%
  and 10% as sensitivity checks. The best per-date pace in hindsight is worth +0.37 pp mean
  (crash +0.76), which makes a regime-conditioned pace the most promising rung.

### 2026-10-27 · BUG · Counterfactual reward did not charge our own impact (mirror of 10-26)
- **What:** the step reward, forced-liquidation and dust settlements were charged against the
  mid of the impact-shifted book. Once impact moved prices (10-26), the price drop caused by
  our own selling disappeared from the reward; only spread and depth (~0.06%) remained.
- **Found by:** M0-e2 (doc 23) lost to TWAP on every regime with a near-constant −0.6% to −1.2%
  IS. Cost split: execution −0.06%, the rest our own impact; the policy finished in ~60 of
  1,920 steps.
- **Impact:** the M0-e2 run is invalid as a test of the recipe. Evaluation IS, fills and
  prices were never affected (golden IS bit-identical); only counterfactual rewards changed.
- **Fix / status:** FIXED. Rewards are charged against the undisturbed mid (the real snapshot
  without our impact), which keeps market drift out of the reward, as before. Regression test
  `counterfactual_reward_charges_our_own_impact` fails without the fix. Golden regenerated
  (counterfactual rewards only). Lesson: the 10-26 impact fix should have triggered a grep of
  every place that reads the book mid — the mirror-bug pattern, fourth occurrence.

### 2026-10-27 · BUG · Rust data tests silently skipped since era-1 data was deleted
- **What:** 12 `rl_env` tests still pointed at deleted `TradeData/BookDepth/...` files and
  returned early with "skipping", which counts as passed.
- **Fix / status:** FIXED. Pointed at the Bybit files; all run and pass. Python data tests had
  been moved to the dataset registry on 10-26; the Rust copies were missed (mirror pattern).

### 2026-10-27 · RESULT · Era-2 RL baseline (M0-e2b): wins crashes, loses calm, sells fast
- M0's recipe on Bybit at 5% ADV, with the reward fixed: validation −0.207% vs TWAP −0.299%;
  by regime vs TWAP, crash +0.49 pp, normal +0.01, calm −0.67 (doc 23). Finishes in 3.8–7.4 h (median start)
  of 16 h on every date, so the gain is front-loading on falling days, not regime-aware pacing.
  Still improving at 1.5M steps. Next rung: TWAP-relative reward (M1).

### 2026-10-28 · RESULT · M1 (TWAP-relative reward) negative: market moves drown the pacing signal
- Charging fills against arrival minus TWAP's market-move gain (doc 25) made the agent sell
  faster (median 281 vs 647 steps), with calm −0.81 pp vs TWAP, and training stalled at
  −0.43% validation from 400k steps. Measured cause: on training dates, the per-episode
  advantage of TWAP's pace over 3× is +0.14 pp mean against 0.92 pp sd (mean/sd 0.15).
  The drift-free mid reward stays the default; M0-e2b remains the best era-2 model.
- **Lesson:** a reward that is right in expectation can still be unlearnable at the sample
  budget. Check the signal-to-noise of a proposed reward from baseline data before training
  (minutes), not after (an hour).

### 2026-10-28 · DECISION · Regime belief head not trained: state cannot predict regime or drift
- Before training, fit the belief head's targets offline (ridge / boosting, 12 training dates →
  4 validation dates, doc 26). Forward drift: validation R² −0.21 to −0.67, worse than
  predicting zero (−0.11). Regime class: 0.13–0.16 balanced accuracy, below chance (0.33),
  against 0.84–0.95 on the training dates. The head would memorise days, not read regimes.
- Found on the way: state dim 4 (depth within 1%) is saturated at 1.0 and dim 6 (10-step
  drift) has sd ≤ 0.004 in era 2, so the only trend input is dead.
- **Lesson:** the doc-25 lesson paid off. A ~20-minute offline fit replaced a training run
  whose result it predicts. "Regime" is a hindsight label (did the price fall), so
  "regime-aware" means forecasting direction.

### 2026-11-24 · BUG · State dim 4 (depth within 1%) saturated by a unit mismatch in counterfactual mode
- **What:** `depth_ref` is measured on the raw book (BTC units), but the state divides the
  rescaled book's depth (raw × 1/btc_scale, ~128× at 5% ADV) by it, so tanh saturates at 1.0
  on every step. Historical mode multiplies `depth_ref` by `qty_factor`; the counterfactual
  path does not. Same class as the earlier `spread_ref` scale fix in that function (mirror
  pattern).
- **Found by:** belief-head pre-check (doc 26) showed dim 4 = 1.000 on all dates; on
  2024-01-15, depth_ref 57,554 vs adjusted depth 52,281 should read tanh(0.91) ≈ 0.72.
- **Impact:** every counterfactual-trained model (era 1 and era 2) had no book-depth input.
  This was the one dim meant to carry the real order book, the reason era 2 moved to Bybit.
- **Fix / status:** FIXED. depth_ref is now in simulation units. The reset book is rescaled too:
  it was raw, so dim 3 read 1.0 on every first observation. Test
  `counterfactual_book_state_dims_use_simulation_units` (mutation-checked against each fix).
  Golden regenerated: only trajectory hashes changed; rewards, IS and TWAP/AC are bit-identical,
  and the dim-4-reading Heuristic and RL model moved as expected. Existing models were trained
  on the saturated dim and are not comparable; the next trained model uses the fixed state.

### 2026-11-24 · MISTAKE · Target leaked into features in the doc-27 pre-check (caught before reporting)
- **What:** `fall_240` was added to the table before `feature_columns()` ran, and the filter
  excluded only `fwd_*`, so the label was an input: AUC 1.00, and validation R² +0.38 for the 2 h return.
- **Found by:** AUC exactly 1.00 on every date, and `fall_240` topping the univariate list.
- **Fix / status:** FIXED. Every target prefix is excluded, an assertion guards it, and the run
  was redone. **Lesson:** a perfect score is a bug until shown otherwise. Derive targets
  after the feature list is fixed, or exclude them by name.

### 2026-11-24 · RESULT · Full-book regime features fail the pre-registered check (doc 27)
- 30 real-time book, trade and price features: 2 h return validation R² −0.04 (ridge); fall-event
  AUC 0.62 on validation but 0.44 mean in leave-one-date-out (2/12 dates ≥ 0.55). Fails
  both criteria, so no belief head. Weak exploratory hint of mean reversion after drops (9/12
  dates). With doc 26, direction is not forecastable from free Bybit data at 1–2 h horizons.

### 2026-11-24 · RESULT · Impact does not scale with local liquidity; daily model stays (doc 28)
- Pre-registered fit on 30 s signed flow: trailing-hour volume normalisation R² 0.365 vs 0.441
  for the daily model (validation); trailing volume + volatility ties (+0.002). Within a day,
  log σ and log V correlate 0.92, so the impact cost per unit order (σ/√V) varies only ±16%:
  busy hours are not cheaper to sell into. Fails the pass criterion.
- Liquidity timing joins direction forecasting (docs 26, 27) as a route to a regime-aware
  edge that this data does not support. Untested: regime-dependent impact decay; more dates.

### 2026-11-24 · MISTAKE · CI red on main for four weeks (formatting), unnoticed
- **What:** every push from 6c8d42c (2026-10-26) through 0acf144 failed CI at `cargo fmt --check`,
  on Rust code written in this period (config.rs, reset.rs, tests.rs). The tests themselves passed.
  Because python-smoke depends on rust-tests, it was skipped on all eleven runs.
- **Found by:** the user forwarded the CI failure emails.
- **Fix / status:** FIXED. `cargo fmt`; the CI steps (tests, `-D warnings` build, fmt check)
  were all run locally before pushing. **Lesson:** run the full CI step list locally before
  pushing Rust changes, and check `gh run list` after each push. A green local test run is
  not a green CI.

### 2026-11-24 · RESULT · Impact decay: fast early reversion; no regime dependence established (doc 29)
- Pre-registered propagator fit: G(30 min)/G(0) = 0.755 [0.60, 0.87] vs the simulator's 0.965,
  with about a quarter reverting within 1–2 min. At 2 h and 6 h the intervals (0.52–1.21, 0.47–1.44) contain the
  simulator, so criterion A does not trigger. High-vol minus low-vol 2 h reversion is +0.35
  [−0.34, +0.97], so criterion B fails. Decay joins docs 26–28 as unproven on 16 days.
- **Lesson:** four pre-checks in a row ended with intervals too wide to decide. Sample size
  (12 training days) is now the binding limit for every route to a regime-aware edge.

### 2026-11-24 · DECISION · Full-year 2024 data, out-of-time split, gzip storage (doc 30)
- 12 training days had become the limit for every pre-check, so the pool grows to every 2024
  day: 256 training (Jan–Sep), 87 validation (Oct–Dec). The 9 V3 held-out dates stay, plus a
  ±2 day buffer that is never downloaded. Jan–Mar 2025 is a new future held-out set,
  downloaded only for the final evaluation.
- New days get a regime label from a fixed rule: the day's mean log-price relative to the open
  (crash ≤ −1.0%, calm ≥ +0.55%). The rule was chosen on the 16 V3 days before any download.
- Data is stored as gzip CSV (6× smaller; golden fixtures unchanged), so a year is ~7 GB.
- Considered and rejected: training RL on the year in sequential chunks (risks forgetting the
  earlier chunks' regimes; disk no longer requires it); a paid historical API (each day is
  replayed thousands of times, so one download is cheaper).

### 2026-11-24 · MISTAKE · Downloader treated a cleanly closed connection as a finished file
- **What:** `fetch_bybit.py` had no retry, so 2 days failed on Bybit connection resets. The
  first fix resumed partial downloads, but a connection that closed without an error mid-file
  was taken as complete: two book zips were saved at half length.
- **Impact:** none on the data. Truncated zip and gzip files cannot be read, so both failed at
  conversion and nothing incomplete was written.
- **Fix / status:** FIXED. Downloads resume until the size matches the server's
  (Content-Length / Content-Range); the 4 days were fetched again in full.
- **Lesson:** a download is complete only when its size has been checked, not when the stream
  ends.

### 2026-11-24 · RESULT · Full-year 2024 pool downloaded; all 4 predictions met (doc 30)
- 343/343 days complete (2,879–2,880 snapshots each). Labels by the pre-registered drift rule:
  54 crash, 180 normal, 109 calm (predicted crash 10–20%, calm 30–45%). Crash days are
  more frequent in Apr–Sep (18.8% vs 12.7%), and the pool's median volatility is 27% below the
  event-picked V3 days'.
- Training crash days go from 3 (V3) to 44; validation has 10 crash, 47 normal, 30 calm days.

### 2026-11-25 · MISTAKE · CI never ran the Python test suite
- **What:** the python-smoke job only imported `evaluate.py` and checked the action map. None
  of `rl/tests` ran in CI, so Python regression tests (golden fixtures, splits, protocol) guarded
  only what was run locally. CI also lacked `orjson`, which the downloader needs.
- **Found by:** adding regression tests for the doc 30 downloader bugs, then checking where
  they would run.
- **Fix / status:** FIXED. CI runs `pytest rl/tests`: 94 data-free tests run, and 37 that need
  TradeData/ or rl/models/ skip (they must be run locally before pushing; all 131 pass locally).
  The downloader bugs now have unit tests (a local HTTP server that closes connections
  mid-file, ignores Range, or never completes) and an integration test (fetch through flaky
  connections equals a clean fetch). Each was mutation-checked: putting the bug back fails it.
- **Lesson:** a regression test only protects if something runs it. Check where a new test
  runs, not only that it passes.

### 2026-11-25 · RESULT · Full-year decay: simulator shape wrong; no regime dependence (doc 31)
- On 256 training days, real impact reverts ~35% within ~8 min and is then flat to 6 h
  (G/G0: 0.65 at 30 min, 0.69 at 2 h, 0.68 at 6 h). The simulator has no fast reversion and
  decays slowly over hours (0.965, 0.873, 0.707). At 2 h the simulator is outside the interval,
  so criterion A triggers: recalibrate decay as a separate, pre-registered, tested change.
- High − low volatility difference: −0.055 [−0.17, +0.07]. Criterion B fails; doc 29's +0.35 was
  noise. Decay is ruled out as a regime-aware edge.
- **Lesson:** doc 29's "same sign on validation" hint did not survive 21× the data. Agreement in
  sign on 4 validation days is weak evidence.

### 2026-11-25 · MISTAKE · gzip move left 14 Rust data tests skipping silently, and book_features broken
- **What:** the .csv.gz change (2026-11-24) updated `market_data.py`, but two readers had their
  own copies of the path. `src/bin/rl_env/tests.rs` named `book-<date>.csv` in 14 places, and its
  data tests return early when the file is missing, so they passed without running (among them
  the dim-4 counterfactual-units test and the protocol-v3 tests). `book_features.py` (doc 27)
  hard-coded the same old path and could not read any day.
- **Found by:** grepping for hard-coded data paths while preparing doc 32.
- **Impact:** no result was affected. Doc 31 does not use these readers, and no test that ran
  failed. But for one day the Rust "all pass" covered 14 tests that never ran.
- **Fix / status:** FIXED. One `bybit_book!` macro defines the fixture path. A guard test fails
  if the dataset folder exists but any fixture file is missing (mutation-checked against the old
  `.csv` pattern). `book_features.py` reads through `market_data`, with unit tests for routing and
  for `drawdown_z` (definition, causality), plus a real-data test. All 50 rl-env tests now run on
  real data; 289 Rust and 136 Python tests pass.
- **Lesson:** this is the recurring mirror-bug pattern (a value changes meaning and not every
  reader follows). After changing where data lives, grep for every hard-coded copy of the path.
  A test that can skip needs a guard that fails when the skip would be wrong.

### 2026-11-25 · RESULT · No bounce-back, no momentum; all four regime-aware routes closed (doc 32)
- Doc 27's hint (drawdown_z vs 2 h return, raw −0.16) is weaker than the Stambaugh bias of a
  random walk (−0.29 raw, 87% of days negative). Pre-registered shuffled-null test: FAIL. Its
  corrected value (+0.071 [+0.046, +0.095]) suggested momentum.
- A post-hoc sign-flip null, which keeps volatility clustering, gives +0.013 [−0.012, +0.038]
  (training) and −0.002 (validation): zero. The momentum came from the shuffled null.
- Direction forecasting (26, 27), liquidity timing (28), decay (29, 31) and drawdown
  conditioning (32) are all closed. Next step to be discussed with the user.
- **Lesson:** a null must preserve every property the statistic responds to. The shuffled null
  dropped volatility clustering and nearly produced a false momentum finding in the opposite
  direction. State a null's limitation in advance, then test it.

### 2026-11-25 · DECISION · Simulator impact decay recalibrated to data (doc 33)
- π 0.3 → 0.435 and ρ 5 → 2,284/day (half-life 3.3 h → 26 s), fitted to the response curve on
  256 V4 training days. Accepted: inside the training band at 30 min / 2 h / 6 h, and
  validation RMSE 11% of the old. The era-1 constants had never been fitted.
- Baselines rerun: TWAP is 0.06 pp cheaper at 5% of ADV. Honest AC still selects λ → 0, so
  AC = TWAP. Faster paces cost ~40% less extra impact than before.
- **Consequence:** all era-2 model results (M0-e2b, M1) were scored on the old decay; re-score
  and retrain on the recalibrated simulator before any comparison.

### 2026-11-25 · MISTAKE · New test fixture leaked its dataset into later tests
- **What:** `test_market_data.py` cleared STRATAEXEC_DATASET with `monkeypatch.delenv` on an
  unset name, which records nothing to undo. The SOL dataset the test selected stayed set, and
  `test_rule_reproduces_14_of_16_v3_labels` then read SOL books (8/16 instead of 14/16).
- **Found by:** the full Python suite, before commit.
- **Fix:** the fixture now calls `setenv` before `delenv`, so teardown restores the unset state.
- **Lesson:** a fixture that guards global state must record the original value, not just
  clear it.

### 2026-11-25 · MISTAKE · Doc 34's two-sided drawdown check could never pass a real effect
- **What:** `bounce_back_test.two_sided` required the correlation and the deep-drawdown excess
  to share a sign. One effect gives them opposite signs (bounce: corr < 0, excess > 0, as in
  doc 32's own criteria). The unit test written with it encoded the same wrong convention, so
  it passed.
- **Found by:** reading the first (shuffled-null) SOL output: corr +0.055 and excess −16 bp,
  one consistent momentum effect, could not satisfy the check. The main sign-flip run had not
  run yet (a quoting error in the run loop skipped it).
- **Fix:** the excess must have the opposite sign to the correlation. A regression test anchored to
  doc 32's convention fails on the old code (mutation-checked). The correction is written into
  the predictions doc before the main run.
- **Lesson:** a test written from the same mental model as the code checks nothing about that
  model. Anchor a new check's test to an independent statement of the convention: here,
  doc 32's own criteria.

### 2026-11-25 · RESULT · BTC findings hold on SOL and DOGE (doc 34)
- Same V4 pool, 343/343 days per coin. Local liquidity, decay-by-volatility and the drawdown
  signal (sign-flip null) all fail on both coins. Pooled decay has BTC's shape (G(30 min)/G(0)
  0.58 SOL, 0.65 DOGE).
- The shuffled null again produces false momentum (+0.043 to +0.058 over the sign-flip null on
  all three coins). On SOL it would have "passed": the pre-registered choice of the sign-flip
  null is what kept it out.
- Every route tested in docs 26–32 is now closed on three crypto perpetuals. Next step goes to
  the user: re-score M0-e2b on the recalibrated simulator, and/or another asset class.
- Process slip: the first run loop glued arguments to script names (zsh does not word-split),
  so only the secondary null ran. Runs now go through an explicit bash script, and the log is
  checked right after launch.
