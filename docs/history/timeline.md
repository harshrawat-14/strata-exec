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
