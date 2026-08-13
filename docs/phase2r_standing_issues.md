# StrataExec — Standing Open Issues (Phase 2R)

Living record of findings that belong in the final writeup's honest-limitations section, or that
are deferred by explicit decision rather than resolved. Kept here so they don't evaporate between
sessions. Cross-referenced from `docs/phase2r_objective_bias_finding.md`.

---

## Simulator provenance boundary (read before comparing ANY two results)

**Every result produced before Phase 2R's Task 2 (impact calibration) and Task 3 (drift
restoration) fixes was measured against a materially different simulator than every result
produced after.** This is not a mode choice (historical vs. counterfactual) — it is a change to
what the counterfactual simulator itself computes. Comparing across this boundary without
accounting for it produces a false read. See `rl/results/PROVENANCE.md` for the per-file table of
which existing results are pre-correction (all of them, as of this entry — v5's headline numbers,
`phase1_tail_risk.csv`, `unified_comparison.csv`, `failure_analysis.csv`, and the invalid
`phase2_ablation.csv`). Task 4's ablation (16-dim vs 12-dim, counterfactual mode) will be the first
post-correction result; tag it and everything after it clearly, and never present a pre/post
number pair as if they were measured on the same ruler.

## Resolved this phase (kept here for the record, not action items)

- **Forced-liquidation reward loophole** (`5630e5d`) — fixed and tested.
- **`price_factor` rescaling in historical mode** (`5630e5d`) — fixed and tested.
- **Objective-bias / VWAP / "observability limit" findings** — retracted; were artifacts of
  measuring IS-vs-arrival instead of the actual training reward. See the finding doc's correction.
- **Late-training policy collapse root cause** — isolated by elimination to device (MPS); resume
  mechanics exonerated directly. CPU adopted as the operational training recipe.
- **Impact-model mislabeling** — relabeled (OW-style decay structure + square-root magnitude, not
  "Obizhaeva-Wang" as a whole). Calibration engine wired in.
- **Counterfactual-mode drift flattening** (`price_factor` in `CounterfactualLob`) — fixed, dim 6
  now live in training.
- **`spread_ref` scale-mismatch bug** in `reset_counterfactual` — fixed; was flooring the
  punitive-liquidation price at ~$0 for 80-90% of any forced dump.
- **Train/eval state-support gap (dims 8/9), formerly issue #2 below** — resolved by decision:
  Task 4's ablation runs entirely in counterfactual mode (both training and eval), so dims 6, 8, 9
  are all live and consistent by construction. Historical mode's asymmetry is unchanged but no
  longer gates anything, since the ablation doesn't use historical mode.
- **Classical-baseline action-fraction table bug** — `TWAPPolicy`/`HeuristicPolicy`/
  `ACOptimalPolicy`/`AdaptiveACPolicy` hand-maintained two mirrors (`V5_FRACTIONS`/
  `V6_FRACTIONS`) of the real environment's action->fraction table that had silently diverged
  (the `n_state_dims=12` mirror matched the real Rust `ACTION_FRACTIONS` at only 3/20 indices).
  Root-fixed: Python now queries the real table from the Rust binary (`--print-action-fractions`)
  instead of hand-copying it, with a cross-language test asserting identity. See the meta-lesson
  below — this is the second instance of this exact failure mode in the project.
- **Third `ACTION_FRACTIONS` mirror in `rl/environment.py`** — found during the Task 4 close-out
  implementation-consistency audit, currently byte-identical to the real table by luck (unprotected
  by any test), still imported by `rl/phase2r_vwap_gate.py`. Root-fixed the same way: `environment.py`
  is now the one canonical Python-side query point; `evaluate.py` imports from it instead of
  querying independently. Identity-asserting regression test added. Third instance of this bug
  class — see the meta-lesson below, sharpened further.
- **Action-granularity/bucketing as the cause of the inverted RL-vs-AC pattern** — investigated and
  mostly ruled out. All classical baselines are bucketed identically to RL (no continuous-execution
  privilege exists for anyone). The subtler "coarse buckets disproportionately distort AC's schedule"
  version was quantified directly and found to be small precisely at AC's actual best λ — the
  un-swept λ (issue #0 above) is the dominant mechanism, not granularity.

## Open — must survive into the final writeup's honest-limitations section

### 0. THE PHASE HEADLINE: RL wins under real regime stress, loses when AC's assumptions hold — recovered honestly, after two strawman baselines were found and removed
**Current headline, stated first because everything below is the history of how it was reached:**
once AC/Adaptive-AC are evaluated the way RL actually is — one λ, chosen without ever looking at a
held-out date's own outcome — **RL-16dim beats them on exactly the two most severe crash dates in the
whole 11-date set, and loses on all other nine (both held-out calm dates, and all 7 training dates)**.
This is the project's original mechanistic thesis (RL earns its keep exactly where a closed-form
heuristic's assumptions break, and has nothing to offer where they hold) — recovered only after two
successive strawman-baseline failures were found and corrected, which is what makes it credible now
rather than merely convenient. See the honest-λ bullet below for the full result and its explicit
n=2-per-regime-on-crash caveat.

**How this was reached (kept for the record — do not skip past this to just the headline, the history
is the evidence the headline survived a real adversarial process):** the comparison rested, in
sequence, on two different unfair baseline-parametrization failures. **Framing, stated precisely for
both:** neither was "weakened" by the next correction — each was never supported in the first place.
Do not carry forward language like "needs more work" or "unresolved pending a rerun" for either layer.
- **Un-swept λ.** `AC_DEFAULT_LAMBDA=0.001` was used identically for every date. A follow-up
  extended the grid to all 11 dates and down to 1e-9 (greenlit item 1, `rl/results/
  phase2r_lambda_grid_full.csv`): **AC/Adaptive-AC at their own best λ beat RL-16dim on 10 of the
  11 dates this phase has ever evaluated on** — RL only wins on 2024-03-05, the milder of the two
  held-out crash dates. See `docs/phase2r_objective_bias_finding.md`, "Task 4 close-out gate —
  Gate B" and the "full 11-date λ-grid" follow-up section for the complete per-date table.
  **SUPERSEDED — see the next bullet before citing "10/11" anywhere.**
- **The per-date λ selection above was itself hindsight — corrected, and it reverses the crash-date
  result.** Every λ in the "10/11" grid was picked using that specific date's own outcome: AC was
  effectively told the day's regime in advance, which RL never is — the same class of error as the
  original un-swept-λ strawman, one level subtler, and it survived the first correction undetected
  until flagged explicitly. The 2024-11-06 anomaly (best λ=1e-9, sitting inside the granularity
  total-failure zone) was corroborating evidence of this same per-date overfitting, not a coincidence.
  **Honest protocol: select one λ from the 7 `TRAINING_DATES`' aggregate outcome only (mean and
  median agree, robust even excluding 2024-11-06: λ=1e-6), apply it unchanged to every other date**
  (pure re-aggregation of the already-collected grid, no new simulation — extended from the 4 anchor
  dates to all 11 once the 4-date result landed, since n=2-per-regime alone isn't enough to lean on).
  **Full-11-date result (`rl/results/phase2r_honest_lambda_all11.csv`): RL-16dim beats honest-λ
  AC/Adaptive-AC on 2/2 held-out crash dates, and loses on all other 9 (0/2 held-out calm, 0/7
  training)** — the regime-conditional pattern holds cleanly at the full n available in this project,
  not just on the original 4-date subset. One nuance stated precisely: 2024-04-15 ("BTC correction,
  moderate stress," a training date) is a near-tie loss (-0.02pp) — the win pattern is about *severe*
  regime breaks specifically (flash liquidation cascade, Yen unwind), not stress in general, which
  sharpens the mechanistic read rather than weakening it. (2024-11-06's honest-λ AC number is large
  and positive, +4.98% — plausibly itself partly a hindsight-trend artifact of the IS-vs-arrival
  metric on a strongly-trending day, per open issue #1; flagged, not resolved here.)
  **Confirmed with the full Task 4 statistical machinery (second cheap confirmation, greenlit):**
  re-ran the complete 11-date ablation with paired bootstrap significance testing at honest λ=1e-6
  (`rl/phase2r_ablation_honest_lambda.py`, `rl/results/phase2r_ablation_honest_lambda.csv`, same
  methodology as the original Task 4 deliverable). **Both held-out crash-date wins are statistically
  strong, not just point estimates: p=0.0000 (paired bootstrap, 10,000 resamples, matched episode
  seeds) against both AC-Optimal and Adaptive-AC on both 2024-03-05 (+1.27pp / +1.87pp) and 2024-08-05
  (+1.41pp / +1.32pp).** Head-to-head against each baseline individually across all 11 dates: RL-16dim
  beats AC-Optimal on 3/11 (both crash dates + a +0.03pp near-tie on 2024-04-15) and beats Adaptive-AC
  on 3/11 (both crash dates + a +0.002pp near-tie on 2024-06-10) — both near-ties small enough to be
  noise, reported exactly rather than rounded into either bucket. The cleaner "vs whichever baseline is
  better that day" framing gives 2/2 crash, 0/9 elsewhere, as above. Mechanism: static
  AC-Optimal's schedule shape is a pure function of (λ, σ, η, T), independent of the date except
  through λ — a single honestly-chosen λ (patient, fit to mostly-calm training conditions) produces a
  patient schedule that gets caught by a real crash exactly as RL does; only *hindsight* per-date λ
  looked regime-adaptive. See `docs/phase2r_objective_bias_finding.md`, "The λ-selection protocol was
  hindsight — corrected, and it changes the headline," for the full tables and the mechanistic-thesis
  framing (why this result is credible now, having survived the same adversarial process that killed
  its two predecessors, with the n=2-per-regime-on-crash caveat stated explicitly rather than implied).
- **Granularity is not second-order — it sets a hard floor on AC's usable λ, corrected from the
  earlier Gate A writeup.** Extending the grid down to 1e-9 revealed that at λ≤3e-7, on every one of
  the 11 dates, 100% of AC's intended schedule rounds down to the zero-action bucket for the entire
  horizon — AC never trades voluntarily and the whole position gets dumped via punitive forced
  liquidation. The original Gate A conclusion ("granularity distortion is second-order, small
  precisely where AC's optimum sits") is corrected, not just updated: AC's actual optimum (λ≈1e-6 on
  9/11 dates) sits immediately above this failure floor, which now has a mechanistic explanation —
  it's the most patient schedule that still clears the 0.1% minimum bucket. See the "Correction" note
  inside Gate A in the finding doc for the full before/after.
- **Hardcoded, uncalibrated market-model parameters.** `ACOptimalPolicy`/`AdaptiveACPolicy` default
  to `sigma=0.20, eta=0.10` and no call site in `rl/evaluate.py` ever overrides either — every
  construction only varies `lam`. The simulator's own per-date calibrated impact coefficient and
  realized volatility (Task 2's `CounterfactualLobConfig::calibrated`, `realized_volatility` in
  `impact_fit.rs`) are never exposed to Python (no CLI flag analogous to
  `--print-action-fractions`), so AC's schedule shape reflects generic textbook assumptions with
  zero connection to what the simulator calibrated for that date. **Not yet fixed** — would need a
  new CLI flag to expose calibrated sigma/eta per date, plus re-running the λ-grid with those values.
- **Adaptive-AC's update frequency is not hardcoded — it's structurally absent as a parameter.**
  Audited (greenlit item 4): `AdaptiveACPolicy` recomputes and renormalizes its full schedule on
  every single environment step (up to 2880×/episode) purely because `predict()` is called that
  often, with no re-estimation-window or smoothing knob anywhere in the class. This is a plausible,
  structurally-supported mechanism for the pre-existing README finding ("AdaptiveOptimal over-trades
  during GARCH volatility clustering," from an earlier, different synthetic simulator) — not
  independently re-verified on the current pipeline. **Not yet fixed** — adding an explicit
  re-estimation-interval parameter and sweeping it is well-scoped future work, same cost order as
  the λ-grid.
- **σ is straightforward to calibrate; η is not — see the dedicated design note.** Audited (greenlit
  item 2, `docs/phase2r_ac_calibration_design_note.md`): σ has a clean per-date calibrated value
  already computed in Rust (`realized_volatility`), no modeling ambiguity, cheap to wire in whenever
  approved. η has no principled calibration at all, because the simulator's impact law is
  square-root and AC's η assumes linear — any single "calibrated" η would silently encode an
  arbitrary choice of which participation rate to match. Recommendation: don't patch η with a
  stopgap; treat it as blocked on the item-5 go/no-go (a numerically-optimized sqrt-impact baseline)
  discussed below.
- **Granularity chattering hypothesis: tested directly, NOT confirmed as originally framed — a
  clearer mechanism found instead.** Predicted mechanism: RL, confined to 20 discrete buckets with no
  schedule prior, would chatter between the zero bucket and an overshoot bucket, worst on crash days.
  Real single-seed trajectories (RL-16dim, AC at best λ, trivial 10%/step) on both held-out crash
  dates and both held-out calm dates show RL trades smoothly and continuously (isolated-burst rate
  0.2–1.3%, essentially no chatter), with per-fill slippage *better* than AC's on 3 of 4 dates tested
  — the specific chattering mechanism and its calm-vs-crash severity prediction are both not
  supported by the data. **What the trajectories show instead: RL uses roughly the same execution
  pace (~33–36% of the horizon) regardless of regime, while tuned-AC's pace adapts sharply by regime**
  (near-full-horizon patience on calm days, hard front-loading on the severe crash). RL's fixed pace
  happens to suit the one date it wins (the milder crash) and is mismatched everywhere else — too
  fast on calm days, too slow on the severe crash. See `docs/phase2r_objective_bias_finding.md`,
  "Gate A follow-up," for the full trajectory data, plots (`rl/results/plots/granularity_*.png`), and
  the crash-date IS decomposition (RL's 2024-08-05 loss is ~85% a drift/"adverse-holding" cost from
  under-urgent pacing, not poor per-trade execution — RL's execution-quality component is actually
  the best of the three policies tested on that date).
- **Reframe (explicit correction to how this should be recorded): this is a diagnosable gap in this
  project's RL setup, not a settled negative result about RL's ceiling.** AC-Optimal is provably
  misspecified for this simulator (linear-impact closed form against a square-root-calibrated
  environment — see the calibration design note). A correctly-functioning learner faces the true
  dynamics directly and should be able to at least match a misspecified heuristic; that RL instead
  loses on 10/11 dates, via a specific and now-identified mechanism (fixed, regime-blind pacing
  against a reward signal that only resolves at/near the horizon), points at the RL setup — not at an
  inherent ceiling on learned execution for this problem. Do not write this phase up as "RL loses to
  AC" without this framing.
- **Design sketch logged (no implementation): residual/delta action space.** Motivated directly by
  the pacing-mismatch mechanism above — reparametrize the action as a multiplier on a reference
  schedule's recommendation (multiplier≡1.0 reproduces AC exactly) so the agent inherits
  regime-appropriate pacing and only has to learn deviations, rather than rediscovering an entire
  2880-step schedule shape per regime from a mostly-terminal reward. Full sketch, including costs and
  risks, in `docs/phase2r_residual_action_space_design_sketch.md`. **Explicitly invalidates the 16-dim
  ablation and needs its own phase** — not scoped into Task 4 or this close-out gate.
- **Headline sharpened and validated with two no-training checks before any item 5/6 decision:**
  "RL learned execution but not urgency" (`execution_pct` -1.19 beats both AC's -1.67 and a trivial
  policy's -1.80 on 2024-08-05; AC's urgency swings 1.00→0.285→0.088 by regime while RL sits at
  0.93–0.99 everywhere). Two checks, `rl/phase2r_urgency_diagnosis.py`:
  - **Task A (does AC-pacing + RL-execution beat tuned AC, no training?): first pass was capped by
    its own construction — corrected framing, then re-run uncapped.** The first hybrid (AC's
    cumulative schedule for urgency, catch-up capped at RL's own median chunk size) was **second-best
    on both dates tested, not a win**: ranking was AC > hybrid > RL on 2024-08-05, and RL > hybrid >
    AC on 2024-03-05 — "beats AC on 2024-03-05" (the original framing) omitted that it still lost to
    pure RL there; corrected. Re-run with the cap removed
    (`rl/phase2r_urgency_hybrid_v2.py`): on 2024-08-05, the uncapped hybrid **beats both AC and RL
    outright** (-3.75 vs AC's -4.79 and RL's -6.58) — confirming the original cap was the binding
    constraint, as suspected. But on 2024-03-05, uncapping makes the hybrid *worse* (-1.44, behind
    both RL and the original capped hybrid) — RL's own organic pacing was already close to right-sized
    there, and forcing AC's urgency onto it without a cap over-commits to costlier chunks than needed.
    **No single cap (tight or none) is uniformly right — the achievable prize needs resolution AND
    range that adapt to how urgent the moment actually is, which is the credit-assignment problem
    itself, not a precondition solved by picking a better constant.** A naive additive estimate (AC's
    drift + RL's execution) was tested against both variants and found unreliable regardless — wrong
    magnitude, sometimes wrong direction. **This experiment's own AC reference schedule (best λ) is
    itself hindsight-selected — see the λ-protocol correction above, which is the more fundamental
    finding and changes what this experiment's result even means** (it describes "RL copying an
    oracle's pacing," not "RL copying a real deployed AC's pacing").
  - **Task B (is regime info present & discriminative early, when urgency must be decided?): mixed,
    precise answer — now elevated to its own named issue, #6 below** (discrete regime dims s10/s11
    flat for the first 100 steps on every date tested, while continuous dims s2/s13 do carry real,
    strengthening signal in that same window). Summary here, full writeup at issue #6.
  - **Recommendation: the residual-action-space premise is SHARPENED, not broken — restated.** The
    honest result doesn't kill the motivation, it corrects *what* needs importing. RL's execution
    component was already shown to be best-in-class (Task A: -1.19 on 2024-08-05, beating AC's -1.67
    and the trivial policy's -1.80) — what RL lacks is specifically regime-appropriate urgency, and
    Task A's uncapped-hybrid re-run shows no single *fixed* urgency rule solves this (uncapped-AC-
    pacing wins on 2024-08-05, loses on 2024-03-05 — the same regime-blindness static-AC-at-one-λ has).
    **The target is adaptive urgency, exactly the shape of thing a learned residual should supply — but
    the reference schedule cannot be static AC at a fixed λ, since that is now shown to be
    regime-blind by construction.** The reference must itself be regime-responsive: Adaptive-AC's live
    volatility recomputation, or item 5's numerically-optimized schedule, are the two live candidates.
    Issue #6's regime-dim lag is a **hard gate** on whichever reference is chosen, not a nice-to-check
    — verify the chosen reference's own regime-responsiveness in the first ~100 steps before building
    anything on top of it. Full detail and the corrected design note in
    `docs/phase2r_residual_action_space_design_sketch.md`.
- **Net effect — this IS the phase's headline finding, stated as such, not as a retraction:** the
  phase found and corrected two layers of the same strawman-baseline failure mode in sequence
  (hardcoded params, then hindsight-selected params) and, after removing both, recovered the original
  mechanistic thesis: **RL-16dim beats an honestly-tuned classical baseline specifically where that
  baseline's assumptions break under real market stress (2/2 severe crash dates), and loses
  specifically where those assumptions are close to correct (0/9 calm, moderate, and training dates).**
  This is credible precisely because it is what survived an adversarial process that killed two
  more-favorable-looking predecessors, not because it was assumed going in. Caveat stated explicitly,
  not implied: the crash-side evidence is n=2 dates — directionally clear and internally consistent,
  not a large-sample claim. The RL-vs-RL ablation (16-dim vs 12-dim) is unaffected by any of this and
  remains a second, independent surviving claim from Task 4.
- **Methodological finding (worth logging on its own, not just as a caveat on the RL number):** four
  instances of the same bug class surfaced this phase — CVaR sign inversion, two independent
  hand-copied `ACTION_FRACTIONS` mirrors, and this hardcoded-baseline-parameter issue. Every one was
  a value duplicated, assumed, or left at a default rather than sourced from its canonical origin or
  properly fitted, and every one biased the comparison toward the project's own method. See the
  "Methodological finding" section in `docs/phase2r_objective_bias_finding.md` for the full
  writeup-ready version of this lesson.

### 1. Hindsight trend-following in single-path historical backtesting (IS-metric only)
Single realized historical paths reward directional timing **in the IS-vs-arrival reporting
metric** (not the training reward, which is drift-neutral — confirmed by reading the code).
Predates this incident; applies to v5's methodology too. Candidate fix: multi-day episodes, or
intraday price-path randomization. **Deferred to post-v2-core** — do not implement speculatively;
not shown to bite training outcomes.

### 2. Train/eval state-support mismatch: dims 8/9 (own-impact fractions) — RESOLVED for Task 4
Dim 6 fixed this phase (Task 3). Dims 8/9 remain architecturally dead in `historical` mode (no
`cf_lob` by design) while live in `counterfactual` mode — this is unchanged and would still gate
any *future* work that needs historical mode specifically. For Task 4, resolved by decision: the
ablation runs entirely in counterfactual mode for both training and eval, so this doesn't bite.
Moved to "resolved this phase" above; kept here only as a pointer for future historical-mode work.

### 3. Punitive-price extrapolation for forced-liquidation residual beyond real book depth
`worst_price − 5×spread_ref`, a simple linear offset, prices whatever portion of a forced dump
can't fill against real snapshot depth (typically 80-92% of a full-inventory dump, since
`total_inventory` defaults to an institutional-sized 50,000 BTC order). The scale bug that made
this floor at ~$0 is fixed; the *shape* of the extrapolation (linear offset from the worst filled
price) remains a simplification. A more principled deep-liquidity/impact model would be worth
building. **Deferred** — not blocking, the current formula produces defensible (if imprecise)
magnitudes post-fix.

### 4. VPIN null result on Binance perpetuals
Structural: 24/7 trading, arbitrage-saturated, no clean informed/uninformed order-flow split to
dilute VPIN's signal on this venue. OFI is the appropriate toxicity signal here instead. A real,
reportable finding, not an implementation failure.

### 5. Start-offset eval bound (fixed-horizon eval only)
Training episodes are safely bounded (`start_offset + horizon ≤ len` always holds for the training
horizon range, verified with a test). The *fixed* eval horizon (2880 steps) does not satisfy this
bound for `start_offset > 0`, but all evals use small seeds (0-19, <1% of the episode), so results
are unaffected in practice. Bounding `start_offset ≤ len − total_steps` would remove the caveat
entirely. **Deferred** — cosmetic, not measured to affect any reported number.

### 6. Discrete regime-indicator state dims (s10, s11) are dead exactly when urgency must be decided
Named and elevated from a sub-finding to its own standing issue, per explicit instruction — this is a
state-representation problem, not merely a footnote on the RL-vs-AC comparison. Measured directly
(`rl/phase2r_urgency_diagnosis.py`, Task B): `s10` (vol_regime) and `s11` (liq_regime), the two state
dims specifically engineered to encode which regime the market is in, are **completely flat and
bit-identical across the severe-crash date and both calm dates for the entire first 100 steps (50
minutes) of the episode** — the window in which a policy would need to decide how urgently to trade.
Mechanism, confirmed by reading `regime_dims()`: both dims require the rolling realized-vol estimate
to cross a fixed 0.7×/1.3× threshold against `sigma_ref` before leaving the "normal" bucket; with a
return buffer that only starts informative after ~5 samples and a `sigma_ref` calibrated at the
whole-day level, the short early-window sample doesn't cross that threshold on any date tested here —
even though the *continuous* vol-ratio dims (s2, s13) do carry real, correctly-directioned, and
strengthening separation in that same window (measured, not assumed). **This is shared machinery, not
an RL-specific defect:** the same rolling-vol estimator that produces s2/s10/s11/s13 also drives
Adaptive-AC's own live schedule recomputation (`current_sigma` in `rl_env.rs`, read by both). Any
future design that hands RL a reference schedule derived from this estimator (see the residual-action-
space design sketch) would inherit this same early-window lag, not escape it. **Explicit hard
requirement for that future phase, not an assumption to carry forward: verify the reference schedule's
own regime-responsiveness in the first ~100 steps before crediting the design with fixing RL's
pacing.** Not fixed here — diagnosis only, per explicit scope.

**CORRECTION (Phase 3, 2026-08-13) — the mechanism above is wrong; the dims were fixed, see
`docs/phase3_plan.md` sections C/D.** The stated mechanism ("`sigma_ref` calibrated at the whole-day
level" — i.e. a lookahead leak) does not match the code: `sigma_ref` was a **hardcoded constant,
0.20**, unrelated to any date's data (`DEFAULT_SIGMA` in `rl_env.rs`) — there was never a lookahead
leak. The real, purely arithmetic cause: `sigma_ref` fixed at 0.20 while `current_sigma`'s
annualisation was separately wrong by ~19x (`sqrt(total_steps)` instead of
`sqrt(total_steps_per_year)`, and worse, keyed to the per-episode horizon rather than a fixed
one-day period) — the two errors combined pinned `current_sigma/sigma_ref` inside the "normal"
bucket band (0.7–1.3) on every real date this project has data for, calm or crash alike. Fixed:
`sigma_ref` now defaults (in historical/counterfactual mode) to `CALIBRATED_SIGMA_REF = 0.4999`,
calibrated the same way as λ — median annualised realised volatility across the 7 `TRAINING_DATES`
only, frozen, never touching held-out or eval dates. Annualisation corrected to a fixed
`TRADING_STEPS_PER_YEAR` constant. The 3-way hard buckets were also replaced with a continuous
`tanh(ln(ratio))` signal (approved decision, since a raw-ratio `tanh(ratio)` saturates at the now-
reachable crash-day ratios of ~2.8–4.1x and would erase the 03-05 vs 08-05 severity difference).
**Result, measured via the real CLI/Python runtime path, first 100 steps, mean s10:** severe crash
(2024-08-05) +0.678, milder crash (2024-03-05) +0.342, held-out calm (2024-07-10) −0.219, held-out
calm (2024-12-10) −0.120 — cleanly ordered by severity, both crash dates positive (above the
training-calibrated reference), both calm dates negative. **s11 (liq_regime) did NOT show the same
improvement** — first-100-step means (−0.012, +0.006, −0.141, +0.032 respectively) show no clean
crash/calm separation, despite `spread_ref` never having had the hardcoded-constant bug (`spread_ref`
is set causally from the episode's own first snapshot, not a global default). Left as an open
question for Phase 3 Step 2 rather than patched speculatively — possibly the percentage-binned
`bookDepth` format genuinely doesn't capture liquidity thinning as cleanly as price captures
volatility, or `spread_ref`'s single-snapshot reference is itself noisy episode-to-episode; not yet
diagnosed. Executable regression test:
`tests::regime_dims_separate_crash_from_calm_in_first_100_steps` in `src/bin/rl_env.rs`.

**CORRECTION 2 (Phase 3, 2026-08-13) — s11 diagnosed to completion: genuine null result, not an
open question.** The "not yet diagnosed" note above is resolved. Two independent checks (Python
directly against the raw `TradeData/BookDepth` CSVs, bypassing the simulator; and the real Rust/
Python CLI pipeline via `rl/environment.py`) confirm both candidate explanations were partially
right, and identify the actual mechanism: (1) `half_spread()`, as synthesized from Binance's
percentage-binned `bookDepth` format (`notional/depth` of the whole cumulative ±1% band, not a
literal top-of-book quote — see `diff_levels()` in `lob_replay.rs`), genuinely is a weak liquidity
proxy — even measured exactly at each date's true minimum-depth snapshot, its ratio to the day's
median is 0.90–1.08 on all four dates tested, no crash signal at any point in the day. (2) Raw depth
(`bid_depth_up_to`, the same quantity already live as s4) is a materially better liquidity signal —
a `tanh(ln(depth_ref/current_depth))` reformulation peaks at +0.85/+0.89 on the two crash dates,
correctly above both calm dates' peaks of +0.46/+0.69 — but only at step 1743/2880 (2024-08-05) and
2395/2879 (2024-03-05), i.e. 60–83% through the trading day, matching this project's own stress
survey almost exactly (independently recovers 2024-03-05's "22% of daily median" finding: measured
0.218). **Within the first 100 steps, neither formula shows any separation** (verified directly,
not assumed) — the liquidity event on both held-out crash dates simply has not happened yet 50
minutes into the day. Unlike volatility (a sustained, whole-day regime property on these same
dates), liquidity thinning here is a sharp, late-arriving, ~1-2-snapshot event that mean-reverts
fast — a fact about market microstructure and about when these two specific real crashes occurred
intraday, not an arithmetic bug reachable by reformulating a same-step, causal (no-lookahead)
signal. **Treated as a genuine null result at the same rigor bar as the VPIN finding (issue #4
below): s11/liq_regime is not a usable early-decision signal at this data's resolution for the two
crash dates this project has ground truth for.** Not patched — swapping in raw depth would not have
passed an s10-style first-100-step regression test either (checked, not assumed), so doing so would
have been a cosmetic fix. Full derivation and numbers in `docs/phase3_plan.md`'s Step 0 section (the
"s11 diagnosed further" paragraph).

## Meta-lesson: hand-maintained mirrors of canonical values (FIFTH occurrence — and #5 was self-inflicted)

**Any value that exists in two places will eventually disagree. Canonical values get read from
their single source, never copied.** This is the fifth time in this project a hardcoded or
hand-maintained value (not always a literal second copy — instance #4 is a single hardcoded default
never overridden per its actual context) caused a silent, real bug — enough occurrences now that
this should be treated as a predictable failure mode, not a series of one-off surprises. **Instance
#5 was introduced by the fix for instance #4, in the session that wrote this warning** — see its
entry below:

1. **CVaR sign inversion** — several call sites reimplemented "CVaR-95" locally (some as
   `np.percentile(arr, 5)`, a single point not a tail mean; the Rust/`parallel_sim.py` siblings had
   an outright sign inversion, averaging the *best* tail instead of the worst). Fixed by
   `compute_cvar_95` becoming the one implementation everything shares.
2. **Action-fraction table divergence** (this phase) — `rl/evaluate.py` hand-maintained
   `V5_FRACTIONS`/`V6_FRACTIONS` as mirrors of the real Rust `ACTION_FRACTIONS`, which silently
   diverged (matched at 3/20 indices for the `n_state_dims=12` case). Fixed by Python querying the
   real table from the Rust binary instead of copying it.
3. **A third copy of the same `ACTION_FRACTIONS` table** (found during the Task 4 close-out audit,
   same phase) — `rl/environment.py` had its own independent hardcoded copy, byte-identical only by
   luck, unprotected by any test, silently still imported by `rl/phase2r_vwap_gate.py`. This one
   hadn't diverged *yet*, but existed under the exact same conditions (hand-copied, untested, two
   places) that caused #2 to diverge. Fixed the same way, plus an identity (`is`) test to make a
   fourth copy structurally harder to introduce by accident.
4. **`sigma_ref` hardcoded at a plausible-looking constant (0.20)** (Phase 3, found while
   diagnosing issue #6 above) — not a second copy of a value, but the same failure mode's sibling:
   a config value defaulted once and never sourced from the data it was implicitly claiming to
   represent, for every real historical date this project runs against. Compounded by a second,
   independent bug (annualisation wrong by ~19x) that happened to mask the first — fixing either
   alone would have moved the dependent state dims to the opposite dead rail and looked like a fix.
   Fixed by calibrating `CALIBRATED_SIGMA_REF` from data (the 7 `TRAINING_DATES`, same protocol as
   λ) and wiring it as the automatic default for historical/counterfactual mode in `parse_cli()` —
   one place, applies to every caller, no per-call-site flag to forget.

5. **`sigma_ref` mirrored in Python, created by the Step 0 fix itself** (Phase 3) — Step 0
   recalibrated `sigma_ref` in Rust (0.20 -> `CALIBRATED_SIGMA_REF`) and left two Python
   copies behind: `rl/evaluate.py`'s `AdaptiveACPolicy(sigma_ref=0.20)`, which was live and
   which no call site overrode, and `rl/batched_environment.py`'s, which was *dead* —
   declared but never stored or forwarded, so a caller passing `sigma_ref=X` silently got the
   default. Fixed by a `--print-env-constants` flag on the Rust binary read at import time,
   mirroring the `--print-action-fractions` mechanism from instances #2/#3, with an AST-based
   test that fails if any float-literal `sigma_ref` default reappears under `rl/`.

   **This instance is the sharpest evidence the meta-lesson is not yet operational.** It was
   introduced *in the same session that rewrote this very section to say "after any fix in
   this class, a grep/audit for siblings should be routine, not something that only happens
   when explicitly requested"* — and then the grep was not run. The rule was known, written
   down, and still skipped. Two consequences worth carrying forward: (a) the sweep has to be
   part of the change itself, not a follow-up step that depends on remembering, which is why
   the check is now an executable test rather than a documented habit; and (b) recording a
   lesson is not the same as adopting one — treat any un-automated process rule in this
   project as *aspirational* until something mechanical enforces it.

All five were structurally similar: a value defined or assumed once, either copied by hand into a
second location or simply left at a default that was never revisited for the context it was
actually used in, with nothing enforcing agreement with reality. Instance #3 was found not by a
table diverging and breaking something, but by **actively grepping for the pattern on request** —
meaning the first two occurrences did not, on their own, prompt a proactive sweep for more copies of
themselves; it took an explicit "assume a third mirror-bug exists until proven otherwise" audit to
surface it. Instance #4 was found the same way, applied to a different question ("assume there's a
hardcoded value masquerading as calibrated somewhere in the state vector"). **Applying this
forward:** any new constant, table, or formula that needs to be used from both Rust and Python (or
from two places in either language, or that implicitly claims to represent real data) should be
read from one canonical, data-derived source at runtime, not hand-transcribed or left at a
plausible-looking default — and after *any* fix in this class, a grep/audit for siblings should be
routine, not something that only happens when explicitly requested.

## Deferred by explicit decision — log only, do not action before post-v2-core

These are named directions, not commitments, so they don't evaporate — none are to be started now:

- **Action-space redesign: uniform vs. non-linear vs. continuous, run as a comparison, not a
  pick.** The current 20-bucket `ACTION_FRACTIONS` ladder is uniform-ish and, per this phase's
  fraction-table investigation, wastes resolution: the difference between selling 1% and 3% matters
  far more to execution quality than between 90% and 95%, and the current ladder spends buckets
  evenly rather than concentrating them where they matter. Three variants worth comparing head to
  head rather than just picking one: (a) the current uniform-ish discretization (baseline), (b)
  non-linear/finer buckets concentrated near small sells, (c) continuous actions (needs a different
  RL algorithm, e.g. SAC/PPO-continuous, and has its own stability considerations). Likely ranking
  by payoff-per-effort: non-linear buckets > continuous > uniform (current) — but that ranking is a
  guess, not a finding; the point of running it as a comparison is to not have to guess. **Hard
  deferral reason, not just scheduling:** changing the action space invalidates every result and
  both collapse diagnoses built this phase (the entropy-floor fix, the device/resume isolation, the
  reward-loophole fix all reason about behavior in terms of the current 20-action space). Doing this
  now would re-litigate the collapse, the reward, and the ablation simultaneously on shifting
  ground. Self-contained future study, arguably its own phase — after v2-core closes.
- **Interval-VWAP as the primary reporting benchmark** (not the training reward, which is already
  drift-neutral — this is about what the *comparison tables* report). Would make reported IS numbers
  drift-neutral too, addressing issue #1 above at the reporting-metric layer specifically. The
  `interval_volume`/`book_mid` telemetry and `rl/phase2r_vwap_gate.py` scaffolding from the
  (retracted) objective-bias investigation remain in the tree and are reusable for this.
- **Multi-day episodes**: span more than one historical day per training episode, or randomize
  which day's price path is spliced into an episode, to reduce any single day's realized drift from
  dominating a training batch. Complements the trend-following issue (#1) rather than replacing it.

## Scope anchor (restate for every future decision on this list)

> This is an **offline execution-cost research study**. The policy is trained and evaluated in a
> counterfactual market simulator built from historical Binance LOB data with a transient own-impact
> model. It is **not** a live-trading system. The sim-to-real transfer gap is a characterized
> limitation and a direction for future work, not a solved problem.
