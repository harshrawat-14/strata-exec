# Simulator provenance for the CSVs in this directory

**Phase 2R (2026-07-15 through 2026-07-17) corrected the counterfactual simulator itself**
(calibrated the impact model instead of a hardcoded constant — Task 2; restored real intraday
price drift instead of flattening every snapshot back to arrival — Task 3; fixed a `spread_ref`
scale bug that mispriced forced liquidation). Every number below predates those fixes and was
produced against the **pre-correction simulator**. Do not compare these numbers directly against
any post-Phase-2R result without accounting for the simulator change underneath them — the
comparability was already broken when the simulator changed, independent of any evaluation-mode
choice.

**Additional reason, found later in Phase 2R, why these are superseded (not a new invalidation —
they were already retired by the note above; recorded here for completeness):** every table below
that includes TWAP/Heuristic/AC-Optimal/Adaptive-AC numbers also predates the fraction-table fix
(the classical-baseline policies used a hand-maintained action-fraction mirror that had silently
diverged from the real environment's table, badly for `n_state_dims=12`). This does not newly
invalidate anything — these numbers were already dead from the pre-correction-simulator issue
above — but it means even a hypothetical re-run against the corrected simulator using the OLD
evaluate.py code would still have been wrong for a second, independent reason. Do not re-open these
files to "fix" them; they stay retired. See `docs/phase2r_objective_bias_finding.md` and
`docs/phase2r_standing_issues.md`.

| file | simulator | mode | status |
|---|---|---|---|
| `phase1_tail_risk.csv` | **pre-correction** | historical | superseded baseline, kept for record |
| `phase1_ac_lambda_grid.csv` | **pre-correction** | historical | superseded baseline, kept for record |
| `unified_comparison.csv` | **pre-correction** | historical | superseded baseline, kept for record |
| `failure_analysis.csv` | **pre-correction** | historical | superseded baseline, kept for record |
| `counterfactual_eval.csv` | **pre-correction** | counterfactual | superseded baseline, kept for record |
| `test_dates_eval.csv` | **pre-correction** | historical | superseded baseline, kept for record |
| `phase2_ablation.csv` | **pre-correction**, and separately invalid on its own terms (v6r's collapsed policy, see `docs/phase2r_objective_bias_finding.md`) | historical | **do not use** |

**Post-Phase-2R results** (the Task 4 16-dim vs 12-dim ablation, once run) will be produced
entirely in **counterfactual mode** against the **corrected** simulator (calibrated impact,
restored drift, fixed `spread_ref`) — see `docs/phase2r_objective_bias_finding.md` for why
historical mode was rejected for that ablation (dims 8/9 would be dead in one arm and not the
other, testing the 16-dim representation with two of its own distinguishing features disabled).
Any future CSV produced under the corrected simulator should be named/tagged clearly as
post-correction and listed in this table.

## Second boundary: Phase 3 Step 0's state-signal repair (2026-08-13) — a per-checkpoint hazard, not a per-file one

**Phase 3 Step 0** (`docs/phase3_plan.md`, commit `0e3356b`) fixed `sigma_ref` (was hardcoded at
0.20, unrelated to any date's data) and the volatility annualization (~19x error) in
`src/bin/rl_env.rs`, and replaced state dims s10/s11's 3-way hard buckets with a continuous
signal. This changes what state dims **s2, s10, s11, s13** contain on every date — a real input
distribution shift, not a cosmetic one.

**This is a boundary on *checkpoints*, not on CSV files the way the Task 2/3 boundary above is.**
Any RL checkpoint trained *before* commit `0e3356b` (this includes `RL16`/`ppo_ablation_16dim_cf_best`
and `RL12`/the 12-dim ablation checkpoint — check training date against the commit if unsure) was
trained on the OLD, broken s2/s10/s11/s13 distribution. Evaluating that checkpoint against the
CURRENT (post-`0e3356b`) `rl_env.rs` binary silently feeds it out-of-distribution state — the model
was never shown these input values during training. This is **not cosmetic**: confirmed directly
(`docs/phase3_tail_validation.md`'s opening section) that it materially changes the checkpoint's
behavior — e.g. 2024-08-05 RL16 mean IS moved from the published −6.48% to −4.49% under the
post-Step-0 binary, no error or warning, a large silent divergence.

**Rule: any further evaluation of a pre-`0e3356b` checkpoint (RL16, RL12) must be run against a
`rl-env` binary built from a commit before `0e3356b`** (e.g. `2246c9f`, the last commit before Step
0 — build via an isolated `git worktree add --detach <commit>` so the current tree's
`src/bin/rl_env.rs` is untouched), **not** the current `target/release/rl-env`. This stays true
until Phase 3 Step 2 retrains RL16/RL12 on the repaired state, at which point a NEW checkpoint
supersedes the old one and should be evaluated against the current (post-Step-0) binary instead.

Verification method for reproducing exactly (from `docs/phase3_tail_validation.md`): AC-Optimal and
TWAP only read `obs[0]`/`obs[1]` (inventory/time fraction), so they reproduce identically regardless
of binary version — use them as a canary. If AC-Optimal/TWAP numbers match a known-good CSV exactly
but RL16/Adaptive-AC (which read `obs[2]`, the recalibrated volatility ratio) don't, you're on the
wrong binary.

**Files produced under the pre-Step-0 binary** (all correctly RL16-checkpoint-consistent, do not
re-derive from a post-Step-0 binary without retraining first): `phase2r_ablation_honest_lambda.csv`,
`phase2r_severity_expansion_full.csv`, `phase2r_severity_expansion_new_days.csv`,
`phase3_tail_validation_episodes.csv`, `phase3_tail_validation_summary.csv`,
`phase3_tail_validation_cvar_bootstrap.csv`, `phase3_tail_validation_crux.csv`, and (per the
severity-expansion doc) `phase2r_honest_lambda_all11.csv`, `phase2r_lambda_grid_full.csv`.
