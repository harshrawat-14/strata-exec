# Simulator provenance for the CSVs in this directory

**Phase 2R (2026-07-15 through 2026-07-17) corrected the counterfactual simulator itself**
(calibrated the impact model instead of a hardcoded constant — Task 2; restored real intraday
price drift instead of flattening every snapshot back to arrival — Task 3; fixed a `spread_ref`
scale bug that mispriced forced liquidation). Every number below predates those fixes and was
produced against the **pre-correction simulator**. Do not compare these numbers directly against
any post-Phase-2R result without accounting for the simulator change underneath them — the
comparability was already broken when the simulator changed, independent of any evaluation-mode
choice.

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
