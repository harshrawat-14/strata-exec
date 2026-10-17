# Phase 1 — Tail-Risk Metrics on Existing Strategies

> **PRE-CORRECTION SIMULATOR NOTICE (added Phase 2R, 2026-07-17):** every number in this report was
> produced against the counterfactual simulator **before** Phase 2R's Task 2 (impact-model
> calibration) and Task 3 (drift restoration) fixes. Do not compare these numbers directly against
> any post-Phase-2R result. See `rl/results/PROVENANCE.md` and `docs/methodology/standing-issues.md`.

No new model was trained in this phase. This measures the existing `ppo_lstm_v5_adaptive_best`
checkpoint, using Phase 0's unified harness (`evaluate_all_strategies_on_date`), across every
date with clean historical data (11 dates), broken out by regime, with a bootstrap CI on every
CVaR-95 estimate (Global Rule 4) and an AC risk-aversion (λ) grid on the two crash dates
(Phase 1 task 3a). Full per-strategy-per-date table: `rl/results/phase1_tail_risk.csv`.
λ-grid table: `rl/results/phase1_ac_lambda_grid.csv`. Reproduce with
`rl/.venv/bin/python rl/phase1_tail_risk_report.py`.

## ⚠️ Honesty note (carried forward from Phase 0, restated here because it directly bears on how to read this table)

`ppo_lstm_v5_adaptive_best` was trained under the **old, uncorrected** date split: both
`2024-08-05` (Yen unwind) and `2024-03-05` (flash liquidation cascade) — the two dates now
declared `HELD_OUT_CRASH_DATES` — were in its training data. Every RL number on either crash
date below is **in-sample**, not a generalization result. The corrected split has been declared
in code (`TRAINING_DATES`/`HELD_OUT_CRASH_DATES` in `rl/evaluate.py`) but not yet trained on —
that's Phase 2. Nothing in this report should be read as "RL beats AC on the crash day"; it's
reported for completeness and comparability with the pre-Phase-0 numbers, not as a validated
claim.

## Regime breakdown (mean CVaR-95, %)

| Strategy | calm | crash | trending | volatile |
|---|---:|---:|---:|---:|
| AC Optimal | -0.4420 | -0.4597 | -0.4457 | -0.4979 |
| Adaptive AC | -0.4472 | -0.4632 | -0.4488 | -0.4976 |
| Heuristic | -0.4454 | -0.5005 | -0.4651 | -0.4929 |
| TWAP | -0.4456 | -0.4690 | -0.4646 | -0.4952 |
| **RL Agent** | **-0.5458** | **-0.5746** | **-0.5558** | **-0.5891** |

"volatile" = 2024-02-20, 09-15, 10-15: meaningful price swings (~2% intraday drawdown) but no
coincident book-depth-thinning signature — i.e. *not* a genuine liquidity-cascade stress event
per Phase 0's survey, just an ordinarily volatile day. These have the worst CVaR-95 for every
strategy, worse than the labeled "crash" bucket — worth noting as-is: on this data, plain
volatility without a liquidity event was harder on every strategy's tail than the two actual
crash days. No causal claim is made here; this is a descriptive observation from 3 dates.

## Crash-day callout (both in-sample for RL — see honesty note)

| Date | Strategy | Mean IS | CVaR-95 | CVaR-95 95% CI |
|---|---|---:|---:|---|
| 2024-03-05 | TWAP | -0.4359% | -0.4547% | [-0.4553%, -0.4511%] |
| 2024-03-05 | Heuristic | -0.4722% | -0.5486% | [-0.5487%, -0.5478%] |
| 2024-03-05 | AC Optimal | -0.4238% | -0.4288% | [-0.4314%, -0.4259%] |
| 2024-03-05 | Adaptive AC | -0.4314% | -0.4361% | [-0.4361%, -0.4354%] |
| 2024-03-05 | RL Agent | -0.5582% | -0.5653% | [-0.5663%, -0.5639%] |
| 2024-08-05 | TWAP | -0.4994% | -0.5032% | [-0.5033%, -0.5029%] |
| 2024-08-05 | Heuristic | -0.5004% | -0.5036% | [-0.5036%, -0.5034%] |
| 2024-08-05 | AC Optimal | -0.5010% | -0.5035% | [-0.5035%, -0.5033%] |
| 2024-08-05 | Adaptive AC | -0.4958% | -0.5011% | [-0.5013%, -0.5005%] |
| 2024-08-05 | RL Agent | -0.5843% | -0.5896% | [-0.5896%, -0.5895%] |

RL's CVaR-95 is numerically worse (more negative) than every classical strategy on both crash
dates under this metric, in-sample. Reported plainly, per Phase 1 task 4 ("no editorializing")
— this is the same direction Phase 0's audit found for mean IS across the board (RL's raw
numbers are worse than the classical baselines on this specific counterfactual-LOB/OW-impact
setup generally, not just on crash days — see the regime table above, RL is worst in every
bucket). Whatever the underlying cause, it is not specific to crash days and this report does
not speculate on it further; that's outside Phase 1's scope.

## AC risk-aversion (λ) grid, crash dates only

Full grid (`AC_LAMBDA_GRID = [1e-5, 1e-4, 1e-3 (=AC_DEFAULT_LAMBDA), 1e-2, 1e-1]`) run on both
crash dates for AC Optimal and Adaptive AC:

| Date | λ | AC Optimal mean_IS | AC Optimal CVaR-95 |
|---|---:|---:|---:|
| 2024-08-05 | 1e-5 | -0.5083% | -0.5085% |
| 2024-08-05 | 1e-4 | -0.5057% | -0.5059% |
| 2024-08-05 | **1e-3 (default)** | **-0.5010%** | **-0.5035%** |
| 2024-08-05 | 1e-2 | -0.4766% | -0.4917% |
| 2024-08-05 | 1e-1 | -0.4731% | -0.5031% |
| 2024-03-05 | 1e-5 | -0.4092% | -0.4143% |
| 2024-03-05 | 1e-4 | -0.4187% | -0.4227% |
| 2024-03-05 | **1e-3 (default)** | **-0.4238%** | **-0.4288%** |
| 2024-03-05 | 1e-2 | -0.4669% | -0.4829% |
| 2024-03-05 | 1e-1 | -0.4720% | -0.4946% |

`AC_DEFAULT_LAMBDA` (1e-3) sits in the middle of the grid, not at an extreme — it is neither the
most patient (best mean IS, worst tail-timing risk) nor the most urgent (worst mean IS from
impact cost) setting available. This is evidence the AC baseline used throughout this project is
not a strawman chosen to make it easy to beat, though it also isn't provably "optimally tuned" —
just not hidden or cherry-picked. Full grid, including Adaptive AC, in
`rl/results/phase1_ac_lambda_grid.csv`.

## CVaR-95 CI width check (Global Rule 4)

No strategy × date row had a CVaR-95 bootstrap CI wider than 50% of the point estimate's own
magnitude at `n_episodes=50`. This is narrower than Global Rule 4's framing anticipated ("CVaR-95
on 50 episodes is the mean of only ~2-3 tail observations — a very high-variance estimator") —
worth stating plainly rather than silently: under this project's `fixed_steps=True,
fixed_size=True` evaluation configuration, per-episode IS values are themselves low-variance
(strategies take near-identical actions episode to episode, differing mainly via the LOB
replay's random start-offset within a fixed window), so even a 2-point tail mean has a fairly
tight bootstrap CI in practice. This should not be read as "CVaR-95 is always stable here" —
it is a property of this specific fixed-horizon/fixed-size evaluation setup, not a general result,
and Phase 4's final tables should re-check CI width under whatever evaluation configuration is
used there.

## Acceptance criteria status

- ✅ CSV/table `strategy × date × {mean_IS, cvar_95, is_p1, is_std, is_sharpe}` — `rl/results/phase1_tail_risk.csv` (11 dates × 5 strategies = 55 rows), plus bootstrap CI columns beyond the minimum spec (Global Rule 4).
- ✅ Aug-05 crash-day row called out explicitly (see callout table above), alongside 2024-03-05 (the second held-out crash date declared in Phase 0).
- ✅ No new model trained in this phase.
- ✅ AC λ pinned and reported across a grid (Phase 1 task 3a), not a single hidden value.
