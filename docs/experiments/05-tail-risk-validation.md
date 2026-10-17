# Phase 3 Step 1 — Tail-risk validation: does RL's CVaR edge survive controlling for its mean edge?

Written 2026-08-13. Answers the four-part evidence bar laid out in `docs/proposals/research-roadmap.md`, "The tail
claim — evidence bar and current status," against the previously-unanalyzed CVaR-95 columns in
`rl/results/phase2r_ablation_honest_lambda.csv`. Follows the reporting conventions of
`docs/methodology/standing-issues.md`: every requirement is reported, including the one that fails, with
exact numbers and CIs, not summarized into a single verdict.

## Bottom line, stated first

**3 of 4 requirements are met. Requirement #3 — the crux — is NOT met.** The raw CVaR-95 advantage on
both crash holdout dates is real and statistically robust (#4), and it scales with day severity the
same way the mean advantage does (#2). But once each date's RL/baseline episode distributions are
**mean-centered** (removing the location shift that is already the known, established mean-cost
result) and the *residual* tail shape is compared, RL's tail is **not distinguishably better than
AC-Optimal's or Adaptive-AC's** — on either individual crash date, or pooled across all 8 V2
crash-regime dates (n=400 episodes). Every centered-CVaR and dispersion (std) test against the two
AC-family baselines is non-significant, and on both individual crash dates the point estimate
actually points slightly the *other* way (RL's residual tail marginally worse, not better, though not
significantly so). **The honest reading: the CVaR advantage is best described as RL's already-known
mean-cost advantage propagating into the tail of the outcome distribution, not a separate,
distinct tail-risk-management mechanism.** Do not write up "RL adapts better to worst-case scenarios"
as a claim distinct from "RL has a larger mean-cost edge on severe days" — the data does not support
treating them as two different findings.

---

## A methodological finding surfaced while building this, stated before the results

**RL16's frozen checkpoint predates Phase 3 Step 0's state-signal repair (`docs/proposals/research-roadmap.md`
section C/D, commit `0e3356b`) — evaluating it with the current `rl_env.rs` binary silently feeds it
an out-of-distribution `s10`/`s11`/`s2`/`s13` it never saw in training, and materially changes its
behavior.** Discovered directly: the first full 25-date eval pass in this session was run against the
current (post-Step-0) `target/release/rl-env` build. `AC-Optimal` and `TWAP` (which only read
`obs[0]`/`obs[1]`, inventory/time fraction) reproduced `phase2r_ablation_honest_lambda.csv` exactly.
`RL16` and `Adaptive-AC` (which both read `obs[2]`, the volatility-ratio dim Step 0 recalibrated) did
**not** — e.g. 2024-08-05 RL16 mean IS went from the published −6.48% to −4.49% under the post-Step-0
binary, a large, silent divergence with no error or warning. This is exactly the "never compare across
a simulator-provenance boundary" failure mode `rl/results/PROVENANCE.md` already warns about for the
Task 2/3 boundary — Step 0 is a second, newly-discovered instance of the same class, specific to any
checkpoint (RL16, RL12) trained before it.

**Fix applied:** built a second `rl-env` binary from commit `2246c9f` (the last commit before Step 0,
via an isolated `git worktree add --detach`, never touching this worktree's `src/bin/rl_env.rs`) and
used *that* binary for every result in this document. Verified by exact reproduction of
`phase2r_ablation_honest_lambda.csv` on both flagship crash dates before trusting any new number:

| date | metric | published (pre-Step-0) | this session's re-run | match |
|---|---|---:|---:|---|
| 2024-03-05 | rl16_mean_IS | −1.186037 | −1.186037 | exact |
| 2024-03-05 | rl16_cvar_95 | −1.475821 | −1.475821 | exact |
| 2024-03-05 | ac_mean_IS | −2.460169 | −2.460169 | exact |
| 2024-03-05 | aac_mean_IS | −3.060564 | −3.060564 | exact |
| 2024-03-05 | tw_mean_IS | −4.227246 | −4.227246 | exact |
| 2024-08-05 | rl16_mean_IS | −6.481457 | −6.481457 | exact |
| 2024-08-05 | rl16_cvar_95 | −7.010132 | −7.010132 | exact |
| 2024-08-05 | ac_mean_IS | −7.888679 | −7.888679 | exact |
| 2024-08-05 | aac_mean_IS | −7.799160 | −7.799160 | exact |
| 2024-08-05 | tw_mean_IS | −9.073939 | −9.073939 | exact |

Every number in this document is from the pre-Step-0 binary. **Standing implication for later
sessions, not actioned here (out of this task's scope — only `docs/experiments/05-tail-risk-validation.md` and the
`docs/proposals/research-roadmap.md` Step 1 line were in scope):** any future eval of the current RL16/RL12
checkpoints must keep using the pre-Step-0 binary until Phase 3 Step 2 retrains on the repaired state;
`rl/results/PROVENANCE.md` likely deserves a new row/section recording this boundary the same way it
records the Task 2/3 one, but that edit was left out of scope here to avoid touching a shared file
outside this task's assigned section.

---

## Data and methodology

- **Model:** RL-16dim only (`rl/models/ppo_ablation_16dim_cf_best/best_model`), honest λ=1e-6, same
  as every other Phase 2R/3 result. RL-12dim not evaluated — not needed for the tail-mechanism
  question, which is stated entirely in terms of RL16 vs AC-Optimal/Adaptive-AC/TWAP.
- **Dates:** all 25 dates in `evaluate.SEVERITY_RANK_V2` — a strict superset of the original 11-date
  honest-λ set (every V1 date is also a V2 date), so this covers the "original 11 at minimum, full 25
  preferred" instruction at the preferred level. Not scoped down: once the harness issue above was
  fixed, the full 25-date, 5-policy, 50-episode-each run completed in **14.8 minutes** wall time (the
  binary swap did not change simulation speed), well within budget.
- **n_episodes = 50 per date**, same as every existing honest-λ result — matched seeds (0–49) across
  every policy on a given date, since `seed` determines `start_offset` into that date's real Binance
  price path (`reset_counterfactual`, `src/bin/rl_env.rs`) identically for every policy evaluated on
  it, which is what makes paired-episode comparison valid (same method every existing
  `phase2r_*_honest_lambda*.py` script already relies on).
- **New code:**
  - `rl/experiments/phase3_tail_validation_eval.py` — thin wrapper around `evaluate.evaluate_all_strategies_on_date`
    (does not reimplement its simulation logic) that additionally persists the raw per-episode
    `is_samples` arrays it already computes but the existing scripts only ever summarized to a mean
    and CVaR-95 before writing to disk. Outputs `rl/results/phase3_tail_validation_episodes.csv`
    (6,250 rows: 25 dates × 5 policies × 50 episodes) and
    `rl/results/phase3_tail_validation_summary.csv` (25 rows).
  - `rl/experiments/phase3_tail_validation_analysis.py` — the four-requirement analysis below. Reuses
    `evaluate.compute_cvar_95` by direct delegation for q=0.95 (not reimplemented) and generalizes its
    same tail-mean method to q=0.90/0.99. Reuses the same paired-index-resample bootstrap mechanics as
    `phase2r_ablation_honest_lambda.paired_bootstrap_delta` (resample the same episode indices for both
    arms, preserving the seed pairing) composed with `evaluate.bootstrap_cvar_95_ci`'s idea of
    bootstrapping a CVaR estimate's own sampling distribution — no new statistical method invented.
    Outputs `rl/results/phase3_tail_validation_cvar_bootstrap.csv` (225 rows: 25 dates × 3 baselines ×
    3 quantiles), `rl/results/phase3_tail_validation_crux.csv` (9 rows: the crux tests below).

---

## Requirement #1 — severity ↑ → RL mean advantage ↑

Already established (`docs/experiments/04-crash-severity-expansion.md`: top 7/17 most-severe days = 7/7 RL wins on
mean IS). Not re-derived here per the task scope.

## Requirement #2 — severity ↑ → RL CVaR advantage ↑ — MET

Spearman correlation between each date's severity (`ac_mean_IS`, the same honest-λ severity proxy the
project already uses — recomputed from this run's own episodes, not the frozen `SEVERITY_RANK_V2`
constant, though the two agree closely) and the RL16-vs-AC-Optimal CVaR-95 delta, across all 25 dates:

**rho = −0.863, p = 0.0000** (more negative severity = more severe; the correlation is strongly
negative, meaning more-severe days pair with larger CVaR deltas — the same direction and, if
anything, a cleaner signal than the mean-based result).

Top-N severity table (AC-Optimal honest-λ mean IS as the severity ranking, most severe first):

| rank | date | regime | ac_mean_is | RL16 CVaR95 | AC CVaR95 | delta | result |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | 2024-08-05 | crash | −7.8887 | −7.0101 | −8.3861 | +1.3760 | **WIN** |
| 2 | 2024-07-04 | crash | −3.1441 | −2.6309 | −3.4013 | +0.7704 | **WIN** |
| 3 | 2024-07-05 | crash | −2.7938 | −2.3403 | −3.0955 | +0.7551 | **WIN** |
| 4 | 2024-03-05 | crash | −2.4602 | −1.4758 | −2.7393 | +1.2635 | **WIN** |
| 5 | 2024-08-02 | crash | −2.3675 | −2.1491 | −2.6035 | +0.4544 | **WIN** |
| 6 | 2024-07-07 | crash | −1.4180 | −1.1309 | −1.6463 | +0.5154 | **WIN** |
| 7 | 2024-08-04 | crash | −1.1021 | −0.8342 | −1.2309 | +0.3966 | **WIN** |
| 8 | 2024-04-15 | crash | −1.0946 | −1.4750 | −1.5320 | +0.0570 | **WIN** |
| 9–25 | (17 more, normal/calm) | | | | | | 2/17 WIN (2024-04-12 rank 13, 2024-06-10 rank 15) |

**Top 7 most-severe dates: RL16 CVaR-95 beats AC-honest on 7/7** — the exact same top-7 clean-sweep
pattern the mean-based severity finding reports, now confirmed to also hold at the CVaR-95 level.
Extends cleanly to rank 8 (04-15) as an 8th win. Remaining 18 (mild/calm) dates: 3/18 wins, and one of
those (2024-04-12) is the same mild-day exception the mean-based severity doc already flagged and
left unsmoothed — a nice small cross-check that the two analyses are seeing the same underlying
pattern, not two unrelated coincidences.

## Requirement #4 — robustness: bootstrap CIs, CVaR at 90/95/99% — MET

Paired bootstrap (10,000 resamples, matched episode seeds, same mechanics as
`phase2r_ablation_honest_lambda.py`) on the raw (non-centered) CVaR delta, both flagship crash dates,
all three quantiles, against all three baselines. Every one of the 18 tests below is significant at
p<0.0001 with a CI that does not cross zero:

**2024-03-05:**

| q | vs AC-Optimal | vs Adaptive-AC | vs TWAP |
|---|---|---|---|
| 90% | +1.2651 [+1.2631, +1.2716] p=0.0000 | +1.8519 [+1.8458, +1.8702] p=0.0000 | +3.0047 [+3.0026, +3.0245] p=0.0000 |
| 95% | +1.2635 [+1.2629, +1.2707] p=0.0000 | +1.8471 [+1.8445, +1.8619] p=0.0000 | +3.0027 [+3.0025, +3.0113] p=0.0000 |
| 99% | +1.2638 [+1.2626, +1.2707] p=0.0000 | +1.8492 [+1.8445, +1.8569] p=0.0000 | +3.0025 [+3.0025, +3.0089] p=0.0000 |

**2024-08-05:**

| q | vs AC-Optimal | vs Adaptive-AC | vs TWAP |
|---|---|---|---|
| 90% | +1.4027 [+1.3535, +1.4520] p=0.0000 | +1.3114 [+1.2650, +1.3588] p=0.0000 | +2.6369 [+2.5267, +2.7381] p=0.0000 |
| 95% | +1.3760 [+1.3479, +1.4590] p=0.0000 | +1.2853 [+1.2609, +1.3648] p=0.0000 | +2.6040 [+2.5019, +2.7442] p=0.0000 |
| 99% | +1.3479 [+1.3479, +1.4590] p=0.0000 | +1.2609 [+1.2609, +1.3648] p=0.0000 | +2.6038 [+2.4989, +2.7442] p=0.0000 |

The raw CVaR-95 headline in `docs/proposals/research-roadmap.md`'s table is not a fragile point estimate — it is
robust across the 90/95/99 quantile choice and has tight, non-zero-crossing CIs on both flagship
dates against every baseline. Full 25-date × 3-baseline × 3-quantile table in
`rl/results/phase3_tail_validation_cvar_bootstrap.csv`.

## Requirement #3 (THE CRUX) — mean-controlled tail advantage — NOT MET

**Method:** for each date and baseline, subtract each policy's own mean from its 50 episode outcomes
(mean-centering — removes the location/mean-cost shift, which is the already-established, separate
finding, without discarding the *shape* of each policy's own distribution around its own mean). Then
compare the residual (mean-removed) distributions': (a) CVaR-95 — is RL's residual left tail still
better once location is matched; (b) plain std — is RL less dispersed around its own mean. Both
tested with the same paired-index-resample bootstrap as requirement #4, with centering recomputed on
each resample (a proper bootstrap of the residual-distribution statistic, not of the raw statistic).

**Per-date, both flagship crash dates, against all three baselines:**

| date | baseline | std delta (RL−base) | p (RL less dispersed) | centered CVaR95 delta | p (RL better tail) |
|---|---|---:|---:|---:|---:|
| 2024-03-05 | AC-Optimal | −0.0016 [−0.0051, +0.0016] | 0.1712 | −0.0106 [−0.0137, −0.0035] | 0.9979 not sig. |
| 2024-03-05 | Adaptive-AC | −0.0055 [−0.0127, +0.0011] | 0.0508 | −0.0274 [−0.0328, −0.0125] | 0.9973 not sig. |
| 2024-03-05 | TWAP | +0.0105 [+0.0051, +0.0147] | 0.9999 | −0.0385 [−0.0417, −0.0303] | 0.9994 not sig. |
| 2024-08-05 | AC-Optimal | +0.0117 [−0.0280, +0.0445] | 0.7183 | −0.0313 [−0.0790, +0.0532] | 0.7389 not sig. |
| 2024-08-05 | Adaptive-AC | +0.0143 [−0.0240, +0.0460] | 0.7696 | −0.0324 [−0.0758, +0.0490] | 0.7579 not sig. |
| 2024-08-05 | TWAP | −0.0478 [−0.1015, +0.0003] | 0.0261 | +0.0115 [−0.0973, +0.1593] | 0.3639 not sig. |

**Every test against the two AC-family baselines (the comparison the phase headline is actually
about) is non-significant, on both dates, in both statistics.** Notably, the centered-CVaR point
estimate is *negative* (i.e. RL's residual left tail is marginally worse, not better, than AC's/AAC's)
on both crash dates against both AC-family baselines — the sign does not even point the direction a
real tail-mechanism claim would need, though the CIs are wide enough that this should not be read as
"AC has a better tail" either; it is genuinely inconclusive.

**Pooled across all 8 V2 crash-regime dates (n=400 paired episodes; each date's arrays centered on
its own mean before pooling, so the pooled test still isolates shape from location):**

| baseline | std delta | p | centered CVaR95 delta | p |
|---|---:|---:|---:|---:|
| AC-Optimal | +0.0050 [−0.0039, +0.0137] | 0.8545 | +0.0106 [−0.0067, +0.0282] | 0.1013 not sig. |
| Adaptive-AC | +0.0046 [−0.0040, +0.0132] | 0.8420 | +0.0096 [−0.0074, +0.0262] | 0.1242 not sig. |
| TWAP | −0.0119 [−0.0249, +0.0002] | 0.0279 | +0.0652 [+0.0235, +0.1056] | **0.0014 SIGNIFICANT** |

Larger sample size (n=400 vs n=50/date) does not change the verdict against the two baselines that
matter: still not significant against AC-Optimal or Adaptive-AC. The one significant result in this
entire section is a small, pooled effect against **TWAP specifically** (+0.065pp) — TWAP is the
weakest baseline, already dominated by RL on every other metric in this project by a wide margin, and
not the baseline the "tail risk management" claim is actually about. It does not rescue requirement
#3 for the comparison that matters.

**Caveats stated directly, not buried:**
- Non-significance is not proof of zero effect — n=50 episodes/date (and even n=400 pooled) with
  CVaR-95 using only the worst ~5% (2–20 points) is a high-variance statistic (the same caveat
  `evaluate.bootstrap_cvar_95_ci`'s own docstring already states for the raw, uncentered case). A
  larger n could in principle sharpen this. But the current evidence gives no positive signal to
  build on, and on two independent tests (dispersion and centered-CVaR) on two independent crash
  dates, the point estimates split roughly evenly in sign against AC/AAC rather than consistently
  favoring RL — that pattern is not what "underpowered but real" typically looks like.
- Episodes within one date are not fully independent draws (they share the same underlying day's real
  price data, differing only in `start_offset`) — a limitation already implicit in every existing
  bootstrap in this project's codebase, not new here, but slightly compounded when pooling across
  dates since it means the pooled n=400 has fewer truly independent degrees of freedom than 400.

---

## What this means for the phase headline

The phase's established mean-cost finding (`docs/methodology/standing-issues.md` issue #0) is unaffected
by anything here — it is a separate, already-robust result. What this document adds and subtracts:

- **Adds:** the same severity-scaling pattern replicates cleanly at the CVaR-95 level (#2), and the
  raw CVaR numbers themselves are statistically solid across quantile choice (#4) — worth citing
  alongside the mean result as corroborating evidence that RL's severe-day advantage is a real,
  broad shift in the outcome distribution, not an artifact of how "mean" happens to be computed.
- **Subtracts:** it forecloses describing this as a *second*, independent capability ("RL specifically
  manages tail/worst-case risk better than AC, beyond its mean advantage"). Requirement #3 is the
  test that would have supported that stronger claim, and it does not clear the bar. The correct,
  defensible statement is the narrower one: RL's mean-cost edge on severe days is large enough that it
  carries the whole outcome distribution — including its worst 5–10% — up with it. That is still a
  real and useful property (a desk cares about CVaR-95 in dollars, regardless of mechanism), but it
  should not be reported as evidence of a distinct risk-shaping behavior, because the data checked
  here does not show one.

## Artifacts

- `rl/experiments/phase3_tail_validation_eval.py` — per-episode eval/dump script (new).
- `rl/experiments/phase3_tail_validation_analysis.py` — four-requirement analysis script (new).
- `rl/results/phase3_tail_validation_episodes.csv` — 6,250 raw per-episode outcomes (25 dates × 5
  policies × 50 episodes), pre-Step-0 binary.
- `rl/results/phase3_tail_validation_summary.csv` — per-date mean/CVaR-95 summary, 25 dates.
- `rl/results/phase3_tail_validation_cvar_bootstrap.csv` — requirement #4, 225 rows (25 dates × 3
  baselines × 3 quantiles).
- `rl/results/phase3_tail_validation_crux.csv` — requirement #3, 9 rows (2 flagship dates + 1 pooled
  scope, × 3 baselines).
- `rl/results/phase3_tail_validation_episodes.POSTSTEP0_DISCARDED.csv.bak` /
  `phase3_tail_validation_summary.POSTSTEP0_DISCARDED.csv.bak` — kept, not deleted, as the evidence
  trail for the provenance finding above (the discarded post-Step-0 run that first revealed the
  binary-version issue). Not used in any result in this document.
