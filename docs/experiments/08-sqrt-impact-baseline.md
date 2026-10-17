# Phase 3 Step 3 — numerically-optimized sqrt-impact baseline

Written 2026-08-22. Answers the question Step 3 was promoted to the critical path to
separate (`docs/proposals/research-roadmap.md`): Step 2b showed RL beats AC-Optimal on both held-out
crash dates while pacing *slower* and *less* urgently than AC overall, so the crash-date
advantage does not come from timing. That leaves two live explanations. (a) AC's
LINEAR-impact assumption is a poor fit to this project's real, empirically
square-root/transient-decay cost structure, making AC a weak strawman independent of RL.
(b) Both AC and RL are far from what a schedule-only policy can actually achieve here.

**`SqrtImpactOptimalPolicy` (`rl/evaluate.py`) answers this directly: it fits a
2-parameter Beta-CDF selling-curve family (a=b=1 is TWAP; a strict superset of AC's
monotone-decay shape) by running real episodes through the exact simulator under
evaluation and minimizing realized IS with Nelder-Mead — not a second closed-form
guess, the real cost model, numerically optimized.**

---

## A load-bearing methodology finding, found while building this

The first fit-seed design (seeds 900,000+, chosen only to be disjoint from the eval
range 0–19) produced a policy that scored far WORSE than AC-Optimal and even TWAP on
2024-08-05 — the opposite of what a superset-of-AC's-shape family fit against the true
cost model should ever do. Diagnosis, not assumption: re-ran the *same fixed schedule*
(and separately, plain TWAP) across several seed ranges on 2024-08-05 alone.

| seed range | policy | mean_IS |
|---|---|---:|
| 0–19 (the eval range) | TWAP | −9.44% to −9.5% |
| 20–29 | TWAP | −8.5% to −9.0% |
| 500–509 | TWAP | −2.7% |
| 1000–1009 | TWAP | −8.9% to −9.1% |
| 5000–5004 | TWAP | −3.8% to −4.0% |
| 900,000–900,019 | TWAP | −0.4% to −1.1% |

**The same policy, same date, varies by more than an order of magnitude (−0.7% to
−9.7%) depending purely on which handful of episodes get sampled.** `start_offset = seed
% max_start` restricts starts to the first third of the day, but that third is far from
homogeneous on this date. This is a sharper, quantified version of the platform audit's
finding 5 ("eval episodes are near-duplicates, crash-date statistics are not credible,"
`docs/methodology/platform-audit.md`) — every mean_IS anywhere in this project for
2024-08-05 (this table, Step 2, Step 2b, RL16) is scored against one arbitrary narrow
slice of a day whose true outcome distribution is far wider than that slice suggests.

**Fix:** fit seeds now start immediately at `n_eval_episodes` (adjacent to, not distant
from, the eval range) — disjoint (no leakage) but empirically the same near-duplicate
regime the eval range itself samples. `SqrtImpactOptimalPolicy.__init__` takes
`n_eval_episodes` explicitly rather than a hardcoded seed constant, so this is
mechanical, not a convention a future caller can forget. After the fix, fit-time and
held-out eval IS agree to within ~0.4pp on every date tested (previously off by up to
9pp) — the schedule now fits the number it is actually being scored on.

---

## Result: sqrt-impact-optimal beats AC-Optimal everywhere, RL beats both on crashes

Canonical harness, counterfactual, honest λ=1e-6, n_episodes=20 matched, model =
`phase3_step2b_allcoverage_best`. Values are mean_IS (higher/less negative = better).

| date | regime | AC-Optimal | Sqrt-Impact Optimal | RL Agent | TWAP |
|---|---|---:|---:|---:|---:|
| 2024-03-05 | crash | −2.3882 | **−2.0114** | −1.2863 | −4.1718 |
| 2024-08-05 | crash | −8.1440 | **−8.0957** | −6.9381 | −9.4373 |
| 2024-07-10 | calm | −0.1614 | **−0.1135** | −1.5832 | −0.1787 |
| 2024-12-10 | normal | −0.6763 | **−0.6195** | −0.8719 | −0.6937 |

Sqrt-Impact Optimal beats AC-Optimal on **all four** dates (+0.38pp on 03-05, +0.05pp on
08-05, +0.05pp on 07-10, +0.06pp on 12-10) — small everywhere except the milder crash
date, where AC's linear-impact mismatch costs a real 0.38pp. This confirms AC's shape is
close to, but not exactly, what the true concave/transient cost model rewards.

**The answer to the question Step 3 exists to settle: RL still beats the best available
schedule-only policy by a wide margin on both crash dates** (RL − Sqrt-Impact-Optimal:
+0.73pp on 03-05, +1.16pp on 08-05 — nearly identical to RL's +1.10pp/+1.21pp margin
over AC-Optimal in Step 2b). **AC was not a strawman.** Fixing AC's linear-impact
assumption recovers at most ~0.4pp of RL's crash-date edge; the rest is not explained by
impact-model mismatch. RL's crash-day advantage survives against the best static
schedule this project can numerically produce under the correct cost model — it must
come from something a fixed schedule structurally cannot do: reacting fill-by-fill to
the real, realized order book rather than committing to a shape in advance.

**On calm and normal days the pattern reverses and sharpens.** Sqrt-Impact Optimal is
the best strategy on 07-10 (beats AC by 0.05pp, TWAP by 0.07pp) and essentially ties
AC-Optimal as the best on 12-10, while RL is the *worst* strategy in the table on both
non-crash dates (−1.5832 on 07-10, more than 1pp behind everything else; −0.8719 on
12-10). Every schedule-only policy — AC, Sqrt-Impact-Optimal, TWAP, Heuristic — clusters
tightly on calm/normal days; RL is the outlier, and not in a good direction.

## What this changes going forward

- **Residual-action-space and imitation-learning proposals were premised on importing
  better pacing/impact-awareness from a stronger baseline.** Step 2b already showed
  RL's crash edge isn't about pacing; this shows it isn't about AC's impact-model
  mismatch either. Neither proposal's premise holds as stated — the advantage most
  plausibly lives in per-fill reactivity to realized book state, which is closer to
  what RL already does natively than what either proposal would add.
- Sqrt-Impact Optimal is now the strongest static/non-adaptive baseline in this
  project's comparison table and should be included (`include_sqrt_impact_optimal=True`
  in `evaluate_all_strategies_on_date`) wherever the question is "how much of AC's gap
  to RL is AC's linear-impact assumption" rather than left implicit.
- The seed-heterogeneity finding on 2024-08-05 is not sqrt-impact-specific — it applies
  to any future date-specific calibration or fitting work in this codebase. Any
  procedure that samples a handful of episodes from a single crash date risks fitting
  (or selecting a model checkpoint, or computing a p-value) against an arbitrary slice
  of that date rather than a representative one.

## Caveats

- `include_sqrt_impact_optimal` defaults to `False` in `evaluate_all_strategies_on_date`
  — fitting costs real wall-clock time (~15-45s/date at the settings used here) that
  every existing caller would otherwise pay silently. Opt in explicitly.
- The Beta-CDF family is a strict superset of AC's shape (a=b=1 is TWAP; AC's monotone
  decay is representable within it) but is still schedule-only and 2-parameter — it is
  the best *this* family can do, not a proof that no schedule could do better. A richer
  family (more control points) was not attempted here; the 2-parameter family was
  sufficient to answer the "is AC a strawman" question this step exists for.
- Inherits audit finding 5 in full: even with the fit-seed fix, 20 near-duplicate
  episodes on a single date still understate true variance (this doc's own seed-range
  experiment is direct evidence of how much). Signs are trustworthy; significance is
  not, for every number in this table including the new one.

## Artifacts

- `rl/evaluate.py`: `SqrtImpactOptimalPolicy`, `query_calibration`.
- `src/bin/rl_env.rs` / `src/market/counterfactual_lob.rs`: `--print-calibration` CLI
  flag and `CounterfactualLob::config()` getter, so Python can read back the exact
  per-date calibrated impact_coefficient/sigma/adv_btc the environment used (no
  hand-copied second estimate).
- `rl/experiments/phase3_step3_sqrt_impact_eval.py`: the script that produced this table.
- `rl/results/phase3_step3_sqrt_impact.csv`
- `rl/tests/test_evaluate.py::TestSqrtImpactOptimalSchedule`,
  `::TestSqrtImpactOptimalFitIntegration`
