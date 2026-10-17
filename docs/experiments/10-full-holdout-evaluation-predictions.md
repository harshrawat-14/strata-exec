# Phase 3 Step A — pre-registered predictions (full 9-date held-out evaluation)

Written 2026-09-01, BEFORE running `rl/phase3_stepA_full_holdout_eval.py`. Purpose: the project has
twice built a comparison that looked like a clean finding and turned out to be a strawman
(hindsight-λ AC, near-duplicate eval episodes). This file exists so the outcome can't be
retrofitted to whatever the numbers say — predictions are locked in first, checked after.

## What's changing

Every prior headline number (`docs/experiments/09-policy-capacity-and-entropy.md`) evaluated the RL-vs-AC comparison on
only 4 of the 9 `HELD_OUT_DATES_V2` dates: `2024-03-05`, `2024-08-05` (crash), `2024-07-10` (calm),
`2024-12-10` (normal). The other 5 held-out dates — `2024-08-02`, `2024-08-03`, `2024-08-04`
(crash/normal, Window C), `2024-07-11`, `2024-07-12` (calm, Window B tail) — were already trained-out
(never in `TRAINING_DATES_V2`) and already used for the pacing/urgency metric, but never run through
the IS comparison table. This step runs exactly that, and nothing else. Same checkpoint
(`phase3_step4_capacity_entropy_best`), same honest λ=1e-6, same 20 episodes, same policy set.

## Severity classification of the 5 new dates (from `SEVERITY_RANK_V2`, `rl/evaluate.py`)

| date | ac_mean_IS | regime |
|---|---:|---|
| 2024-08-02 | -2.368 | crash |
| 2024-08-04 | -1.102 | crash |
| 2024-08-03 | -0.783 | normal |
| 2024-07-12 | -0.577 | normal |
| 2024-07-11 | -0.063 | calm |

## Predictions (RL − AC-Optimal mean_IS, positive = RL better)

Given the standing thesis (RL wins where AC's static-schedule/linear-impact assumptions break under
genuine severe stress; loses where they hold):

- **2024-08-02** (crash, -2.368, close to 08-05's own -7.9 in rank if not magnitude): predict RL
  **wins**, comparable in kind to the 03-05/08-05 wins, though 08-02 is 3 days before the flagship
  crash and its own severity score is far milder than 08-05's -7.889 — treat as "elevated stress",
  not a second flagship crash.
- **2024-08-04** (crash by rank, -1.102, almost identical severity to 2024-04-15's -1.095 which was
  a near-tie -0.02pp loss in the V1 headline): predict a **narrow RL loss or near-tie**, NOT a clean
  win. A clean RL win here would be inconsistent with the existing "severe stress only" framing and
  would need explaining, not just reporting.
- **2024-08-03** (normal, -0.783): predict RL **loses**, consistent with every other normal/calm date
  on record.
- **2024-07-11** (calm, -0.063): predict RL **loses**.
- **2024-07-12** (normal, -0.577): predict RL **loses**.

## What would falsify or complicate the standing thesis

- RL winning on 2024-08-03, 2024-07-11, or 2024-07-12 (calm/normal dates) — would break the clean
  regime-conditional pattern and require re-examining the mechanism claim, not just updating a count.
- RL losing clearly on 2024-08-02 (a genuine crash-tier date) — would suggest the 2/2 crash-win
  pattern was itself a small-sample artifact rather than a regime effect.
- 2024-08-04 landing as a clean win rather than a narrow loss/tie would be the specific outcome that
  needs the most scrutiny, since it's the closest severity-matched date to a known near-tie (04-15)
  and a clean win there would be the most likely place for a hidden leakage or measurement issue to
  show up first.

## Known limitations, stated up front (not discovered after)

- 2024-08-02/03/04/05 are 4 days of ONE macro event (the Yen-unwind window). They are not 4
  independent samples of "a crash" — moving from n=2 to n=4 crash dates here is really moving from
  n=2 independent events to n=2 independent events with denser sampling around one of them. State
  this explicitly in the result writeup; do not describe it as quadrupling the crash-day evidence.
- This does not touch the near-duplicate-eval-episode problem (audit finding 5, still open). More
  dates does not fix within-date episode correlation.
- The RL checkpoint (`phase3_step4_capacity_entropy_best`) was trained on `TRAINING_DATES_V2`, which
  excludes all of Window C (08-02..08-05) and the late half of Window B (07-10..07-12) by
  construction — confirmed via the disjointness assert in `rl/evaluate.py`. No leakage risk from
  training on these 5 new dates.
