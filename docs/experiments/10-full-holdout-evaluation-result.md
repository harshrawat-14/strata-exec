# Phase 3 Step A — full 9-date held-out evaluation (result)

Written 2026-09-01. Predictions pre-registered in `docs/experiments/10-full-holdout-evaluation-predictions.md` before
`rl/phase3_stepA_full_holdout_eval.py` was run — see that file for the reasoning locked in beforehand.

## What changed

Every prior headline number (`docs/experiments/09-policy-capacity-and-entropy.md`, and the phase-headline memory) evaluated
RL-vs-AC on only 4 of the 9 `HELD_OUT_DATES_V2` dates. The other 5 (`2024-08-02`, `2024-08-03`,
`2024-08-04`, `2024-07-11`, `2024-07-12`) were already correctly held out of training and already used
for the pacing/urgency metric, but never run through the IS comparison table. This step ran exactly
that — same checkpoint (`phase3_step4_capacity_entropy_best`), same honest λ=1e-6, same 20 episodes,
same policy set, same harness (`rl/evaluate.py::evaluate_all_strategies_on_date`) as Steps 3/4. Single
variable changed: date coverage. Full output: `rl/results/phase3_stepA_full_holdout.csv`.

## Result: RL − AC-Optimal (mean_IS, all 9 held-out dates)

| date | regime | severity (ac_mean_IS) | RL − AC | predicted | matched? |
|---|---|---:|---:|---|---|
| 2024-08-05 | crash | −7.89 | **+0.92** | win | ✓ |
| 2024-03-05 | crash | −2.46 | **+0.92** | win | ✓ |
| 2024-08-02 | crash | −2.37 | **+0.37** | win | ✓ |
| 2024-08-04 | crash | −1.10 | **+0.26** | narrow loss/tie | ✗ (see below) |
| 2024-08-03 | normal | −0.78 | −0.31 | loss | ✓ |
| 2024-07-12 | normal | −0.58 | −0.58 | loss | ✓ |
| 2024-12-10 | normal | −0.72 | −0.46 | loss | ✓ |
| 2024-07-11 | calm | −0.06 | −0.59 | loss | ✓ |
| 2024-07-10 | calm | +0.03 | −0.80 | loss | ✓ |

**4/5 new-date predictions landed exactly. The headline pattern is now 4/4 crash-tier wins, 0/5
calm/normal losses — cleaner than the previous 2/2-vs-0/9 (same direction, denser evidence, not a
different story).**

## The one miss, investigated rather than waved through

`2024-08-04` was predicted a narrow loss/tie (matched to `2024-04-15`'s near-tie precedent, both at
~−1.1 day-averaged severity) but came in a clean win. Two checks before accepting it:

1. **Leakage check:** `2024-08-04` was never in `TRAINING_DATES_V2` (Window C kept entirely held out
   by construction, verified via the disjointness assert in `rl/evaluate.py`). No training leak.
2. **Cross-model replication:** `docs/experiments/04-crash-severity-expansion.md` already evaluated this exact
   date months earlier under a completely different, independently trained checkpoint (`RL16`,
   pre-state-repair) and got the same qualitative result: a clean win (+0.38pp, p=0.0000). Two
   unrelated models, two project phases, same direction — the win is not an artifact of this specific
   checkpoint.
3. **Depth-thinning check** (`rl/phase3_stepA_full_holdout_eval.py` is the eval; the depth check was
   ad hoc, not committed as a script): a local-window search for the bid-depth minimum around each
   day's price minimum (calibrated loosely against the documented 03-05 signature — offset 59s here
   vs. the originally-documented 47s, ratio 0.56 here vs. 0.22 documented; directionally consistent,
   not an exact reproduction, different depth-aggregation choice). Result: `2024-08-04`'s book thins
   to **66% of its daily median depth within 3.5 minutes of its price low** — much closer to the two
   known crash signatures (56%, 78%) than to `2024-04-15`, which was the comparison date used to make
   the prediction and which actually *thickens* to 143% at its own low (the opposite of a liquidity
   event). **The day-averaged severity score treats 04-15 and 08-04 as equivalent; their intraday
   liquidity-stress profiles are not equivalent.** That is the source of the missed prediction — the
   wrong comparison date was picked, not a flaw in the underlying pattern.

**Not fully explained:** `2024-08-02` shows flat depth (ratio ≈1.04, no real thinning) yet still won
(+0.37). Depth-thinning is a partial explanation, not a complete one — consistent with a similar
unexplained mild-day win already on record (`2024-04-12`, `docs/experiments/04-crash-severity-expansion.md`).
Recorded honestly as an open loose end, not smoothed into the narrative.

## What this changes

- Phase headline strengthens: **4/4 crash-tier wins, 0/5 calm/normal losses** across all held-out
  dates, not a 4-date subset. Update the standing "n=2 crash dates" caveat everywhere it's cited —
  it should now read n=4 (with the caveat that 3 of the 4 belong to one macro event window, so this is
  n=2 independent events at denser within-event sampling, not 4 independent trials).
- `2024-08-04` and `2024-08-02` should be added to any future headline table alongside `03-05`/`08-05`.
- Does NOT touch the near-duplicate-episode problem (audit finding 5, still open) — more dates does
  not address within-date episode correlation.
- Does NOT resolve the pacing dissociation (RL wins crashes while pacing less urgently) — untouched by
  this step, still the standing unexplained mechanism question.

## Known limitations carried forward

- 2024-08-02/03/04/05 are 4 days of one macro event (Yen-unwind window); not 4 independent crash
  samples.
- Depth-thinning replication is approximate, not an exact match to the original 03-05 finding's
  numbers — useful directionally, not as a precise, reusable metric without further calibration.
- 2024-08-02's win remains unexplained by the depth-thinning proxy.
