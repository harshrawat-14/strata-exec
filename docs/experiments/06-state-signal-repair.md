# Phase 3 Step 2 — Retrain on the repaired state: negative result

Written 2026-08-13. Step 2 asked one question: with the volatility signal repaired (Step 0 took
`s10` from bit-identical-across-regimes to cleanly severity-ordered), does the retrained policy's
execution pace finally respond to regime?

**Answer: no. It got slightly worse, and it lost the crash-date advantage that is the project's
headline. Reported as a negative result, not softened.**

## Run

Counterfactual, 16 V2 training dates, CPU, 500k timesteps, `phase3_step2_16dim_v2`.
Best-model selection used **training dates only** (4 crash-regime) — the selection leak present in
the RL16 run was fixed first (see `docs/proposals/research-roadmap.md`). Policy did **not** collapse: 4–8 distinct
actions per episode, varied over the horizon.

## Result 1 — pace is still regime-blind (the question Step 2 exists to answer)

`t50` = fraction of horizon elapsed when 50% of inventory is sold. Lower = more urgent.
9 V2 held-out dates × 3 seeds.

| policy | crash | normal | calm | **responsiveness** (calm − crash) |
|---|---:|---:|---:|---:|
| AC-Optimal (honest λ) | 0.2611 | 0.2611 | 0.2611 | **+0.0000** |
| TWAP | 0.5461 | 0.5461 | 0.5461 | **−0.0000** |
| RL16 (pre-Step-0) | 0.2515 | 0.2882 | 0.2793 | **+0.0278** |
| **Step 2 (repaired state)** | 0.4646 | 0.5041 | 0.4668 | **+0.0022** |

Two readings, both uncomfortable:

1. **The repair did not produce regime-responsive pacing.** Step 2's responsiveness is +0.0022 —
   indistinguishable from zero, and *lower* than the model trained on the broken signal (+0.0278).
   Giving the policy a working regime signal did not make it use one.
2. **Step 2 is far more patient overall** (t50 ≈ 0.47 vs RL16's ≈ 0.26). It holds inventory much
   longer on every regime. This is the opposite of what the ~58× larger wait-penalty predicts, and
   points at the training-set change (V2 is 6 calm + 6 normal + 4 crash; patience pays on 12 of
   those 16) rather than the reward.

**Metric sanity checks pass:** AC-Optimal reads exactly +0.0000, matching the documented fact that
static AC's schedule is a pure function of (λ, σ, η, T) and date-independent except through λ;
TWAP reads −0.0000, uniform by construction. The metric detects responsiveness where it exists and
zero where it does not.

**Correction to an earlier note in this session:** a 1-seed/2-date smoke run put RL16's
responsiveness at +0.1014 and prompted the observation that RL16 might already be the responsive
one. At n=9 dates it is **+0.0278** — the direction survives (RL16 does respond more than AC's
literal zero) but the magnitude was overstated ~4x by the small sample. The sharper framing still
holds: honest-λ AC is *structurally incapable* of regime response, so "RL is regime-blind" was
always measured against hindsight-λ AC, which only looked adaptive because λ was picked per date.

## Result 2 — the crash-date advantage did not survive

Canonical harness (`evaluate_all_strategies_on_date`), counterfactual, honest λ=1e-6,
**n_episodes=20 matched across both models**. Each model run against the binary whose state
encoding it was trained on.

| date | regime | RL16 | Step 2 | AC-Optimal | RL16 vs AC | Step 2 vs AC |
|---|---|---:|---:|---:|---:|---:|
| 2024-03-05 | crash | −1.122 | −1.397 | −2.388 | **+1.27** | **+0.99** |
| 2024-08-05 | crash | −6.661 | −11.081 | −8.144 | **+1.48** | **−2.94** |
| 2024-07-10 | calm | −0.411 | +0.273 | −0.161 | −0.25 | **+0.43** |
| 2024-12-10 | normal | −0.744 | −1.173 | −0.676 | −0.07 | −0.50 |

Step 2 **loses 4.4pp of edge on 2024-08-05**, the most severe date in the dataset and the flagship
result — turning a +1.48pp win into a −2.94pp loss. It holds a narrower win on 2024-03-05 and gains
on one calm date.

**Cross-binary comparability confirmed empirically, not just by argument:** AC-Optimal and TWAP
return *bit-identical* values on both binaries (−2.388206109043805 and −4.171769065913786). Nothing
in the counterfactual fill/impact/price path depends on the Step 0 changes, exactly as the
code-classification in `../proposals/research-roadmap.md` predicted.

## Why this cannot be attributed — four simultaneous changes

This run differs from RL16 in **four** ways at once, so no single cause can be assigned:

1. **State signal** — repaired (the intended change).
2. **Reward scale** — wait-penalty ~58× larger (Step 0 side effect, recorded in `PROVENANCE.md`).
3. **Training set** — V1's 7 dates → V2's 16 dates, a different regime mix.
4. **Selection protocol** — RL16 was selected with 2024-08-05 and 2024-03-05 *in the eval set*;
   Step 2 was not.

**(4) deserves emphasis, because it is the honest candidate for much of the regression.** RL16's
+1.48pp on 2024-08-05 was obtained by a checkpoint chosen partly *by its score on 2024-08-05*.
Step 2 never saw that date in selection. Some of the apparent "regression" is plausibly RL16's
crash-date number deflating back toward its true held-out value once the leak is removed — which is
precisely the concern that motivated fixing the leak. **That is a real possibility, not an excuse,
and it is not yet separated from (1)–(3).**

## What this does and does not change

**Does not change:** the Phase 2R result stands as reported *for the model and protocol it was
measured under*. The AC-Optimal arithmetic is untouched, and the severity-scaling finding is
unaffected by anything here.

**Does change:** "RL16 beats AC on 2/2 held-out crash dates" now carries a second qualifier
alongside the n=2 caveat — the checkpoint was selected using those dates, and a cleanly-selected
retrain does not reproduce the effect on the more severe of the two. Until the confounds are
separated, the honest statement is that **the crash-date advantage is not established under clean
selection**.

## Next — the ablation is now required, not optional

Separating (1)–(4) needs targeted runs, cheapest and most informative first:

1. **Clean-selection RL16 replica** — V1 training set, old reward scale, pre-Step-0 state, but
   training-dates-only selection. Isolates (4) alone, and directly tests whether the crash-date
   advantage was a selection artifact. **This is the highest-value single run**: if the advantage
   disappears here, (1)–(3) are largely moot.
2. **V2 training set, old reward scale** — isolates (2) from (3).
3. Only then attribute anything to the state repair itself.

Do not run imitation learning or the residual action space until this is resolved — both were
premised on "RL learned execution but not urgency," and the pacing evidence underneath that framing
is now in question.

## Artifacts

- `rl/models/phase3_step2_16dim_v2_best/best_model`, `..._final.zip`
- `rl/experiments/phase3_step2_urgency_eval.py` — pace metric (t50/t90), defined explicitly in its docstring
- `rl/results/phase3_step2_urgency_step2.csv`, `rl/results/phase3_step2_urgency_rl16.csv`
