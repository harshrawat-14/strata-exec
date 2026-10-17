# Phase 3 Step 2b — retrain with full date coverage

Written 2026-08-16. Rerun of Step 2 after the audit found that the original run trained on 3 of 16
dates and **zero crash dates** (`docs/methodology/platform-audit.md` §1). This run covers all 16 V2
training dates, regime mix {calm 6, normal 6, crash 4}, with clean best-model selection.

**Two questions, two different answers. The performance regression was the data bug. The urgency
finding was not — it survives the fix.**

---

## Result 1 — implementation shortfall: crash advantage restored

Canonical harness, counterfactual, honest λ=1e-6, n_episodes=20 matched, each model on the binary
whose state encoding it was trained on. Values are RL − AC-Optimal (positive = RL better).

| date | regime | RL16 | Step 2 (3 dates, 0 crash) | **Step 2b (16 dates)** |
|---|---|---:|---:|---:|
| 2024-03-05 | crash | +1.27 | +0.99 | **+1.10** |
| 2024-08-05 | crash | +1.48 | **−2.94** | **+1.21** |
| 2024-07-10 | calm | −0.25 | +0.43 | −1.42 |
| 2024-12-10 | normal | −0.07 | −0.50 | −0.20 |

**2024-08-05 recovered 4.15pp** (−2.94 → +1.21), confirming the date-coverage bug as the cause of
the Step 2 regression rather than the state repair.

**The recovered result is stronger than RL16's despite the smaller number.** RL16's +1.48 came from
a checkpoint selected partly *by its score on 2024-08-05* (the selection leak, fixed before this
run). Step 2b's +1.21 is a genuinely held-out number: the date was never in selection. A slightly
smaller effect measured cleanly is worth more than a larger one measured with the answer in hand.

**The regime-conditional profile sharpened.** Step 2b is better than RL16 on neither crash date but
clearly worse on calm (−1.42 vs −0.25 on 2024-07-10). Wins concentrate on crashes, losses on calm —
which is the project's mechanistic thesis stated more sharply, not more weakly.

## Result 2 — pacing is still NOT regime-responsive (Step 2's actual question)

`t50` = fraction of horizon to 50% sold. 9 V2 held-out dates × 3 seeds.

| policy | crash | normal | calm | **responsiveness** (calm − crash) |
|---|---:|---:|---:|---:|
| AC-Optimal (honest λ) | 0.2611 | 0.2611 | 0.2611 | +0.0000 |
| TWAP | 0.5461 | 0.5461 | 0.5461 | −0.0000 |
| RL16 | 0.2515 | 0.2882 | 0.2793 | **+0.0278** |
| Step 2 (broken data) | 0.4646 | 0.5041 | 0.4668 | +0.0022 |
| **Step 2b (fixed data)** | 0.4584 | 0.4145 | 0.4368 | **−0.0216** |

**Step 2b is slightly *less* urgent on crash days than on calm days — the wrong direction.** Giving
the policy a working regime signal AND four crash dates in training still did not produce
regime-responsive pacing. **The Step 2 conclusion on urgency stands; only its IS numbers were
invalidated by the data bug.**

### The dissociation worth naming

Step 2b **beats AC on both crash dates while being less urgent on crash days, and while pacing far
slower than AC overall** (t50 ≈ 0.44 vs AC's 0.26). So its crash-date advantage **does not come from
timing**. That is direct evidence for the project's existing "RL learned execution but not urgency"
framing — and it is now measured on a policy that had every opportunity to learn urgency and didn't.

This should be treated as a finding, not a footnote: the mechanism behind RL's crash advantage is
**not** the one the residual-action-space and imitation-learning proposals were designed to supply.
Both were premised on importing better pacing. A policy that wins on crashes with *worse* pacing
suggests the payoff from fixing pacing is smaller than assumed, or that the advantage lives
somewhere else entirely (per-fill execution quality being the standing candidate).

## Caveats — what still cannot be attributed

Step 2b differs from RL16 in four ways (state repair, ~58× reward scale, V1→V2 training set,
leaky→clean selection). The **IS recovery** is safely attributed to the date-coverage fix, because
Step 2 and Step 2b differ *only* in date coverage. Nothing else here is isolated:

- Whether the state repair helps at all is still unmeasured — Step 2b confounds it with three
  other changes.
- The clean-selection RL16 replica (V1 data, old reward, pre-Step-0 state, clean selection) remains
  the run that isolates the selection effect.

**Also unresolved and material:** every number here inherits audit finding §5 — all eval episodes
start within 9.5 minutes of each other, so crash-date variance is understated ~30× and p-values are
not credible. The *signs* above are trustworthy; their significance is not.

Finding §6 applies too: the policy sat at ~98% of uniform entropy for the whole run, so all of these
numbers likely sit below what this setup can reach with more timesteps.

## Artifacts

- `rl/models/phase3_step2b_allcoverage_best/best_model`
- `rl/results/phase3_step2_urgency_step2b.csv`
