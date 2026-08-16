# Phase 3 — Full platform audit: why the signal repair didn't improve RL

Written 2026-08-16, triggered by the Step 2 negative result. The question asked was blunt and
correct: the changes were meant to improve RL and the score went backwards, so what broke between
planning and implementation?

**Answer: one severe pre-existing bug, three secondary defects. The plan was sound; the training
pipeline never delivered the data the plan specified.**

---

## 1. CRITICAL — training saw 3 of 16 dates and ZERO crash dates (FIXED)

`train.py`'s `make_env`:

```python
def _init():
    date = random.choice(multi_dates)   # runs ONCE per worker
```

`SubprocVecEnv` calls `_init` exactly once per worker, so each worker was pinned to a single
randomly-drawn date **for the entire run**. An N-date training set was really trained on at most
`n_envs` dates, chosen by luck.

The Step 2 run (16 V2 dates, `n_envs=4`) trained on **2024-06-10, 2024-09-15, 2024-04-13 (×2)** —
three distinct dates, **zero crash dates**. The whole purpose of the V2 split (4 crash dates in
training) never reached the learner.

**This explains every Step 2 symptom:**

| symptom | explanation |
|---|---|
| responsiveness +0.0022 (regime-blind) | never saw a regime change |
| t50 0.47 vs RL16's 0.26 (over-patient) | all 3 training dates reward patience |
| −2.94pp on 2024-08-05 | crash dynamics were out-of-distribution |

**The state repair was NOT at fault.** `s10` spans −0.99…+0.99 and separates cleanly. But the
trained dates sit at s10 mean −0.75 / −0.81 / +0.12 while 2024-08-05 sits at **+0.85** — the policy
was never exposed to the positive region it needed. The signal worked; the pipeline starved it.

**Scope: this degraded every prior multi-date run in the project, RL16 included.** RL16 trained on
at most 4 of V1's 7 dates.

**Fix:** each worker owns one date, assigned round-robin, with `n_envs` auto-raised to the date
count — so workers jointly cover every date *and* every PPO batch contains all regimes rather than a
random subset. Verified 16/16 dates, mix {calm 6, normal 6, crash 4}. Faster than before (fps 7892
vs 7067, no respawns) at unchanged memory. A per-episode-resample fallback (`StrataExecEnv.lob_dates`)
covers `n_envs < n_dates`; measured ~15× slower on short episodes, so it is the fallback only.

---

## 2. Train/eval horizon mismatch (NOT fixed — deliberate)

Training randomizes the horizon to **200–1440** steps (`RANDOM_HORIZON_MIN`/`MAX`); evaluation runs
at **2880** (`HISTORICAL_DEFAULT_STEPS`). The policy is never trained at the horizon it is scored on
— at minimum a 2× extrapolation, at worst 14×.

Partially mitigated: `s1` is normalized time remaining, and actions are fractions of *remaining*
inventory, so both self-normalize somewhat. Not fully — pacing learned over ≤1440 steps is being
applied to 2880.

**Left unfixed on purpose:** changing it now would add a second variable to the corrective retrain
and repeat the attribution problem that made Step 2 uninterpretable. It is a live candidate for
further gains and should be its own change.

---

## 3. `CALIBRATED_SIGMA_REF` calibrated on V1, now used with V2 (NOT fixed — recorded)

Self-inflicted, from the Step 0 work. The constant is `0.4999`, the median annualized realized vol
across V1's **7** `TRAINING_DATES`. Training now uses V2's **16** dates, whose median is **0.6444**
— the reference is ~22% low relative to the distribution it now normalizes.

Consequence is mild, not breaking: ratios inflate by ~1.29×, shifting `s10` positive by ln(1.29) ≈
0.26 pre-tanh. The signal still spans the full range and still separates by regime, so nothing is
destroyed — but "s10 = 0" no longer means "a typical training day," which is the property the
constant was chosen to give.

**Not changed mid-run**, because it would alter the state encoding underneath the training job now
executing. Revisit after: either recalibrate to the V2 median or state explicitly that the reference
is a frozen V1 constant and not a V2-centred one.

---

## 4. Reward is terminal-heavy, but NOT misaligned (no bug — quantified)

Measured per-step reward magnitudes:

| date | mean wait reward | mean trade reward | ratio |
|---|---:|---:|---:|
| 2024-06-10 (normal) | −3.0e−08 | −3.2e−05 | trading **1,062×** costlier |
| 2024-08-05 (crash) | −4.9e−06 | −4.8e−05 | trading **10×** costlier |

Waiting is nearly free per step on a calm day, and the asymmetry correctly *shrinks* by ~100× on a
crash day — the shaping does encode regime. The counterweight to near-free waiting is the terminal
forced-liquidation penalty, and it does work directionally:

| date | policy | cumulative reward | IS (raw, +ve = shortfall) |
|---|---|---:|---:|
| 2024-06-10 | never trade | −0.03207 | +2.845% |
| 2024-06-10 | TWAP | −0.00465 | +0.676% |
| 2024-08-05 | never trade | −0.07790 | +9.622% |
| 2024-08-05 | TWAP | −0.02760 | +9.465% |

Reward and IS agree in direction on both dates — never-trading is worse on both measures. **There is
no sign or alignment bug.** What this does confirm, quantitatively, is the credit-assignment problem
already recorded in the standing issues: almost all of the signal against over-patience arrives at
the horizon, while the per-step gradient on a calm day says "don't trade" by three orders of
magnitude. That is hard to propagate over 2880 steps and is a plausible structural limit on learned
urgency — independent of the data bug in §1.

**Note for readers of both code paths:** raw `info["total_IS_pct"]` is positive-is-worse (it is a
shortfall), while `evaluate.py`'s harness negates it so higher-is-better. Both are internally
consistent; do not compare them directly.

---

## What this changes

- **The Step 2 negative result is retracted as evidence about the state repair.** It measured a
  policy that never saw a crash date. It says nothing about whether a working regime signal helps.
- **The Step 2 conclusion about RL16's selection leak still stands** — that was measured on RL16's
  own published numbers, not on the retrained model.
- **§1 also weakens confidence in RL16's training itself**, which drew at most 4 of 7 V1 dates at
  random. Any claim about what RL16 "learned from the training set" should name the dates it
  actually received.

## Status

Corrective retrain (`phase3_step2b_allcoverage`) running with full 16-date coverage confirmed in
its log. It still differs from RL16 in reward scale, training set, and selection protocol, so a
good result will still not be attributable to the state repair alone — the clean-selection replica
remains the run that isolates that.
