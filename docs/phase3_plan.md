# Phase 3 plan — state-signal repair, tail-risk validation, and a readable baseline triangle

Written 2026-08-13. This is the working plan and the reasoning behind it, recorded so it survives
between sessions. Supersedes nothing; it sits downstream of `docs/phase2r_standing_issues.md`.

---

## Where Phase 2R left us (one paragraph)

RL-16dim's advantage **scales with realized day severity**, not with membership in a named stress
event. Top 7 of 17 days ranked by severity: 7/7 RL wins, all significant. The coarse
"stress-window" version is dead (8/17, a coin flip) and is reported as dead. The result survived
two rounds of self-caught strawman baselines (hardcoded `AC_DEFAULT_LAMBDA=0.001`, then
hindsight-per-date λ selection). Diagnosed but unfixed: RL paces at ~33-36% of the horizon
regardless of regime while tuned AC's pace swings hard by regime — "RL learned execution but not
urgency."

---

## Findings established 2026-08-13 (new, not in the Phase 2R docs)

### A. The λ calibration set contains zero severe crashes — structural, not fixable

`TRAINING_DATES` is 7 dates: 2024-01-15, 02-20, 04-15, 06-10, 09-15, 10-15, 11-06. Per the
stress survey in `rl/evaluate.py`, only **2024-04-15** is even moderately stressed (depth-thinning
ratio 0.63); 02-20 / 09-15 / 10-15 were explicitly checked and rejected as liquidity events
(ratios >= 0.9). The only two severe crashes in the dataset are the held-out set.

**This is a bind, not a bug.** Calibrating λ on the crash dates is the hindsight leak Phase 2R
already killed. So AC is necessarily tuned on a crash-free sample, gets a patient λ=1e-6, and gets
caught by crashes. That is the honest deployment condition — a desk tuning λ in July 2024 also had
no August 5th to tune on.

**Required action: state this as the mechanism, not as a footnote.** The docs currently present
λ=1e-6 as a neutral technical choice. It is the entire reason AC gets caught, and it is the honest
reason. Also record the regime composition (0/7 severe) explicitly next to the λ value.

### B. λ and η are different kinds of object — current framing conflates them

- **η** is a physical estimate: how much does trading at this intensity move price? Estimable from
  execution data. Empirical literature reports a *range* of concave exponents, not a universal 0.5.
- **λ** is a preference: how much does this decision-maker trade off execution risk against cost?
  AC's own framework builds an efficient frontier and lets the user pick a point. There is no
  "correct BTC λ."

**Secondary issue:** λ=1e-6 was selected by minimizing training-set *mean* cost. A mean-only
objective quietly assumes risk-neutrality for the parameter that *is* the risk-aversion knob. A
defensible benchmark states the calibration objective before seeing test results — e.g. minimize
mean subject to a stated risk constraint.

### C. `sigma_ref` is a hardcoded constant — and the regime buckets can never fire

**This corrects `docs/phase2r_standing_issues.md` issue #6, whose stated mechanism is wrong.**
Issue #6 attributes the dead s10/s11 dims to a whole-day-calibrated `sigma_ref` plus a 5-sample
warmup. The code says otherwise:

```rust
const DEFAULT_SIGMA: f64 = 0.20;   // src/bin/rl_env.rs:168
sigma_ref: DEFAULT_SIGMA,          // :251 — no call site ever overrides it per date
```

`sigma_ref` is a fixed 0.20, unrelated to any date's data. **There is therefore no look-ahead leak**
(a natural worry, but not what the code does). The real defect is arithmetic. Measured per-step
realized volatility from `TradeData/AggTrades` (30s buckets, log returns):

| date | regime | per-step sd | `current_sigma` | ratio vs sigma_ref | true annualized |
|---|---|---:|---:|---:|---:|
| 2024-06-10 | quiet consolidation | 0.000253 | 0.0136 | **0.068** | 0.26 |
| 2024-01-15 | calm bull | 0.000435 | 0.0233 | **0.117** | 0.45 |
| 2024-07-10 | held-out calm | 0.000507 | 0.0272 | **0.136** | 0.52 |
| 2024-03-05 | CRASH cascade | 0.001366 | 0.0733 | **0.367** | 1.40 |
| 2024-08-05 | CRASH Yen unwind | 0.002009 | 0.1078 | **0.539** | 2.06 |

`regime_dims()` buckets at ratio < 0.7 (LOW) and > 1.3 (HIGH). **The ratio never exceeds 0.54 on
any date, calm or crash — so `vol_regime` is pinned at LOW everywhere, all day, by arithmetic.**
That is the complete mechanical explanation for the dims being flat and bit-identical across a
crash and a calm day.

Two compounding causes:

1. **`sigma_ref` hardcoded at 0.20** — instance **#4** of the project's canonical-value bug class
   (see the meta-lesson in `phase2r_standing_issues.md`). `realized_volatility` already exists in
   `src/calibration/impact_fit.rs` and is never wired to it.
2. **Annualization is wrong by ~19x** — `realised_vol()` multiplies per-step sd by
   `sqrt(total_steps)` = sqrt(2880) ~ 53.7. A year is ~2880 steps x 365 days, so the factor should
   be ~1025. `current_sigma` is not annualized volatility.

**Trap to avoid:** fixing only (2) puts calm at 0.45/0.20 = 2.25 and crash at 2.06/0.20 = 10.3 —
**both pinned at HIGH.** The dims would move from one dead rail to the other and look fixed. Both
causes must be fixed together.

### D. The continuous form already works — s2/s13 are the existence proof

`s2 = tanh(current_sigma / sigma_ref)` and `s13 = tanh(ratio - 1)` are already the continuous ratio
form of the same quantity. Against the table above, s2 = 0.068 on the quiet day vs 0.492 on the
crash — a ~7x separation, live in the state vector today. Same underlying signal; one pair kept it
continuous and works, the other rounded it into a bucket that can never flip.

**Decision (approved): replace s10/s11's 3-way hard buckets with a continuous ratio/proportion.**
Watch for tanh saturation: with a correct causal reference the crash ratio lands near 4, and
tanh(4) = 0.9993 — replacing a bucketing bug with a compression bug. The squashing needs revisiting
alongside the reference.

### E. Framing correction on linear-vs-sqrt

Do not write "AC is provably solving the wrong problem." Correct form:

> Our counterfactual execution environment uses an empirically-motivated concave impact
> specification; the AC benchmark internally assumes linear temporary impact. AC is therefore
> model-misspecified *relative to our chosen environment* — not relative to a proven law of markets.

Supporting distinctions worth keeping explicit in any writeup:

- Real historical price path is replayed, not modeled. The **counterfactual execution response**
  ("what would my hypothetical order have received, and how much of the move do I own?") is
  unobservable and must be modeled. Both are true simultaneously; real data does not eliminate
  simulation.
- Impact models do not predict the market path. They attribute price movement to one's own order.
- Cross-strategy fairness comes from a **shared environment**, not shared internal beliefs: same
  data, same execution model, same cost metric; AC may believe linear, RL may believe nothing.

---

## The tail claim — evidence bar and current status

Post-correction CVaR-95 from `rl/results/phase2r_ablation_honest_lambda.csv` (more negative =
worse):

| date | regime | RL16 | AC | Adaptive-AC | TWAP |
|---|---|---:|---:|---:|---:|
| 2024-03-05 | held-out crash | **-1.48** | -2.74 | -3.32 | -4.48 |
| 2024-08-05 | held-out crash | **-7.01** | -8.39 | -8.30 | -9.61 |
| 2024-07-10 | held-out calm | -0.61 | -0.36 | -0.34 | -0.35 |
| 2024-12-10 | held-out calm | -1.03 | -0.95 | -0.92 | -1.00 |
| 2024-10-15 | training | -1.38 | -1.16 | -1.15 | -0.72 |

This data exists but has **never been analyzed or written up** — the standing-issues doc and the
17-day severity run both report mean IS only; CVaR is not even a column in the severity CSV.

**Four-part bar before claiming "RL adapts better to worst-case scenarios":**

| # | requirement | status |
|---|---|---|
| 1 | severity up -> RL mean advantage up | **have it** (7/7 top-severity days) |
| 2 | severity up -> RL CVaR advantage up | data exists on 11 dates, unanalyzed; absent from the 17-day set |
| 3 | mean-controlled tail advantage | **missing — the crux** |
| 4 | robustness: bootstrap CIs, seeds, CVaR at 90/95/99 | missing |

**On #3, stated precisely.** The challenge is whether RL's CVaR edge is a distinct tail mechanism
or merely its mean edge restated. Subtracting the mean is the right concept, but the real question
is whether **dispersion and lower-quantile shape** still favour RL once location is matched.
**This requires per-episode outcomes, not the summary CSV** (which stores only mean and CVaR per
date) — so it needs an eval re-run with per-episode dumping. No training required, but not free.

---

## Plan

### Step 0 — Repair the regime signal (gates the retrain)
- Causal trailing-window `sigma_ref` (prior sessions/days), never today's completed day.
- Fix the annualization factor.
- Replace s10/s11 hard buckets with the continuous ratio; re-check tanh saturation.
- **Measure first-100-step crash-vs-calm separation before retraining anything.**
- Log `sigma_ref` as bug-class instance #4; grep for further hardcoded references per the
  meta-lesson (after any mirror/default fix, sweeping for siblings is routine, not on request).

### Step 1 — Tail validation (gates IQN) — independent of Step 0
- Add per-episode outcome dumping to the eval harness.
- Requirements #2, #3, #4 above.
- Add CVaR columns to the 17-day severity run.

### Step 2 — Retrain on the repaired state
- Re-measure whether RL's urgency now varies by regime.
- **Imitation learning only if it does not.** Note there is currently no clean expert to imitate:
  hindsight-λ AC leaks the oracle, honest-λ AC is regime-blind by construction, Adaptive-AC reads
  the same broken estimator, and the numerical optimizer is item 5. That alone sequences imitation
  after the state fix.

### Step 3 — Item 5: numerically-optimized sqrt-impact baseline
Promoted to the critical path. It is simultaneously (a) the readability leg of the comparison —
without it, "RL loses" cannot be distinguished from "AC's linear assumption costs little" or "both
are far from optimal"; (b) the only principled route out of the uncalibratable-η problem; and
(c) the only clean imitation expert.

### Doc fixes (cheap, do alongside)
- Correct issue #6's mechanism in `phase2r_standing_issues.md` (currently wrong — see C above).
- Restate the λ framing as preference-not-physics, with the 0/7-severe composition explicit.
- Soften "provably wrong problem" to the wording in E.

### Explicitly not now
IQN/DQN (neither exists in the codebase; the only trace is a comment at `rl/train.py:55` — gated on
Step 1's outcome). Action-space redesign. Multi-day episodes. LUNA/FTX (still blocked on the
Tardis overlap-date validation).

### Also cheap and unblocked, whenever wanted
- Wire per-date calibrated sigma into the AC baselines (CLI flag; the value already exists in Rust).
- Add a re-estimation-interval parameter to `AdaptiveACPolicy` and sweep it — it currently
  recomputes its entire schedule up to 2880x per episode, a plausible and never-tested mechanism
  for the long-standing "Adaptive over-trades during volatility clustering" observation.

---

## Standing rules carried forward
- Never compare across the Phase 2R Task 2/3 simulator provenance boundary. See
  `rl/results/PROVENANCE.md`.
- Canonical values are read from their single source at runtime, never hand-copied or defaulted.
  Four instances of this bug class so far, every one biasing the comparison toward this project's
  own method.
- Report every date, including mild and losing ones. Selection honesty at the pull stage is undone
  by a table that quietly drops the unremarkable days.
- Scope anchor: this is an offline execution-cost research study, not a live-trading system.
