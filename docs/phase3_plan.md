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

## V2 date split — 25-date pool, regime-balanced — DONE 2026-08-15

The `TradeData` pool grew from 11 to 25 dates (the 14 severity-expansion dates were always real
data, just never assigned a train/held-out role). Implemented in `rl/evaluate.py` as
`SEVERITY_RANK_V2` / `TRAINING_DATES_V2` / `HELD_OUT_DATES_V2`, additive — **V1's `TRAINING_DATES`/
`HELD_OUT_CRASH_DATES` are untouched**, so every existing result remains reproducible exactly as
before. V2 is for new work (this session's Step 1/s11 work, and Step 2's eventual retrain) going
forward.

**Why:** V1's training set has 0/7 severe-crash days by construction (finding A above) — a real,
earned result for "how does a desk calibrate λ with no future crash to tune on," but it cannot
answer "does a model that has seen *some* crash dynamics generalize to a crash it hasn't seen,"
because it never sees any. V2 answers that second question.

**Method:** classified all 25 dates by AC's own honest-λ realized cost (`ac_mean_IS`, the same
severity proxy the severity-expansion doc already validated — not a new metric), tercile split into
crash/normal/calm. The three multi-day event windows (A: 04-12–15, B: 07-04–12, C: 08-02–05) were
assigned to train/held-out as *whole windows or at one temporal boundary*, never interleaved
day-by-day, to avoid leaking one event's dynamics across the train/held-out boundary:
- Window A: kept entirely in training (extends the existing 04-15 precedent).
- Window B: split at one boundary — early/acute six days (07-04–09, includes 3 of its 4 crash days)
  to training, later three (07-10–12, calmer) stay held-out (07-10 was already a V1 anchor).
- Window C: kept entirely held-out — these are the direct run-up to the 08-05 flagship crash
  holdout; training on any of them would leak into the comparison this project's headline result
  depends on most.

**Result — crash exposure is now balanced, not absent:**

| regime | train | held-out |
|---|---:|---:|
| crash | 4 | 4 |
| normal | 6 | 3 |
| calm | 6 | 2 |

16 train / 9 held-out. Both crash holdout anchors (2024-08-05, 2024-03-05) remain held out.
Guarded by `rl/tests/test_date_split_v2.py` (9 tests, all passing) — asserts no overlap, full
25-date coverage, training crash exposure ≥3, held-out crash exposure ≥2 with both flagship dates
present, every regime ≥2 on both sides, and window C's no-leakage property specifically.

---

## Plan

### Step 0 — Repair the regime signal (gates the retrain) — DONE 2026-08-13
Implemented in `src/bin/rl_env.rs`:
- `CALIBRATED_SIGMA_REF = 0.4999` — median annualized realized vol across the 7 `TRAINING_DATES`
  only (never held-out/eval dates — same protocol as λ). Wired as the automatic default for
  historical/counterfactual mode in `parse_cli()`, so every caller (all of `rl/environment.py`'s
  launches included) gets it with no per-call-site change; `--sigma-ref` still overrides explicitly.
- `TRADING_STEPS_PER_YEAR` fixed constant replaces the old `sqrt(cfg.total_steps)` annualization,
  which was both off by ~19x AND kept varying with the randomized per-episode training horizon
  (200–1440 steps) — a second bug that would have masked the first if fixed alone (see the trap
  noted in section C).
- `regime_dims()` (s10, s11) rewritten from 3-way hard buckets to `tanh(ln(ratio))` — continuous,
  doesn't saturate at the now-reachable crash-day ratios (~2.8–4.1x), sign convention: positive =
  more stressed than the training-calibrated reference on both dims.
- New executable regression test: `tests::regime_dims_separate_crash_from_calm_in_first_100_steps`.
- 29/29 existing `rl-env` tests + 208/208 lib tests still pass (one stale test literally asserting
  the old `{0.0, 0.5, 1.0}` set was updated to assert the new signed-tanh range instead).

**Result, measured via the real CLI/Python runtime path (`rl/environment.py`, no test-only
shortcuts), first-100-step mean:**

| date | regime | s10 (vol) | s11 (liq) |
|---|---|---:|---:|
| 2024-08-05 | severe crash | **+0.678** | -0.012 |
| 2024-03-05 | milder crash | **+0.342** | +0.006 |
| 2024-07-10 | held-out calm | **-0.219** | -0.141 |
| 2024-12-10 | held-out calm | **-0.120** | +0.032 |

**s10 is fixed and clean:** both crash dates positive, both calm dates negative, ordered by
severity (severe > milder crash > either calm day). This is the signal issue #6 needed.

**s11 diagnosed further (2026-08-13) — genuine null result for early-window liquidity regime
detection, not patched.** Root cause is now understood and independently confirmed two ways
(Python straight off the raw `TradeData/BookDepth` CSVs, bypassing the simulator entirely, and the
real Rust/Python CLI pipeline via `rl/environment.py` driving `./target/release/rl-env`) — two
compounding factors, not one:

1. **`half_spread()` is a weak liquidity-thinning proxy for this data.** Binance's `bookDepth`
   format bins depth at *fixed percentage distances from mid* (±1%, ±2%, ...), cumulative from the
   mid outward. `LobReplay`'s `diff_levels()` recovers a "best bid/ask price" for the innermost band
   as `notional/depth` — a volume-weighted average price of *everything resting within that whole
   1%-wide band*, not a literal top-of-book quote. That quantity tracks the distributional *shape*
   of resting liquidity inside a fixed-width window, not the *total depth* available — so real
   depth withdrawal barely moves it. Measured directly: even evaluated exactly at each date's true
   minimum-depth snapshot, `half_spread`'s ratio to its day median is 0.90–1.08 on all four dates
   tested (both crashes included) — no crash signal at all, at any point in the day.
2. **Raw depth (`bid_depth_up_to`, the same quantity already live as s4) is a substantially better
   liquidity signal, but the real event it detects is late and extremely localized.** A
   `tanh(ln(depth_ref/current_depth))` reformulation peaks at +0.85 (2024-08-05) and +0.89
   (2024-03-05) — but at step **1743/2880** and **2395/2879** respectively (60% and 83% through the
   trading day), matching the project's own stress survey almost exactly (independently recovers
   2024-03-05's "depth drops to 22% of daily median" finding: measured ratio 0.218). Both held-out
   calm dates show a much lower peak (+0.46, +0.69) at similarly late points, confirming the
   direction is right. **But within the first 100 steps — the window this project tests and needs
   for early urgency decisions — neither `half_spread` nor this raw-depth reformulation shows any
   separation** (first-100-step means all sit in ±0.15 noise band, confirmed on the real CLI path:
   -0.04/-0.07/-0.00/+0.15 for 08-05/03-05/07-10/12-10), because on both held-out crash dates the
   liquidity-thinning event genuinely has not happened yet 50 minutes into the day. This is a fact
   about *when* these two real crashes occurred intraday, not an arithmetic bug reachable by
   reformulating a same-step signal — no causal (no-lookahead) quantity computed from data up to the
   current step can report an event that hasn't occurred yet.

This explains why s10 and s11 diverge post-fix even though both had a real underlying signal to
recover: **volatility is a sustained, whole-day regime property on these dates** (current_sigma
reads elevated from the first 100 steps onward, consistent with pre-crash jitters / GARCH-style
persistence), while **liquidity thinning on Binance BTCUSDT is a sharp, late-arriving, ~1-2-snapshot
event that mean-reverts fast** — not a regime the way volatility is one. Swapping the s11 formula
(half_spread → raw depth) would not have passed an s10-style first-100-step regression test either
(verified directly, not assumed), so implementing it would have been exactly the cosmetic fix this
phase was warned against forcing. **Treated as a genuine null result, same rigor bar as the VPIN
finding (`docs/phase2r_standing_issues.md` issue #4): s11/liq_regime is not a usable early-decision
signal at this data's resolution, for the two crash dates this project has ground truth for — left
unfixed, not patched to look better. Do not build anything (residual-action-space reference,
Adaptive-AC diagnosis) that depends on s11 being informative early in an episode.** A same-step
"is the book thin right now" reading (raw depth vs. `depth_ref`, already close to what s4 exposes)
remains a materially truer quantity than the current `half_spread`-based s11 and could be worth
swapping in as a coincident (not early-warning) liquidity reading in a future phase if one is
motivated to do so — but that is a different, weaker claim than "regime detection," and is not
implemented here since it does not fix the diagnosed problem.

Corrected in `docs/phase2r_standing_issues.md` issue #6 (append-only correction, original wrong
mechanism kept visible) and logged as meta-lesson bug-class instance #4.

### Step 1 — Tail validation (gates IQN) — independent of Step 0 — DONE 2026-08-13, mixed result
Full writeup: `docs/phase3_tail_validation.md`. Per-episode dumping added
(`rl/phase3_tail_validation_eval.py`, all 25 V2 dates, not just 17). Requirement #2 (severity → CVaR
advantage) and #4 (bootstrap robustness, CVaR-90/95/99) **MET** — top-7-severity CVaR clean sweep,
same pattern as the mean result; raw CVaR deltas significant at p<0.0001 across all quantiles on both
crash holdouts. **Requirement #3 (the crux — mean-controlled tail advantage) NOT MET**: once each
date's episodes are mean-centered, RL's residual tail/dispersion is not distinguishably better than
AC-Optimal's or Adaptive-AC's, per-date or pooled (n=400, 8 crash-regime dates). Read as: the CVaR
edge is RL's already-known mean edge propagating into the tail, not a separate risk-shaping
mechanism — do not claim "RL manages tail risk" as a finding distinct from the mean-cost result.
Also surfaced a standing methodological hazard: RL16's checkpoint predates Phase 3 Step 0's state
repair, so evaluating it with the current (post-Step-0) `rl_env.rs` silently feeds it
out-of-distribution state and materially changes its numbers — this document's results all use a
pre-Step-0 binary built in an isolated worktree; see the doc's opening section before running any
further eval of the current RL16/RL12 checkpoints.

### Step 1.5 — Consistency repair after Step 0 (DONE 2026-08-13) — required before Step 2

An audit of the Step 0 change found four wiring defects. Three were live; one was a false alarm
caught during verification and is recorded so it is not "rediscovered" later.

**Fixed:**
1. **`sigma_ref` mirrored in Python (bug-class instance #5, self-inflicted).** Step 0 recalibrated
   the Rust constant and left `rl/evaluate.py`'s `AdaptiveACPolicy(sigma_ref=0.20)` (live, never
   overridden) and `rl/batched_environment.py`'s (dead — declared, never forwarded, so callers'
   values were silently ignored). Now read from the binary via a new `--print-env-constants`
   flag, mirroring `--print-action-fractions`. `batched_environment` now actually forwards
   `--sigma-ref`, defaulting to `None` so the binary applies its own per-mode default.
2. **Adaptive-AC's volatility inversion was wrong** — `s2` is `tanh(ratio)`, not
   `tanh(ratio − 1)` (that is `s13`), but the code inverted with a spurious `1.0 +`. Pre-existing,
   not introduced by Step 0.
3. **Guardrails mis-scaled after recalibration** — the arctanh clip (0.999) saturated at ratio
   3.80 and the sigma clamp at 2.0, both binding on the severe-crash range.

**Also found: `s2` cannot represent crash-day volatility at all.** Once the policy trades, own
impact feeds back into realized vol and 2024-08-05 reaches sigma **6.57** (ratio ~13), where
`tanh` is 1.0 to within 6e-12. Inverting `s2` there caps at 4.20 — a **36% understatement on the
single most severe date**. `s10 = tanh(ln(ratio))` applies the log first and inverts exactly, so
`AdaptiveACPolicy` now prefers `s10`, falling back to `s2` only on pre-Step-0 binaries (where
`s10` is the old dead bucket). This is an unplanned second payoff from the Step 0 encoding choice.

**Measured impact on Adaptive-AC** (5 seeds, honest λ; deltas only — absolute values use a
shorter horizon than the canonical eval): −0.062 pp on 2024-08-05, +0.079 pp on 2024-03-05,
−0.001 / +0.056 pp on the held-out calm dates. **The headline is unaffected** — RL16's margins
over Adaptive-AC on the crash dates are +1.32 / +1.87 pp, ~20× the shift. But no Adaptive-AC
number should be described as "AC with live volatility re-estimation" until re-run.

**Not a bug (recorded to prevent rediscovery):** a raw-protocol probe appeared to show
`current_sigma` frozen at 0.2 for a whole episode. That was an artifact of the probe using
`{"cmd":"step"}` instead of the real wire format `{"action":N}`. With the correct protocol,
`current_sigma` warms up at `sigma_ref` and tracks realized volatility correctly.

**New regression coverage** (`rl/tests/test_evaluate.py`): Python/Rust `sigma_ref` identity; an
AST-based sweep that fails if any float-literal `sigma_ref` default reappears under `rl/`; and an
inversion check that verifies the reconstruction against the environment's own `current_sigma`
(now emitted in the info dict) rather than against itself — including the high-sigma trading
regime. Verified the inversion test *would* have caught the original bug: it errs 20–38% against
a 0.1% tolerance.

**Reward-scale boundary recorded** in `rl/results/PROVENANCE.md`: Step 0 raised the wait-penalty
term ~58×, so Step 2's retrain optimizes a different objective than RL16/RL12 did. State repair
and reward change are confounded unless separated by an ablation.

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
- ~~Correct issue #6's mechanism in `phase2r_standing_issues.md`~~ — done alongside Step 0.
- Restate the λ framing as preference-not-physics, with the 0/7-severe composition explicit. Not
  yet done — still owed in `phase2r_standing_issues.md`/`phase2r_objective_bias_finding.md`.
- Soften "provably wrong problem" to the wording in E. Not yet done.

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
