# StrataExec — Standing Open Issues (Phase 2R)

Living record of findings that belong in the final writeup's honest-limitations section, or that
are deferred by explicit decision rather than resolved. Kept here so they don't evaporate between
sessions. Cross-referenced from `docs/phase2r_objective_bias_finding.md`.

---

## Simulator provenance boundary (read before comparing ANY two results)

**Every result produced before Phase 2R's Task 2 (impact calibration) and Task 3 (drift
restoration) fixes was measured against a materially different simulator than every result
produced after.** This is not a mode choice (historical vs. counterfactual) — it is a change to
what the counterfactual simulator itself computes. Comparing across this boundary without
accounting for it produces a false read. See `rl/results/PROVENANCE.md` for the per-file table of
which existing results are pre-correction (all of them, as of this entry — v5's headline numbers,
`phase1_tail_risk.csv`, `unified_comparison.csv`, `failure_analysis.csv`, and the invalid
`phase2_ablation.csv`). Task 4's ablation (16-dim vs 12-dim, counterfactual mode) will be the first
post-correction result; tag it and everything after it clearly, and never present a pre/post
number pair as if they were measured on the same ruler.

## Resolved this phase (kept here for the record, not action items)

- **Forced-liquidation reward loophole** (`5630e5d`) — fixed and tested.
- **`price_factor` rescaling in historical mode** (`5630e5d`) — fixed and tested.
- **Objective-bias / VWAP / "observability limit" findings** — retracted; were artifacts of
  measuring IS-vs-arrival instead of the actual training reward. See the finding doc's correction.
- **Late-training policy collapse root cause** — isolated by elimination to device (MPS); resume
  mechanics exonerated directly. CPU adopted as the operational training recipe.
- **Impact-model mislabeling** — relabeled (OW-style decay structure + square-root magnitude, not
  "Obizhaeva-Wang" as a whole). Calibration engine wired in.
- **Counterfactual-mode drift flattening** (`price_factor` in `CounterfactualLob`) — fixed, dim 6
  now live in training.
- **`spread_ref` scale-mismatch bug** in `reset_counterfactual` — fixed; was flooring the
  punitive-liquidation price at ~$0 for 80-90% of any forced dump.
- **Train/eval state-support gap (dims 8/9), formerly issue #2 below** — resolved by decision:
  Task 4's ablation runs entirely in counterfactual mode (both training and eval), so dims 6, 8, 9
  are all live and consistent by construction. Historical mode's asymmetry is unchanged but no
  longer gates anything, since the ablation doesn't use historical mode.
- **Classical-baseline action-fraction table bug** — `TWAPPolicy`/`HeuristicPolicy`/
  `ACOptimalPolicy`/`AdaptiveACPolicy` hand-maintained two mirrors (`V5_FRACTIONS`/
  `V6_FRACTIONS`) of the real environment's action->fraction table that had silently diverged
  (the `n_state_dims=12` mirror matched the real Rust `ACTION_FRACTIONS` at only 3/20 indices).
  Root-fixed: Python now queries the real table from the Rust binary (`--print-action-fractions`)
  instead of hand-copying it, with a cross-language test asserting identity. See the meta-lesson
  below — this is the second instance of this exact failure mode in the project.
- **Third `ACTION_FRACTIONS` mirror in `rl/environment.py`** — found during the Task 4 close-out
  implementation-consistency audit, currently byte-identical to the real table by luck (unprotected
  by any test), still imported by `rl/phase2r_vwap_gate.py`. Root-fixed the same way: `environment.py`
  is now the one canonical Python-side query point; `evaluate.py` imports from it instead of
  querying independently. Identity-asserting regression test added. Third instance of this bug
  class — see the meta-lesson below, sharpened further.
- **Action-granularity/bucketing as the cause of the inverted RL-vs-AC pattern** — investigated and
  mostly ruled out. All classical baselines are bucketed identically to RL (no continuous-execution
  privilege exists for anyone). The subtler "coarse buckets disproportionately distort AC's schedule"
  version was quantified directly and found to be small precisely at AC's actual best λ — the
  un-swept λ (issue #0 above) is the dominant mechanism, not granularity.

## Open — must survive into the final writeup's honest-limitations section

### 0. AC/Adaptive-AC baseline was not fairly parametrized — the close-out gate's headline finding
Two independent issues, both found while investigating why the Task 4 corrected ablation showed RL
beating AC/AAC on calm dates but losing on the harder crash date (a pattern inverted from the
project's working thesis):
- **Un-swept λ.** `AC_DEFAULT_LAMBDA=0.001` was used identically for every date; a wide λ-grid sweep
  on the 4 anchor dates (post-correction simulator, counterfactual mode) shows AC/AAC at their own
  best λ tie-or-beat RL-16dim on 3 of 4 dates (both held-out calm dates, and the harder held-out
  crash date), reversing the un-swept table's story. See `docs/phase2r_objective_bias_finding.md`,
  "Task 4 close-out gate — Gate B."
- **Hardcoded, uncalibrated market-model parameters.** `ACOptimalPolicy`/`AdaptiveACPolicy` default
  to `sigma=0.20, eta=0.10` and no call site in `rl/evaluate.py` ever overrides either — every
  construction only varies `lam`. The simulator's own per-date calibrated impact coefficient and
  realized volatility (Task 2's `CounterfactualLobConfig::calibrated`, `realized_volatility` in
  `impact_fit.rs`) are never exposed to Python (no CLI flag analogous to
  `--print-action-fractions`), so AC's schedule shape reflects generic textbook assumptions with
  zero connection to what the simulator calibrated for that date. **Not yet fixed** — would need a
  new CLI flag to expose calibrated sigma/eta per date, plus re-running the λ-grid with those values.
- **Net effect:** any "RL beats classical execution baselines" claim from Phase 2R should be treated
  as unresolved, not established, until both of the above are addressed. The RL-vs-RL ablation
  (16-dim vs 12-dim) is unaffected by either issue.

### 1. Hindsight trend-following in single-path historical backtesting (IS-metric only)
Single realized historical paths reward directional timing **in the IS-vs-arrival reporting
metric** (not the training reward, which is drift-neutral — confirmed by reading the code).
Predates this incident; applies to v5's methodology too. Candidate fix: multi-day episodes, or
intraday price-path randomization. **Deferred to post-v2-core** — do not implement speculatively;
not shown to bite training outcomes.

### 2. Train/eval state-support mismatch: dims 8/9 (own-impact fractions) — RESOLVED for Task 4
Dim 6 fixed this phase (Task 3). Dims 8/9 remain architecturally dead in `historical` mode (no
`cf_lob` by design) while live in `counterfactual` mode — this is unchanged and would still gate
any *future* work that needs historical mode specifically. For Task 4, resolved by decision: the
ablation runs entirely in counterfactual mode for both training and eval, so this doesn't bite.
Moved to "resolved this phase" above; kept here only as a pointer for future historical-mode work.

### 3. Punitive-price extrapolation for forced-liquidation residual beyond real book depth
`worst_price − 5×spread_ref`, a simple linear offset, prices whatever portion of a forced dump
can't fill against real snapshot depth (typically 80-92% of a full-inventory dump, since
`total_inventory` defaults to an institutional-sized 50,000 BTC order). The scale bug that made
this floor at ~$0 is fixed; the *shape* of the extrapolation (linear offset from the worst filled
price) remains a simplification. A more principled deep-liquidity/impact model would be worth
building. **Deferred** — not blocking, the current formula produces defensible (if imprecise)
magnitudes post-fix.

### 4. VPIN null result on Binance perpetuals
Structural: 24/7 trading, arbitrage-saturated, no clean informed/uninformed order-flow split to
dilute VPIN's signal on this venue. OFI is the appropriate toxicity signal here instead. A real,
reportable finding, not an implementation failure.

### 5. Start-offset eval bound (fixed-horizon eval only)
Training episodes are safely bounded (`start_offset + horizon ≤ len` always holds for the training
horizon range, verified with a test). The *fixed* eval horizon (2880 steps) does not satisfy this
bound for `start_offset > 0`, but all evals use small seeds (0-19, <1% of the episode), so results
are unaffected in practice. Bounding `start_offset ≤ len − total_steps` would remove the caveat
entirely. **Deferred** — cosmetic, not measured to affect any reported number.

## Meta-lesson: hand-maintained mirrors of canonical values (THIRD occurrence — pattern confirmed)

**Any value that exists in two places will eventually disagree. Canonical values get read from
their single source, never copied.** This is the third time in this project a hand-maintained
mirror of a canonical value caused (or risked) a silent, real bug — enough occurrences now that this
should be treated as a predictable failure mode, not a series of one-off surprises:

1. **CVaR sign inversion** — several call sites reimplemented "CVaR-95" locally (some as
   `np.percentile(arr, 5)`, a single point not a tail mean; the Rust/`parallel_sim.py` siblings had
   an outright sign inversion, averaging the *best* tail instead of the worst). Fixed by
   `compute_cvar_95` becoming the one implementation everything shares.
2. **Action-fraction table divergence** (this phase) — `rl/evaluate.py` hand-maintained
   `V5_FRACTIONS`/`V6_FRACTIONS` as mirrors of the real Rust `ACTION_FRACTIONS`, which silently
   diverged (matched at 3/20 indices for the `n_state_dims=12` case). Fixed by Python querying the
   real table from the Rust binary instead of copying it.
3. **A third copy of the same `ACTION_FRACTIONS` table** (found during the Task 4 close-out audit,
   same phase) — `rl/environment.py` had its own independent hardcoded copy, byte-identical only by
   luck, unprotected by any test, silently still imported by `rl/phase2r_vwap_gate.py`. This one
   hadn't diverged *yet*, but existed under the exact same conditions (hand-copied, untested, two
   places) that caused #2 to diverge. Fixed the same way, plus an identity (`is`) test to make a
   fourth copy structurally harder to introduce by accident.

All three were structurally identical: a value defined once canonically, copied by hand into a
second (or third) location for convenience, with nothing enforcing agreement. Instance #3 is the
sharpest data point yet: it was found not by a table diverging and breaking something, but by
**actively grepping for the pattern on request** — meaning the first two occurrences did not, on
their own, prompt a proactive sweep for more copies of themselves; it took an explicit "assume a
third mirror-bug exists until proven otherwise" audit to surface it. **Applying this forward:** any
new constant, table, or formula that needs to be used from both Rust and Python (or from two places
in either language) should be read from one canonical source at runtime, not hand-transcribed — and
after *any* mirror-bug fix, a grep for other copies of the same value should be routine, not
something that only happens when explicitly requested.

## Deferred by explicit decision — log only, do not action before post-v2-core

These are named directions, not commitments, so they don't evaporate — none are to be started now:

- **Action-space redesign: uniform vs. non-linear vs. continuous, run as a comparison, not a
  pick.** The current 20-bucket `ACTION_FRACTIONS` ladder is uniform-ish and, per this phase's
  fraction-table investigation, wastes resolution: the difference between selling 1% and 3% matters
  far more to execution quality than between 90% and 95%, and the current ladder spends buckets
  evenly rather than concentrating them where they matter. Three variants worth comparing head to
  head rather than just picking one: (a) the current uniform-ish discretization (baseline), (b)
  non-linear/finer buckets concentrated near small sells, (c) continuous actions (needs a different
  RL algorithm, e.g. SAC/PPO-continuous, and has its own stability considerations). Likely ranking
  by payoff-per-effort: non-linear buckets > continuous > uniform (current) — but that ranking is a
  guess, not a finding; the point of running it as a comparison is to not have to guess. **Hard
  deferral reason, not just scheduling:** changing the action space invalidates every result and
  both collapse diagnoses built this phase (the entropy-floor fix, the device/resume isolation, the
  reward-loophole fix all reason about behavior in terms of the current 20-action space). Doing this
  now would re-litigate the collapse, the reward, and the ablation simultaneously on shifting
  ground. Self-contained future study, arguably its own phase — after v2-core closes.
- **Interval-VWAP as the primary reporting benchmark** (not the training reward, which is already
  drift-neutral — this is about what the *comparison tables* report). Would make reported IS numbers
  drift-neutral too, addressing issue #1 above at the reporting-metric layer specifically. The
  `interval_volume`/`book_mid` telemetry and `rl/phase2r_vwap_gate.py` scaffolding from the
  (retracted) objective-bias investigation remain in the tree and are reusable for this.
- **Multi-day episodes**: span more than one historical day per training episode, or randomize
  which day's price path is spliced into an episode, to reduce any single day's realized drift from
  dominating a training batch. Complements the trend-following issue (#1) rather than replacing it.

## Scope anchor (restate for every future decision on this list)

> This is an **offline execution-cost research study**. The policy is trained and evaluated in a
> counterfactual market simulator built from historical Binance LOB data with a transient own-impact
> model. It is **not** a live-trading system. The sim-to-real transfer gap is a characterized
> limitation and a direction for future work, not a solved problem.
