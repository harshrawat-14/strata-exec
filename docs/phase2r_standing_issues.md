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

## Open — must survive into the final writeup's honest-limitations section

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

## Meta-lesson: hand-maintained mirrors of canonical values (second occurrence)

**Any value that exists in two places will eventually disagree. Canonical values get read from
their single source, never copied.** This is the second time in this project a hand-maintained
mirror of a canonical value caused a silent, real bug:

1. **CVaR sign inversion** — several call sites reimplemented "CVaR-95" locally (some as
   `np.percentile(arr, 5)`, a single point not a tail mean; the Rust/`parallel_sim.py` siblings had
   an outright sign inversion, averaging the *best* tail instead of the worst). Fixed by
   `compute_cvar_95` becoming the one implementation everything shares.
2. **Action-fraction table divergence** (this phase) — `rl/evaluate.py` hand-maintained
   `V5_FRACTIONS`/`V6_FRACTIONS` as mirrors of the real Rust `ACTION_FRACTIONS`, which silently
   diverged (matched at 3/20 indices for the `n_state_dims=12` case). Fixed by Python querying the
   real table from the Rust binary instead of copying it.

Both bugs were structurally identical: a value defined once canonically, copied by hand into a
second location for convenience, and the copy silently rotted. **Applying this forward:** any new
constant, table, or formula that needs to be used from both Rust and Python (or from two places in
either language) should be read from one canonical source at runtime, not hand-transcribed. If a
review ever finds a second copy of a value that has an authoritative source elsewhere, that is a
bug by construction, not a style nit — fix it the same way these two were fixed.

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
