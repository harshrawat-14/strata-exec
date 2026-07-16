# StrataExec — Standing Open Issues (Phase 2R)

Living record of findings that belong in the final writeup's honest-limitations section, or that
are deferred by explicit decision rather than resolved. Kept here so they don't evaporate between
sessions. Cross-referenced from `docs/phase2r_objective_bias_finding.md`.

---

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

## Open — must survive into the final writeup's honest-limitations section

### 1. Hindsight trend-following in single-path historical backtesting (IS-metric only)
Single realized historical paths reward directional timing **in the IS-vs-arrival reporting
metric** (not the training reward, which is drift-neutral — confirmed by reading the code).
Predates this incident; applies to v5's methodology too. Candidate fix: multi-day episodes, or
intraday price-path randomization. **Deferred to post-v2-core** — do not implement speculatively;
not shown to bite training outcomes.

### 2. Train/eval state-support mismatch: dims 8/9 (own-impact fractions)
Dim 6 fixed this phase (Task 3). Dims 8/9 remain dead in `historical` mode (the eval/ablation
harness) while live in `counterfactual` mode (training) — architectural, not a bug: historical mode
has no `cf_lob` by design ("pure replay, no causal feedback"). **Requires an explicit decision**
before any state-design claim: either switch the eval harness to counterfactual mode (breaks
comparability with all prior historical-mode-based reports back to Phase 0/1), or accept and
document the asymmetry. Not resolved as of this entry — gates Task 4.

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

## Deferred by explicit decision — log only, do not action before post-v2-core

These are named directions, not commitments, so they don't evaporate — none are to be started now:

- **Action-space redesign**: continuous actions, non-linear/finer buckets near the low end, or
  actual limit orders instead of the current 20-bucket discrete market-order fractions. Would give
  the agent finer control than the current `ACTION_FRACTIONS` ladder allows, particularly around
  the "small continuous slicing" region.
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
