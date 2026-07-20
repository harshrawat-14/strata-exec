# Design sketch: residual/delta action space (multiplier on a reference schedule)

**Status: design sketch only, per explicit instruction — no implementation.** This is a candidate
direction for a future, separate phase, not a Task 4 or Phase 2R deliverable. It is logged here so
the reasoning behind it survives, motivated directly by the granularity investigation above (RL's
regime-blind fixed pacing vs AC's regime-adaptive tuned pacing).

## The idea

Replace the current action semantics ("sell X% of remaining inventory, X ∈ 20 fixed buckets") with:

```
action = multiplier ∈ [0, ~2.0]   (fine-grained near 1.0)
executed_fraction = reference_schedule(t, regime) × multiplier
```

where `reference_schedule` is AC-Optimal's own closed-form recommendation at the current step,
re-derived per date/regime the same way `ACOptimalPolicy._get_schedule` already does (or its
Adaptive-AC variant, using live volatility). The agent no longer outputs an absolute per-step size —
it outputs a *deviation* from what a tuned classical schedule would already do at that instant.

## Why this directly targets what was just found

The granularity investigation (above) found RL's actual failure mode is not bucket-forced chatter —
it's that RL learns one roughly-fixed execution pace and applies it regardless of regime, while
tuned-AC's pace varies correctly by regime (extremely patient on calm days, sharply front-loaded on
severe-crash days). Under the current action space, matching AC's regime-adaptive pace would require
the agent to discover, from a delayed and mostly-terminal reward, an entire 2880-step schedule *shape*
for every regime it might encounter — effectively re-deriving AC's own closed-form result from scratch,
per regime, through trial and error. A residual/delta action space removes that burden: the
regime-appropriate pace is *given* to the agent as the reference schedule, and the only thing left to
learn is when and how much to deviate from it — a much lower-dimensional, more localized credit
assignment problem than learning absolute sizing over a full multi-thousand-step horizon.

## What this buys, specifically

- **Inherits AC's schedule shape.** The agent starts from a regime-appropriate pace by construction,
  rather than needing to rediscover pacing from a sparse terminal reward — directly addressing the
  mechanism the granularity investigation identified.
- **Puts resolution where it matters.** A multiplier space fine-grained near 1.0 (e.g. denser buckets
  or continuous in [0.7, 1.3], coarser toward the [0, 2.0] extremes) concentrates precision on "trade
  slightly more/less than the reference recommends," which is plausibly the economically important
  range, rather than the current ladder's coarse 0.5→0.7→0.9→1.0 top end mattering only when someone
  wants to dump most of the order in one shot.
- **Guarantees an AC-parity floor.** `multiplier ≡ 1.0` for every step reproduces AC's schedule
  exactly (whichever reference AC variant is chosen). This means a policy that learns nothing useful
  at all still matches the (properly-tuned) classical baseline — the failure mode this whole close-out
  gate has been diagnosing (a learner doing *worse* than a correctly-tuned heuristic) becomes
  structurally much harder to reach by construction, rather than being an emergent property that has
  to be hoped for.
- **Structurally removes the absolute action-0 corner.** Both of this phase's collapse diagnoses (the
  v6r wait-collapse, and the earlier forced-liquidation reward loophole) were the policy converging
  onto literal action-index-0 ("do nothing") as a degenerate optimum. In multiplier space, action-0
  equivalent (`multiplier=0`) is just as reachable numerically, but it no longer coincides with "the
  simplest constant output a collapsing policy might drift toward" in the same way, since the
  reference schedule itself already ensures *something* is always being sold at `multiplier=1`; a
  collapse toward a constant multiplier near 1.0 is much less catastrophic than a collapse toward
  constant action-0 under the current space.
- **Composes with the imitation warm-start (item 6, still on hold) rather than competing with it.**
  Behavior-cloning a tuned AC policy is nearly trivial to define in this space (target multiplier
  ≡ 1.0 everywhere) and gives a well-defined, low-variance BC target, unlike cloning AC's absolute
  bucket choices under the current 20-action space (which would need to imitate AC's own bucket-
  rounding artifacts, including the granularity-floor cliff documented above).

## What this costs / risks

- **Invalidates the 16-dim vs 12-dim ablation and both collapse diagnoses' framing.** Every result
  produced this phase reasons about behavior in terms of the current 20-bucket absolute action space
  (the entropy-floor fix, the device/resume isolation, the reward-loophole fix, the fraction-table
  bug, and now this granularity investigation). Changing the action space doesn't just add a new
  result, it changes the ground every existing result was measured on — the same reasoning already
  used to defer the earlier "uniform vs non-linear vs continuous buckets" idea in
  `docs/phase2r_standing_issues.md`. **This needs to be its own phase**, not a Task 4 addendum.
- **Reference-schedule choice is itself a design decision, not free of the sqrt-vs-linear question.**
  If the reference is AC-Optimal (linear-impact-misspecified per the calibration design note), the
  agent inherits a misspecified prior — better than no prior, but not obviously the *right* prior. If
  item 5's numerically-optimized sqrt-impact schedule is ever built, that would be the more
  principled reference to use instead. Worth deciding in sequence, not in parallel, if both are ever
  pursued.
- **Multiplier bounds and resolution need their own tuning.** `[0, ~2.0]` is a placeholder; the right
  range depends on how far the truly-optimal deviation from AC ever needs to go, which isn't known
  yet — plausibly informed by whatever item 5 finds about the sqrt-vs-linear gap, if that work
  happens.
- **New failure modes are possible, not just removed.** E.g., a multiplier space could make it easier
  for a policy to *drift* the schedule cumulatively off-track over many steps in a way that's harder
  to diagnose from action histograms than the current space's discrete, directly-interpretable bucket
  choices (this design sketch does not address that; would need its own analysis before
  implementation).

## Not decided here

Whether to pursue this at all, which reference schedule to use, exact multiplier bounds/resolution,
and how it interacts with the imitation-warm-start idea are all open — this document exists to record
the design and its motivating evidence, not to commit to building it.
