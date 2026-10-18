# 13 — M0 under protocol v3: predictions (pre-registered)

Written 2026-10-18 while M0 trains, before any M0 number exists.

## What is being measured

**M0** = the Step 4 recipe (1.5M steps, `ent_coef` 0.01, constant LR, 16-dim state,
counterfactual, CPU), retrained under protocol v3
(`docs/methodology/evaluation-protocol-v3.md`): a working per-episode seed stream,
1,920-step episodes with random starts and sizes, the 12 `TRAINING_DATES_V3`, and
best-model selection on the 4 `VALIDATION_DATES_V3`.

Scored on the 4 validation dates (04-15 crash, 02-20 normal, 01-15 calm, 09-15 calm)
under v3: 20 evenly spaced starts per date, against AC (λ=1e-6), Adaptive-AC (λ=1e-9),
TWAP, Heuristic and the sqrt-impact baseline. The legacy-trained **Step 4** checkpoint is
scored the same way as a diagnostic. Held-out dates are not used.

Two caveats are known in advance:
- The validation dates were *training* dates for Step 4, so Step 4 has seen their
  00:21–07:27 UTC window. That favours Step 4 on these dates.
- 3 of M0's 12 training dates are crash days from one event window (B), so M0 has seen
  crash dynamics from only one event.

## Predictions

1. **M0 beats Step 4 under v3 on at least 3 of the 4 validation dates.** Step 4 never
   trained past ~07:27 UTC or on any order size but 58.85%, and v3 episodes span 16
   hours from starts up to 8 hours into the day.
2. **M0's gap to AC on the two calm dates is smaller than Step 4's** (M0 − AC higher than
   Step 4 − AC on 01-15 and 09-15).
3. **On the crash date (04-15), M0 is within 0.3 pp of AC or better.** 04-15 was a
   near-tie for RL under the legacy protocol.
4. **M0 does not beat AC on both calm dates.** The calm-day gap is believed to be
   structural (regime-blind pace), which M0 does not change; that is what M1–M3 target.
5. **No strategy force-liquidates under v3** (rate 0 for all six).

Each will be marked met / not met in the result document, including the misses.
