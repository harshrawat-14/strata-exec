# Phase 3 Step 4 — audit finding 6 (underfit policy): fixed, but doesn't change the story

Written 2026-08-24. Addresses `docs/phase3_platform_audit.md` finding 6: across Step 2b's 500k-step
run, `train/entropy_loss` moved from -2.99 to -2.92 against a max of -ln(20) = -2.996 for a 20-action
space — the policy sat at ~98% of uniform-random for the entire run. The audit's "cheapest next
experiments": raise timesteps to 1.5-2M (matching the historical v6r run that only began sharpening
between 1.1M-2.0M steps) and sweep `ent_coef` down toward 0.01 (from 0.05, high for PPO).

**Retrained (`phase3_step4_capacity_entropy`) with both changes at once** — same 16 V2 training
dates, same everything else as Step 2b (counterfactual, 16-dim, n_envs auto-raised to 16,
btc_target=50k) — `--timesteps 1500000 --ent-coef 0.01 --device cpu`. 45 minutes wall-clock.

**Both changes were applied together, not isolated — this result cannot attribute the effect to
timesteps vs ent_coef individually, only to the combination.** Recorded as a limitation, not
overlooked: isolating them is the natural follow-up if either factor needs to be understood on its
own, but the audit itself proposed both as "cheapest next experiments" together, and the aggregate
question ("does fixing the underfit change the headline") doesn't require the split.

## The entropy problem is real and fixable

| run | timesteps | ent_coef | final entropy_loss | % of max entropy |
|---|---:|---:|---:|---:|
| Step 2b | 500,000 | 0.05 | -2.85 | ~95% |
| Step 4 | 1,500,000 | 0.01 | **-1.27** | ~42% |

A substantial, genuine sharpening — confirms the audit's diagnosis, not a measurement artifact.

## But it does not resolve the core findings — it shifts magnitudes, mixed direction

Canonical harness, counterfactual, honest λ=1e-6, n_episodes=20 matched. IS = RL − AC-Optimal
(positive = RL better):

| date | regime | Step 2b RL − AC | Step 4 RL − AC | change |
|---|---|---:|---:|---:|
| 2024-03-05 | crash | +1.10 | +0.92 | crash edge **shrank** |
| 2024-08-05 | crash | +1.21 | +0.92 | crash edge **shrank** |
| 2024-07-10 | calm | −1.42 | −0.80 | calm loss **shrank** (improved) |
| 2024-12-10 | normal | −0.20 | −0.46 | normal loss **grew** (worse) |

The regime-conditional pattern survives qualitatively (RL still wins both crash dates, still loses
calm and normal), but the sharper policy is not uniformly better — it trades some crash-date edge
for a smaller calm-date loss, and gets worse on the one "normal" date tested. There is no clean
"more capacity = better" story here; the sharper policy redistributes performance across regimes
rather than improving on all of them.

## Pacing got *more* wrong-direction, not less

Step 2b's headline nuance: RL beats AC on both crash dates while pacing *less* urgently on crashes
than on calm days (responsiveness −0.0216, wrong sign) — direct evidence the crash-date advantage
does not come from timing. If that pattern were an artifact of an underfit, near-random policy, a
sharper one should move toward zero or positive (regime-responsive).

**It moved further negative instead: −0.0892, more than 4× Step 2b's value** (9 V2 held-out dates ×
20 seeds; t50 calm=0.5145, crash=0.6037 — the sharpened policy sells *slower* under crash than under
calm, more emphatically than the underfit one did). AC-Optimal and TWAP are both ~0.0000 by
construction (regime-blind by design, included as reference points).

**This strengthens, not weakens, Step 2b's conclusion.** A 98%-uniform, barely-trained policy
producing regime-blind pacing could plausibly have been "hasn't learned yet." A policy that used 3×
the training budget at a lower entropy floor, sharpened its action distribution by more than half,
and became *more* pronounced in the wrong direction is not plausibly explained by undertraining.
Whatever RL's crash-date advantage is built from, it is not on a trajectory toward learning
regime-adaptive urgency with more of the same training recipe.

## What this changes

- Audit finding 6 is closed: the underfit-policy hypothesis was real, is now fixed, and the fix does
  not change either headline conclusion (Step 2b's pacing dissociation, Step 3's ruling-out of AC as
  a strawman). Both survive a genuine capacity/entropy increase, not just the original underfit run.
- The residual-action-space/imitation-learning proposals lose more ground: "the policy hasn't had
  enough training to find urgency yet" is no longer a live objection to Step 2b's finding — a
  meaningfully sharper policy trained on the same data moved further from regime-responsive pacing,
  not closer.
- `phase3_step4_capacity_entropy_best` is now the strongest available checkpoint by entropy/training
  budget and is a reasonable default for future comparisons, but Step 2b's checkpoint remains the one
  every prior document's numbers cite — cite the specific checkpoint explicitly going forward.

## Caveats

- Timesteps and ent_coef were changed together (see above) — this result answers "does fixing the
  underfit change the story," not "which factor did it."
- Inherits audit finding 5 in full (near-duplicate eval episodes, crash-date significance not
  credible) and the seed-heterogeneity finding from Step 3 — signs are directionally informative,
  magnitudes and significance are not.
- Only one seed/training run at the new settings — no variance estimate across independent training
  runs exists for either config, so "the crash edge shrank" is a single point comparison, not a
  distribution.

## Artifacts

- `rl/models/phase3_step4_capacity_entropy_best/best_model`
- `rl/results/phase3_step4_sqrt_impact.csv`
- `rl/results/phase3_step2_urgency_step4.csv`
- MLflow run: `phase3_step4_capacity_entropy` (`rl/mlruns`)
