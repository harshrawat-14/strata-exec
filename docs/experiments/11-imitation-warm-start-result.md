# Phase 3 Step B — imitation warm-start: NEGATIVE RESULT

Written 2026-09-07. Predictions pre-registered in `docs/experiments/11-imitation-warm-start-predictions.md` before the
fine-tune was launched. **This is a negative result and is reported as one** — per the
pre-registration's own instruction not to bury it behind the improved calm-day numbers.

## Summary

Behavior-cloning honest-λ AC-Optimal into the policy, then RL fine-tuning from that initialization,
**eliminated the calm-day deficit and destroyed the crash-day edge at the same time.** The resulting
model is, for practical purposes, a slightly noisier reimplementation of Almgren-Chriss.

The project's standing regime-conditional finding (Step A: 4/4 crash wins, 0/5 calm/normal losses)
is **not overturned** — it belongs to a different checkpoint
(`phase3_step4_capacity_entropy_best`). Step B is a separate variant that failed to improve on it.

## What was run

- **Stage 1 (BC):** `rl/experiments/phase3_stepB_bc_pretrain.py` — 48 demonstration episodes (16
  `TRAINING_DATES_V2` × 3 seeds), honest λ=1e-6 AC-Optimal as the sole expert, teacher-forced
  cross-entropy, truncated BPTT (chunk=128). NLL 2.996 (uniform) → 0.313 over 8 epochs.
- **Floor check:** `rl/experiments/phase3_stepB_verify_floor.py` — the clone tracked AC-Optimal within
  **0.088pp mean absolute deviation** on all 9 `HELD_OUT_DATES_V2`. Floor guarantee achieved.
- **Stage 2 (RL fine-tune):** `train.py --resume-from` the BC checkpoint, 1.5M timesteps, `n_envs=16`,
  `device=cpu`, `base_lr=1e-4` (reduced from 3e-4 specifically to protect the floor). 28m55s.
- **Evaluation:** `rl/experiments/phase3_stepA_full_holdout_eval.py` unchanged (same 9 dates, same n_episodes,
  same policy set) → `rl/results/phase3_stepB_finetuned_holdout.csv`.

## Result: RL − AC-Optimal (mean_IS)

| date | regime | Step 4 | Step B | change |
|---|---|---:|---:|---:|
| 2024-03-05 | crash | +0.9174 | +0.0353 | −0.8821 |
| 2024-08-05 | crash | +0.9224 | +0.1713 | −0.7512 |
| 2024-08-02 | crash | +0.3701 | −0.1106 | −0.4808 |
| 2024-08-04 | crash | +0.2625 | −0.0743 | −0.3368 |
| 2024-07-10 | calm | −0.8017 | +0.0102 | +0.8119 |
| 2024-07-11 | calm | −0.5885 | −0.0579 | +0.5306 |
| 2024-07-12 | normal | −0.5814 | −0.0008 | +0.5805 |
| 2024-08-03 | normal | −0.3133 | +0.0144 | +0.3276 |
| 2024-12-10 | normal | −0.4559 | +0.0556 | +0.5114 |

**Crash mean +0.618 → +0.005. Calm/normal mean −0.548 → +0.004. Crash wins 4/4 → 2/4.**

## Against the pre-registered predictions

| prediction | outcome |
|---|---|
| calm/normal deficit improves past −0.30pp | **met** (+0.004, far exceeded) |
| crash edge shrinks, stays positive on ≥2 of 4 | **met at the boundary** (2/4, margins +0.035/+0.171) |
| overall flattening, less regime-conditional | **met** |

All three predictions were technically correct, which is worth noting for calibration: the failure
mode was anticipated in advance, not discovered after the fact. This maps to pre-registered outcome
3 ("floor holds, crash edge disappears — report plainly as a negative result").

## The policy barely moved off its initialization

This is the load-bearing diagnostic, and it is why the result should be read as "the fine-tune did
nothing" rather than "RL learned a different trade-off":

| checkpoint | mean \|delta from AC-Optimal\| across 9 dates |
|---|---:|
| BC clone (zero RL steps) | 0.0883 |
| after 1.5M RL timesteps | 0.0589 |

Per-date movement over the entire 1.5M-step fine-tune was 0.05–0.19pp, and mostly *toward* AC, not
away from it. Corroborating evidence from the training log (`/tmp/stepB_finetune.log`):

- Best model selected at **450,000 steps** and never beaten across the remaining 1,050,000.
- Eval `episode_reward` pinned at **−0.01 ± 0.00** from the first eval (50k) through the last.
- Final policy entropy ≈0.5 vs max ln(20)=2.996 — BC handed over a near-deterministic policy and
  fine-tuning kept it there.

## Interpretation

The tension identified before launching resolved in the pessimistic direction. The anchor that
protected the calm-day floor (low LR + initialization toward a regime-blind expert) is the same
force that prevented rediscovering the per-fill reactivity that produces the crash-day edge
(`docs/experiments/08-sqrt-impact-baseline.md` established reactivity, not pacing or impact-model shape, as the
standing explanation for that edge).

**Cloning AC did not give RL a better starting point to improve from — it gave RL a basin it did not
climb out of at this learning rate.** The floor guarantee and the regime-conditional edge turned out
to be in direct competition, not complementary.

## Confound not resolved: LR vs. imitation itself

**This run cannot distinguish two explanations**, and must not be cited as if it can:

1. Imitation warm-start from a regime-blind expert is fundamentally wrong for this problem, or
2. `base_lr=1e-4` was simply too low to escape the BC basin, and imitation might work at 3e-4.

Both are consistent with everything observed. A single-variable re-run at `base_lr=3e-4` would
separate them. It was NOT run here because of test-set erosion (see below) — recorded as an open
question, not silently omitted.

## Test-set erosion (methodological, applies beyond this step)

`HELD_OUT_DATES_V2` has now been evaluated three times — Step 4's model, the BC clone, and the
fine-tune. Best-model selection itself was verified clean (the fine-tune's `EvalCallback` used only
the 16 `TRAINING_DATES_V2`; the crash-blindness fallback that would have injected held-out crash
dates did not trigger, since V2 training has 4 crash dates). So this is not training leakage.

But repeated evaluation followed by decisions informed by what was seen gradually converts a test set
into a validation set. Future model-variant selection should carve a validation split out of the 16
training dates rather than spending further looks at the 9 held-out dates.

## What this does and does not change

- **Does not change** Step A's headline (4/4 crash, 0/5 calm/normal) — different checkpoint.
- **Does close** the long-deferred imitation-learning proposal as tested-and-negative *at this LR*.
  `docs/proposals/research-roadmap.md`'s standing objection ("no clean expert to imitate") is no longer the blocker;
  the expert was clean and the clone verified. The problem is what fine-tuning does afterward.
- **Sharpens** the residual-action-space proposal's risk profile
  (`docs/proposals/residual-action-space.md`): it shares the premise that anchoring to a
  reference schedule is safe. This result is direct evidence that anchoring to a *regime-blind*
  reference costs the crash-day edge. Any residual design must use a regime-responsive reference.

## Artifacts

- `rl/experiments/phase3_stepB_bc_pretrain.py`, `rl/experiments/phase3_stepB_verify_floor.py`
- `rl/models/phase3_stepB_bc_pretrain/bc_pretrained.zip`
- `rl/models/phase3_stepB_warmstart_finetune_best/best_model.zip` (450k steps)
- `rl/results/phase3_stepB_floor_check.csv`, `rl/results/phase3_stepB_finetuned_holdout.csv`

## Known logging bug found during this run (not fixed, no code touched mid-run)

`train.py` prints and logs to mlflow the *config default* architecture (`lstm_hidden=128`,
`lstm_layers=2`) even on a `--resume-from` run, where the actual loaded architecture comes from the
checkpoint (here 64×1, verified directly). **Every resumed run in this project mis-records its own
architecture in mlflow.** Cosmetic for this result (architecture was confirmed correct out-of-band)
but worth fixing before it misleads a future comparison.
