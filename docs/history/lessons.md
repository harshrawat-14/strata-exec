# Recurring patterns

Most entries in [timeline.md](timeline.md) are instances of a handful of patterns.
Each pattern lists its instances and the check that now guards against it. When you fix
something, look up its pattern here and search for the sibling copies before closing it.

---

## L1 · A value or a piece of logic exists in two places, and the copies drift

**Instances:** CVaR sign bug duplicated in Rust and `parallel_sim.py` (07-11) ·
`price_factor` fix applied to historical mode but not counterfactual (07-13 → 07-16) ·
`spread_ref` correct in historical, wrong in counterfactual (07-16) · three hand-copied
`ACTION_FRACTIONS` tables, one matching at 3 of 20 indices (07-17/18) · Python
`sigma_ref` mirrors left behind by the Rust recalibration, a week after writing this very
lesson down (08-13) · `step()` vs `step_counterfactual()`: different info fields, and the
adversary present in only one (10-17).

**Guard:** one source of truth, queried rather than copied (`--print-action-fractions`,
`--print-env-constants`, `--print-calibration`). Shared Rust helpers for fill, reward,
forced liquidation and episode summary. An AST test rejects literal `sigma_ref` defaults.
**After any fix, grep for the sibling before calling it done.**

## L2 · Measuring a quantity other than the one being optimised

**Instances:** the flat forced-liquidation penalty made "never trade" reward-optimal
(07-13) · "the objective rewards speculation" concluded from IS-vs-arrival, which is not
the reward (07-14) · the punitive dump at data exhaustion let a fitted baseline optimise
the loophole instead of execution (10-18) · argmax decoding scored a policy that
never acts that way (10-18).

**Guard:** read the reward code and measure cumulative reward directly before
theorising. Invariant test: never trading must not beat competent execution, on reward
*and* on IS.

## L3 · The comparison is tilted toward our own method

**Instances:** RL compared against baselines from a different simulator (07-11) · crash
test days inside the training set (07-11) · AC's λ, σ and η hard-coded to handicapping
values (07-11, 07-18) · AC's λ chosen per date with hindsight (07-20) · checkpoint
selection scored on the held-out crash days (08-13) · open-loop AC traded at
the smallest action bucket instead of its schedule (10-18).

**Guard:** every strategy runs through the same environment and seeds; baselines are
tuned only on training dates (honest-λ protocol); model selection uses training dates
only; executable split-overlap tests. **If a result favours RL, look for the leak first.**

## L4 · Silent defaults and silently ignored inputs

**Instances:** `no_regime_impact` dead scaffolding (06-11) · counterfactual state
truncated to 12 dims (07-11) · `n_envs` hard-coded to 8 (07-13) · `sigma_ref` hard-coded
to 0.20 (08-13) · `random.choice` once per worker pinned training to 3 dates (08-16) ·
MLflow logged 128×2 for a 64×1 LSTM, and zero metrics before 07-11 · the binary ignored
unknown flags, unknown modes and bad numbers, so `--jump-diffusion` did nothing (10-17) ·
every runner overwrote `results/comparison.csv` by default (10-17) · seedless resets reused one seed, so every training episode was the same episode (10-18).

**Guard:** strict CLI parsing (unknown input exits 2). Kwargs validated against the
callee's real signature. Logged config is derived from what is actually constructed.
Training reports which dates and regimes it really sampled.

## L5 · A test or gate that passes without testing anything

**Instances:** the sinh-overflow test held "on average" despite NaN steps (07-11) ·
`train.py` "passed" for months because it was never run (07-10) · CI's warning gate
re-ran a cached build that printed nothing (10-17) · a crate-wide `allow(dead_code)`
(10-17) · "random horizon and size" training was never random, and no test sampled
more than one training episode (10-18).

**Guard:** before trusting a test, make it fail once against the bug it targets (every
regression test added since 07-13 records this). Golden byte-hash guard for refactors,
mutation-checked. CI denies warnings explicitly.

## L6 · Too many variables changed at once, so the result cannot be attributed

**Instances:** wait-collapse investigated across LR, resume and device in turn
(07-14…16) · Step 2 changed state, reward scale, training set and selection together, and
the critical date bug hid inside that bundle (08-13/16) · Step 4 changed timesteps and
`ent_coef` together (08-24) · Step B's imitation failure is confounded with its low
learning rate (09-07).

**Guard:** one change per experiment, stated as such in the pre-registration. This is
the rule behind the M0 → M1 → M2 → M3 model ladder.

## L7 · Results silently depend on which binary or checkpoint produced them

**Instances:** `price_factor` erasing drift made forced-liquidation IS read ~0% (07-13) ·
pre-Step-0 checkpoints evaluated on the post-Step-0 binary (08-13) · `ppo_adversarial_v1`
has no recorded training mode (10-18).

**Guard:** `rl/results/PROVENANCE.md` boundaries; AC/TWAP used as a canary (they don't
read volatility state); binary chosen explicitly via `STRATAEXEC_RL_ENV_BIN`.

## L8 · Small or correlated samples read as strong evidence

**Instances:** a 1-seed, 2-date smoke overstated responsiveness ~4× (08-13) · seeds 0–19
start within 9.5 minutes of each other, so effective n ≈ 1 (08-16) · one date's IS varies
−0.7% to −9.7% by seed range (08-22) · 4 crash days that are really 2 events · a 400-step
probe underestimated an Adaptive-AC shift ~100× (08-13) · held-out dates looked at
repeatedly (09-07).

**Guard:** state the effective n, not the episode count. Fit seeds adjacent to the eval
range. Select variants on a validation split carved from training dates; spend the
held-out set once.

## L9 · An issue judged "cosmetic" without measuring it

**Instances:** start-offset bounds called harmless because seeds are small (07-15), which
hid both the near-duplicate episodes (08-16) and the empty-book dump (10-18) for months. · "γ = 0.99 limits planning to ~100 steps" acted on without checking that GAE's
λ = 0.95 caps credit at ~20 steps either way (10-19).

**Guard:** a "doesn't matter" verdict needs a number from every policy it could touch,
including the ones that behave differently from the one you checked.

## L10 · Operational slips during a session

**Instances:** committing with pre-staged files included · passing `--help` to a script
with no argument parser (it ran and rewrote a CSV) · diffing against a baseline binary two
commits old (10-18).

**Guard:** stage named paths only; check a script's interface before invoking it; build
comparison baselines from the exact parent commit; restore anything an accidental run
writes, and say so.
