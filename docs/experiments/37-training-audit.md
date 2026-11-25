# 37 — Why the training cannot produce regime awareness: an audit of reward, state and actions

Written 2026-11-25. An audit, not a model run: no RL is trained and no predictions are scored.
Each finding was checked in code and on real episodes, not assumed. The user asked for exactly
this before any redesign: the counterfactual LOB, the market move and the reward interact in
ways that are easy to get wrong.

## The question

Docs 23–36 tried many training techniques. None produced a model that conditions its pace on
the regime: M0-e2b and M0-e2c are constant fast paces (docs 35–36). This audit asks how the
model is trained, and whether that training could ever reward regime awareness.

## How M0 is trained (from `rl/train.py` and `src/bin/rl_env`)

| Part | Setting |
|---|---|
| Data | 12 V3 BTC days (3 crash, 4 calm, 5 normal), one worker per day |
| Episode | 1,920 steps × 30 s, random start in the day; sell 5% of 20-day ADV |
| Agent | RecurrentPPO, LSTM 64; γ 0.99, GAE λ 0.95 (credit horizon about 17 steps); 1.5M steps ≈ 780 episodes |
| Actions | 20 fractions of the *remaining* inventory: 0, 0.1%, 0.3% … 90%, 100% |
| Reward, sale | (fill − mid without our impact) × qty ÷ (arrival × order) |
| Reward, wait | −10⁻⁴ × (0.01 (σ/σ_ref)² q² + 0.0002 q²), with q = share of the order left |
| Selection | Best IS against the arrival price on 4 validation days |
| Inputs | 16 raw numbers, no normalisation (no `VecNormalize`) |

## Findings

### 1. The accounting is exact (`rl/tests/test_reward_accounting.py`)

Every sale splits exactly into three parts:
- market move: arrival − mid without our impact;
- our impact: mid without our impact − impacted mid;
- spread and book walk: impacted mid − fill.

On real episodes, these parts sum to IS, and the fill rewards equal the impact + spread parts,
to 10⁻⁹. Under `--reward-baseline twap`, reward − IS is the same constant for every schedule on a
price path, so that reward ranks schedules exactly as IS does.

The test fails when doc 24's bug (charging against the impacted mid) is put back. To make the
reward's reference visible, step info now carries `reward_reference_mid`. On 37,487 golden steps,
the old and new binaries are identical apart from that field.

### 2. The reward leaves out the only thing that separates regimes

Cost parts on real episodes (%, sign = gain to the seller; pace × TWAP):

| day | pace | market move | our impact | spread/walk | wait penalty |
|---|---:|---:|---:|---:|---:|
| 2024-04-13 | 1 | +2.046 | −0.507 | −0.004 | −0.020 |
| 2024-04-13 | 3 | +1.755 | −0.507 | −0.009 | 0 |
| 2024-04-13 | 10 | +0.800 | −0.503 | −0.018 | 0 |
| 2024-07-08 | 1 | +2.476 | −0.338 | −0.003 | −0.038 |
| 2024-07-08 | 3 | +0.837 | −0.333 | −0.007 | 0 |
| 2024-07-08 | 10 | +0.492 | −0.333 | −0.018 | 0 |

- **Our impact hardly depends on pace** (within 0.005 on a day). 43.5% of it is permanent,
  and it grows with √(total sold), so its cost depends on how much is sold, not when. The
  temporary part has a 26 s half-life, shorter than one step. This is a property of the doc 33
  calibration that had not been written down. Empirical metaorder studies also find that impact
  depends mainly on size, so this is plausible, not a bug.
- **The reward is therefore almost flat in pace.** Faster selling costs about 0.014 more in spread
  and saves 0.02–0.04 in waiting penalty, so **fast wins by a hair**. That is why the learned pace
  is a constant fast one (docs 35–36). The waiting penalty, not the market, sets it.
- **What separates schedules by 1–2 pp is the market move, and the default reward excludes it
  by design.** Regime awareness (sell early on a falling day, wait on a rising one) is worth
  exactly the market-move term. The training signal never contains it.
- The only time it was included (M1, doc 25, step by step against TWAP), the per-step noise
  drowned it (mean/sd 0.15).

### 3. The state carries no price trend

Standard deviation of each input over the same 192 snapshots (2024-04-13), and its shift
under 10× selling against TWAP:

| dim | input | sd | shift from our own selling |
|---:|---|---:|---:|
| 2 | volatility | 0.054 | +0.002 |
| 3 | spread | 0.0002 | −0.001 |
| 4 | depth within 1% | 0.035 | 0 |
| 5 | order-flow imbalance | 0.463 | 0 |
| **6** | **5-min return (the only trend input)** | **0.003** | −0.0004 |
| 9 | temporary impact | 0.001 | +0.003 |
| 10 | volatility regime | 0.132 | +0.004 |
| 11 | liquidity regime | 0.0006 | −0.003 |

- **Dim 6 is dead.** It is divided by an annualised σ (doc 26), which was flagged and never fixed.
  Dims 3, 9 and 11 are also almost constant.
- **Scale matters here because the inputs are not normalised.** A dim with sd 0.003 needs a
  weight about 250× larger to matter, and its gradient is 250× smaller. So it is unreadable in
  practice, though not empty. The fix is to standardise every input to its own range.
- Nothing tells the agent how far the price has moved since the episode or day began.
- **Own impact barely leaks into the market inputs** (shifts ≤ 0.004). The concern was real but
  small.

### 4. The action set favours dumping

Each action is a fraction of the *remaining* inventory. TWAP needs 0.052% at the start, which
falls between the 0 and 0.1% buckets, so even selling cannot be held as one action (doc 18: 837
switches per episode). Dumping is one action. Pace actions (doc 19: k × TWAP) or a continuous
pace make TWAP a single value and regime awareness a change in that value.

### 5. A hindsight input to the simulator (recorded, not changed)

The impact model's σ and V are the **whole day's** realised volatility and volume
(`CounterfactualLobConfig::calibrated`). A crash day's selling is charged impact based on volatility
that has not happened yet at the time of the sale. The agent never sees these values, and doc 21
fitted the impact law with daily σ and V. But it is a hindsight element of the simulated world,
and any regime result must be robust to it: for example, check it with a trailing σ.

## Conclusion

The current training cannot produce regime awareness, whatever the algorithm:
- the reward excludes the market move, which is the only thing regime awareness changes;
- the state has no live price-trend input;
- the action set makes constant fast selling the easy choice.

Docs 14–25 changed the learner. None of them changed these three things together.

## Next (discussed with the user)

1. **Learnability first, and supervised** (the user's idea). From history, learn when and why
   selling faster or slower was right. The target is the hindsight-best pace and delay at each
   moment; the inputs are conditions visible at that moment, including support/resistance and
   round-number levels, never tested before. Use an interpretable model with feature attribution,
   tested on unseen days. If it predicts, its output becomes StrataExec's regime signal. If it
   does not, no training can make this state regime-aware.
2. **Only then** retrain StrataExec with all of the following:
   - a reward that contains the market move, scored per episode against TWAP on the same path;
   - normalised inputs, with dim 6 fixed;
   - a continuous pace action;
   - the critic allowed to see hindsight information, which reduces the noise that sank M1;
   - the 256-day V4 pool, sampled equally across regimes.
