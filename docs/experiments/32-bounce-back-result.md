# 32 — Bounce-back after a drawdown: result

Predictions in [32-bounce-back-predictions.md](32-bounce-back-predictions.md).

**Runs:**
- pre-registered: `python rl/experiments/bounce_back_test.py` → `rl/results/bounce_back_test.csv`
- post-hoc robustness check: `... signflip` → `rl/results/bounce_back_test_signflip.csv`

256 training and 87 validation days; held-out and future dates not read.

## Pre-registered test (shuffled-returns null): FAIL, all four criteria

| | raw corr | null | corrected (90% interval) | deep-drawdown excess, corrected (bp) |
|---|---:|---:|---|---|
| training | −0.229 | −0.299 | **+0.071** [+0.046, +0.095] | −8.1 [−13.3, −3.3] |
| validation | −0.236 | −0.311 | **+0.076** [+0.035, +0.123] | −4.1 [−15.8, +8.8] |
| crash days (54) | −0.247 | −0.313 | +0.066 [+0.007, +0.125] | +1.0 [−4.4, +7.2] |

- **Raw statistic:** negative, as doc 27 found, and less negative than the null. There is no
  bounce-back.
- **Corrected statistic:** taken at face value, it says **momentum**: after a drawdown the price
  keeps falling. That would favour *faster* selling after drops.

## Robustness check (post-hoc): the momentum is an artefact of the shuffled null

**Why the check was needed:** the doc stated in advance that shuffling also destroys volatility
clustering, which drawdown_z's trailing-vol normaliser responds to.

**The check:** a sign-flip null keeps every return's size in its original order and randomises
only its sign. A unit test confirms it is unbiased on volatility-clustered random walks.

**Result:**

| | raw corr | sign-flip null | corrected (90% interval) | deep-drawdown excess, corrected (bp) |
|---|---:|---:|---|---|
| training | −0.229 | −0.241 | **+0.013** [−0.012, +0.038] | −0.9 [−7.4, +5.5] |
| validation | −0.236 | −0.234 | **−0.002** [−0.043, +0.041] | +2.5 [−8.4, +16.2] |
| crash days (54) | −0.247 | −0.240 | −0.008 [−0.065, +0.056] | +15.8 [+9.2, +23.7]* |

With volatility clustering kept, the corrected correlation is zero in training and validation.
**Neither bounce-back nor momentum is present.** The raw −0.23 is the bias that a price path
with this volatility pattern produces by itself.

\*The crash-day excess is a hindsight subgroup. A day labelled crash by its own drift
selects for paths whose average falls, so its conditional returns are not usable in real
time. It is reported, not used.

## Predictions

| # | prediction | result | |
|---|---|---|---|
| 1 | raw training corr between −0.35 and −0.10 | −0.229 | met |
| 2 | corrected corr within ±0.05 (criterion 1 fails) | shuffled null: +0.071 (outside); criterion 1 fails. Sign-flip: +0.013 | half met |
| 3 | corrected deep excess below +5 bp | −8.1 (shuffled), −0.9 (sign-flip) | met |
| 4 | overall FAIL; doc 27's hint is mostly Stambaugh bias | FAIL; the hint is entirely bias | met |

## Conclusion

**All four routes to a regime-aware edge on risk-neutral cost are now closed on this data:**

| route | doc | result |
|---|---|---|
| forecast direction from the state or full book | 26, 27 | no |
| time liquidity | 28 | no |
| volatility-dependent impact decay | 29, 31 | no |
| bounce-back or momentum after drawdowns | 32 | no |

As pre-registered, the next step is discussed with the user before any change of direction.
