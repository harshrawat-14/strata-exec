# 22 — Era 2 baselines: AC, TWAP and constant paces on the real Bybit book

Date: 2026-10-26. Script: `rl/experiments/era2_baselines.py`. Raw data:
`rl/results/era2_baselines.csv`. The simulator is the Bybit real 500-level book with
price-moving impact (doc 21). Protocol v3: 1,920-step (16 h) episodes, 20 evenly spaced
starts per date. 12 training and 4 validation dates; held-out dates untouched.

Rules fixed before running: orders of 1%, 5% and 10% of the date's 20-day Bybit ADV; one
AC λ per size, chosen on training dates only by mean IS; constant paces 0.5–3× TWAP.

## 1. Honest AC is TWAP

| λ | 1% ADV | 5% ADV | 10% ADV |
|---|---:|---:|---:|
| 1e-9 | −0.002 | −0.205 | −0.358 |
| 1e-7 | −0.003 | −0.206 | −0.359 |
| 1e-6 (era-1 choice) | −0.016 | −0.219 | −0.372 |
| 1e-5 | −0.082 | −0.300 | −0.464 |
| 1e-3 | −0.207 | −0.467 | −0.670 |

Training-date mean IS (%) falls monotonically with λ: any risk aversion means selling
earlier, which costs impact and is not repaid on average. The selected λ (3e-9) is the
λ → 0 limit, where AC's schedule is uniform, and its IS matches TWAP to 0.0001 pp at every
size. This is a result of the impact model and these dates. It is not a grid edge: lowering
λ further changes nothing.

## 2. The cost of size

TWAP training mean IS is −0.002%, −0.205% and −0.358% at 1%, 5% and 10% of ADV. Market
drift is the same at every size, so roughly 0.2 pp of impact is added from 1% to 5% and
0.15 pp more to 10%. That is concave, as the square-root law says.

## 3. No single pace wins every regime

Mean IS (%) by pace, 5% of ADV:

| split / regime | 0.5× | 1× (TWAP) | 1.5× | 2× | 3× |
|---|---:|---:|---:|---:|---:|
| train calm | +1.03 | +0.91 | +0.64 | +0.49 | +0.29 |
| train normal | −1.24 | −0.43 | −0.25 | −0.26 | −0.31 |
| train crash | −1.30 | −1.31 | −1.44 | −1.41 | −1.25 |
| validation crash (04-15) | −2.39 | −0.77 | +0.02 | +0.13 | −0.05 |
| validation normal | −0.36 | −0.06 | +0.00 | −0.05 | −0.20 |
| validation calm | −0.70 | −0.18 | −0.19 | −0.18 | −0.21 |

0.5× cannot finish and dumps about 44% of the order at the end.

Slow is best on calm training days (prices rose), and 1.5–2× is best on normal days and on
the validation crash. On training crash days no pace helps reliably.

## 4. Headroom: the best pace for each date, chosen in hindsight

| order | calm | normal | crash | mean |
|---|---:|---:|---:|---:|
| 1% ADV | +0.32 | +0.42 | +0.83 | +0.48 |
| 5% ADV | +0.15 | +0.33 | +0.76 | +0.37 |
| 10% ADV | +0.07 | +0.29 | +0.67 | +0.30 |

Gain over TWAP in pp. This uses hindsight, so it is an upper bound for a policy that picks a
constant pace per date, not an achievable target. A policy that adjusts its pace within the
day can exceed it.

## What this means for the RL ladder

- **AC to beat = TWAP.** The goal "beat AC in every regime" becomes "beat TWAP in every
  regime".
- **The edge is choosing the pace by regime.** The hindsight-best pace is worth about
  0.37 pp at 5% of ADV, roughly twice as much on crash days as on normal ones and twice
  again over calm ones. That supports step 3 of the v2 plan (a belief-conditioned pace) over
  per-step order-placement tricks: crossing the spread costs under 1 bp now.
- **Operating size for the ladder: 5% of ADV** (4k–8k BTC per order), with 1% and 10%
  reported as sensitivity checks.
