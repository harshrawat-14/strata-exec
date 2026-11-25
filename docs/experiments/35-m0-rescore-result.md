# 35 — M0-e2b re-scored and out of sample. Result — a fast constant seller; goal not met

Written 2026-11-25. Predictions were pre-registered in
[35-m0-rescore-predictions.md](35-m0-rescore-predictions.md) (commit aaf491e). Held-out
dates are untouched.

## Summary

- **On the corrected simulator, doc 23's result stands.** M0-e2b still beats TWAP on crash
  days and loses on calm days.
- **Out of sample (85 days), the same split holds:**
  - M0-e2b beats TWAP on all 10 crash days, by +0.67 pp.
  - It loses on 27 of 29 calm days, by −0.79 pp.
  - Overall it is 0.12 pp worse than TWAP.
- **It carries no regime information beyond speed.** Its scores sit between selling at a
  constant 3× and 5× TWAP pace, in every regime. A constant 5× pace beats it overall and on
  crash days.
- **The project goal (beat TWAP = honest AC in every regime) is not met.**

## Result (`rl/experiments/m0_rescore.py`, `rl/results/m0_e2b_rescore_doc35.csv`)

### v3: doc 23's 16 dates on the corrected simulator

| | doc 23 (old simulator) | doc 35 (corrected) | change |
|---|---:|---:|---:|
| M0-e2b, mean IS | −0.354 | −0.242 | **+0.112** (every date +0.005 to +0.250) |
| TWAP, mean IS | −0.228 | −0.170 | +0.058 |

By doc 30's regime labels, M0-e2b − TWAP:

| regime | doc 23 | doc 35 | 90% interval | days M0-e2b wins |
|---|---:|---:|---|---:|
| crash (3) | +0.477 | **+0.533** | [+0.102, +0.963] | 2/3 |
| normal (8) | +0.084 | +0.136 | [−0.167, +0.461] | 4/8 |
| calm (5) | −0.823 | **−0.770** | [−1.092, −0.467] | 0/5 |

Doc 30's labels differ from doc 23's on a few dates, so these means differ from doc 23's
table (+0.489 / +0.007 / −0.669), as stated in advance.

### oos: 85 V4 validation days M0-e2b never trained on or was selected on

| regime | days | M0-e2b | TWAP | pace 1.5 | pace 2 | pace 3 | pace 5 | M0-e2b − TWAP [90%] | days M0-e2b wins |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| crash | 10 | −0.578 | −1.251 | −0.982 | −0.822 | −0.657 | **−0.521** | **+0.673** [+0.508, +0.849] | 10/10 |
| normal | 46 | −0.292 | −0.421 | −0.315 | −0.274 | **−0.259** | −0.269 | +0.130 [+0.049, +0.214] | 32/46 |
| calm | 29 | −0.067 | **+0.717** | +0.352 | +0.175 | +0.035 | −0.077 | **−0.785** [−0.930, −0.631] | 2/29 |
| **all** | 85 | −0.249 | **−0.130** | −0.166 | −0.186 | −0.205 | −0.233 | −0.118 | |

- **Speed alone explains it.** M0-e2b minus the constant paces, by regime:
  - vs pace 3: crash +0.079, normal −0.032, calm −0.103.
  - vs pace 5: crash −0.057, normal −0.022, calm +0.010.

  A regime-aware seller would beat *both* paces in *both* regimes. It would sell slower than
  pace 3 on calm days and faster than pace 5 on crash days. M0-e2b beats pace 3 only on crash
  days and pace 5 only on calm days. Any constant pace between the two does the same.
- **TWAP is the best constant pace overall** (−0.130), in line with doc 33: honest AC = TWAP.
  Every faster pace trades calm-day losses for crash-day gains, and the calm losses are larger.
- **Execution:** no forced liquidation on any date. Median steps to finish are 676 of 1,920
  (TWAP 1,918).

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | v3 mean improves by 0.08–0.20 pp, more than TWAP's 0.06 | **Met** (+0.112 vs +0.058) |
| 2 | v3 crash gap stays positive and grows, to +0.50–0.70 | **Met** (+0.477 → +0.533) |
| 3 | v3 calm: still loses to TWAP | **Met** (−0.770) |
| 4 | oos: wins crash (interval > 0), loses calm (interval < 0); goal fails | **Met** |
| 5 | oos: between pace 3 and 5 (± 0.05) in every regime; median steps 450–650 | **Half met**: inside the band in every regime (normal 0.03 below both); median steps 676 |
| 6 | No forced liquidation | **Met** |

## What this means

- The era-2 headline, "RL wins crashes, loses calm days", is now confirmed on 85
  out-of-sample days. It is also explained: the model learned one fast pace, not a regime-
  dependent one. With docs 26–34 finding no real-time regime signal in BTC, SOL or DOGE,
  that is the expected outcome. There is nothing for the agent's pace to condition on.
- **Next (decision rule):** retrain the M0 recipe on the corrected simulator (doc 36, one
  change). The model learned under impact costs that were too high for fast selling, so the
  pace it would learn now may differ. Docs 26–34 predict it will still be one constant pace.
  Then equities.
