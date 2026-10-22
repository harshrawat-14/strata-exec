# 19 — M4 (rate-based actions): result — best validation number, but not learned skill

Written 2026-10-22. Predictions in [19-m4-rate-actions-predictions.md](19-m4-rate-actions-predictions.md).
Validation dates only.

## Result

M0's recipe with `--action-space rate`; 1.5M steps, no errors. Offline selection
(`rl/results/m4_v3_rate_checkpoint_selection.csv`) picks the **200k** checkpoint at
**−0.311%** mean validation IS — better than AC (−0.4425%) and M0 (−0.4651%). Checkpoints
up to 400k sit at −0.31% to −0.35%; from 500k on, validation degrades with forced dumps
(10–57% of episodes), ending at −0.86% (final). Training reward improved to ~−0.0070 by
~1.1M and then worsened to ~−0.0108.

## Is it learned? Compared with constant paces on the same validation episodes

| | 01-15 calm | 02-20 normal | 04-15 crash | 09-15 calm | mean |
|---|---:|---:|---:|---:|---:|
| AC | −0.093 | −0.288 | −0.652 | −0.737 | −0.4425 |
| M4 (200k) | −0.226 | −0.316 | −0.003 | −0.699 | −0.3110 |
| constant 1× | −0.098 | −0.300 | −0.834 | −0.781 | −0.5031 |
| **constant 1.5×** | −0.146 | −0.200 | +0.040 | −0.675 | **−0.2451** |

M4 averages ~2× the steady pace on every date (2.05× on the crash date, 1.94× and 2.08×
on the calm dates); its average sell time is 0.21–0.23 of the episode (AC 0.45). A fixed
1.5× pace beats it. On the **training** dates, faster is not better (doc 19 bound:
constant 1× −0.340 ≈ AC −0.331; 2× −0.441). So M4's validation advantage is explained by
(i) the action design giving near-AC execution and (ii) selling faster than AC on four
dates where prices fell — not by state-dependent pacing. The selected 200k checkpoint is
close to its random initialisation (whose mean pace is ~4×); selecting the best of 16
checkpoints on four falling dates favours fast sellers. **Treat −0.311% as optimistic.**

Execution vs AC (`rl/results/m4_v3_is_decomposition.csv`): −0.08, −0.03, −0.11, −0.06 pp
— much closer than M0 (−0.13 to −0.24), not yet AC-level.

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | Beats M0 | Met (−0.311 vs −0.465) |
| 2 | Execution within 0.05 pp of AC on ≥3/4 | **Not met** (1/4) |
| 3 | Mean ≥ AC | Met — explained by pace and the dates' drift, not skill |
| 4 | Beats AC on 04-15 | Met (+0.65) — same caveat |
| 5 | Largest trade < 1.1% of order | Met (0.30%) |
| 6 | Higher pace on 04-15 than both calm dates | **Not met** |

## What it establishes

- The rate menu works as designed: bounded trades, near-AC execution, no bursts.
- Learning still does not produce state-dependent pacing, and later in training the
  policy drifts to pausing and forced dumps — the same "deferring looks free" pattern as
  M2.
- Constant-pace schedules are a legitimate baseline and belong in the final comparison,
  with the pace chosen on training dates only (there: 1×, ≈ AC).
