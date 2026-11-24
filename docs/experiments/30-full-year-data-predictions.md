# 30 — Full-year 2024 data: split, regime rule and predictions

Written 2026-11-24, before any new day is downloaded. No RL is trained.

## Why

Docs 26–29 each ended with intervals too wide to decide, using 12 training days (2–4 per
regime). Two of them left a hint that this sample can neither confirm nor rule out: a
bounce-back after drops (doc 27) and slower reversion in high volatility (doc 29). The aim is
unchanged (regime-aware RL beats TWAP in every regime). This doc only adds the data.

## Storage

Bybit's archive is downloaded and converted by `rl/experiments/fetch_bybit.py`, which now
writes gzip CSV. On the 25 existing days, gzip cut 2.98 GB to 483 MB, and the golden
fixtures were unchanged. A year of data is therefore about 7 GB, against 48 GB free. Raw
downloads (about 250–450 MB per day) are deleted after conversion.

## Split (`rl/evaluation/dates.py`, V4; tested in `rl/tests/test_dates_v4.py`)

| set | days | use |
|---|---:|---|
| **training** | 256 | Jan–Sep 2024 |
| **validation** | 87 | Oct–Dec 2024; selection only |
| **held-out, V3** | 9 | unchanged; spent once at the end |
| **buffer** | 14 | ±2 days around each held-out date; never downloaded |
| **held-out, future** | 90 | Jan–Mar 2025; not downloaded until the final evaluation |

- **Buffer:** it applies only to days outside the old 25-day pool. V2 training days next to
  held-out window B (07-08, 07-09) keep their place, as in V2.
- **Out-of-time split:** training precedes validation, so validation tests the next quarter,
  which is the setting the agent would face in use.
- **V3 is not replaced.** Earlier results (M0, M1) stay comparable on V3. V4 is additive.

## Regime rule, fixed now

`drift` is the day's mean of ln(mid_t / mid_open) over the 30 s snapshots, in %. That is roughly
what a full-day TWAP seller loses.

- **crash:** drift ≤ −1.0%
- **calm:** drift ≥ +0.55%
- **normal:** otherwise

**How the rule was chosen:**
- On the 25 old days, drift has rank correlation 0.955 with the old labels (AC cost in the
  era-1 simulator, which cannot be computed for new days).
- The cuts were chosen on the 16 V3 training and validation days only and reproduce 14 of
  their 16 labels. The misses are 04-13 (−0.90%, old label normal) and 09-15 (−0.02%, old
  label calm); both sat at the old tercile boundary.
- "calm" means favourable for a seller, and strong up days count as calm. Volatility is
  reported next to the label but is not part of it.

## What follows (each with its own pre-registration before it is run)

1. Rerun doc 29's decay-by-volatility test on 256 training and 87 validation days.
2. Rerun doc 27's bounce-back hint (drawdown_z) as a single pre-registered hypothesis.
3. Only if one passes: put the mechanism in the simulator and check that a simple rule
   beats TWAP in every regime, then move to RL.

## Predictions

1. **Regime counts on the 343-day pool:** crash 10–20%, calm 30–45%, normal the rest. 2024
   was a bull year, so calm (up) days outnumber crashes.
2. **Crash days are more frequent in Apr–Sep than in Q1 and Q4** (share of days labelled
   crash), because Q1 and Q4 2024 were the strongest rallies.
3. **Data completeness:** at least 99% of the 343 days convert with 2,879–2,880 book
   snapshots. Any day with fewer than 2,800 is listed and excluded, not repaired.
4. **The pool's median daily volatility is 10–40% lower than the 16 V3 days' median.** The
   existing days were picked around named events, so they are more volatile than a typical day.
