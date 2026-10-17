# Phase 2R severity-expansion: event-window data pull, counting rules, LUNA/FTX scoping

Tracks the data-expansion work that follows the honest-λ headline (see
`docs/experiments/02-reward-objective-audit.md`, "The λ-selection protocol was hindsight") — raising n on
the severe-stress side honestly, before any claim about whether the 2-crash-date pattern generalizes.

## Status: data pulled AND evaluated (`rl/experiments/phase2r_severity_expansion_eval.py`,
`rl/results/phase2r_severity_expansion_full.csv`) — see "Results" below. Counting rules and
data-quality findings (further down) were fixed *before* these results existed, per explicit
instruction, not decided after seeing them.

## Results: the coarse pattern does not survive dilution to the full window set — a sharper,
severity-scaling pattern does

**Both guardrails enforced in what follows: every one of the 17 days is reported below (not just the
sharp ones), and 2024-04-15 / 07-10 / 08-05 are folded into their windows as the pre-existing episode
members they are — pulled from the original honest-λ ablation, not re-run, not counted as extra
independent observations.**

| date | window | status | RL16 | AC (honest) | AAC (honest) | Δ(RL16−AC) | p | result |
|---|---|---|---|---|---|---|---|---|
| 2024-04-12 | A | new | -0.15 | -0.65 | -0.57 | +0.50 | 0.0000 | **BEATS** |
| 2024-04-13 | A | new | -1.16 | -0.89 | -0.85 | -0.27 | 1.0000 | loses |
| 2024-04-14 | A | new | -1.25 | -0.51 | -0.39 | -0.74 | 1.0000 | loses |
| 2024-04-15 | A | existing | -1.06 | -1.09 | -1.04 | +0.03 | 0.0000 | **BEATS** |
| **Window A: 2/4** | | | | | | | | |
| 2024-07-04 | B | new | -2.33 | -3.14 | -3.15 | +0.82 | 0.0000 | **BEATS** |
| 2024-07-05 | B | new | -2.02 | -2.79 | -2.78 | +0.78 | 0.0000 | **BEATS** |
| 2024-07-06 | B | new | -0.77 | -0.21 | -0.20 | -0.56 | 1.0000 | loses |
| 2024-07-07 | B | new | -0.91 | -1.42 | -1.44 | +0.51 | 0.0000 | **BEATS** |
| 2024-07-08 | B | new | -1.58 | -0.26 | -0.23 | -1.33 | 1.0000 | loses |
| 2024-07-09 | B | new | -0.36 | +0.47 | +0.48 | -0.83 | 1.0000 | loses |
| 2024-07-10 | B | existing | -0.22 | +0.03 | +0.04 | -0.25 | 1.0000 | loses |
| 2024-07-11 | B | new | -0.43 | -0.06 | -0.08 | -0.37 | 1.0000 | loses |
| 2024-07-12 | B | new | -0.93 | -0.58 | -0.58 | -0.35 | 1.0000 | loses |
| **Window B: 3/9** | | | | | | | | |
| 2024-08-02 | C | new | -1.90 | -2.37 | -2.41 | +0.47 | 0.0000 | **BEATS** |
| 2024-08-03 | C | new | -0.94 | -0.78 | -0.74 | -0.15 | 1.0000 | loses |
| 2024-08-04 | C | new | -0.72 | -1.10 | -1.05 | +0.38 | 0.0000 | **BEATS** |
| 2024-08-05 | C | existing | -6.48 | -7.89 | -7.80 | +1.41 | 0.0000 | **BEATS** |
| **Window C: 3/4** | | | | | | | | |

**Overall: RL16 beats honest-λ AC-Optimal on 8/17 days (47%), Adaptive-AC on 7/17 (41%).**

**The coarse "any day in a named stress-event window" version of the finding does not hold — stated
plainly, per the pre-committed rule that a wash-out means LUNA/FTX is moot.** 8/17 is statistically
indistinguishable from a coin flip. The clean 2/2 sweep on the original two crash dates does not
replicate at this larger, honestly-selected n.

**But a sharper, quantitatively strong pattern replaces it, found by ranking all 17 days (plus the
original two) by AC's own honest-λ realized cost — a market-severity measure, not an RL-outcome
measure, so this is explanatory analysis on already-fully-reported data, not a second round of date
selection:**

| rank by severity | date | AC honest-λ cost | RL16 result |
|---|---|---|---|
| 1 (most severe) | 2024-08-05 | 7.89% | **BEATS** (+1.41pp, p=0.0000) |
| 2 | 2024-07-04 | 3.14% | **BEATS** (+0.82pp, p=0.0000) |
| 3 | 2024-07-05 | 2.79% | **BEATS** (+0.78pp, p=0.0000) |
| 4 | 2024-08-02 | 2.37% | **BEATS** (+0.47pp, p=0.0000) |
| 5 | 2024-07-07 | 1.42% | **BEATS** (+0.51pp, p=0.0000) |
| 6 | 2024-08-04 | 1.10% | **BEATS** (+0.38pp, p=0.0000) |
| 7 | 2024-04-15 | 1.09% | **BEATS** (+0.03pp, p=0.0000) |
| 8 | 2024-04-13 | 0.89% | loses |
| ... | (9 more days, decreasing severity) | ... | 1 more win (2024-04-12, an exception — see below), 8 more losses |

**The top 7 most severe days in the entire 17-day set are 7/7 wins, every one statistically
significant.** Mean severity on win days (2.56%) is roughly **8× the mean severity on loss days
(0.31%)**. One clean exception: 2024-04-12 is a comparatively mild day (rank 10 of 17 by severity) but
still a decisive win — noted honestly, not smoothed over; a single day at n=17 could be noise, or could
indicate the day-level severity proxy misses acute intraday moves that don't show up in the full-day
average.

**Revised mechanism, more precise than the original: RL's advantage appears to scale with how severe
that specific day's realized move was, not with whether the day belongs to a publicly-named stress
narrative.** Multi-day event windows are mostly mild days with a few genuinely sharp ones inside them —
averaging over the whole window is what diluted the original signal. This isn't a retreat from the
finding; it's a sharper, better-supported version of it, found by honoring the guardrail (report every
day) rather than only the peaks.

**What this means for LUNA/FTX, stated as evidence for the user's decision, not a unilateral
re-approval:** the coarse version of "does the effect hold across new 2024 days" reads as a wash-out.
The sharper, severity-scaling version reads as a *strengthened, more precise* mechanism that — if it
extrapolates outside the 2024 severity range, which is untested — would predict an even larger RL edge
on genuinely more extreme days than anything in this dataset. LUNA and FTX were selected, before any
of this, as outcome-independent, publicly-extreme events; that framing is now more directly motivated
than it was, not less. Still gated on the previously-agreed de-risking step (validate the Tardis
pipeline against a known-good 2024 date first) before any subscription — that condition is unchanged.

## Data acquisition record

14 new dates acquired from Binance's public archive (`data.binance.vision`, futures/um/daily/{bookDepth,aggTrades}/BTCUSDT/),
selected entirely by public-event window, never by price outcome. `TradeData/` is gitignored — this is
local-only acquisition, nothing to commit for the data itself.

| window | dates | trigger | selection method |
|---|---|---|---|
| A — Iran–Israel escalation | 2024-04-12, 04-13, 04-14 | Iran's strike on Israel, night of Apr 13–14; tension public Apr 12 | event-date bounded |
| B — German govt BTC sales / Mt. Gox fear | 2024-07-04 through 07-12 (07-10 already present) | Boundary verified via news archive: first market-facing coverage [CoinDesk Jul 4](https://www.coindesk.com/markets/2024/07/04/bitcoin-nosedives-under-58k-amid-mt-gox-german-government-wallet-movements); German wallet reached zero balance Jul 12, [confirmed Jul 11](https://www.coindesk.com/markets/2024/07/11/germany-almost-done-selling-bitcoin-holding-less-than-5k-tokens-after-latest-moves) | window edges only, never day-level |
| C — Jobs-report fear → Aug 5 culmination | 2024-08-02, 08-03, 08-04 (08-05 already present) | Weak US jobs report, a scheduled release, Fri Aug 2 | event-date bounded |

Window B widened from the originally-proposed 07-01–07-08 (8 days) to the news-verified 07-04–07-12
(9 days) — a +1 net day, reported here rather than silently absorbed.

## Counting rule for 2024-04-15 — fixed now, before results exist

**2024-04-15 counts as ONE observation: the tail/aftermath day of window A's Iran–Israel episode, not
a second, independent data point.** It is already in the dataset as a `TRAINING_DATES` member labeled
"BTC correction (moderate stress)." That existing role in the train/held-out split is unchanged. But
when window A's severity pattern is reported (e.g. "how many of the N severe-stress days did RL win"),
**04-12/13/14/15 together are one episode** — reporting 04-15 separately alongside window A's other
three days would double-count a single event's aftermath as two independent observations, inflating
the apparent severe-stress n without adding real information. Any future summary table must either (a)
report window A as a single labeled episode spanning all four days with one outcome judgment, or (b)
if reporting day-by-day for transparency, explicitly annotate that 04-12/13/14/15 share one underlying
event and are not to be summed as four independent data points toward "n severe-stress days."

## Guardrails for when results come back (restated, not yet exercised)

1. **Report every day in each window**, including mild or losing ones — pulling by window instead of
   by peak only prevents hindsight if the reporting honors it too. A table that quietly drops the
   unremarkable days reintroduces the same selection bias one layer up.
2. **Window B's boundary is news-verified, not outcome-trimmed.** The 07-04–07-12 range came from when
   the event was publicly reported/active, never from which day inside it moved the most.

## 2024-07-05 row-count gap — resolved, not just spot-checked

Original concern: `BTCUSDT-bookDepth-2024-07-05.csv` has 28,541 rows vs. the standard 28,801 (13
dates × 10 percentage bins × 2880 nominal snapshots/day). Investigated properly:

- **Not scattered.** Of 26 missing snapshots (2880 nominal → 2854 actual), 25 are trivial ~31–37s
  jitter around the nominal 30s cadence (normal collection noise, present in varying degrees on most
  days) — not real gaps. **Exactly one real, contiguous gap: 08:17:30 → 08:31:00 UTC, 13.5 minutes**
  (snapshot index 995 → 996 of 2854).
- **Every episode drawn on this date will cross it.** `start_offset` is sampled from `[0, len//3)` =
  `[0, 951)`, which sits just before index 995 — no episode *starts* exactly at the gap, but any
  horizon long enough to reach index 995 (nearly all of them, since 995 ≪ 2880) crosses it partway
  through.
- **Measured price impact across the gap is small — smaller than ordinary single-step noise.** Checked
  against aggTrades ground truth (not bookDepth-derived, so independent of the gap itself): price moved
  **-0.036%** across the full 13.5-minute gap, versus a mean 30-second |return| of 0.056% (p95 0.14%,
  max 0.36%) in the surrounding hours that morning. A step that silently spans 13.5 real minutes will
  not register as an artificial volatility spike in the realized-vol estimator (`current_sigma`) — if
  anything the single compressed step looks quieter than typical, not louder.
- **Residual uncertainty, stated rather than waved away:** the price check does not rule out that
  order-book *depth* (not just price) evolved unusually across those 13.5 minutes and snapped back —
  bookDepth-derived state dims (s4 depth ratio, the liquidity half of s10/s11) aren't independently
  verifiable against a ground-truth source the way price was against aggTrades. Not fully resolved;
  judged low-risk given the price evidence, not certified clean.
- **Day boundaries are clean** (00:00:05 → 23:59:30, no start/end truncation).

**Conclusion: usable, with the above understood** — not silently accepted as "probably fine," not
excluded either.

## LUNA / FTX go/no-go — on hold pending cost + recalibration-risk answer

COVID dropped from consideration per explicit instruction (vendor-flagged data-quality issues on
Tardis.dev fall inside the exact pre-2020-05-14 window the COVID crash sits in — building on
acknowledged-unreliable data on the days that matter was judged indefensible).

### Tardis.dev pricing — incomplete, stated with the actual confidence level, not asserted as certain

Could not get a single authoritative, fully-itemized price list — `tardis.dev/pricing` did not return
readable pricing content via fetch, and secondary sources partially disagree:
- **Professional tier: ~$599/month**, corroborated by two independent sources.
- A cheaper tier (~$199/month per one source, using different tier-naming than Tardis's own docs —
  possibly a reseller/aggregator's own labels, not directly verified against Tardis's own site) —
  **lower confidence, not independently corroborated.**
- Tardis's own docs describe four tiers (Academic, Solo, Professional, Business) with **no dollar
  figures disclosed** in the FAQ pages fetched — pricing lives behind an order form.
- **A specific, material risk found while checking this, not just a pricing gap:** historical lookback
  appears to be measured *backward from the current billing date*, not as a fixed allotment —
  "yearly billing: 4 years of historical data," "quarterly: 12 months," "monthly: 4 months." From
  today (mid-2026), only the **annual** tier's 4-year lookback would reach back far enough to touch
  2022 at all, and even then, **LUNA (May 2022) sits close to or just past the edge of a strict 4-year
  window** — FTX (Nov 2022) is more comfortably inside it. Only the **Business** tier ("all available
  historical data since 2019-03-30") would guarantee both regardless of this arithmetic, and Business
  pricing was not found anywhere (typically the most expensive, "contact sales" tier).
- Docs explicitly state no one-time/non-subscription purchase option exists.
- **Recommendation: a direct quote/scoping request to Tardis is needed before this number is treated
  as real** — what's here is a bounded estimate (~$600–1000+/month is plausible depending on tier
  forced by the lookback issue above), not a quote.

### Recalibration-risk estimate — the substantive question, answered with reasoning shown

Read `CalibrationEngine::calibrate` and `build_calibration_trades` (`src/calibration/impact_fit.rs`)
against what Tardis would actually supply, to give a real confidence level rather than a guess:

- **The calibration fit itself (Y) is probably reasonably transferable — moderate-to-high confidence.**
  `Y = |Δprice/price| / (σ·√(Q/V))` is fundamentally a regression on trade price/size/timing data
  (aggTrades-equivalent), which Tardis provides in a comparable shape. This part of the pipeline is
  largely source-agnostic.
- **What is NOT transferable without new code: the book-depth input format itself.** Binance's
  `bookDepth` is a specific, coarse, pre-binned summary (10 fixed percentage-distance bins from mid,
  sampled roughly every 30s) — this exact shape is what `to_order_book()`, `bid_depth_up_to()`,
  `half_spread()`, and the fill/impact-walking logic all assume. Tardis supplies raw, full-depth L2
  snapshots/updates at a different (likely much finer, event-driven) granularity — a materially
  different data shape, not a relabeled version of the same thing. Using it requires genuinely new
  reconstruction code: aggregating raw L2 levels into equivalent percentage bins, and deciding a
  sampling cadence to match the simulator's `dt`/step-cadence assumptions (`total_steps=2880`, ~30s
  implied per step).
- **The real risk isn't the calibration math — it's whether a reconstructed bin, built by a different
  method than Binance's own, behaves the same way inside the simulator.** This project's own Gate A
  investigation (see the finding doc) already demonstrated, at length, that *discretization/binning
  granularity has first-order effects* on simulated fills, not second-order ones (the entire
  action-bucket granularity-floor story). By identical logic, a differently-sourced, differently-binned
  order-book input is a live, untested confound — even a well-calibrated Y could still produce
  simulated fills that are not apples-to-apples comparable to the 2024 Binance-native results, for
  reasons having nothing to do with whether LUNA/FTX were genuinely more severe.
- **Honest confidence level: LOW-TO-MODERATE that the existing pipeline produces directly comparable
  numbers on Tardis data without validation — not because the idea is unsound, but because nothing in
  this codebase has ever tested cross-source consistency.** No existing test checks "does a
  differently-sourced LOB snapshot produce comparable simulated outcomes to Binance's native
  bookDepth."
- **The concrete, cheap way to convert this from a guess into a measurement, before spending the
  subscription cost on LUNA/FTX specifically:** pull Tardis data for a date **already in the current
  2024 dataset**, run it through the same reconstruction-and-calibration pipeline that would be built
  for LUNA/FTX, and compare the resulting simulated outcome against the already-known-good
  Binance-native result for that same date. If they agree closely, confidence in LUNA/FTX numbers goes
  up cheaply and directly. If they diverge, that's the answer *before* any money or effort goes into
  the two target crashes specifically — the exact failure mode being asked about, caught at the
  cheapest possible point.

**Net recommendation, not a decision:** the recalibration risk is real and currently unmeasured, not
disqualifying. The overlap-date validation above is the way to measure it before committing. Cost is
still not pinned down precisely enough to greenlight on its own.
