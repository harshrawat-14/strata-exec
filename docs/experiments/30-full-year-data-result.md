# 30 — Full-year 2024 data: result

Predictions in [30-full-year-data-predictions.md](30-full-year-data-predictions.md).
Check: `rl/experiments/v4_data_check.py` → `rl/results/v4_day_stats.csv`. Held-out and
future dates were not downloaded or read.

## Data

- **Complete:** all 343 pool days downloaded and converted. Every day has 2,879–2,880 book
  snapshots, so none was excluded.
- **Trades:** 256,846 to 5.32 M per day (median 1.46 M; busiest 2024-11-06).
- **Size:** 5.3 GB as gzip CSV for all 2024 days on disk (the V4 pool plus the old 25-day pool).

**Downloader bugs found and fixed (`fetch_bybit.py`):**
1. Bybit's servers reset connections often. Without a retry, 2 days failed.
2. The first retry fix resumed partial downloads, but a connection that closed cleanly
   mid-file was treated as a finished download. Two zips were half-length and failed to open.
   Downloads now resume until the size matches the server's.

No converted day is affected by either bug. A truncated zip cannot be opened (its index is
at the end), and a truncated gzip raises an error when read, so every incomplete file failed
loudly at conversion. All 4 failed days were then fetched again in full.

## Predictions

| # | prediction | result | |
|---|---|---|---|
| 1 | crash 10–20%, calm 30–45% | crash 15.7% (54), normal 52.5% (180), calm 31.8% (109) | met |
| 2 | crash share higher Apr–Sep than Q1 + Q4 | 18.8% vs 12.7% | met |
| 3 | ≥ 99% of days complete | 100% | met |
| 4 | pool median vol 10–40% below the V3 days' | 2.47% vs 3.37% (−27%) | met |

## What changes

| | V3 training | V4 training | V4 validation |
|---|---:|---:|---:|
| crash days | 3 | **44** | 10 |
| normal days | 5 | 133 | 47 |
| calm days | 4 | 79 | 30 |

Every regime now has at least 10 validation days and 44 training days. The next step is the
pre-registered rerun of doc 29's decay-by-volatility test on this pool.
