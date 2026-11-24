"""
Doc 30: data completeness and regime labels for the V4 pool (343 days), scored against
the doc's pre-registered predictions. Held-out and future dates are not read.

    python rl/experiments/v4_data_check.py
"""
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/

import pandas as pd

from evaluation.dates import TRAINING_DATES_V3, TRAINING_DATES_V4, VALIDATION_DATES_V3, VALIDATION_DATES_V4
from evaluation.day_regime import day_stats
from market_data import trades_path

OUT = "rl/results/v4_day_stats.csv"


def row(date):
    r = day_stats(date)
    r["n_trades"] = len(pd.read_csv(trades_path(date), usecols=["quantity"]))
    r["split"] = "train" if date in TRAINING_DATES_V4 else "validation"
    return r


def main():
    dates = TRAINING_DATES_V4 + VALIDATION_DATES_V4
    with Pool(8) as pool:
        df = pd.DataFrame(pool.map(row, dates))
    df.to_csv(OUT, index=False)
    n = len(df)

    full = df.n_snapshots.between(2879, 2880)
    print(f"1/3. completeness: {full.sum()}/{n} days with 2,879-2,880 snapshots "
          f"({full.mean():.1%}; predicted >= 99%)")
    short = df[df.n_snapshots < 2800]
    print(f"     days below 2,800 (excluded): {short.date.tolist() or 'none'}")
    print(f"     trades per day: min {df.n_trades.min():,}, median {int(df.n_trades.median()):,}")

    share = df.regime.value_counts(normalize=True)
    print("\n1. regime shares (predicted crash 10-20%, calm 30-45%):")
    for reg in ("crash", "normal", "calm"):
        print(f"     {reg:>6}: {share.get(reg, 0):.1%}  ({(df.regime == reg).sum()} days)")
    print("     by split:", df.groupby("split").regime.value_counts().unstack().to_dict("index"))

    m = pd.to_datetime(df.date).dt.month
    mid_year = df[m.between(4, 9)]
    ends = df[~m.between(4, 9)]
    a, b = (mid_year.regime == "crash").mean(), (ends.regime == "crash").mean()
    print(f"\n2. crash share Apr-Sep {a:.1%} vs Q1+Q4 {b:.1%}  (predicted Apr-Sep higher: {a > b})")

    v3 = df[df.date.isin(list(TRAINING_DATES_V3) + list(VALIDATION_DATES_V3))]
    ratio = df.sigma_pct.median() / v3.sigma_pct.median() - 1
    print(f"\n4. median daily vol: pool {df.sigma_pct.median():.2f}% vs V3 days {v3.sigma_pct.median():.2f}% "
          f"({ratio:+.0%}; predicted -10% to -40%)")
    print("\nby month (days per regime):")
    print(pd.crosstab(m.rename("month"), df.regime).to_string())


if __name__ == "__main__":
    main()
