"""
Day regime labels for any downloaded date (doc 30): drift = mean of ln(mid_t / mid_open)
over the day's 30 s book snapshots, in %, labelled with dates.regime_v4.

    python rl/evaluation/day_regime.py 2024-05-01 2024-05-02 ...
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/

import numpy as np
import pandas as pd

from evaluation.dates import regime_v4
from market_data import book_path


def day_stats(date: str) -> dict:
    book = pd.read_csv(book_path(date), usecols=["b0_p", "a0_p"])
    mid = ((book.b0_p + book.a0_p) / 2).to_numpy()
    lr = np.log(mid / mid[0])
    r = np.diff(np.log(mid))
    drift = lr.mean() * 100
    return dict(date=date, drift_pct=drift, return_pct=lr[-1] * 100,
                sigma_pct=r.std(ddof=1) * np.sqrt(len(r)) * 100, regime=regime_v4(drift))


if __name__ == "__main__":
    print(pd.DataFrame([day_stats(d) for d in sys.argv[1:]]).round(3).to_string(index=False))
