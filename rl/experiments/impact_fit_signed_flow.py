"""
Fit the square-root impact law on real signed order flow (era 2, Bybit).

Per date, per bucket (30 s and 5 min): r = ln(mid_end / mid_start) and the net
taker flow Q = buy volume - sell volume. Model:  r = Y * sigma_d * sign(Q) * sqrt(|Q| / V)
with sigma_d the date's daily volatility (from 30 s mids) and V its daily volume.
Y is the least-squares slope (no intercept). If impact really is square-root,
Y is roughly the same at both bucket sizes.

    python rl/experiments/impact_fit_signed_flow.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/

import numpy as np
import pandas as pd

from evaluation.dates import SEVERITY_RANK_V2, TRAINING_DATES_V3
from market_data import book_path, trades_path

OUT = "rl/results/impact_fit_signed_flow.csv"


def fit_date(date: str) -> list:
    book = pd.read_csv(book_path(date), usecols=["timestamp", "b0_p", "a0_p"])
    mid = ((book.b0_p + book.a0_p) / 2).to_numpy()
    ts = book.timestamp.to_numpy()
    t = pd.read_csv(trades_path(date))
    signed = np.where(t.is_buyer_maker.astype(str).str.lower() == "true", -1.0, 1.0) * t.quantity.to_numpy()
    volume = t.quantity.sum()
    sigma_d = np.std(np.diff(np.log(mid)), ddof=1) * np.sqrt(len(mid))
    idx = np.searchsorted(ts, t.transact_time.to_numpy(), side="left")  # trade belongs to bucket ending at ts[idx]
    flow30 = np.bincount(idx, weights=signed, minlength=len(ts) + 1)[1:len(ts)]
    r30 = np.diff(np.log(mid))
    rows = []
    for label, k in (("30s", 1), ("5min", 10)):
        n = (len(r30) // k) * k
        r = r30[:n].reshape(-1, k).sum(1)
        q = flow30[:n].reshape(-1, k).sum(1)
        x = sigma_d * np.sign(q) * np.sqrt(np.abs(q) / volume)
        y = float((x @ r) / (x @ x))
        r2 = 1 - np.sum((r - y * x) ** 2) / np.sum((r - r.mean()) ** 2)
        rows.append(dict(date=date, bucket=label, Y=y, r2=r2, sigma_daily=sigma_d, volume_btc=volume))
    return rows


def main():
    rows = [r for d in sorted(SEVERITY_RANK_V2) for r in fit_date(d)]
    df = pd.DataFrame(rows)
    df["regime"] = df.date.map(lambda d: SEVERITY_RANK_V2[d][1] if isinstance(SEVERITY_RANK_V2[d][1], str) else "")
    df.to_csv(OUT, index=False)
    wide = df.pivot(index="date", columns="bucket", values=["Y", "r2"]).round(3)
    print(wide.to_string())
    train = df[df.date.isin(TRAINING_DATES_V3)]
    print("\ntraining-date median Y:", train.groupby("bucket").Y.median().round(3).to_dict())


if __name__ == "__main__":
    main()
