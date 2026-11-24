"""
Candidate real-time regime features from the raw Bybit L2 book and trades (doc 27).

One row per 30 s book snapshot. Every feature uses only data at or before that
snapshot, within the same day, and is a ratio or a normalised quantity rather
than a price or volume level, so it cannot identify a day by its level.

    python rl/experiments/belief_precheck/book_features.py 2024-01-15   # prints a summary
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("TradeData/Bybit/BTCUSDT")
# The free 500-level book spans only ~13-20 bp each side (p10 13 bp), so
# "all" is the whole visible book and wider bands would all equal it.
BANDS_BP = (2, 5, 10, "all")
SNAPSHOT_MS = 30_000


def _load_book(date):
    df = pd.read_csv(DATA / f"book-{date}.csv")
    ts = df.timestamp.to_numpy()
    n_levels = sum(c.startswith("b") and c.endswith("_p") for c in df.columns)
    bp = df[[f"b{i}_p" for i in range(n_levels)]].to_numpy(float)
    bq = df[[f"b{i}_q" for i in range(n_levels)]].to_numpy(float)
    ap = df[[f"a{i}_p" for i in range(n_levels)]].to_numpy(float)
    aq = df[[f"a{i}_q" for i in range(n_levels)]].to_numpy(float)
    return ts, bp, np.nan_to_num(bq), ap, np.nan_to_num(aq)


def _trade_flow(date, ts):
    """Taker buy and sell volume in the 30 s window ending at each snapshot."""
    tr = pd.read_csv(DATA / f"trades-{date}.csv")
    idx = np.searchsorted(ts, tr.transact_time.to_numpy(), side="left")
    keep = idx < len(ts)
    idx, q = idx[keep], tr.quantity.to_numpy(float)[keep]
    sell = tr.is_buyer_maker.to_numpy()[keep]  # buyer is maker -> seller took liquidity
    buy_v = np.bincount(idx[~sell], q[~sell], minlength=len(ts))
    sell_v = np.bincount(idx[sell], q[sell], minlength=len(ts))
    return buy_v, sell_v


def _trailing_sum(x, n):
    c = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(1, len(x) + 1)
    return c[i] - c[np.maximum(i - n, 0)]


def _trailing_mean(x, n):
    return pd.Series(x).rolling(n, min_periods=1).mean().to_numpy()


def build(date):
    ts, bp, bq, ap, aq = _load_book(date)
    mid = (bp[:, 0] + ap[:, 0]) / 2
    f = {"date": date, "k": np.arange(len(ts)), "mid": mid}
    f["spread_bp"] = (ap[:, 0] - bp[:, 0]) / mid * 1e4
    micro = (bp[:, 0] * aq[:, 0] + ap[:, 0] * bq[:, 0]) / (bq[:, 0] + aq[:, 0] + 1e-12)
    f["micro_dev"] = (micro - mid) / (ap[:, 0] - bp[:, 0] + 1e-12)

    for d in BANDS_BP:
        if d == "all":
            bid, ask = bq.sum(1), aq.sum(1)
        else:
            bid = np.where(bp >= mid[:, None] * (1 - d / 1e4), bq, 0).sum(1)
            ask = np.where(ap <= mid[:, None] * (1 + d / 1e4), aq, 0).sum(1)
        f[f"imb_{d}"] = (bid - ask) / (bid + ask + 1e-12)
        tot = bid + ask
        f[f"depth_rel_{d}"] = np.log((tot + 1e-9) / (_trailing_mean(tot, 240) + 1e-9))
        f[f"bid_thin_{d}"] = np.log((bid + 1e-9) / (np.roll(bid, 20) + 1e-9))
        f[f"bid_thin_{d}"][:20] = 0.0

    buy_v, sell_v = _trade_flow(date, ts)
    vol = buy_v + sell_v
    for n in (2, 10, 60, 240):
        b, s = _trailing_sum(buy_v, n), _trailing_sum(sell_v, n)
        f[f"flow_{n}"] = (b - s) / (b + s + 1e-12)
    f["intensity_10"] = np.log((_trailing_sum(vol, 10) / 10 + 1e-9) / (_trailing_mean(vol, 240) + 1e-9))
    f["intensity_60"] = np.log((_trailing_sum(vol, 60) / 60 + 1e-9) / (_trailing_mean(vol, 960) + 1e-9))

    r = np.concatenate([[0.0], np.diff(np.log(mid))])
    sd240 = np.sqrt(_trailing_mean(r ** 2, 240)) + 1e-12
    for n in (10, 60, 240, 720):
        f[f"ret_z_{n}"] = _trailing_sum(r, n) / (sd240 * np.sqrt(n))
    sd60 = np.sqrt(_trailing_mean(r ** 2, 60)) + 1e-12
    f["vol_ratio_60_240"] = np.log(sd60 / sd240)
    f["vol_ratio_240_day"] = np.log(sd240 / (np.sqrt(pd.Series(r ** 2).expanding().mean().to_numpy()) + 1e-12))
    # drawdown from the running high, in trailing-vol units
    f["drawdown_z"] = np.log(mid / np.maximum.accumulate(mid)) / (sd240 * np.sqrt(240))

    out = pd.DataFrame(f)
    for h in (120, 240):
        out[f"fwd_{h}"] = (np.roll(mid, -h) / mid - 1) * 100
        out.loc[out.index[-h:], f"fwd_{h}"] = np.nan
    return out




def feature_columns(df):
    """Inputs only: never ids, prices or any target (fwd_*, fall_*)."""
    return [c for c in df.columns
            if c not in ("date", "k", "mid") and not c.startswith(("fwd_", "fall_"))]


if __name__ == "__main__":
    df = build(sys.argv[1])
    print(df[feature_columns(df)].describe().T.round(3).to_string())
