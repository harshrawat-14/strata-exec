"""
Doc 28: does square-root impact scale with local liquidity rather than the day's?

Extends impact_fit_signed_flow.py (doc 21). For each bucket, r = ln(mid_end / mid_start)
and Q = net taker flow. Model: r = Y * sigma * sign(Q) * sqrt(|Q| / V), with sigma and V
either the date's daily values (current simulator) or trailing-window values that end
before the bucket (causal), scaled to a daily rate. Y is frozen at the training-date
median and scored out of sample on the validation dates. Held-out dates are not read.

    python rl/experiments/impact_fit_local_liquidity.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/

import numpy as np
import pandas as pd

from evaluation.dates import TRAINING_DATES_V3, VALIDATION_DATES_V3
from market_data import book_path, trades_path

OUT = "rl/results/impact_fit_local_liquidity.csv"
MODELS = ("day", "local_V", "local_sigma", "local_both")
MAIN_WINDOW_MIN, MAIN_BUCKET = 60, "30s"


def _trailing(x, w):
    """Sum of x over the w entries strictly before each index (0 where history is short)."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(len(x))
    return c[i] - c[np.maximum(i - w, 0)]


def date_rows(date, window_min):
    """Per-bucket regressors for every model, 30 s and 5 min buckets."""
    book = pd.read_csv(book_path(date), usecols=["timestamp", "b0_p", "a0_p"])
    mid = ((book.b0_p + book.a0_p) / 2).to_numpy()
    ts = book.timestamp.to_numpy()
    t = pd.read_csv(trades_path(date))
    q = t.quantity.to_numpy(float)
    signed = np.where(t.is_buyer_maker.astype(str).str.lower() == "true", -1.0, 1.0) * q
    idx = np.searchsorted(ts, t.transact_time.to_numpy(), side="left")
    flow30 = np.bincount(idx, weights=signed, minlength=len(ts) + 1)[1:len(ts)]
    vol30 = np.bincount(idx, weights=q, minlength=len(ts) + 1)[1:len(ts)]
    r30 = np.diff(np.log(mid))
    per_day = len(r30)  # 30 s buckets in the day

    sigma_d = np.std(r30, ddof=1) * np.sqrt(per_day)
    v_day = q.sum()
    w = window_min * 2
    v_loc = _trailing(vol30, w) / w * per_day
    s_loc = np.sqrt(_trailing(r30 ** 2, w) / w * per_day)

    out = []
    for label, k in (("30s", 1), ("5min", 10)):
        n = (len(r30) // k) * k
        starts = np.arange(0, n, k)
        r = r30[:n].reshape(-1, k).sum(1)
        flow = flow30[:n].reshape(-1, k).sum(1)
        keep = starts >= max(w, 120)  # same rows for every model
        norm = {
            "day": (np.full(len(starts), sigma_d), np.full(len(starts), v_day)),
            "local_V": (np.full(len(starts), sigma_d), v_loc[starts]),
            "local_sigma": (s_loc[starts], np.full(len(starts), v_day)),
            "local_both": (s_loc[starts], v_loc[starts]),
        }
        df = pd.DataFrame({"date": date, "bucket": label, "r": r})
        for m, (s, v) in norm.items():
            df[m] = s * np.sign(flow) * np.sqrt(np.abs(flow) / np.maximum(v, 1e-9))
        out.append(df[keep])
    return pd.concat(out)


def _r2(r, x, y):
    return 1 - np.sum((r - y * x) ** 2) / np.sum((r - r.mean()) ** 2)


def evaluate(window_min):
    from multiprocessing import Pool
    dates = list(TRAINING_DATES_V3) + list(VALIDATION_DATES_V3)
    with Pool(8) as pool:
        df = pd.concat(pool.starmap(date_rows, [(d, window_min) for d in dates]), ignore_index=True)
    rows = []
    for bucket, b in df.groupby("bucket"):
        for m in MODELS:
            per_y = {d: (g[m] @ g.r) / (g[m] @ g[m]) for d, g in b.groupby("date")}
            y_frozen = np.median([per_y[d] for d in TRAINING_DATES_V3])
            ys = np.array(list(per_y.values()))
            val = b[b.date.isin(VALIDATION_DATES_V3)]
            row = dict(window_min=window_min, bucket=bucket, model=m, Y_frozen=y_frozen,
                       Y_cv=ys.std() / ys.mean(), pooled_val_r2=_r2(val.r, val[m], y_frozen))
            for d, g in b.groupby("date"):
                row[f"r2_{d}"] = _r2(g.r, g[m], y_frozen)
                row[f"Y_{d}"] = per_y[d]
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    from evaluation.dates import SEVERITY_RANK_V2
    res = pd.concat([evaluate(w) for w in (MAIN_WINDOW_MIN, 15, 120)], ignore_index=True)
    res.to_csv(OUT, index=False)
    for (w, bucket), g in res.groupby(["window_min", "bucket"], sort=False):
        tag = "MAIN" if (w, bucket) == (MAIN_WINDOW_MIN, MAIN_BUCKET) else "sensitivity"
        base = g[g.model == "day"].iloc[0]
        print(f"\n== window {w} min, bucket {bucket} ({tag})")
        for _, r in g.iterrows():
            wins_v = sum(r[f"r2_{d}"] > base[f"r2_{d}"] for d in VALIDATION_DATES_V3)
            wins_t = sum(r[f"r2_{d}"] > base[f"r2_{d}"] for d in TRAINING_DATES_V3)
            print(f"  {r.model:<12} Y {r.Y_frozen:.3f}  Y_cv {r.Y_cv:.3f}  pooled val R2 {r.pooled_val_r2:.4f}"
                  f" ({r.pooled_val_r2 - base.pooled_val_r2:+.4f})  beats day: val {wins_v}/4, train {wins_t}/12")
    main_ = res[(res.window_min == MAIN_WINDOW_MIN) & (res.bucket == MAIN_BUCKET)]
    print("\nper-date fitted Y (main setting), by regime:")
    dates = list(TRAINING_DATES_V3) + list(VALIDATION_DATES_V3)
    tab = pd.DataFrame({m: {d: main_[main_.model == m][f"Y_{d}"].iloc[0] for d in dates} for m in MODELS})
    tab["regime"] = tab.index.map(lambda d: SEVERITY_RANK_V2[d][1])
    print(tab.sort_values("regime").round(3).to_string())


if __name__ == "__main__":
    main()
