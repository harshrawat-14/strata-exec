"""
Doc 29: how does the price response to taker flow decay, and does the decay
depend on an observable volatility regime?

Propagator-style regression on 30 s buckets: r_t = sum_b beta_b * sum_{l in bin b} x_{t-l},
x = sigma_d * sign(Q) * sqrt(|Q| / V) (doc 28's daily normalisation). The response to one
unit of flow tau steps later is G(tau) = sum_{l <= tau} beta(l). Held-out dates are not read.

    python rl/experiments/impact_decay_fit.py        # doc 29: V3 (12 training, 4 validation dates)
    python rl/experiments/impact_decay_fit.py v4     # doc 31: V4 (256 training, 87 validation dates)
    python rl/experiments/impact_decay_fit.py v4 --dataset bybit_solusdt_perp   # doc 34
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/

import numpy as np
import pandas as pd

from evaluation.dates import (SEVERITY_RANK_V2, TRAINING_DATES_V3, TRAINING_DATES_V4,
                              VALIDATION_DATES_V3, VALIDATION_DATES_V4)
from market_data import book_path, results_tag, trades_path, use_dataset_from_argv

POOLS = {  # name: (training, validation, output, hindsight labels)
    "v3": (TRAINING_DATES_V3, VALIDATION_DATES_V3, "rl/results/impact_decay_fit.csv",
           lambda: {d: v[1] for d, v in SEVERITY_RANK_V2.items()}),
    "v4": (TRAINING_DATES_V4, VALIDATION_DATES_V4, "rl/results/impact_decay_fit_v4{tag}.csv",
           lambda: pd.read_csv(f"rl/results/v4_day_stats{results_tag()}.csv").set_index("date").regime.to_dict()),
}
MIN_EFFECT = 0.05  # doc 31: |high - low| in G(2h)/G(0) must be at least this to count
BINS = [(0, 0), (1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 127),
        (128, 255), (256, 511), (512, 720)]
MAX_LAG = 720
TAUS = {"30min": 60, "2h": 240, "6h": 720}
VOL_WINDOW = 240  # trailing 2 h
SIM = {"30min": 0.965, "2h": 0.873, "6h": 0.707}  # sqrt(0.3 + 0.7 exp(-5 tau))
N_BOOT = 1000


def _lag_sum(v, lo, hi):
    """sum_{l=lo..hi} v[t-l] for every t (0 where t-l < 0)."""
    c = np.concatenate([[0.0], np.cumsum(v)])
    t = np.arange(len(v))
    return c[np.clip(t - lo + 1, 0, None)] - c[np.clip(t - hi, 0, None)]


def date_data(date):
    book = pd.read_csv(book_path(date), usecols=["timestamp", "b0_p", "a0_p"])
    mid = ((book.b0_p + book.a0_p) / 2).to_numpy()
    ts = book.timestamp.to_numpy()
    t = pd.read_csv(trades_path(date))
    signed = np.where(t.is_buyer_maker.astype(str).str.lower() == "true", -1.0, 1.0) * t.quantity.to_numpy()
    idx = np.searchsorted(ts, t.transact_time.to_numpy(), side="left")
    flow = np.bincount(idx, weights=signed, minlength=len(ts) + 1)[1:len(ts)]
    r = np.diff(np.log(mid))
    sigma_d = np.std(r, ddof=1) * np.sqrt(len(r))
    x = sigma_d * np.sign(flow) * np.sqrt(np.abs(flow) / t.quantity.sum())
    # trailing realised vol BEFORE bucket t (causal); NaN until the window is full
    c2 = np.concatenate([[0.0], np.cumsum(r ** 2)])
    i = np.arange(len(r))
    tv = np.where(i >= VOL_WINDOW, np.sqrt((c2[i] - c2[np.maximum(i - VOL_WINDOW, 0)]) / VOL_WINDOW), np.nan)
    return dict(date=date, r=r, x=x, tv=tv)


def design(d, cuts, model):
    """Columns for 'pooled' (one kernel), 'shared' (tercile lag 0, shared decay) or
    'tercile' (every lag bin per tercile of the vol at the FLOW's time)."""
    x, tv = d["x"], d["tv"]
    terc = np.where(np.isnan(tv), -1, np.digitize(tv, cuts))  # -1: no vol yet
    cols = []
    for b, (lo, hi) in enumerate(BINS):
        if model == "pooled" or (model == "shared" and b > 0):
            cols.append(_lag_sum(x, lo, hi))
        else:
            # flows before the vol window is full belong to tercile 1 (middle)
            for k in range(3):
                cols.append(_lag_sum(x * ((terc == k) | ((terc == -1) & (k == 1))), lo, hi))
    X = np.column_stack(cols)
    keep = np.arange(len(x)) >= MAX_LAG
    return X[keep], d["r"][keep]


def curve(beta_per_bin):
    """G(tau)/G(0) at TAUS from per-lag coefficients by bin."""
    g0 = beta_per_bin[0]
    out = {}
    for name, tau in TAUS.items():
        g = sum(beta_per_bin[b] * (min(hi, tau) - lo + 1) for b, (lo, hi) in enumerate(BINS) if lo <= tau)
        out[name] = g / g0
    return out


def split_beta(beta, model):
    """Per-tercile per-bin coefficient arrays (pooled -> same array three times)."""
    if model == "pooled":
        return [beta] * 3
    per = [[], [], []]
    j = 0
    for b in range(len(BINS)):
        if model == "shared" and b > 0:
            for k in range(3):
                per[k].append(beta[j])
            j += 1
        else:
            for k in range(3):
                per[k].append(beta[j + k])
            j += 3
    return [np.array(p) for p in per]


def main(pool_name="v3"):
    from multiprocessing import Pool
    TRAIN, VAL, OUT, labels = POOLS[pool_name]
    dates = list(TRAIN) + list(VAL)
    with Pool(8) as pool:
        data = {d["date"]: d for d in pool.map(date_data, dates)}
    tv_train = np.concatenate([data[d]["tv"][MAX_LAG:] for d in TRAIN])
    cuts = np.nanpercentile(tv_train, [100 / 3, 200 / 3])

    # Gram matrices for every date; design matrices kept only for validation scoring
    mats, gram = {}, {}
    for m in ("pooled", "shared", "tercile"):
        mats[m], gram[m] = {}, {}
        for d in dates:
            X, y = design(data[d], cuts, m)
            gram[m][d] = (X.T @ X, X.T @ y)
            if d in VAL:
                mats[m][d] = (X, y)

    def fit(m, ds):
        A = sum(gram[m][d][0] for d in ds)
        b = sum(gram[m][d][1] for d in ds)
        return np.linalg.solve(A, b)

    def r2(m, beta, ds):
        X = np.vstack([mats[m][d][0] for d in ds])
        y = np.concatenate([mats[m][d][1] for d in ds])
        return 1 - np.sum((y - X @ beta) ** 2) / np.sum((y - y.mean()) ** 2)

    rng = np.random.default_rng(0)
    train = list(TRAIN)
    boots = [list(rng.choice(train, len(train))) for _ in range(N_BOOT)]
    rows = []

    # A: pooled kernel vs the simulator
    print("A. pooled decay G(tau)/G(0), training dates (90% date-bootstrap interval) vs simulator")
    point = curve(fit("pooled", train))
    bs = pd.DataFrame([curve(fit("pooled", s)) for s in boots])
    for name in TAUS:
        lo, hi = bs[name].quantile([0.05, 0.95])
        inside = lo <= SIM[name] <= hi
        print(f"  {name:>6}: {point[name]:.3f}  [{lo:.3f}, {hi:.3f}]  simulator {SIM[name]:.3f} "
              f"{'inside' if inside else 'OUTSIDE'}")
        rows.append(dict(test="A", tau=name, point=point[name], lo=lo, hi=hi, sim=SIM[name]))
    vpoint = curve(fit("pooled", VAL))
    print("  validation dates:", {k: round(v, 3) for k, v in vpoint.items()})
    print("  kernel per lag (training):", np.round(fit("pooled", train) * 1e4, 3).tolist(), "x1e-4")

    # B: does decay depend on the trailing-vol tercile?
    print(f"\nB. decay by trailing 2 h vol tercile (cuts {cuts[0]:.2e}, {cuts[1]:.2e} per 30 s)")
    per = split_beta(fit("tercile", train), "tercile")
    pv = split_beta(fit("tercile", VAL), "tercile")
    bdiff = []
    for s in boots:
        p = split_beta(fit("tercile", s), "tercile")
        bdiff.append(curve(p[2])["2h"] - curve(p[0])["2h"])
    for k, name in enumerate(("low", "mid", "high")):
        c = curve(per[k])
        print(f"  {name:>4} vol: G0 {per[k][0]*1e4:.3f}e-4  " + "  ".join(f"{t} {v:.3f}" for t, v in c.items())
              + f"   | validation 2h {curve(pv[k])['2h']:.3f}")
    diff = curve(per[2])["2h"] - curve(per[0])["2h"]
    lo, hi = np.quantile(bdiff, [0.05, 0.95])
    vdiff = curve(pv[2])["2h"] - curve(pv[0])["2h"]
    r2_shared = r2("shared", fit("shared", train), VAL)
    r2_terc = r2("tercile", fit("tercile", train), VAL)
    r2_pool = r2("pooled", fit("pooled", train), VAL)
    c1 = lo > 0 or hi < 0
    c2 = np.sign(vdiff) == np.sign(diff)
    c3 = r2_terc >= r2_shared
    c4 = abs(diff) >= MIN_EFFECT
    print(f"  high - low, G(2h)/G(0): {diff:+.3f}  [{lo:+.3f}, {hi:+.3f}]  -> excludes 0: {c1}")
    print(f"  validation high - low: {vdiff:+.3f}  -> same sign: {c2}")
    print(f"  validation R2: pooled {r2_pool:.5f}  shared {r2_shared:.5f}  tercile {r2_terc:.5f}  -> tercile >= shared: {c3}")
    if pool_name == "v4":  # doc 31 adds a minimum effect size
        print(f"  |high - low| >= {MIN_EFFECT}: {c4}")
        print(f"  criterion B: {'PASS' if (c1 and c2 and c3 and c4) else 'FAIL'}")
    else:
        print(f"  criterion B: {'PASS' if (c1 and c2 and c3) else 'FAIL'}")
    rows.append(dict(test="B", tau="2h", point=diff, lo=lo, hi=hi, validation=vdiff,
                     r2_pooled=r2_pool, r2_shared=r2_shared, r2_tercile=r2_terc))

    # descriptive: hindsight day labels, all 16 dates
    print(f"\ndescriptive (hindsight labels, all {len(dates)} dates; not a criterion):")
    lab = labels()
    for reg in ("calm", "normal", "crash"):
        ds = [d for d in dates if lab[d] == reg]
        c = curve(fit("pooled", ds))
        print(f"  {reg:>6} ({len(ds)}): " + "  ".join(f"{t} {v:.3f}" for t, v in c.items()))
        rows.append(dict(test="hindsight", tau=reg, **{f"G_{t}": v for t, v in c.items()}))
    pd.DataFrame(rows).to_csv(OUT.format(tag=results_tag()), index=False)


if __name__ == "__main__":
    main(*use_dataset_from_argv(sys.argv[1:]))
