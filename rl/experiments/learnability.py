"""
Doc 38: from what a trader can see at time t, can a model say whether selling the next 2 h
faster or slower pays, on unseen days? Pre-registered in
docs/experiments/38-learnability-predictions.md.

Target y_t = -(mean mid over (t, t+H] / mid_t - 1) in bp: positive when selling now beat
spreading the slice over the next 2 h. Inputs are round-number and support/resistance levels,
past returns, volatility, book imbalance and time of day, all in trailing-sigma units.

Level features come from the same path as the target, so they correlate with it even on a
random walk (doc 32). Every statistic is also computed on sign-flipped null paths (doc 34) and
reported as real minus null. Held-out and future dates are not read.

    python rl/experiments/learnability.py
"""
import sys
from datetime import date as Date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from evaluation.dates import TRAINING_DATES_V4, VALIDATION_DATES_V4
from market_data import book_path

H = 240  # forward horizon: 2 h of 30 s steps
STEP = 60  # a decision every 30 min
WARMUP = 480  # 4 h of history before the first decision
N_LEVELS = 5  # book levels in the imbalance
N_NULL = 20
N_BOOT = 1000
MIN_GAIN_BP = 3.0
TOUCH_BAND = 0.25  # sigma units
TOUCH_GAP = 10  # steps between separate visits
OUT = "rl/results/learnability_days.csv"

ROUND = ["d_1000", "d_500", "near_round", "side_1000"]
LEVELS = ["d_day_high", "d_day_low", "d_4h_high", "d_4h_low", "d_prev_high", "d_prev_low",
          "d_prev_close", "touch_4h_high", "touch_4h_low", "pos_4h", "prev_missing"]
TREND = ["ret_10", "ret_60", "ret_240", "ret_480"]
VOL = ["log_sigma", "vol_ratio"]
OTHER = ["imbalance", "tod_sin", "tod_cos"]
ALL = ROUND + LEVELS + TREND + VOL + OTHER
GROUPS = {"round numbers": ROUND, "support/resistance": LEVELS, "trend": TREND,
          "volatility": VOL, "book + time": OTHER}


def load_day(date):
    """(mid, top-5 depth imbalance) per 30 s snapshot, or None when the day is not on disk."""
    path = Path(book_path(date))
    if not path.exists():
        return None
    cols = [f"{s}{i}_{k}" for s in "ba" for i in range(N_LEVELS) for k in "pq"]
    b = pd.read_csv(path, usecols=cols)
    mid = ((b.b0_p + b.a0_p) / 2).to_numpy(float)
    bq = b[[f"b{i}_q" for i in range(N_LEVELS)]].fillna(0).to_numpy().sum(1)
    aq = b[[f"a{i}_q" for i in range(N_LEVELS)]].fillna(0).to_numpy().sum(1)
    return mid, (bq - aq) / np.maximum(bq + aq, 1e-12)


def prev_levels(date):
    """Previous calendar day's (high, low, close) of mid; NaN when that day is not on disk."""
    prev = (Date.fromisoformat(date) - timedelta(days=1)).isoformat()
    day = load_day(prev)
    if day is None:
        return np.nan, np.nan, np.nan
    m = day[0]
    return m.max(), m.min(), m[-1]


def _trailing_mean(x, n):
    return pd.Series(x).rolling(n, min_periods=1).mean().to_numpy()


def _touches(mid, t, level, band):
    """Separate visits within `band` (log units) of `level` in the WARMUP steps up to t."""
    near = np.abs(np.log(mid[t - WARMUP + 1:t + 1] / level)) <= band
    idx = np.flatnonzero(near)
    return 0 if len(idx) == 0 else 1 + int((np.diff(idx) >= TOUCH_GAP).sum())


def decision_points(n):
    return np.arange(WARMUP, n - H, STEP)


def target(mid, k):
    """y in bp: gain from selling at t instead of at the mean mid of the next H steps."""
    c = np.concatenate([[0.0], np.cumsum(mid)])
    return -((c[k + H + 1] - c[k + 1]) / H / mid[k] - 1) * 1e4


def features(mid, imb, prev):
    """One row per decision point. Row t uses mid[:t+1], imb[:t+1] and the previous day only."""
    n = len(mid)
    k = decision_points(n)
    r = np.concatenate([[0.0], np.diff(np.log(mid))])
    sig = np.sqrt(_trailing_mean(r ** 2, 240)) * np.sqrt(240) + 1e-12
    sd1h = np.sqrt(_trailing_mean(r ** 2, 120))
    sd4h = np.sqrt(_trailing_mean(r ** 2, 480)) + 1e-12
    hi4 = pd.Series(mid).rolling(WARMUP, min_periods=1).max().to_numpy()
    lo4 = pd.Series(mid).rolling(WARMUP, min_periods=1).min().to_numpy()
    hid, lod = np.maximum.accumulate(mid), np.minimum.accumulate(mid)
    m, s = mid[k], sig[k]

    def d(level):
        return np.log(m / level) / s

    r1000, r500 = np.round(m / 1000) * 1000, np.round(m / 500) * 500
    p_hi, p_lo, p_cl = prev
    missing = np.isnan(p_hi)
    tod = 2 * np.pi * k / 2880
    f = pd.DataFrame({
        "d_1000": d(r1000), "d_500": d(r500),
        "near_round": (np.abs(d(r500)) < 0.5).astype(float),
        "side_1000": np.sign(m - r1000),
        "d_day_high": d(hid[k]), "d_day_low": d(lod[k]),
        "d_4h_high": d(hi4[k]), "d_4h_low": d(lo4[k]),
        "d_prev_high": d(p_hi) if not missing else np.nan,
        "d_prev_low": d(p_lo) if not missing else np.nan,
        "d_prev_close": d(p_cl) if not missing else np.nan,
        "touch_4h_high": [_touches(mid, t, hi4[t], TOUCH_BAND * sig[t]) for t in k],
        "touch_4h_low": [_touches(mid, t, lo4[t], TOUCH_BAND * sig[t]) for t in k],
        "pos_4h": (m - lo4[k]) / np.maximum(hi4[k] - lo4[k], 1e-9),
        "prev_missing": float(missing),
        **{f"ret_{j}": np.log(m / mid[k - j]) / s for j in (10, 60, 240, 480)},
        "log_sigma": np.log(s), "vol_ratio": sd1h[k] / sd4h[k],
        "imbalance": _trailing_mean(imb, 10)[k],
        "tod_sin": np.sin(tod), "tod_cos": np.cos(tod),
    })
    f["step"] = k
    f["y"] = target(mid, k)
    return f


def signflip(mid, rng):
    r = np.diff(np.log(mid))
    return mid[0] * np.exp(np.concatenate([[0.0], np.cumsum(r * rng.choice([-1.0, 1.0], len(r)))]))


def _day(args):
    """Real rows plus n_null sign-flipped rows for one day (same imbalance and previous day)."""
    date, n_null, seed = args
    mid, imb = load_day(date)
    prev = prev_levels(date)
    rng = np.random.default_rng(seed)
    real = features(mid, imb, prev).assign(date=date, path=-1)
    nulls = [features(signflip(mid, rng), imb, prev).assign(date=date, path=i) for i in range(n_null)]
    return pd.concat([real] + nulls, ignore_index=True)


class Ridge:
    """Ridge on standardised inputs; missing values become 0 after standardising."""

    def __init__(self, cols, alpha=1.0):
        self.cols, self.alpha = cols, alpha

    def fit(self, df):
        from sklearn.linear_model import Ridge as R
        x = df[self.cols].to_numpy(float)
        self.mu, self.sd = np.nanmean(x, 0), np.nanstd(x, 0) + 1e-12
        self.m = R(alpha=self.alpha).fit(self._x(df), df.y.to_numpy())
        return self

    def _x(self, df):
        return np.nan_to_num((df[self.cols].to_numpy(float) - self.mu) / self.sd)

    def predict(self, df):
        return self.m.predict(self._x(df))

    def coefs(self):
        return pd.Series(self.m.coef_, index=self.cols)


class Boost:
    def __init__(self, cols):
        self.cols = cols

    def fit(self, df):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.m = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05,
                                               random_state=0).fit(df[self.cols], df.y)
        return self

    def predict(self, df):
        return self.m.predict(df[self.cols])


def day_scores(df, pred, cut):
    """Per (date, path): Spearman IC of pred vs y, and timing gain mean(sign(pred - cut) * y)."""
    d = df[["date", "path", "y"]].assign(pred=pred)

    def one(g):
        ic = spearmanr(g.pred, g.y)[0] if g.pred.std() > 0 else 0.0
        return pd.Series({"ic": ic, "gain": float(np.mean(np.sign(g.pred - cut) * g.y))})
    return d.groupby(["date", "path"])[["pred", "y"]].apply(one).reset_index()


def corrected(scores):
    """Per date: real, null mean and real - null for ic and gain."""
    real = scores[scores.path == -1].set_index("date")[["ic", "gain"]]
    null = scores[scores.path >= 0].groupby("date")[["ic", "gain"]].mean()
    out = real.join(null, lsuffix="_real", rsuffix="_null")
    out["ic"] = out.ic_real - out.ic_null
    out["gain"] = out.gain_real - out.gain_null
    return out


def boot(s, rng, n=N_BOOT):
    """Mean and 90% bootstrap interval over days."""
    v = s.to_numpy()
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return v.mean(), np.quantile(means, 0.05), np.quantile(means, 0.95)


def passes(c, regime, rng):
    """Doc 38's three conditions on corrected validation scores (c indexed by date)."""
    ic, ic_lo, _ = boot(c.ic, rng)
    g, g_lo, _ = boot(c.gain, rng)
    crash = c.gain[regime.reindex(c.index) == "crash"].mean()
    calm = c.gain[regime.reindex(c.index) == "calm"].mean()
    cond = [ic > 0 and ic_lo > 0, g >= MIN_GAIN_BP and g_lo > 0, crash > 0 and calm > 0]
    return all(cond), cond, crash, calm


def main():
    from multiprocessing import Pool
    dates = TRAINING_DATES_V4 + VALIDATION_DATES_V4
    n_null = {d: (1 if d in TRAINING_DATES_V4 else N_NULL) for d in dates}
    with Pool(8) as pool:
        rows = pool.map(_day, [(d, n_null[d], i) for i, d in enumerate(dates)])
    data = pd.concat(rows, ignore_index=True)
    train_all = data[data.date.isin(TRAINING_DATES_V4)]
    train, train_null = train_all[train_all.path == -1], train_all[train_all.path == 0]
    val = data[data.date.isin(VALIDATION_DATES_V4)]
    regime = pd.read_csv("rl/results/v4_day_stats.csv").set_index("date").regime
    print(f"rows: training {len(train)}, validation {len(val[val.path == -1])} "
          f"(+{N_NULL} null paths per validation day)")
    print(f"y (bp): training sd {train.y.std():.1f}, validation sd {val[val.path == -1].y.std():.1f}")

    out, rng = [], np.random.default_rng(0)
    for hyp, cols in (("H1 all inputs", ALL), ("H2 levels only", ROUND + LEVELS)):
        for name, model in (("ridge", Ridge(cols)), ("boosting", Boost(cols))):
            model.fit(train)
            cut = float(np.median(model.predict(train)))
            tr = day_scores(train, model.predict(train), cut)
            c = corrected(day_scores(val, model.predict(val), cut))
            ok, cond, crash, calm = passes(c, regime, rng)
            ic, ic_lo, ic_hi = boot(c.ic, rng)
            g, g_lo, g_hi = boot(c.gain, rng)
            print(f"\n{hyp}, {name}")
            print(f"  training IC (in sample, raw): {tr.ic.mean():+.4f}")
            print(f"  validation IC   raw {c.ic_real.mean():+.4f}  null {c.ic_null.mean():+.4f}  "
                  f"corrected {ic:+.4f} [{ic_lo:+.4f}, {ic_hi:+.4f}]")
            print(f"  validation gain raw {c.gain_real.mean():+.2f}  null {c.gain_null.mean():+.2f}  "
                  f"corrected {g:+.2f} bp [{g_lo:+.2f}, {g_hi:+.2f}]")
            print(f"  corrected gain by regime: crash {crash:+.2f}  calm {calm:+.2f}  "
                  f"normal {c.gain[regime.reindex(c.index) == 'normal'].mean():+.2f}")
            verdict = ("PASS" if ok else "FAIL") if name == "ridge" else "reported only"
            print(f"  conditions {cond} -> {verdict}")
            out.append(c.assign(hypothesis=hyp, model=name, regime=regime.reindex(c.index)))
            if hyp.startswith("H1"):
                importance(model, val, cut, regime, name)
            if hyp.startswith("H1") and name == "ridge":
                null_fit = Ridge(cols).fit(train_null)
                coef = pd.DataFrame({"real": model.coefs(), "null": null_fit.coefs()})
                print("  ridge coefficients (bp per sd), training real vs one sign-flipped copy:")
                for f, r in coef.reindex(coef.real.abs().sort_values(ascending=False).index).iterrows():
                    print(f"    {f:<14} {r.real:+7.2f}  {r.null:+7.2f}")
    pd.concat(out).reset_index().to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


def importance(model, val, cut, regime, name):
    """Drop in corrected validation IC when one input group is shuffled across rows."""
    rng = np.random.default_rng(1)
    base = corrected(day_scores(val, model.predict(val), cut)).ic.mean()
    print(f"  permutation importance ({name}, drop in corrected validation IC):")
    for g, cols in GROUPS.items():
        cols = [c for c in cols if c in model.cols]
        if not cols:
            continue
        shuf = val.copy()
        perm = rng.permutation(len(shuf))
        shuf[cols] = shuf[cols].to_numpy()[perm]
        ic = corrected(day_scores(shuf, model.predict(shuf), cut)).ic.mean()
        print(f"    {g:<20} {base - ic:+.4f}")


if __name__ == "__main__":
    main()
