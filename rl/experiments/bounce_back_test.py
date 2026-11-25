"""
Doc 32: after a drawdown, does the price bounce back within 2 h? Single pre-registered
feature (drawdown_z, doc 27) and horizon (240 steps), on the V4 pool.

A drawdown from the running high moves with the price, so a within-day predictive
correlation with forward returns is biased negative even for a random walk (Stambaugh
bias), with the same sign every day. Every statistic is therefore compared with a null:
the same day's 30 s returns shuffled N_NULL times (same return distribution, no
predictability). Held-out and future dates are not read.

    python rl/experiments/bounce_back_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/

import numpy as np
import pandas as pd

from evaluation.dates import TRAINING_DATES_V4, VALIDATION_DATES_V4
from market_data import book_path

sys.path.insert(0, str(Path(__file__).resolve().parent / "belief_precheck"))
from book_features import drawdown_z  # noqa: E402

H = 240  # forward horizon, 2 h of 30 s steps
WARMUP = 240  # trailing-vol window must be full
N_NULL = 20
N_BOOT = 1000
DEEP_Q = 10  # deep drawdown: at or below the training rows' 10th percentile
MIN_EXCESS_BP = 5.0
OUT = "rl/results/bounce_back_test.csv"


def rows(mid):
    """(drawdown_z, forward 2 h return in bp) for every usable row of one day."""
    dz = drawdown_z(mid)
    fwd = (mid[H:] / mid[:-H] - 1) * 1e4
    k = np.arange(WARMUP, len(mid) - H)
    return dz[k], fwd[k]


def shuffled(mid, rng):
    r = np.diff(np.log(mid))
    return mid[0] * np.exp(np.concatenate([[0.0], np.cumsum(rng.permutation(r))]))


def day_stats(mid, cut, rng, n_null=N_NULL):
    """Per-day correlation and deep-drawdown sums, real and averaged over shuffled nulls."""
    def one(m):
        dz, fwd = rows(m)
        deep = dz <= cut
        c = np.corrcoef(dz, fwd)[0, 1] if dz.std() > 0 else 0.0
        return c, fwd[deep].sum(), deep.sum(), fwd.sum(), len(fwd)
    real = one(mid)
    null = np.mean([one(shuffled(mid, rng)) for _ in range(n_null)], axis=0)
    return real, null


def summarise(df):
    """Bias-corrected statistics for a set of days (rows of per-day results)."""
    corr = (df.corr_real - df.corr_null).mean()
    def excess(deep_sum, deep_n, all_sum, all_n):
        return deep_sum.sum() / max(deep_n.sum(), 1) - all_sum.sum() / all_n.sum()
    ex_real = excess(df.deep_sum_real, df.deep_n_real, df.all_sum_real, df.all_n_real)
    ex_null = excess(df.deep_sum_null, df.deep_n_null, df.all_sum_null, df.all_n_null)
    return dict(corr_raw=df.corr_real.mean(), corr_null=df.corr_null.mean(), corr=corr,
                excess_raw=ex_real, excess_null=ex_null, excess=ex_real - ex_null)


def boot(df, rng, n=N_BOOT):
    s = pd.DataFrame([summarise(df.iloc[rng.integers(0, len(df), len(df))]) for _ in range(n)])
    return s.quantile(0.05), s.quantile(0.95)


def load_mid(date):
    b = pd.read_csv(book_path(date), usecols=["b0_p", "a0_p"])
    return ((b.b0_p + b.a0_p) / 2).to_numpy()


def _day(args):
    date, cut, seed = args
    real, null = day_stats(load_mid(date), cut, np.random.default_rng(seed))
    keys = ("corr", "deep_sum", "deep_n", "all_sum", "all_n")
    return dict(date=date, **{f"{k}_real": v for k, v in zip(keys, real)},
                **{f"{k}_null": v for k, v in zip(keys, null)})


def main():
    from multiprocessing import Pool
    with Pool(8) as pool:
        mids = dict(zip(TRAINING_DATES_V4, pool.map(load_mid, TRAINING_DATES_V4)))
        cut = np.percentile(np.concatenate([rows(m)[0] for m in mids.values()]), DEEP_Q)
        dates = TRAINING_DATES_V4 + VALIDATION_DATES_V4
        res = pd.DataFrame(pool.map(_day, [(d, cut, i) for i, d in enumerate(dates)]))
    labels = pd.read_csv("rl/results/v4_day_stats.csv").set_index("date").regime
    res["regime"] = res.date.map(labels)
    res["split"] = np.where(res.date.isin(TRAINING_DATES_V4), "train", "validation")
    res.to_csv(OUT, index=False)

    rng = np.random.default_rng(0)
    print(f"deep-drawdown cut (training q{DEEP_Q}): drawdown_z <= {cut:.3f}")
    out = {}
    for name, sub in (("train", res[res.split == "train"]), ("validation", res[res.split == "validation"]),
                      ("crash days (all)", res[res.regime == "crash"])):
        s, (lo, hi) = summarise(sub), boot(sub, rng)
        out[name] = (s, lo, hi)
        print(f"\n{name} ({len(sub)} days)")
        print(f"  per-day corr: raw {s['corr_raw']:+.4f}  null {s['corr_null']:+.4f}  "
              f"corrected {s['corr']:+.4f}  [{lo['corr']:+.4f}, {hi['corr']:+.4f}]")
        print(f"  deep-drawdown excess 2 h return (bp): raw {s['excess_raw']:+.2f}  null {s['excess_null']:+.2f}  "
              f"corrected {s['excess']:+.2f}  [{lo['excess']:+.2f}, {hi['excess']:+.2f}]")

    (st, lt, ht), (sv, lv, hv), (sc, _, _) = out["train"], out["validation"], out["crash days (all)"]
    c1 = ht["corr"] < 0
    c2 = hv["corr"] < 0
    c3 = st["excess"] >= MIN_EXCESS_BP and sv["excess"] >= MIN_EXCESS_BP and lt["excess"] > 0
    c4 = sc["corr"] <= 0
    print(f"\n1. training corrected corr < 0, interval excludes 0: {c1}")
    print(f"2. validation corrected corr < 0, interval excludes 0: {c2}")
    print(f"3. corrected deep excess >= {MIN_EXCESS_BP} bp (train and validation), train interval > 0: {c3}")
    print(f"4. crash days corrected corr <= 0: {c4}")
    print(f"PASS: {c1 and c2 and c3 and c4}")


if __name__ == "__main__":
    main()
