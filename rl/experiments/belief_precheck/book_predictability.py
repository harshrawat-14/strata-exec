"""
Doc 27 pre-check: do real-time book/trade features predict the next 1-2 h?

Fits on the 12 training dates and tests on the 4 validation dates, then
leave-one-date-out over the training dates. Held-out dates are never read.

    python rl/experiments/belief_precheck/book_predictability.py
"""
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # rl/

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from book_features import build, feature_columns
from evaluation.dates import TRAINING_DATES_V3, VALIDATION_DATES_V3

# Doc-26 state equivalents: short flow, short return, volatility ratio.
REFERENCE = ["flow_2", "ret_z_10", "vol_ratio_60_240"]
FALL = -0.5  # pp over the next 2 h


def _models(kind):
    if kind == "reg":
        return {"ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
                "boost": HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200)}
    return {"logit": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)),
            "boost": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200)}


def _score(train, test, cols, target):
    """Per-model metrics on `test`: pooled R^2 vs 0 / AUC, and per-date values."""
    out = {}
    if target == "fall_240":
        for name, m in _models("cls").items():
            m.fit(train[cols], train[target])
            p = m.predict_proba(test[cols])[:, 1]
            per = {d: roc_auc_score(g[target], p[test.date.values == d])
                   for d, g in test.groupby("date") if g[target].nunique() == 2}
            out[name] = (roc_auc_score(test[target], p), per)
    else:
        for name, m in _models("reg").items():
            m.fit(train[cols], train[target])
            p = m.predict(test[cols])
            y = test[target].to_numpy()
            r2 = 1 - ((y - p) ** 2).sum() / (y ** 2).sum()
            per = {d: np.corrcoef(p[test.date.values == d], y[test.date.values == d])[0, 1]
                   for d in test.date.unique()}
            out[name] = (r2, per)
    return out


def main():
    dates = list(TRAINING_DATES_V3) + list(VALIDATION_DATES_V3)
    with Pool(8) as pool:
        df = pd.concat(pool.map(build, dates), ignore_index=True)
    df["fall_240"] = (df.fwd_240 < FALL).astype(int)
    df = df.dropna(subset=["fwd_120", "fwd_240"]).reset_index(drop=True)
    every = feature_columns(df)
    assert not any(c.startswith(("fwd_", "fall_")) for c in every), "target leaked into features"
    sets = {"reference (doc-26 equiv.)": REFERENCE, "all book+trade+price": every}

    train = df[df.date.isin(TRAINING_DATES_V3)]
    val = df[df.date.isin(VALIDATION_DATES_V3)]
    print(f"rows: train {len(train)}, validation {len(val)}; "
          f"fall_240 base rate train {train.fall_240.mean():.3f}, val {val.fall_240.mean():.3f}")

    rows = []
    for target in ("fwd_120", "fwd_240", "fall_240"):
        for set_name, cols in sets.items():
            res = _score(train, val, cols, target)
            for model, (pooled, per) in res.items():
                rows.append(dict(split="validation", target=target, features=set_name, model=model,
                                 pooled=pooled, **{d: v for d, v in per.items()}))
            # leave one training date out
            lodo = {}
            for d in TRAINING_DATES_V3:
                r = _score(train[train.date != d], train[train.date == d], cols, target)
                for model, (pooled, per) in r.items():
                    lodo.setdefault(model, {})[d] = per.get(d, np.nan)
            for model, per in lodo.items():
                rows.append(dict(split="lodo", target=target, features=set_name, model=model,
                                 pooled=np.nanmean(list(per.values())), **per))
    res = pd.DataFrame(rows)
    res.to_csv("rl/results/book_regime_features_precheck.csv", index=False)

    vd = list(VALIDATION_DATES_V3)
    for target in ("fwd_120", "fwd_240", "fall_240"):
        metric = "AUC" if target == "fall_240" else "R^2 vs 0"
        per_metric = "AUC" if target == "fall_240" else "corr"
        good = 0.55 if target == "fall_240" else 0.0
        print(f"\n== {target}  (pooled = {metric}; per-date = {per_metric})")
        for _, r in res[res.target == target].iterrows():
            ds = vd if r.split == "validation" else list(TRAINING_DATES_V3)
            vals = [r[d] for d in ds]
            n_good = sum(v > good for v in vals if pd.notna(v))
            print(f"  {r.split:<10} {r.features:<26} {r.model:<6} pooled {r.pooled:+.3f}  "
                  f"dates > {good}: {n_good}/{sum(pd.notna(v) for v in vals)}  "
                  + " ".join(f"{v:+.2f}" for v in vals))

    # univariate sign stability on the training dates, 2 h return
    print("\nunivariate corr with fwd_240: mean over training dates, and dates with the same sign")
    stab = []
    for c in every:
        cs = [np.corrcoef(g[c], g.fwd_240)[0, 1] for _, g in train.groupby("date")]
        m = np.nanmean(cs)
        stab.append((c, m, sum(np.sign(x) == np.sign(m) for x in cs)))
    for c, m, n in sorted(stab, key=lambda t: -abs(t[1]))[:10]:
        print(f"  {c:<18} {m:+.3f}  {n}/12")


if __name__ == "__main__":
    main()
