"""
Doc 35: re-score M0-e2b (doc 23) on the recalibrated simulator (doc 33), against TWAP and
constant paces on the same starts and order sizes (5% of ADV, protocol v3).

Two date sets:
  v3     -- the 16 V3 training + validation dates doc 23 scored (continuity);
  oos    -- the V4 validation days (Oct-Dec 2024) minus the V3 dates M0-e2b was trained or
            selected on: out of sample for this model. Regimes are doc 30's drift labels.
Honest-lambda AC is TWAP on this simulator (doc 33, lambda -> 0), so TWAP stands for both.
Held-out dates are not read.

    python rl/experiments/m0_rescore.py                     # doc 35: M0-e2b
    python rl/experiments/m0_rescore.py --model rl/models/checkpoints/m0_e2c/ckpt_N_steps \
        --name M0-e2c --out rl/results/m0_e2c_rescore_doc36.csv   # doc 36: retrain
"""
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from evaluation.dates import TRAINING_DATES_V3, VALIDATION_DATES_V3, VALIDATION_DATES_V4

MODEL = "rl/models/checkpoints/m0_e2b/ckpt_1499940_steps"  # doc 23's selected checkpoint
SHARE = 0.05
PACES = [1.0, 1.5, 2.0, 3.0, 5.0]
OUT = "rl/results/m0_e2b_rescore_doc35.csv"
V3 = list(TRAINING_DATES_V3) + list(VALIDATION_DATES_V3)
OOS = [d for d in VALIDATION_DATES_V4 if d not in set(V3)]


def _paces(date):
    import era2_baselines as eb
    from environment import StrataExecEnv
    from evaluation.protocol import EPISODE_STEPS, N_EVAL_EPISODES, eval_start_fracs
    from evaluation.rate import RATES, RateAgent

    class Fixed:
        def __init__(self, pace):
            self.a = RATES.index(pace)

        def predict(self, obs, state=None, episode_start=None, deterministic=True):
            return np.array([self.a]), state

    env = StrataExecEnv(mode="counterfactual", lob_date=date, agg_date=date, order_share_adv=SHARE,
                        n_state_dims=16, fixed_steps=True, fixed_size=True, episode_steps=EPISODE_STEPS)
    rows = []
    try:
        for p in PACES:
            for ep, f in enumerate(eval_start_fracs(N_EVAL_EPISODES)):
                r = eb._episode(env, RateAgent(Fixed(p)), ep, f)
                rows.append(dict(date=date, strategy=f"pace {p}", episode=ep, IS=r["IS"],
                                 forced=r["forced"], steps=r["steps"]))
    finally:
        env.close()
    return rows


def _model(date, model=MODEL, name="M0-e2b"):
    from select_checkpoint_v3 import _evaluate
    return [dict(date=r["date"], strategy=name, episode=i, IS=r["IS"], forced=float(r["forced"]),
                 steps=r["steps"])
            for i, r in enumerate(_evaluate((name, model, date, "raw", SHARE)))]


def _job(args):
    kind, date, model, name = args
    rows = _model(date, model, name) if kind == "model" else _paces(date)
    print(f"  {date} {kind} done", flush=True)
    return rows


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--name", default="M0-e2b")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    dates = V3 + OOS
    with Pool(8) as pool:
        rows = pool.map(_job, [(k, d, args.model, args.name) for d in dates for k in ("model", "paces")],
                        chunksize=1)
    df = pd.DataFrame([r for rs in rows for r in rs])
    df["set"] = np.where(df.date.isin(V3), "v3", "oos")
    df["regime"] = df.date.map(pd.read_csv("rl/results/v4_day_stats.csv").set_index("date").regime)
    df.to_csv(args.out, index=False)
    report(df, args.name)


def report(df, name="M0-e2b"):
    df = df.assign(forced=df.forced.map(lambda v: float(v == "True") if isinstance(v, str) else float(v)))
    per = df.pivot_table(index=["set", "regime", "date"], columns="strategy", values="IS").reset_index()
    per["vs TWAP"] = per[name] - per["pace 1.0"]
    for s in ("v3", "oos"):
        p = per[per.set == s]
        print(f"\n=== {s} ({len(p)} dates), mean IS % by regime (doc 30 drift labels)")
        cols = [name] + [f"pace {x}" for x in PACES] + ["vs TWAP"]
        t = p.groupby("regime")[cols].mean()
        t["n"] = p.groupby("regime").size()
        t["model beats TWAP (days)"] = p.groupby("regime")["vs TWAP"].apply(lambda x: f"{(x > 0).sum()}/{len(x)}")
        print(t.round(3).to_string())
        print(f"  all: {name} {p[name].mean():+.3f}  TWAP {p['pace 1.0'].mean():+.3f}  "
              f"vs TWAP {p['vs TWAP'].mean():+.3f}")
        rng = np.random.default_rng(0)
        for reg, g in p.groupby("regime"):
            v = g["vs TWAP"].to_numpy()
            bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
            lo, hi = np.quantile(bs, [0.05, 0.95])
            print(f"  {reg:>6}: model - TWAP {v.mean():+.3f}  90% [{lo:+.3f}, {hi:+.3f}]")
    m = df[df.strategy == name]
    print(f"\n{name} forced-liquidation share: max {m.forced.max():.4f}; median steps to finish "
          f"{m.steps.median():.0f} (TWAP 1,918)")


if __name__ == "__main__":
    main()
