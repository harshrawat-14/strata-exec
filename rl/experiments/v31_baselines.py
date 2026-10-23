"""
Protocol v3.1 baselines: orders sized as a share of 20-day trailing ADV.

For each size (0.5%, 1%, 2% of ADV):
  - re-select AC's honest lambda on TRAINING_DATES_V3 only (best mean IS),
  - score AC (closed-loop) and constant paces 1x / 1.5x / 2x (rate menu)
    on training dates (5 starts) and validation dates (20 starts).
No RL. Held-out dates untouched.

    python rl/experiments/v31_baselines.py
"""
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/, for bare imports like `import evaluate`

import numpy as np
import pandas as pd

SIZES = [0.005, 0.01, 0.02]
LAMBDAS = [1e-9, 1e-7, 1e-6, 3e-6, 1e-5, 1e-4, 1e-3]
PACES = [1.0, 1.5, 2.0]
OUT = "rl/results/v31_baselines.csv"


def _run(args):
    size, date, split, policy_name = args
    from environment import StrataExecEnv
    from evaluation.policies.almgren_chriss import ACOptimalPolicy
    from evaluation.protocol import EPISODE_STEPS, eval_start_fracs
    from evaluation.rate import RATES, RateAgent

    class Fixed:
        def __init__(self, pace):
            self.i = RATES.index(pace)

        def predict(self, obs, state=None, episode_start=None, deterministic=True):
            return np.array([self.i]), state

    kind, value = policy_name.split("=")
    env = StrataExecEnv(mode="counterfactual", lob_date=date, agg_date=date, n_state_dims=16,
                        fixed_steps=True, fixed_size=True, episode_steps=EPISODE_STEPS,
                        order_share_adv=size)
    rows = []
    try:
        for ep, frac in enumerate(eval_start_fracs(5 if split == "train" else 20)):
            policy = (ACOptimalPolicy(16, lam=float(value), track_schedule=True) if kind == "AC"
                      else RateAgent(Fixed(float(value))))
            obs, info = env.reset(seed=ep, options={"start_frac": frac})
            policy.set_horizon(info["episode_horizon"])
            arrival, total = info["arrival_price"], info["total_inventory"]
            done, execution = False, 0.0
            while not done:
                a, _ = policy.predict(obs)
                obs, _, done, _, info = env.step(int(np.asarray(a).item()))
                q = info.get("fill_qty", 0.0)
                if q > 0:
                    execution += q * (info["fill_price"] - info["book_mid"])
            rows.append(dict(size=size, date=date, split=split, policy=policy_name,
                             IS=-info["total_IS_pct"], execution=100 * execution / (arrival * total),
                             forced=info.get("forced_liquidation_qty", 0.0) / total))
    finally:
        env.close()
    return rows


def main():
    from evaluation.dates import SEVERITY_RANK_V2, TRAINING_DATES_V3, VALIDATION_DATES_V3
    policies = [f"AC={l}" for l in LAMBDAS] + [f"pace={p}" for p in PACES]
    jobs = [(s, d, "train", p) for s in SIZES for d in TRAINING_DATES_V3 for p in policies]
    jobs += [(s, d, "validation", p) for s in SIZES for d in VALIDATION_DATES_V3 for p in policies]
    with Pool(8) as pool:
        df = pd.DataFrame([r for rs in pool.map(_run, jobs, chunksize=2) for r in rs])
    df["regime"] = df.date.map(lambda d: SEVERITY_RANK_V2[d][1])
    df.to_csv(OUT, index=False)

    for size in SIZES:
        sub = df[df["size"] == size]
        train = sub[sub.split == "train"].groupby("policy").IS.mean()
        ac_train = train[[p for p in train.index if p.startswith("AC=")]]
        best = ac_train.idxmax()
        print(f"\n=== order = {size:.1%} of ADV | honest AC lambda (training dates): {best[3:]} ===")
        keep = [best] + [f"pace={p}" for p in PACES]
        tab = (sub[sub.policy.isin(keep)].groupby(["split", "policy"])[["IS", "execution", "forced"]]
               .mean().round(4))
        print(tab.to_string())
        val = sub[(sub.split == "validation") & sub.policy.isin(keep)]
        print(val.pivot_table(index=["date", "regime"], columns="policy", values="IS").round(4).to_string())
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
