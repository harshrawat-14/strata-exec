"""
Constant paces with the M4 rate-based action space (no learning), on
TRAINING_DATES_V3, protocol v3, 5 starts per date -- the same setting as
doc 18 so AC (-0.33% IS, -0.46% execution) is directly comparable.

    python rl/experiments/rate_constant_paces.py
"""
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/, for bare imports like `import evaluate`

import numpy as np
import pandas as pd

PACES = [0.5, 1.0, 1.5, 2.0, 3.0]


def _run(args):
    date, pace = args
    from environment import StrataExecEnv
    from evaluation.protocol import EPISODE_STEPS, eval_start_fracs
    from evaluation.rate import RATES, RateAgent

    class Fixed:
        def predict(self, obs, state=None, episode_start=None, deterministic=True):
            return np.array([RATES.index(pace)]), state

    env = StrataExecEnv(mode="counterfactual", lob_date=date, agg_date=date, btc_target=50_000.0,
                        n_state_dims=16, fixed_steps=True, fixed_size=True, episode_steps=EPISODE_STEPS)
    rows = []
    try:
        for ep, frac in enumerate(eval_start_fracs(5)):
            agent = RateAgent(Fixed())
            obs, info = env.reset(seed=ep, options={"start_frac": frac})
            agent.set_horizon(info["episode_horizon"])
            arrival, total = info["arrival_price"], info["total_inventory"]
            done, execution = False, 0.0
            while not done:
                a, _ = agent.predict(obs)
                obs, _, done, _, info = env.step(int(a[0]))
                q = info.get("fill_qty", 0.0)
                if q > 0:
                    execution += q * (info["fill_price"] - info["book_mid"])
            rows.append(dict(date=date, pace=pace, IS=-info["total_IS_pct"],
                             execution=100 * execution / (arrival * total),
                             forced=info.get("forced_liquidation_qty", 0.0) / total))
    finally:
        env.close()
    return rows


def main():
    from evaluation.dates import TRAINING_DATES_V3
    with Pool(8) as pool:
        df = pd.DataFrame([r for rs in pool.map(_run, [(d, p) for d in TRAINING_DATES_V3 for p in PACES]) for r in rs])
    df.to_csv("rl/results/rate_constant_paces.csv", index=False)
    print(df.groupby("pace")[["IS", "execution", "forced"]].mean().round(4).to_string())
    print("reference, same setting (doc 18): AC IS -0.3312, execution -0.4605; M0 IS -0.5396, execution -0.6372")


if __name__ == "__main__":
    main()
