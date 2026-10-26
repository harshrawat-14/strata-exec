"""
Era-2 baselines on the real Bybit book with price-moving impact (doc 21).

Rules, fixed before running:
  * Order sizes: 1%, 5% and 10% of the date's 20-day trailing Bybit ADV.
  * Protocol v3: 1,920-step episodes, 20 evenly spaced starts per date.
  * AC lambda: ONE value per order size, the one maximising AC-Optimal's mean
    IS over TRAINING_DATES_V3 only. Validation dates are evaluated at that
    lambda for later comparison; held-out dates are never touched.
  * Constant paces (rate actions, no learning): 0.5x-3x the TWAP pace;
    pace 1.0 is TWAP.

    python rl/experiments/era2_baselines.py
"""
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/

import numpy as np
import pandas as pd

SHARES = [0.01, 0.05, 0.10]
LAMBDA_GRID = sorted([1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5,
                      1e-4, 3e-4, 1e-3, 1e-2, 1e-1, 1.0])
PACES = [0.5, 1.0, 1.5, 2.0, 3.0]
OUT = "rl/results/era2_baselines.csv"


def _episode(env, agent, seed, frac):
    obs, info = env.reset(seed=seed, options={"start_frac": frac})
    agent.set_horizon(info["episode_horizon"])
    arrival, total = info["arrival_price"], info["total_inventory"]
    done, execution, steps = False, 0.0, 0
    state = None
    while not done:
        action, state = agent.predict(obs, state=state, deterministic=True)
        obs, _, done, _, info = env.step(int(np.asarray(action).item()))
        steps += 1
        q = info.get("fill_qty", 0.0)
        if q > 0:
            execution += q * (info["fill_price"] - info["book_mid"])
    return dict(IS=-info["total_IS_pct"], execution=100 * execution / (arrival * total),
                forced=info.get("forced_liquidation_qty", 0.0) / total, steps=steps,
                order_btc=info.get("btc_scale", 0.0) * total)


def _run(args):
    date, share = args
    from environment import StrataExecEnv
    from evaluation.dates import SEVERITY_RANK_V2
    from evaluation.policies.almgren_chriss import ACOptimalPolicy
    from evaluation.protocol import EPISODE_STEPS, N_EVAL_EPISODES, eval_start_fracs
    from evaluation.rate import RATES, RateAgent

    class Fixed:
        def __init__(self, pace):
            self.a = RATES.index(pace)

        def predict(self, obs, state=None, episode_start=None, deterministic=True):
            return np.array([self.a]), state

    env = StrataExecEnv(mode="counterfactual", lob_date=date, agg_date=date, order_share_adv=share,
                        n_state_dims=16, fixed_steps=True, fixed_size=True, episode_steps=EPISODE_STEPS)
    fracs = eval_start_fracs(N_EVAL_EPISODES)
    agents = [("AC", lam, lambda lam=lam: ACOptimalPolicy(16, lam=lam, track_schedule=True))
              for lam in LAMBDA_GRID]
    agents += [("pace", p, lambda p=p: RateAgent(Fixed(p))) for p in PACES]
    rows = []
    try:
        for kind, param, make in agents:
            for ep, frac in enumerate(fracs):
                r = _episode(env, make(), ep, frac)
                rows.append(dict(date=date, regime=SEVERITY_RANK_V2[date][1], share=share,
                                 kind=kind, param=param, episode=ep, **r))
    finally:
        env.close()
    print(f"  {date} {share:.0%} done", flush=True)
    return rows


def main():
    from evaluation.dates import TRAINING_DATES_V3, VALIDATION_DATES_V3

    dates = list(TRAINING_DATES_V3) + list(VALIDATION_DATES_V3)
    with Pool(8) as pool:
        df = pd.DataFrame([r for rs in pool.map(_run, [(d, s) for s in SHARES for d in dates]) for r in rs])
    df["split"] = np.where(df.date.isin(TRAINING_DATES_V3), "train", "validation")
    df.to_csv(OUT, index=False)

    per_date = df.groupby(["share", "split", "date", "regime", "kind", "param"], as_index=False)[
        ["IS", "execution", "forced"]].mean()
    for share in SHARES:
        d = per_date[per_date.share == share]
        train_ac = d[(d.split == "train") & (d.kind == "AC")].groupby("param").IS.mean()
        lam = train_ac.idxmax()
        print(f"\n=== order {share:.0%} of ADV: selected lambda {lam:.0e} "
              f"(train mean IS {train_ac.max():+.4f}%; lambda=1e-9 {train_ac.iloc[0]:+.4f}%) ===")
        pick = d[((d.kind == "AC") & (d.param == lam)) | (d.kind == "pace")].copy()
        pick["strategy"] = np.where(pick.kind == "AC", "AC", "pace " + pick.param.astype(str))
        print(pick.pivot_table(index="strategy", columns="split", values=["IS", "forced"], aggfunc="mean")
              .round(4).to_string())
        print(pick[pick.split == "validation"].pivot_table(index="date", columns="strategy", values="IS")
              .round(3).to_string())
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
