"""
Split implementation shortfall into execution and timing, RL model vs AC, under v3.

  execution = sum fill_qty * (fill_price - contemporaneous mid)   -- what the RL reward measures
  drift     = sum fill_qty * (contemporaneous mid - arrival price) -- timing: price moves while holding
  IS        = execution + drift (+ any forced-dump residual)
both as % of arrival notional. Also reports the episode's summed training reward
and the inventory-weighted average sell time (fraction of the 1,920-step episode).

VALIDATION_DATES_V3, 20 evenly spaced starts, RL decoded by MeanFractionDecoder,
AC at the v3 honest lambda, closed-loop.

    python rl/experiments/decompose_is_v3.py --model rl/models/checkpoints/m0_v3/ckpt_1199952_steps --label M0
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/, for bare imports like `import evaluate`

import numpy as np
import pandas as pd


def _agent(model_path, action_space):
    """RL model decoded the protocol-v3 way for its action space."""
    from evaluation.decoding import MeanFractionDecoder, MeanValueDecoder
    from evaluation.episode import load_model
    from evaluation.rate import RATES, RateAgent
    from evaluation.residual import DELTAS, ResidualACAgent
    model = load_model(model_path)
    if action_space == "residual_ac":
        return ResidualACAgent(MeanValueDecoder(model, DELTAS))
    if action_space == "rate":
        return RateAgent(MeanValueDecoder(model, RATES))
    return MeanFractionDecoder(model)


def _run(args):
    date, who, model_path, action_space = args
    from environment import StrataExecEnv
    from evaluation.policies.almgren_chriss import ACOptimalPolicy
    from evaluation.protocol import AC_LAMBDA, EPISODE_STEPS, eval_start_fracs

    env = StrataExecEnv(mode="counterfactual", lob_date=date, agg_date=date, btc_target=50_000.0,
                        n_state_dims=16, fixed_steps=True, fixed_size=True, episode_steps=EPISODE_STEPS)
    policy = (ACOptimalPolicy(16, lam=AC_LAMBDA, track_schedule=True) if who == "AC"
              else _agent(model_path, action_space))
    rows = []
    try:
        for ep, frac in enumerate(eval_start_fracs(20)):
            obs, info = env.reset(seed=ep, options={"start_frac": frac})
            arrival, total = info["arrival_price"], info["total_inventory"]
            if hasattr(policy, "set_horizon"):
                policy.set_horizon(info["episode_horizon"])
            state, start, done, step = None, np.ones((1,), bool), False, 0
            execution = drift = reward = weighted_t = sold = 0.0
            while not done:
                action, state = policy.predict(obs, state=state, episode_start=start, deterministic=True)
                start = np.zeros((1,), bool)
                obs, r, done, _, info = env.step(int(np.asarray(action).item()))
                step += 1
                reward += r
                q = info.get("fill_qty", 0.0)
                if q > 0:
                    execution += q * (info["fill_price"] - info["book_mid"])
                    drift += q * (info["book_mid"] - arrival)
                    weighted_t += q * step / EPISODE_STEPS
                    sold += q
            rows.append(dict(date=date, who=who, IS=-info["total_IS_pct"],
                             execution=100 * execution / (arrival * total),
                             drift=100 * drift / (arrival * total), reward=reward,
                             avg_sell_time=weighted_t / sold if sold else 1.0,
                             forced=info.get("forced_liquidation_qty", 0.0) / total))
    finally:
        env.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--action-space", choices=["raw", "residual_ac", "rate"], default="raw")
    args = ap.parse_args()
    from evaluation.dates import VALIDATION_DATES_V3

    jobs = [(d, w, args.model, args.action_space) for d in VALIDATION_DATES_V3 for w in (args.label, "AC")]
    with Pool(8) as pool:
        df = pd.DataFrame([r for rs in pool.map(_run, jobs) for r in rs])
    out = f"rl/results/{args.label.lower()}_v3_is_decomposition.csv"
    df.to_csv(out, index=False)
    cols = ["IS", "execution", "drift", "reward", "avg_sell_time", "forced"]
    g = df.groupby(["date", "who"])[cols].mean().unstack("who")
    print(g.round(4).to_string())
    diff = pd.DataFrame({c: g[(c, args.label)] - g[(c, "AC")] for c in cols[:4]})
    print(f"\n{args.label} - AC:")
    print(diff.round(4).to_string())
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
