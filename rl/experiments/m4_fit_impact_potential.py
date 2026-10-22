"""
Fit k for the M4 impact potential (training/shaping.py) on TRAINING_DATES_V3 only.

For behaviour policies with different impact profiles (AC, AC run ahead of
schedule by 0.1 and 0.3, and M0), record at every step the impact exposure
x = q * (P + T * D(n)) and the realised cost-to-go G (sum of all later env
rewards), then fit  G = c0 + a*q + b*x  by least squares. The q term absorbs
the ordinary cost of selling what is left; b is the extra cost per unit of
impact exposure, so k = -b. Validation and held-out dates are not touched.

Rule (fixed before running): use the fitted k if b < 0 and its t-statistic
exceeds 5; otherwise stop and report, because the potential would not be
measuring a real cost.

    python rl/experiments/m4_fit_impact_potential.py
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/, for bare imports like `import evaluate`

import numpy as np

OUT = "rl/results/m4_impact_potential_fit.json"
POLICIES = ("AC", "AC+0.1", "AC+0.3", "M0")
M0_PATH = "rl/models/checkpoints/m0_v3/ckpt_1199952_steps"


def _collect(args):
    date, who = args
    from environment import StrataExecEnv
    from evaluation.decoding import MeanFractionDecoder
    from evaluation.episode import load_model
    from evaluation.policies.sqrt_impact import query_calibration
    from evaluation.protocol import EPISODE_STEPS, eval_start_fracs
    from evaluation.residual import DELTAS, ResidualACAgent
    from training.shaping import impact_exposure

    cal = query_calibration(date, btc_target=50_000.0)
    rho_dt = cal["rho"] * cal["dt"]

    class Fixed:
        def __init__(self, delta):
            self.idx = DELTAS.index(delta)

        def predict(self, obs, state=None, episode_start=None, deterministic=True):
            return np.array([self.idx]), state

    if who == "M0":
        policy = MeanFractionDecoder(load_model(M0_PATH))
    else:
        policy = ResidualACAgent(Fixed({"AC": 0.0, "AC+0.1": 0.1, "AC+0.3": 0.3}[who]))
    env = StrataExecEnv(mode="counterfactual", lob_date=date, agg_date=date, btc_target=50_000.0,
                        n_state_dims=16, fixed_steps=True, fixed_size=True, episode_steps=EPISODE_STEPS)
    rows = []
    try:
        for ep, frac in enumerate(eval_start_fracs(5)):
            obs, info = env.reset(seed=ep, options={"start_frac": frac})
            if hasattr(policy, "set_horizon"):
                policy.set_horizon(info["episode_horizon"])
            horizon = info["episode_horizon"]
            state, start, done = None, np.ones((1,), bool), False
            qs, xs, rewards = [], [], []
            while not done:
                action, state = policy.predict(obs, state=state, episode_start=start, deterministic=True)
                start = np.zeros((1,), bool)
                obs, r, done, _, info = env.step(int(np.asarray(action).item()))
                rewards.append(r)
                q = float(obs[0])
                qs.append(q)
                xs.append(0.0 if done else impact_exposure(q, info, float(obs[1]) * horizon, rho_dt))
            future = np.cumsum(rewards[::-1])[::-1]
            cost_to_go = np.append(future[1:], 0.0)  # rewards strictly after each step
            for t in range(0, len(rewards), 10):
                rows.append((qs[t], xs[t], cost_to_go[t]))
    finally:
        env.close()
    return rows


def main():
    from evaluation.dates import TRAINING_DATES_V3

    jobs = [(d, w) for d in TRAINING_DATES_V3 for w in POLICIES]
    with Pool(8) as pool:
        data = np.array([r for rs in pool.map(_collect, jobs) for r in rs])
    q, x, g = data[:, 0], data[:, 1], data[:, 2]
    X = np.column_stack([np.ones_like(q), q, x])
    coef, *_ = np.linalg.lstsq(X, g, rcond=None)
    resid = g - X @ coef
    sigma2 = resid @ resid / (len(g) - X.shape[1])
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - (resid @ resid) / ((g - g.mean()) @ (g - g.mean()))
    c0, a, b = coef
    t_b = b / se[2]
    ok = b < 0 and abs(t_b) > 5
    result = dict(n=int(len(g)), c0=float(c0), a=float(a), b=float(b), t_b=float(t_b), r2=float(r2),
                  k=float(-b) if ok else None, accepted=bool(ok),
                  exposure_mean=float(x.mean()), exposure_p99=float(np.percentile(x, 99)))
    Path(OUT).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    print("ACCEPTED" if ok else "REJECTED -- potential does not track a real cost; stop and report")


if __name__ == "__main__":
    main()
