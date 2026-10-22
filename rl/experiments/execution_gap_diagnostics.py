"""
Why does RL execute worse than AC? Three diagnostics (docs/experiments/18-*),
TRAINING_DATES_V3 only, protocol v3, 5 evenly spaced starts per date.

  A. Policies side by side -- AC (closed-loop), M0, and constant actions
     1-4 (sell a fixed 0.1/0.3/0.5/0.8% of remaining every step): IS,
     execution, summed training reward, average sell time, and how AC uses
     the action table (actions chosen, switches per episode).
  B. M0's value function vs what actually happened: V(s_t) against the
     realised discounted return (gamma 0.99, the training objective), by
     remaining-inventory bucket. A systematic bias would mean the agent
     misjudges the cost of holding inventory.
  C. Per-trade cost vs trade size: slippage per unit (bp of mid) by fill
     size, over every fill from A. Strong convexity means large trades are
     expensive immediately -- a cost RL does observe.

    python rl/experiments/execution_gap_diagnostics.py
"""
import collections
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/, for bare imports like `import evaluate`

import numpy as np
import pandas as pd

M0_PATH = "rl/models/checkpoints/m0_v3/ckpt_1199952_steps"
POLICIES = ["AC", "M0", "const1", "const2", "const3", "const4"]
GAMMA = 0.99
OUT = "rl/results/execution_gap_diagnostics"


class _ConstantAction:
    def __init__(self, action):
        self.action = action

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        return np.array([self.action]), state


class _Critic:
    """Steps M0's recurrent critic alongside the episode to read V(s_t)."""

    def __init__(self, model):
        import torch as th
        from stable_baselines3.common.policies import ActorCriticPolicy
        self.th, self.ACP, self.policy = th, ActorCriticPolicy, model.policy
        self.reset()

    def reset(self):
        z = self.th.zeros(self.policy.lstm_hidden_state_shape)
        self.states, self.start = (z.clone(), z.clone()), True

    def value(self, obs):
        th, p = self.th, self.policy
        with th.no_grad():
            obs_t, _ = p.obs_to_tensor(obs)
            feats = super(self.ACP, p).extract_features(obs_t, p.vf_features_extractor)
            starts = th.tensor([1.0 if self.start else 0.0])
            latent, self.states = p._process_sequence(feats, self.states, starts, p.lstm_critic)
            self.start = False
            return float(p.value_net(p.mlp_extractor.forward_critic(latent)).item())


def _run(args):
    date, who = args
    from environment import StrataExecEnv
    from evaluation.decoding import MeanFractionDecoder
    from evaluation.episode import load_model
    from evaluation.policies.almgren_chriss import ACOptimalPolicy
    from evaluation.protocol import AC_LAMBDA, EPISODE_STEPS, eval_start_fracs

    critic = None
    if who == "AC":
        policy = ACOptimalPolicy(16, lam=AC_LAMBDA, track_schedule=True)
    elif who == "M0":
        model = load_model(M0_PATH)
        policy, critic = MeanFractionDecoder(model), _Critic(model)
    else:
        policy = _ConstantAction(int(who[-1]))
    env = StrataExecEnv(mode="counterfactual", lob_date=date, agg_date=date, btc_target=50_000.0,
                        n_state_dims=16, fixed_steps=True, fixed_size=True, episode_steps=EPISODE_STEPS)
    episodes, fills, values = [], [], []
    try:
        for ep, frac in enumerate(eval_start_fracs(5)):
            obs, info = env.reset(seed=ep, options={"start_frac": frac})
            arrival, total = info["arrival_price"], info["total_inventory"]
            if hasattr(policy, "set_horizon"):
                policy.set_horizon(info["episode_horizon"])
            if critic:
                critic.reset()
            state, start, done, step = None, np.ones((1,), bool), False, 0
            execution = reward = weighted_t = sold = 0.0
            actions, rewards, v_trace, q_trace = [], [], [], []
            while not done:
                if critic:
                    v_trace.append(critic.value(obs))
                    q_trace.append(float(obs[0]))
                a, state = policy.predict(obs, state=state, episode_start=start, deterministic=True)
                start = np.zeros((1,), bool)
                a = int(np.asarray(a).item())
                actions.append(a)
                obs, r, done, _, info = env.step(a)
                step += 1
                reward += r
                rewards.append(r)
                q = info.get("fill_qty", 0.0)
                if q > 0:
                    slip_bp = 1e4 * (info["fill_price"] - info["book_mid"]) / info["book_mid"]
                    fills.append((who, q / total, slip_bp))
                    execution += q * (info["fill_price"] - info["book_mid"])
                    weighted_t += q * step / EPISODE_STEPS
                    sold += q
            if critic:
                g, ret = 0.0, []
                for r in reversed(rewards):
                    g = r + GAMMA * g
                    ret.append(g)
                ret = ret[::-1]
                values += [(q_trace[t], v_trace[t], ret[t]) for t in range(0, len(ret), 5)]
            counts = collections.Counter(actions)
            episodes.append(dict(
                date=date, policy=who, IS=-info["total_IS_pct"],
                execution=100 * execution / (arrival * total), reward=reward,
                avg_sell_time=weighted_t / sold if sold else 1.0,
                forced=info.get("forced_liquidation_qty", 0.0) / total,
                switches=sum(1 for i in range(1, len(actions)) if actions[i] != actions[i - 1]),
                top_actions=json.dumps(counts.most_common(4))))
    finally:
        env.close()
    return episodes, fills, values


def main():
    from evaluation.dates import TRAINING_DATES_V3

    jobs = [(d, w) for d in TRAINING_DATES_V3 for w in POLICIES]
    with Pool(8) as pool:
        out = pool.map(_run, jobs)
    eps = pd.DataFrame([e for o in out for e in o[0]])
    fills = pd.DataFrame([f for o in out for f in o[1]], columns=["policy", "size", "slip_bp"])
    vals = pd.DataFrame([v for o in out for v in o[2]], columns=["q", "V", "G"])
    eps.to_csv(f"{OUT}_episodes.csv", index=False)

    print("A. Policies on the 12 training dates (means over dates x 5 starts)")
    print(eps.groupby("policy")[["IS", "execution", "reward", "avg_sell_time", "forced", "switches"]]
          .mean().reindex(POLICIES).round(4).to_string())
    print("\n   AC's most used actions (first episode on each of 3 dates):")
    for _, r in eps[eps.policy == "AC"].groupby("date").head(1).head(3).iterrows():
        print(f"   {r.date}: {r.top_actions}")

    print("\nB. M0 value function vs realised discounted return (gamma 0.99), by remaining inventory")
    vals["bucket"] = pd.cut(vals.q, [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0001])
    b = vals.groupby("bucket", observed=True).agg(V=("V", "mean"), G=("G", "mean"), n=("V", "size"))
    b["V_minus_G"] = b.V - b.G
    print(b.round(5).to_string())
    print(f"   corr(V, G) = {vals.V.corr(vals.G):.3f};  mean bias V-G = {(vals.V - vals.G).mean():+.5f}")

    print("\nC. Slippage per unit vs fill size (bp below mid; all fills from A)")
    fills["size_bin"] = pd.cut(fills["size"], [0, 0.0005, 0.001, 0.003, 0.01, 0.03, 1.0])
    c = fills.groupby("size_bin", observed=True).agg(slip_bp=("slip_bp", "mean"), n=("slip_bp", "size"))
    print(c.round(2).to_string())
    print("   fill-size share by policy (fraction of fills > 0.3% of the order):")
    print("   " + str(fills.assign(big=fills["size"] > 0.003).groupby("policy").big.mean().round(3).to_dict()))
    vals.to_csv(f"{OUT}_values.csv", index=False)
    c.to_csv(f"{OUT}_slippage_by_size.csv")


if __name__ == "__main__":
    main()
