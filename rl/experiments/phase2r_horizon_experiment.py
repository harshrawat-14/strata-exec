"""
Phase 2R credit-assignment test (cheap, discriminating).

Hypothesis: the wait-collapse is a long-horizon CREDIT-ASSIGNMENT failure. With
gamma=0.99 and training episodes up to 1440 steps, the terminal forced-liquidation
penalty is discounted to near-invisibility from early wait decisions
(0.99^800 ~ 3e-4), so the agent never attributes it back and collapses onto the
do-nothing action.

Prediction: holding EVERYTHING else identical, a SHORT fixed horizon (terminal
penalty temporally close, 0.99^60 ~ 0.55) should let the agent learn to EXECUTE,
while a LONG fixed horizon should still collapse toward WAIT.

Falsifiable: if the short-horizon arm ALSO collapses to wait, the hypothesis is
wrong (the cause is not horizon/credit-assignment). Two independent arms, same
seed, same model config, same dates -- only the horizon differs.

Run: rl/.venv/bin/python rl/experiments/phase2r_horizon_experiment.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/, for bare imports like `import evaluate`

import sys
import numpy as np
from collections import Counter

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from environment import StrataExecEnv

DATES = ["2024-03-05", "2024-11-06"]   # a crash (exec 17x better) + a rally
N_ENVS = 2
TIMESTEPS = 80_000
N_STATE_DIMS = 16


def make_env_fn(total_steps, seed):
    import random

    def _init():
        date = random.choice(DATES)
        env = StrataExecEnv(
            mode="counterfactual", lob_date=date, agg_date=date, btc_target=50000.0,
            n_state_dims=N_STATE_DIMS, seed=seed, fixed_steps=True, fixed_size=True,
            no_time_features=False, total_steps=total_steps,
        )
        return Monitor(env)
    return _init


def train_arm(total_steps, tag):
    print(f"\n=== ARM {tag}: fixed horizon = {total_steps} steps ===", flush=True)
    venv = SubprocVecEnv([make_env_fn(total_steps, 42 + i) for i in range(N_ENVS)])
    model = RecurrentPPO(
        policy="MlpLstmPolicy", env=venv, device="cpu",
        learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.05,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]),
                           lstm_hidden_size=64, n_lstm_layers=1,
                           activation_fn=__import__("torch").nn.Tanh,
                           enable_critic_lstm=True),
        verbose=0, seed=42,
    )
    model.learn(total_timesteps=TIMESTEPS, progress_bar=False)
    venv.close()
    return model


def eval_wait_fraction(model, total_steps):
    """Deterministic wait-fraction + mean cumulative reward per date."""
    out = {}
    for date in DATES:
        env = StrataExecEnv(mode="counterfactual", lob_date=date, agg_date=date,
                            btc_target=50000.0, n_state_dims=N_STATE_DIMS, seed=0,
                            fixed_steps=True, fixed_size=True, no_time_features=False,
                            total_steps=total_steps)
        actions, rewards = [], []
        for seed in range(6):
            obs, info = env.reset(seed=seed)
            lstm, es, done, tr = None, np.ones((1,), bool), False, 0.0
            while not done:
                a, lstm = model.predict(obs, state=lstm, episode_start=es, deterministic=True)
                es = np.zeros((1,), bool)
                actions.append(int(a))
                obs, r, done, _, info = env.step(int(a))
                tr += r
            rewards.append(tr)
        env.close()
        c = Counter(actions)
        out[date] = (c.get(0, 0) / sum(c.values()), float(np.mean(rewards)))
    return out


def main():
    print("Phase 2R credit-assignment test: short vs long fixed horizon, all else equal.")
    print(f"gamma=0.99 -> terminal penalty discount at horizon: "
          f"0.99^60={0.99**60:.3f}  0.99^800={0.99**800:.2e}")
    results = {}
    for total_steps, tag in [(60, "SHORT"), (800, "LONG")]:
        model = train_arm(total_steps, tag)
        results[tag] = eval_wait_fraction(model, total_steps)

    print("\n" + "=" * 70)
    print("RESULT -- deterministic wait-fraction (and cum reward) after 80k steps")
    print("=" * 70)
    for tag in ("SHORT", "LONG"):
        print(f"  {tag} horizon:")
        for date, (wf, rew) in results[tag].items():
            print(f"    {date}: wait_frac={wf:.3f}  cum_reward={rew:+.5f}")
    sw = np.mean([results["SHORT"][d][0] for d in DATES])
    lw = np.mean([results["LONG"][d][0] for d in DATES])
    print(f"\n  mean wait_frac: SHORT={sw:.3f}  LONG={lw:.3f}")
    if sw < 0.5 <= lw:
        print("  -> SUPPORTS credit-assignment hypothesis (short executes, long collapses).")
    elif sw > 0.5 and lw > 0.5:
        print("  -> Both collapse: hypothesis NOT supported (cause is not horizon).")
    else:
        print("  -> Mixed/ambiguous: needs a closer look, do not adopt hypothesis yet.")


if __name__ == "__main__":
    main()
