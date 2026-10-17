"""
Phase 3 Step B: imitation warm-start -- behavior-clone honest-lambda AC-Optimal
into a fresh RecurrentPPO policy via supervised (teacher-forced) cross-entropy,
BEFORE any RL fine-tuning. This produces the "floor guarantee" the project has
repeatedly proposed and deferred (docs/proposals/residual-action-space.md,
docs/proposals/research-roadmap.md Section on imitation) now that AC itself has been cleaned
up (honest lambda, fixed impact/volatility bugs) -- the prior objection ("no
clean expert to imitate") no longer applies to the SPECIFIC expert used here.

LEAKAGE GUARDRAIL (the one most likely way this experiment silently produces a
fake result, flagged before writing any training code): the expert MUST be
honest-lambda AC (lambda=1e-6, selected from TRAINING_DATES aggregate only,
docs/experiments/10-full-holdout-evaluation-predictions.md / docs/experiments/02-reward-objective-audit.md), never
per-date hindsight-lambda AC. Cloning hindsight-lambda AC would inject each
date's own outcome into the RL agent's initialization -- exactly the failure
mode that killed the project's second major correction. Data collection below
is restricted to TRAINING_DATES_V2 only; HELD_OUT_DATES_V2 is never touched
during BC training, only used afterward to verify the floor.

Method: for each training date, roll ACOptimalPolicy episodes through the real
counterfactual env (not synthetic obs) to get real (obs, action) sequences.
Train the policy's recurrent action distribution via truncated BPTT
(chunk=128 steps, matching the env's LSTM statefulness) with cross-entropy
against AC's chosen discrete action at each step. This is supervised learning
only -- no reward, no PPO update, no simulator-outcome-based label of any kind.

Usage:
    rl/.venv/bin/python rl/experiments/phase3_stepB_bc_pretrain.py --seeds-per-date 3 --epochs 3
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/, for bare imports like `import evaluate`

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

from environment import StrataExecEnv
from evaluate import ACOptimalPolicy, TRAINING_DATES_V2, HELD_OUT_DATES_V2

HONEST_LAMBDA = 1e-6
BTC_TARGET = 50_000.0
N_STATE_DIMS = 16
TOTAL_STEPS = 2880
CHUNK = 128

# Same architecture as rl/train.py's fresh-construction path, kept identical
# on purpose so the resulting checkpoint is a drop-in --warm-start-from
# candidate for further RL fine-tuning, not a differently-shaped model.
POLICY_KWARGS = dict(
    net_arch=dict(pi=[64, 64], vf=[64, 64]),
    lstm_hidden_size=64,
    n_lstm_layers=1,
    activation_fn=torch.nn.Tanh,
    enable_critic_lstm=True,
)


def make_env(date: str):
    def _init():
        return StrataExecEnv(
            mode="counterfactual", lob_date=date, agg_date=date,
            btc_target=BTC_TARGET, n_state_dims=N_STATE_DIMS,
            total_steps=TOTAL_STEPS, fixed_steps=True, fixed_size=True,
            no_time_features=False,
        )
    return _init


def collect_episode(date: str, seed: int, ac: ACOptimalPolicy):
    """Roll AC-Optimal through the real env; return (obs[T,D], actions[T])."""
    env = StrataExecEnv(
        mode="counterfactual", lob_date=date, agg_date=date,
        btc_target=BTC_TARGET, n_state_dims=N_STATE_DIMS,
        total_steps=TOTAL_STEPS, fixed_steps=True, fixed_size=True,
        no_time_features=False,
    )
    try:
        obs, _info = env.reset(seed=seed)
        obs_list, act_list, done = [], [], False
        while not done:
            action, _ = ac.predict(obs)
            a = int(action[0]) if hasattr(action, "__len__") else int(action)
            obs_list.append(obs.copy())
            act_list.append(a)
            obs, _r, done, _t, _info = env.step(a)
        return np.asarray(obs_list, dtype=np.float32), np.asarray(act_list, dtype=np.int64)
    finally:
        env.close()


def bc_loss_for_episode(policy, obs_seq: np.ndarray, act_seq: np.ndarray, device) -> torch.Tensor:
    """
    Teacher-forced cross-entropy over one episode, truncated-BPTT in chunks
    of CHUNK steps (hidden state carried across chunks, detached between
    them -- bounds backprop memory without breaking episode-long statefulness
    at inference time, since only the GRADIENT history is truncated, not the
    hidden state itself).
    """
    T = obs_seq.shape[0]
    obs_t = torch.as_tensor(obs_seq, device=device)
    act_t = torch.as_tensor(act_seq, device=device)

    lstm_states = (
        torch.zeros(policy.lstm_actor.num_layers, 1, policy.lstm_actor.hidden_size, device=device),
        torch.zeros(policy.lstm_actor.num_layers, 1, policy.lstm_actor.hidden_size, device=device),
    )

    total_loss = torch.zeros((), device=device)
    n_chunks = 0
    for start in range(0, T, CHUNK):
        end = min(start + CHUNK, T)
        chunk_obs = obs_t[start:end].unsqueeze(1)          # (seq_len, n_envs=1, D)
        chunk_acts = act_t[start:end]
        episode_starts = torch.zeros(end - start, device=device)
        if start == 0:
            episode_starts[0] = 1.0

        dist, lstm_states = policy.get_distribution(chunk_obs, lstm_states, episode_starts)
        log_prob = dist.log_prob(chunk_acts)
        total_loss = total_loss + (-log_prob.mean())
        n_chunks += 1

        # Truncate BPTT: detach hidden state between chunks so gradients
        # don't accumulate across the full 2880-step episode.
        lstm_states = (lstm_states[0].detach(), lstm_states[1].detach())

    return total_loss / max(n_chunks, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-per-date", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="rl/models/phase3_stepB_bc_pretrain/bc_pretrained.zip")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    assert not (set(TRAINING_DATES_V2) & set(HELD_OUT_DATES_V2)), (
        "TRAINING_DATES_V2 / HELD_OUT_DATES_V2 overlap -- refusing to run"
    )

    device = args.device
    ac = ACOptimalPolicy(n_state_dims=N_STATE_DIMS, lam=HONEST_LAMBDA, total_steps=TOTAL_STEPS)

    print(f"Collecting BC data: {len(TRAINING_DATES_V2)} training dates x "
          f"{args.seeds_per_date} seeds, honest-lambda AC (lambda={HONEST_LAMBDA}) "
          f"as the ONLY expert. TRAINING_DATES_V2 only -- held-out dates never touched.")
    episodes = []
    for date in TRAINING_DATES_V2:
        for seed in range(args.seeds_per_date):
            obs_seq, act_seq = collect_episode(date, seed, ac)
            episodes.append((date, seed, obs_seq, act_seq))
        print(f"  collected {date}: {args.seeds_per_date} episodes")

    dummy_env = DummyVecEnv([make_env(TRAINING_DATES_V2[0])])
    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=dummy_env,
        device=device,
        policy_kwargs=POLICY_KWARGS,
        verbose=0,
        seed=42,
    )
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    print(f"\nBC training: {args.epochs} epochs over {len(episodes)} episodes "
          f"(chunk={CHUNK} truncated-BPTT)")
    rng = np.random.default_rng(42)
    for epoch in range(args.epochs):
        order = rng.permutation(len(episodes))
        epoch_loss = 0.0
        for idx in order:
            date, seed, obs_seq, act_seq = episodes[idx]
            optimizer.zero_grad()
            loss = bc_loss_for_episode(policy, obs_seq, act_seq, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            epoch_loss += loss.item()
        print(f"  epoch {epoch+1}/{args.epochs}: mean NLL = {epoch_loss / len(episodes):.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save(args.out.replace(".zip", ""))
    print(f"\nSaved BC-pretrained model to {args.out}")
    print("Next: verify this checkpoint's held-out performance is close to AC's "
          "BEFORE any RL fine-tuning (the floor guarantee) -- "
          "rl/experiments/phase3_stepB_verify_floor.py")


if __name__ == "__main__":
    main()
