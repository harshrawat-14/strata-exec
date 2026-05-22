from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback)
from stable_baselines3.common.monitor import Monitor
from environment import StrataExecEnv
import torch
import argparse
import os

def make_env(
    mode="synthetic",
    adversarial=False,
    lob_date=None,
    agg_date=None,
    btc_target=500.0,
    multi_dates=None,
    n_state_dims=8,
    seed=42,
):
    import random

    def _init():
        date = lob_date
        agg = agg_date
        if multi_dates:
            date = random.choice(multi_dates)
            agg = date  # assume agg-trades file has the same date name
        env = StrataExecEnv(
            mode=mode,
            adversarial=adversarial,
            lob_date=date,
            agg_date=agg,
            btc_target=btc_target,
            n_state_dims=n_state_dims,
            seed=seed,
        )
        return Monitor(env)
    return _init

def get_lr_schedule(schedule, initial_lr, total_timesteps):
    if schedule == "linear":
        # Returns a function that decays lr linearly to 0
        def lr_fn(progress_remaining: float) -> float:
            return progress_remaining * initial_lr
        return lr_fn
    return initial_lr  # constant

def train(
    total_timesteps: int = 500_000,
    adversarial: bool = False,
    mode: str = "synthetic",
    lob_date=None,
    agg_date=None,
    btc_target: float = 500.0,
    multi_dates=None,
    save_path: str = "rl/models/",
    run_name: str = "ppo_passive",
    lr_schedule: str = "constant",
):
    os.makedirs(save_path, exist_ok=True)
    os.makedirs("rl/logs/", exist_ok=True)

    print("=" * 60)
    print(f"Training: {run_name}")
    print(f"Mode: {mode}")
    print(f"Adversarial: {adversarial}")
    print(f"Timesteps: {total_timesteps:,}")
    if lob_date:
        print(f"LOB date: {lob_date}")
    if multi_dates:
        print(f"Multi-dates: {multi_dates}")
    print("=" * 60)

    n_state_dims = 10 if mode == "counterfactual" else 8

    env_kwargs = dict(
        mode=mode,
        adversarial=adversarial,
        lob_date=lob_date,
        agg_date=agg_date,
        btc_target=btc_target,
        multi_dates=multi_dates,
        n_state_dims=n_state_dims,
    )

    train_env = make_vec_env(
        make_env(**env_kwargs),
        n_envs=4,
    )

    # For counterfactual mode, eval env needs a lob_date.
    # Use the first date from multi_dates list, or lob_date directly.
    eval_lob_date = None
    eval_agg_date = None
    if mode == "counterfactual":
        if multi_dates:
            eval_lob_date = multi_dates[0]
            eval_agg_date = multi_dates[0]
        else:
            eval_lob_date = lob_date
            eval_agg_date = agg_date

    eval_env = Monitor(StrataExecEnv(
        mode=mode,
        adversarial=adversarial,
        lob_date=eval_lob_date,
        agg_date=eval_agg_date,
        btc_target=btc_target,
        n_state_dims=n_state_dims,
        seed=9999,
    ))

    # Infer obs size from training env.
    n_obs = train_env.observation_space.shape[0]

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=train_env,
        learning_rate=get_lr_schedule(lr_schedule, 3e-4, total_timesteps),
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            lstm_hidden_size=64,
            n_lstm_layers=1,
            activation_fn=torch.nn.Tanh,
            enable_critic_lstm=True,
        ),
        verbose=1,
        tensorboard_log=f"rl/logs/{run_name}",
        seed=42,
    )

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=f"{save_path}{run_name}_best",
            eval_freq=50_000 // 4,
            n_eval_episodes=50,
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=100_000 // 4,
            save_path=f"{save_path}checkpoints/{run_name}/",
            name_prefix="ckpt",
        ),
    ]

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    final_path = f"{save_path}{run_name}_final"
    model.save(final_path)
    print(f"\nModel saved: {final_path}")
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lr-schedule", type=str, default="constant",
    choices=["constant", "linear"],
    help="Learning rate schedule. linear decays to 0 by end of training.")
    p.add_argument("--adversarial", action="store_true")
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--mode", type=str, default="synthetic",
                   choices=["synthetic", "historical", "counterfactual"])
    p.add_argument("--lob-date", type=str, default=None)
    p.add_argument("--agg-date", type=str, default=None)
    p.add_argument("--btc-target", type=float, default=500.0)
    p.add_argument("--multi-date", type=str, default=None,
                   help="Comma-separated dates: 2024-01-15,2024-03-05")
    args = p.parse_args()

    multi_dates = None
    if args.multi_date:
        multi_dates = [d.strip() for d in args.multi_date.split(",")]

    if args.name is None:
        if args.mode == "counterfactual":
            args.name = "ppo_counterfactual"
        elif args.adversarial:
            args.name = "ppo_adversarial"
        else:
            args.name = "ppo_passive"

    train(
        total_timesteps=args.timesteps,
        adversarial=args.adversarial,
        mode=args.mode,
        lob_date=args.lob_date,
        agg_date=args.agg_date,
        btc_target=args.btc_target,
        multi_dates=multi_dates,
        run_name=args.name,
        lr_schedule=args.lr_schedule,
    )
