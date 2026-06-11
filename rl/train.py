from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback)
from stable_baselines3.common.monitor import Monitor
from environment import StrataExecEnv
import torch
import torch.nn as nn
import argparse
import math
import os


def get_device() -> str:
    if torch.backends.mps.is_available():
        print("Device: Apple MPS (GPU) + CPU LSTM")
        return "mps"
    print("Device: CPU")
    return "cpu"


class _CPULSTMWrapper(nn.Module):
    """
    Forces an nn.LSTM to run on CPU while the surrounding model lives on MPS.

    PyTorch 2.1 MPS LSTM kernel crashes with MPSNDArrayDescriptor
    sliceDimension error when sequence_length >= lstm_hidden_size. Moving
    only the LSTM to CPU sidesteps the broken Metal kernel; the MLP layers
    stay on MPS and dominate compute, so throughput loss is small.

    Exposes all nn.LSTM attributes that sb3-contrib reads directly so the
    policy code sees a transparent drop-in replacement.

    Permanent fix: upgrade to PyTorch >= 2.3 where the MPS LSTM kernel was
    rewritten. This wrapper is safe to leave in place after upgrading — it
    just adds negligible CPU<->MPS transfer overhead.
    """
    def __init__(self, lstm: nn.LSTM):
        super().__init__()
        self.lstm = lstm.cpu()
        # Proxy every attribute sb3-contrib / PyTorch may read directly
        self.input_size    = lstm.input_size
        self.hidden_size   = lstm.hidden_size
        self.num_layers    = lstm.num_layers
        self.batch_first   = lstm.batch_first
        self.dropout       = lstm.dropout
        self.bidirectional = lstm.bidirectional

    def forward(self, x, hx=None):
        dev = x.device
        x_cpu = x.cpu()
        hx_cpu = (hx[0].cpu(), hx[1].cpu()) if hx is not None else None
        out, (h_n, c_n) = self.lstm(x_cpu, hx_cpu)
        return out.to(dev), (h_n.to(dev), c_n.to(dev))


def _patch_mps_lstm(policy) -> None:
    """
    Replace lstm_actor and lstm_critic with _CPULSTMWrapper instances.
    No-op if the attributes are already wrapped or do not exist.
    Only called when device == 'mps'.
    """
    for attr in ("lstm_actor", "lstm_critic"):
        lstm = getattr(policy, attr, None)
        if lstm is not None and isinstance(lstm, nn.LSTM):
            setattr(policy, attr, _CPULSTMWrapper(lstm))
            print(f"  [MPS fix] {attr} → CPU")


def cosine_with_restarts(progress_remaining: float) -> float:
    """
    Cosine annealing with 3 cycles over full training.
    progress_remaining: 1.0 at step 0, 0.0 at final step.

    Cycle boundaries (progress elapsed):
      Cycle 0: 0%  -> 33%  peak multiplier = 1.000
      Cycle 1: 33% -> 66%  peak multiplier = 0.500
      Cycle 2: 66% -> 100% peak multiplier = 0.250

    Each cycle: cosine decay from peak to ~0.
    Halving peaks prevents late-training instability while
    still allowing meaningful exploration after each restart.
    """
    progress = 1.0 - progress_remaining
    cycle_length = 1.0 / 3.0
    cycle_num = min(int(progress / cycle_length), 2)
    cycle_progress = (progress - cycle_num * cycle_length) / cycle_length
    peak = 0.5 ** cycle_num
    return peak * 0.5 * (1.0 + math.cos(math.pi * cycle_progress))


assert abs(cosine_with_restarts(1.0) - 1.0) < 0.01, "cosine_with_restarts(1.0) must be ~1.0"
assert abs(cosine_with_restarts(0.67) - 0.5) < 0.05, "cosine_with_restarts(0.67) must be ~0.5"
assert abs(cosine_with_restarts(0.34) - 0.25) < 0.05, "cosine_with_restarts(0.34) must be ~0.25"


def make_env(
    mode="synthetic",
    adversarial=False,
    lob_date=None,
    agg_date=None,
    btc_target=50000.0,
    multi_dates=None,
    n_state_dims=8,
    seed=42,
    fixed_steps=False,
    fixed_size=False,
    jump_diffusion=False,
    no_regime_impact=False,
):
    import random

    def _init():
        date = lob_date
        agg = agg_date
        if multi_dates:
            date = random.choice(multi_dates)
            agg = date
        env = StrataExecEnv(
            mode=mode,
            adversarial=adversarial,
            lob_date=date,
            agg_date=agg,
            btc_target=btc_target,
            n_state_dims=n_state_dims,
            seed=seed,
            fixed_steps=fixed_steps,
            fixed_size=fixed_size,
            jump_diffusion=jump_diffusion,
            no_regime_impact=no_regime_impact,
        )
        return Monitor(env)
    return _init


def train(
    total_timesteps: int = 500_000,
    adversarial: bool = False,
    mode: str = "synthetic",
    lob_date=None,
    agg_date=None,
    btc_target: float = 50000.0,
    multi_dates=None,
    save_path: str = "rl/models/",
    run_name: str = "ppo_passive",
    lr_schedule: str = "cosine_restarts",
    ent_coef: float = 0.05,
    jump_diffusion: bool = False,
    lstm_hidden: int = 128,
    lstm_layers: int = 2,
    n_envs: int = 4,
    no_regime_impact: bool = False,
):
    os.makedirs(save_path, exist_ok=True)
    os.makedirs("rl/logs/", exist_ok=True)

    print("=" * 60)
    print(f"Training: {run_name}")
    print(f"Mode: {mode}")
    print(f"Adversarial: {adversarial}")
    print(f"Timesteps: {total_timesteps:,}")
    print(f"  btc_target: {btc_target}")
    if lob_date:
        print(f"LOB date: {lob_date}")
    if multi_dates:
        print(f"Multi-dates: {multi_dates}")
    if adversarial:
        print("  Adversarial agents: FrontRunner + LiquidityWithdrawer ACTIVE")
    if jump_diffusion:
        print("  Jump diffusion: Poisson jump events ACTIVE")
    print(f"  ent_coef: {ent_coef}")
    print(f"  lr_schedule: {lr_schedule} (base_lr=3e-4)")
    print(f"  LSTM: {lstm_hidden} hidden × {lstm_layers} layers")
    if no_regime_impact:
        print("  Regime impact scaling: DISABLED (eval mode)")
    else:
        print("  Regime impact scaling: ACTIVE (crash=1.8×, quiet=0.7×)")
    print("=" * 60)

    n_state_dims = 12 if mode == "counterfactual" else 8

    env_kwargs = dict(
        mode=mode,
        adversarial=adversarial,
        lob_date=lob_date,
        agg_date=agg_date,
        btc_target=btc_target,
        multi_dates=multi_dates,
        n_state_dims=n_state_dims,
        fixed_steps=False,
        fixed_size=False,
        jump_diffusion=jump_diffusion,
    )

    n_envs = 8
    env_fns = [make_env(**env_kwargs) for _ in range(n_envs)]
    train_env = SubprocVecEnv(env_fns)

    eval_lob_date = None
    eval_agg_date = None
    if mode == "counterfactual":
        if multi_dates:
            eval_lob_date = multi_dates[0]
            eval_agg_date = multi_dates[0]
        else:
            eval_lob_date = lob_date
            eval_agg_date = agg_date

    eval_env = DummyVecEnv([lambda: Monitor(StrataExecEnv(
        mode=mode,
        adversarial=adversarial,
        lob_date=eval_lob_date,
        agg_date=eval_agg_date,
        btc_target=btc_target,
        n_state_dims=n_state_dims,
        seed=9999,
        fixed_steps=True,
        fixed_size=True,
        jump_diffusion=jump_diffusion,
    ))])

    device = get_device()

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=train_env,
        device=device,
        learning_rate=get_lr_schedule(lr_schedule, 3e-4, total_timesteps),
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
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

    # PyTorch 2.1 MPS LSTM crashes when sequence_length >= lstm_hidden_size.
    # Pin only the LSTM submodules to CPU; all MLP layers stay on MPS.
    if device == "mps":
        _patch_mps_lstm(model.policy)

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
                   help="Learning rate schedule. linear decays lr to 0.")
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
    p.add_argument("--ent-coef", type=float, default=0.05,
                   help="Entropy coefficient. Higher -> more exploration.")
    p.add_argument("--jump-diffusion", action="store_true",
                   help="Enable Poisson jump diffusion in GARCH price model.")
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
        ent_coef=args.ent_coef,
        jump_diffusion=args.jump_diffusion,
    )