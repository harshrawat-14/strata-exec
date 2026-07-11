from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback)
from stable_baselines3.common.monitor import Monitor
from environment import StrataExecEnv
import mlflow
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


# Regression note (Phase 2): this module could not previously be imported at
# all -- these two assertions were failing with the ORIGINAL 0.67/0.34
# inputs (cosine_with_restarts(0.67) ~= 0.0002, not ~0.5). The function
# itself is correct and matches the docstring; 0.67/0.34 simply landed on
# the wrong side of the 1/3 and 2/3 progress boundaries (e.g.
# progress_remaining=0.67 -> progress=0.33, which is fractionally BEFORE
# the exact 1/3 boundary -- still the tail end of cycle 0, where cosine
# decays to ~0, not the start of cycle 1's peak). 0.66/0.32 land safely
# inside cycles 1 and 2 respectively, at their intended peaks.
assert abs(cosine_with_restarts(1.0) - 1.0) < 0.01, "cosine_with_restarts(1.0) must be ~1.0"
assert abs(cosine_with_restarts(0.66) - 0.5) < 0.05, "cosine_with_restarts(0.66) must be ~0.5"
assert abs(cosine_with_restarts(0.32) - 0.25) < 0.05, "cosine_with_restarts(0.32) must be ~0.25"


def get_lr_schedule(lr_schedule: str, base_lr: float, total_timesteps: int):
    """
    Map the --lr-schedule CLI choice to what RecurrentPPO's learning_rate
    argument expects: either a constant float, or a callable
    (progress_remaining: float) -> float, where progress_remaining goes
    1.0 -> 0.0 over training regardless of total_timesteps (SB3's own
    convention -- total_timesteps is accepted here for interface symmetry
    with the other two branches and isn't otherwise needed).

    Regression note (Phase 2): this function was called but never defined
    anywhere in this file -- train() raised NameError on the first line
    that needed a learning rate, meaning training had never been run to
    completion in this state. cosine_restarts here reuses
    cosine_with_restarts (already implemented, already exercised at module
    import time by the assertions above) scaled by base_lr.
    """
    if lr_schedule == "constant":
        return base_lr
    if lr_schedule == "linear":
        return lambda progress_remaining: base_lr * progress_remaining
    if lr_schedule == "cosine_restarts":
        return lambda progress_remaining: base_lr * cosine_with_restarts(progress_remaining)
    raise ValueError(
        f"unknown lr_schedule {lr_schedule!r}; expected one of "
        f"'constant', 'linear', 'cosine_restarts'"
    )


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
    no_time_features=False,
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
            no_time_features=no_time_features,
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
    no_time_features: bool = False,
):
    os.makedirs(save_path, exist_ok=True)
    os.makedirs("rl/logs/", exist_ok=True)

    # Phase 2 (intraday time features): counterfactual mode's real state width
    # is 12 dims (no_time_features=True) or 16 dims (time features s12-s15
    # included, the default). Previously this was hardcoded to 12 regardless
    # of no_time_features, and no_time_features was never threaded through
    # make_env/env_kwargs at all -- meaning every prior "counterfactual"
    # training run silently trained on a 12-dim slice of whatever the Rust
    # subprocess actually emitted (16-dim by default), discarding the time
    # features entirely. See docs/system_design/ADR.md and the Phase 2 report
    # for detail.
    if mode == "counterfactual":
        n_state_dims = 12 if no_time_features else 16
    else:
        n_state_dims = 8

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(dict(
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            n_state_dims=n_state_dims,
            ent_coef=ent_coef,
            lr_schedule=lr_schedule,
            n_envs=n_envs,
            total_timesteps=total_timesteps,
            adversarial=adversarial,
            jump_diffusion=jump_diffusion,
            btc_target=btc_target,
            action_space_version="v6",
            mode=mode,
            no_time_features=no_time_features,
        ))

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
        print(f"  n_state_dims: {n_state_dims}"
              f"{' (time features s12-s15 included)' if n_state_dims == 16 else ''}")
        print("=" * 60)

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
            no_time_features=no_time_features,
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
            no_time_features=no_time_features,
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
        mlflow.log_artifact(f"{final_path}.zip")
        print(f"\nModel saved: {final_path}")
        return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lr-schedule", type=str, default="constant",
                   choices=["constant", "linear", "cosine_restarts"],
                   help="Learning rate schedule. linear decays lr to 0; "
                        "cosine_restarts uses 3 cosine-annealed cycles with "
                        "halving peaks (see cosine_with_restarts).")
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
    p.add_argument("--no-time-features", action="store_true",
                   help="Counterfactual mode only: use the 12-dim state "
                        "(omit intraday time features s12-s15). Default is "
                        "16-dim (time features included) -- pass this flag "
                        "to reproduce the pre-Phase-2 12-dim baseline for an "
                        "ablation comparison.")
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
        no_time_features=args.no_time_features,
    )