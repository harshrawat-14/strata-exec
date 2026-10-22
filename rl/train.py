"""
Train the RecurrentPPO execution agent (CLI entry point).

Training components live in the `training` package; they are re-exported
here so existing `from train import X` callers keep working.
"""

import argparse
import os

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
import mlflow
import torch

from training.callbacks import (  # noqa: F401
    MLflowMetricsCallback,
    MemoryLoggingCallback,
    AdaptiveEntropyCallback,
    ResumeSafeEvalCallback,
)
from training.device import (  # noqa: F401
    get_device,
    _CPULSTMWrapper,
    _patch_mps_lstm,
)
from training.schedules import (  # noqa: F401
    cosine_with_restarts,
    get_lr_schedule,
)
from training.envs import (  # noqa: F401
    make_multi_date_eval_env,
    make_env,
)


# Policy LSTM architecture for fresh runs. The single source for both model
# construction and what gets logged: train() previously logged
# lstm_hidden=128/lstm_layers=2 from unused parameters while building 64x1,
# so every run's MLflow params misrecorded its architecture.
LSTM_HIDDEN_SIZE = 64
LSTM_LAYERS = 1


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
    n_envs: int = 4,
    no_time_features: bool = False,
    resume_from: str = None,
    resume_mlflow_run_id: str = None,
    force_device: str = None,
    base_lr: float = 3e-4,
    episode_steps: int = None,
    eval_dates: list = None,
    gamma: float = 0.99,
    action_space: str = "raw",
    gae_lambda: float = 0.95,
):
    """
    episode_steps: evaluation protocol v3 (evaluation/protocol.py) -- every
    training episode is this long, with a fresh random start and order size.
    eval_dates: dates for best-model selection (protocol v3: the validation
    dates). Held-out test dates are refused. None keeps the legacy choice.
    gamma: PPO discount. 0.99 plans ~100 steps ahead (~50 min), short against
    a 1,920-step v3 episode; M1 tests 0.999 (docs/experiments/14-*).
    gae_lambda: GAE lambda. PPO's credit horizon is ~1/(1 - gamma*lambda):
    ~17 steps at the defaults; M3 tests 0.99 with gamma 0.999 (~90 steps,
    docs/experiments/16-*).

    resume_from: path to a saved checkpoint (e.g. from CheckpointCallback)
    to resume training from, instead of constructing a fresh model. When
    set, `total_timesteps` is interpreted as the REMAINING steps to train
    (SB3's own convention for reset_num_timesteps=False -- see
    _setup_learn, which does `total_timesteps += self.num_timesteps`), not
    the original run's full target. Both policy weights AND optimizer
    state are restored from the checkpoint (confirmed present in
    CheckpointCallback's saved .zip via policy.optimizer.pth), and the LR
    schedule continues from its saved progress_remaining rather than
    restarting -- this reproduces the same trajectory as one uninterrupted
    run, not a discontinuity spliced into the metrics.
    """
    os.makedirs(save_path, exist_ok=True)
    os.makedirs("rl/logs/", exist_ok=True)

    # Phase 2 (intraday time features): counterfactual mode's real state width
    # is 12 dims (no_time_features=True) or 16 dims (time features s12-s15
    # included, the default). Previously this was hardcoded to 12 regardless
    # of no_time_features, and no_time_features was never threaded through
    # make_env/env_kwargs at all -- meaning every prior "counterfactual"
    # training run silently trained on a 12-dim slice of whatever the Rust
    # subprocess actually emitted (16-dim by default), discarding the time
    # features entirely. See docs/architecture/decision-records.md and the Phase 2 report
    # for detail.
    if mode == "counterfactual":
        n_state_dims = 12 if no_time_features else 16
    else:
        n_state_dims = 8

    # Resuming an existing mlflow run (by run_id) rather than always starting
    # a fresh one -- otherwise a resumed training run fragments its tracked
    # history across two run_ids, which is exactly what Phase 3's mlflow
    # integration exists to avoid. mlflow.start_run accepts EITHER run_id OR
    # run_name, not both.
    mlflow_run_ctx = (
        mlflow.start_run(run_id=resume_mlflow_run_id)
        if resume_mlflow_run_id
        else mlflow.start_run(run_name=run_name)
    )
    with mlflow_run_ctx:
        # Re-logging as PARAMS to an already-populated run_id would raise on
        # any key whose value legitimately differs this time (params are
        # write-once in mlflow) -- e.g. a second resume of the same run_id
        # would collide with the first resume's own resumed_from record.
        # Tags are mutable, so use those instead for anything that can
        # reasonably change across repeated resumes of one run.
        if resume_mlflow_run_id:
            mlflow.set_tag("resumed_from", resume_from)
            mlflow.set_tag("resumed_remaining_timesteps", total_timesteps)
            mlflow.set_tag("resumed_n_envs", n_envs)
        else:
            mlflow.log_params(dict(
                lstm_hidden=LSTM_HIDDEN_SIZE,
                lstm_layers=LSTM_LAYERS,
                n_state_dims=n_state_dims,
                ent_coef=ent_coef,
                lr_schedule=lr_schedule,
                n_envs=n_envs,
                total_timesteps=total_timesteps,
                adversarial=adversarial,
                btc_target=btc_target,
                action_space_version="v6",
                mode=mode,
                no_time_features=no_time_features,
                episode_steps=episode_steps,
                eval_dates=",".join(eval_dates) if eval_dates else "",
                gamma=gamma,
                gae_lambda=gae_lambda,
                action_space_kind=action_space,
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
        print(f"  ent_coef: {ent_coef}")
        print(f"  lr_schedule: {lr_schedule} (base_lr={base_lr:.2e})")
        print(f"  LSTM: {LSTM_HIDDEN_SIZE} hidden × {LSTM_LAYERS} layers"
              f"{' (resumed runs keep the checkpoint architecture)' if resume_from else ''}")
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
            no_time_features=no_time_features,
            episode_steps=episode_steps,
            action_space=action_space,
        )

        # Regression note: this was hardcoded to 8, silently ignoring the
        # n_envs parameter train() actually accepts (default 4) -- found
        # while reducing parallelism after ppo_lstm_v6_time_features was
        # killed under sustained near-zero free memory. Now respects the
        # parameter so a caller can actually control it.
        # Date coverage (Phase 3 audit fix). With multi_dates, raise n_envs to
        # cover every date so each worker owns one and no date goes unseen --
        # the failure that made the Step 2 run train on 3 of 16 dates and zero
        # crash dates. Subprocesses are cheap here (~70MB each, measured), so
        # one per date is affordable; the per-episode-respawn fallback is
        # ~15x slower on short episodes and only used if the caller pins n_envs
        # below the date count.
        if multi_dates and n_envs < len(multi_dates):
            print(
                f"n_envs={n_envs} < {len(multi_dates)} training dates: raising "
                f"n_envs to {len(multi_dates)} so every date is covered by a "
                f"dedicated worker (was: {n_envs} random dates, rest unseen)."
            )
            n_envs = len(multi_dates)

        if multi_dates:
            env_fns = [
                make_env(**env_kwargs, worker_date=multi_dates[i % len(multi_dates)],
                         seed=42 + i)
                for i in range(n_envs)
            ]
            from collections import Counter
            try:
                from evaluate import SEVERITY_RANK_V2
                mix = Counter(
                    SEVERITY_RANK_V2.get(multi_dates[i % len(multi_dates)],
                                         (0, "unknown", ""))[1]
                    for i in range(n_envs)
                )
                print(f"Training worker date coverage: {len(set(multi_dates))} dates, regime mix {dict(mix)}")
            except ImportError:
                pass
        else:
            env_fns = [make_env(**env_kwargs, seed=42 + i) for i in range(n_envs)]
        train_env = SubprocVecEnv(env_fns)

        # Phase 2R fix (2): eval across ALL dates (training + held-out crash),
        # not the single trending-up date multi_dates[0]. See
        # make_multi_date_eval_env for the rationale and the held-out-in-selection
        # caveat.
        if eval_dates:
            from evaluation.dates import HELD_OUT_DATES_V2
            leaked = sorted(set(eval_dates) & set(HELD_OUT_DATES_V2))
            if leaked:
                raise ValueError(f"eval_dates include held-out test dates: {leaked}")
            eval_dates = list(dict.fromkeys(eval_dates))
        elif mode == "counterfactual" and multi_dates:
            # Phase 3 fix: select on the TRAINING dates only, so held-out dates
            # stay genuinely held out.
            #
            # The previous behavior appended HELD_OUT_CRASH_DATES to the
            # selection set. That was a deliberate, documented tradeoff, but it
            # is the same class of error this project already caught twice with
            # lambda (un-swept, then hindsight-per-date): information from the
            # evaluation set leaking into a selection step, which tilts the
            # comparison toward this project's own method. It made the phrase
            # "held-out crash dates" in the headline do less work than it looks
            # like it does -- the model was chosen partly BY its score on them.
            #
            # It existed for a real reason: the V1 training set contains only
            # ONE crash-regime date, so selecting on training dates alone was
            # nearly blind to crash behavior and could not tell a wait-collapsed
            # policy from a good one. The V2 split (4 crash dates in training)
            # removes that constraint, so the leak is no longer a necessary
            # price. The guard below keeps the old behavior -- loudly -- for any
            # training set that still lacks crash exposure, rather than silently
            # reintroducing the blindness the append was protecting against.
            eval_dates = list(dict.fromkeys(multi_dates))
            try:
                from evaluate import SEVERITY_RANK_V2
                n_crash = sum(
                    1 for d in eval_dates
                    if SEVERITY_RANK_V2.get(d, (0, "unknown", ""))[1] == "crash"
                )
            except ImportError:
                n_crash = 0
            if n_crash < 2:
                from evaluate import HELD_OUT_CRASH_DATES
                eval_dates = list(dict.fromkeys(eval_dates + list(HELD_OUT_CRASH_DATES)))
                print(
                    f"WARNING: training set has only {n_crash} crash-regime date(s); "
                    f"falling back to including HELD_OUT_CRASH_DATES in best-model "
                    f"selection so crash behavior is not invisible to it. Held-out "
                    f"crash results from this run are NOT a clean generalization "
                    f"measure -- use the V2 split to avoid this."
                )
            else:
                print(
                    f"Best-model selection uses training dates only "
                    f"({n_crash} crash-regime); held-out dates stay clean."
                )
        elif mode == "counterfactual":
            eval_dates = [lob_date]
        else:
            eval_dates = [lob_date] if lob_date else [None]
        print(f"Eval (best-model selection) dates: {eval_dates}")
        eval_env = make_multi_date_eval_env(
            mode=mode, adversarial=adversarial, eval_dates=eval_dates,
            btc_target=btc_target, n_state_dims=n_state_dims,
            no_time_features=no_time_features, episode_steps=episode_steps,
            action_space=action_space,
        )

        device = force_device if force_device else get_device()
        if force_device:
            print(f"Device: forced to {force_device} (--device override)")

        if resume_from:
            # Force CPU for resumed runs, regardless of get_device(). Found
            # via a smoke test (always run before committing a resume to a
            # multi-hour job -- this is the second real bug it caught):
            # loading a checkpoint with device="mps" places the LSTM's
            # saved optimizer state (Adam momentum/variance) on mps, but
            # _patch_mps_lstm then moves the LSTM's PARAMETERS to cpu via
            # `.cpu()`, which returns new tensor objects -- the optimizer's
            # state dict still references the pre-move (mps-resident)
            # tensor identities, so the first optimizer.step() after resume
            # raises "Expected all tensors to be on the same device, but
            # found at least two devices, mps:0 and cpu". This device split
            # is only safe on FRESH construction, where the optimizer has
            # taken zero steps and has no pre-existing state to invalidate.
            # The remaining step count after a resume is bounded (a
            # fraction of the original run), so losing MPS acceleration for
            # just that tail is an acceptable, simple, correct-by-
            # construction trade against re-deriving the optimizer's device
            # placement by hand.
            resume_device = "cpu"
            print(f"Resuming from checkpoint: {resume_from} (forcing device=cpu -- "
                  f"see comment above for why MPS resume is unsafe)")
            from evaluate import load_model
            # env passed directly to load(), not a separate set_env() call
            # afterward -- set_env() hard-asserts num_envs matches the
            # checkpoint's original env count and refuses to resize the
            # rollout buffers, which breaks resuming with a different
            # n_envs (e.g. reduced parallelism after a memory incident).
            model = load_model(resume_from, device=resume_device, env=train_env)
            device = resume_device  # so the _patch_mps_lstm check below is skipped
            print(f"  Restored num_timesteps={model.num_timesteps}, "
                  f"progress_remaining={model._current_progress_remaining:.4f}")
        else:
            model = RecurrentPPO(
                policy="MlpLstmPolicy",
                env=train_env,
                device=device,
                learning_rate=get_lr_schedule(lr_schedule, base_lr, total_timesteps),
                n_steps=2048,
                batch_size=256,
                n_epochs=10,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=0.2,
                ent_coef=ent_coef,
                policy_kwargs=dict(
                    net_arch=dict(pi=[64, 64], vf=[64, 64]),
                    lstm_hidden_size=LSTM_HIDDEN_SIZE,
                    n_lstm_layers=LSTM_LAYERS,
                    activation_fn=torch.nn.Tanh,
                    enable_critic_lstm=True,
                ),
                verbose=1,
                tensorboard_log=f"rl/logs/{run_name}",
                seed=42,
            )

        # PyTorch 2.1 MPS LSTM crashes when sequence_length >= lstm_hidden_size.
        # Pin only the LSTM submodules to CPU; all MLP layers stay on MPS.
        # Applies whether the model was just constructed or resumed from a
        # checkpoint -- load_model(device="mps") loads the LSTM weights onto
        # MPS in plain nn.LSTM form (unwrapped keys), so this patch step is
        # still needed after loading, same as after fresh construction.
        if device == "mps":
            _patch_mps_lstm(model.policy)

        # eval enough episodes that every date gets a fair sample (>=5 each).
        # Protocol v3: exactly one pass over each date's 5 start positions, so
        # every evaluation round sees the same 5 episodes per date.
        n_eval_episodes = (len(eval_dates) * 5 if episode_steps
                           else max(50, len(eval_dates) * 5))
        callbacks = [
            # Phase 2R fix (2)+(3): reward-based selection across ALL dates, with
            # best_mean_reward persisted across resume.
            ResumeSafeEvalCallback(
                eval_env,
                best_model_save_path=f"{save_path}{run_name}_best",
                best_reward_path=f"{save_path}{run_name}_best/best_mean_reward.json",
                load_prior_best=bool(resume_from),
                eval_freq=max(50_000 // n_envs, 1),
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                verbose=1,
            ),
            # Phase 2R fix (1): entropy floor to prevent the late-training collapse.
            AdaptiveEntropyCallback(base_coef=ent_coef, floor=1.2, verbose=1),
            CheckpointCallback(
                save_freq=100_000 // n_envs,
                save_path=f"{save_path}checkpoints/{run_name}/",
                name_prefix="ckpt",
            ),
            MLflowMetricsCallback(),
            MemoryLoggingCallback(),
        ]

        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=(resume_from is None),
        )

        final_metrics = {
            f"final_{k.replace('/', '_')}": v
            for k, v in model.logger.name_to_value.items()
            if isinstance(v, (int, float))
        }
        if final_metrics:
            mlflow.log_metrics(final_metrics)

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
    p.add_argument("--no-time-features", action="store_true",
                   help="Counterfactual mode only: use the 12-dim state "
                        "(omit intraday time features s12-s15). Default is "
                        "16-dim (time features included) -- pass this flag "
                        "to reproduce the pre-Phase-2 12-dim baseline for an "
                        "ablation comparison.")
    p.add_argument("--n-envs", type=int, default=4,
                   help="Number of parallel SubprocVecEnv workers. Was "
                        "silently hardcoded to 8 regardless of this flag "
                        "until the fix that added this CLI option.")
    p.add_argument("--resume-from", type=str, default=None,
                   help="Path to a checkpoint to resume from (e.g. "
                        "rl/models/checkpoints/<run>/ckpt_N_steps). "
                        "--timesteps is then the REMAINING steps to train, "
                        "not the original run's total (SB3's own "
                        "reset_num_timesteps=False convention).")
    p.add_argument("--resume-mlflow-run-id", type=str, default=None,
                   help="Reopen this existing mlflow run_id instead of "
                        "starting a fresh one, so a resumed run's metrics "
                        "land in the same tracked history as the original "
                        "(interrupted) run.")
    p.add_argument("--device", type=str, default=None,
                   choices=["cpu", "mps", "cuda"],
                   help="Force a torch device instead of auto-detecting. Useful "
                        "for fast CPU smoke tests (MPS has no throughput edge for "
                        "this small LSTM and degrades on long runs).")
    p.add_argument("--base-lr", type=float, default=3e-4,
                   help="Base learning rate the lr-schedule scales. Default 3e-4. "
                        "Set with --lr-schedule constant to hold a fixed LR (e.g. "
                        "the Phase 2R collapse-region isolation test at 1.9e-4).")
    p.add_argument("--episode-steps", type=int, default=None,
                   help="Evaluation protocol v3: fixed episode length in steps "
                        "(1920 = 16h) with a fresh random start per episode. "
                        "Omit for the legacy protocol.")
    p.add_argument("--action-space", choices=["raw", "residual_ac", "rate"], default="raw",
                   help="raw: choose sell fractions directly. residual_ac: choose an "
                        "offset ahead of/behind AC's schedule (M2). rate: multiples "
                        "of the steady selling rate (M4).")
    p.add_argument("--gamma", type=float, default=0.99,
                   help="PPO discount factor (default 0.99).")
    p.add_argument("--gae-lambda", type=float, default=0.95,
                   help="GAE lambda (default 0.95). Credit horizon ~1/(1-gamma*lambda).")
    p.add_argument("--eval-dates", type=str, default=None,
                   help="Comma-separated dates for best-model selection "
                        "(protocol v3: the validation dates). Held-out test "
                        "dates are refused.")
    args = p.parse_args()
    if args.adversarial and args.mode == "counterfactual":
        p.error("--adversarial is not supported with --mode counterfactual "
                "(adversarial agents exist only in historical and synthetic modes)")

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
        no_time_features=args.no_time_features,
        n_envs=args.n_envs,
        resume_from=args.resume_from,
        resume_mlflow_run_id=args.resume_mlflow_run_id,
        force_device=args.device,
        base_lr=args.base_lr,
        episode_steps=args.episode_steps,
        eval_dates=[d.strip() for d in args.eval_dates.split(",")] if args.eval_dates else None,
        gamma=args.gamma,
        action_space=args.action_space,
        gae_lambda=args.gae_lambda,
    )
