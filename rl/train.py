from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback)
from stable_baselines3.common.monitor import Monitor
from environment import StrataExecEnv
import mlflow
import torch
import torch.nn as nn
import argparse
import math
import os


class MLflowMetricsCallback(BaseCallback):
    """
    Forwards SB3's own rollout/train metrics (entropy, ep_len_mean, loss,
    approx_kl, explained_variance, etc.) to mlflow.log_metric at the end of
    every rollout, so training-time telemetry is captured continuously
    rather than only as a final snapshot after model.learn() returns.

    Phase 3 gap this closes: train.py previously had mlflow.start_run /
    log_params / log_artifact but ZERO log_metric calls anywhere -- every
    training run's loss curves, entropy decay, and episode-length growth
    were computed by SB3 internally (visible only in the TensorBoard log
    directory) and never reached mlflow, which is why
    experiment_tracker.py's get_best_run() always returned {} (nothing was
    ever logged for it to find).
    """

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        for key, value in self.model.logger.name_to_value.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key.replace("/", "_"), value, step=self.num_timesteps)


class MemoryLoggingCallback(BaseCallback):
    """
    Logs main-process RSS and the summed RSS of every direct child process
    (the SubprocVecEnv rl-env workers) to both stdout and mlflow at every
    rollout end -- an early-warning line for the next multi-hour run.

    Added after the ppo_lstm_v6_time_features run was killed by something
    external at ~92% through its 3M-step schedule with free system memory
    sitting near-zero for most of the run (RSS fluctuated 403MB->1.32GB->
    898MB rather than climbing monotonically, more consistent with
    oversubscription -- other processes + 8 parallel envs on a
    memory-constrained machine -- than a leak in this codebase, but that
    conclusion was reached after the fact by reading process listings taken
    ad hoc during the incident, not from anything the training loop itself
    recorded). Phase 4's IQN run will likely be longer and heavier; this
    callback exists so a memory problem shows up as a line in the log
    within the first few rollouts, not as a silent kill discovered hours in.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._process = None

    def _on_training_start(self) -> None:
        import psutil
        self._process = psutil.Process()

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        main_rss_mb = self._process.memory_info().rss / (1024 * 1024)
        children_rss_mb = 0.0
        for child in self._process.children(recursive=True):
            try:
                children_rss_mb += child.memory_info().rss / (1024 * 1024)
            except Exception:
                # Child may have exited between listing and sampling.
                pass
        total_rss_mb = main_rss_mb + children_rss_mb

        print(f"  [memory] main={main_rss_mb:.0f}MB  children={children_rss_mb:.0f}MB  "
              f"total={total_rss_mb:.0f}MB  at timestep={self.num_timesteps}")
        mlflow.log_metrics({
            "mem_main_rss_mb": main_rss_mb,
            "mem_children_rss_mb": children_rss_mb,
            "mem_total_rss_mb": total_rss_mb,
        }, step=self.num_timesteps)


class AdaptiveEntropyCallback(BaseCallback):
    """
    Phase 2R fix (1): an entropy FLOOR that actively opposes the late-training
    policy collapse.

    The v6r run collapsed from an execution policy (entropy ~2.1) to blanket-wait
    (entropy ~0.38) between 1.1M and 2.0M steps, a self-reinforcing entropy-decay
    slide -- and it did so DESPITE a constant ent_coef=0.05 (SB3 does not anneal
    ent_coef; a fixed coefficient was not enough to hold entropy up). So a fixed
    coefficient is insufficient; this callback adapts it: when the policy's
    entropy drops below `floor`, ent_coef is raised (up to `max_coef`); when
    entropy recovers above `floor + margin`, ent_coef relaxes back toward
    `base_coef`. This is a standard adaptive-entropy control loop (cf. SAC target
    entropy), targeting a minimum exploration level rather than a fixed bonus.

    Entropy is read from SB3's own `train/entropy_loss` (= -entropy) logged each
    update, so no extra forward passes. For a 20-action space, max entropy is
    ln(20) ~ 3.0; a floor around 1.0-1.3 keeps the policy meaningfully stochastic
    without forcing uniformity.
    """

    def __init__(self, base_coef, floor=1.2, margin=0.3, max_coef=0.5,
                 up=1.5, down=0.9, verbose=0):
        super().__init__(verbose)
        self.base_coef = float(base_coef)
        self.floor = float(floor)
        self.margin = float(margin)
        self.max_coef = float(max_coef)
        self.up = float(up)
        self.down = float(down)

    def _current_entropy(self):
        ent_loss = self.model.logger.name_to_value.get("train/entropy_loss")
        if ent_loss is None:
            return None
        return -float(ent_loss)  # entropy_loss is -entropy

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        entropy = self._current_entropy()
        if entropy is None:
            return  # no update has happened yet (first rollout)
        coef = float(self.model.ent_coef)
        if entropy < self.floor:
            coef = min(coef * self.up, self.max_coef)
        elif entropy > self.floor + self.margin:
            coef = max(coef * self.down, self.base_coef)
        self.model.ent_coef = coef
        mlflow.log_metrics(
            {"policy_entropy": entropy, "ent_coef": coef}, step=self.num_timesteps)
        if self.verbose:
            print(f"  [entropy-floor] entropy={entropy:.3f} -> ent_coef={coef:.4f}")


class ResumeSafeEvalCallback(EvalCallback):
    """
    Phase 2R fix (3): persist `best_mean_reward` across a resume.

    SB3's EvalCallback initializes `best_mean_reward = -inf` in __init__. Because
    train() rebuilds its callback list every call, a resumed run gets a FRESH
    EvalCallback with best = -inf -- so the first post-resume eval overwrites
    `best_model` regardless of how bad the resumed policy is. In the v6r run the
    resume happened at ~1.7M when the policy was already collapsed, so the good
    ~900k-era best_model was discarded and best_model became a collapsed policy.

    This subclass writes `best_mean_reward` to a sidecar JSON whenever it improves,
    and (when `load_prior_best=True`, i.e. on resume) loads it back at training
    start so the resumed run must BEAT the pre-resume best before overwriting
    best_model.
    """

    def __init__(self, *args, best_reward_path=None, load_prior_best=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._best_reward_path = best_reward_path
        self._load_prior_best = load_prior_best

    def _on_training_start(self) -> None:
        if (self._load_prior_best and self._best_reward_path
                and os.path.exists(self._best_reward_path)):
            import json
            with open(self._best_reward_path) as f:
                prior = json.load(f).get("best_mean_reward")
            if prior is not None:
                self.best_mean_reward = float(prior)
                print(f"  [resume-safe-eval] loaded prior best_mean_reward="
                      f"{self.best_mean_reward:.6f}")

    def _on_step(self) -> bool:
        prev_best = self.best_mean_reward
        result = super()._on_step()
        if (self._best_reward_path is not None
                and self.best_mean_reward > prev_best):
            import json
            with open(self._best_reward_path, "w") as f:
                json.dump({"best_mean_reward": self.best_mean_reward}, f)
        return result


def make_multi_date_eval_env(mode, adversarial, eval_dates, btc_target,
                             n_state_dims, jump_diffusion, no_time_features):
    """
    Phase 2R fix (2): a MULTI-date eval env for best-model selection.

    The original eval env was a single date (multi_dates[0] = 2024-01-15, a
    trending-up date where waiting scores acceptably), so best-model selection --
    even though it selects on reward -- was blind to crash-day behavior and could
    not distinguish a wait-collapsed policy from a good one on the dimension that
    matters most. This builds one sub-env per eval date (all training dates PLUS
    the held-out crash dates), so EvalCallback's mean reward is averaged across
    all regimes and a policy that collapses on crash days is correctly penalized.

    CAVEAT (documented, for the writeup): including the held-out crash dates in
    best-model SELECTION means their eval numbers are no longer a clean held-out
    generalization measure for the selected model. Per explicit instruction, crash
    behavior must count in selection (or the single-trending-date pathology
    returns); if a pure generalization claim on the crash dates is later needed, a
    separate untouched test date must be reserved.
    """
    def _one(date):
        return lambda: Monitor(StrataExecEnv(
            mode=mode, adversarial=adversarial, lob_date=date, agg_date=date,
            btc_target=btc_target, n_state_dims=n_state_dims, seed=9999,
            fixed_steps=True, fixed_size=True, jump_diffusion=jump_diffusion,
            no_time_features=no_time_features,
        ))
    return DummyVecEnv([_one(d) for d in eval_dates])


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
    resume_from: str = None,
    resume_mlflow_run_id: str = None,
    force_device: str = None,
    base_lr: float = 3e-4,
):
    """
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
    # features entirely. See docs/system_design/ADR.md and the Phase 2 report
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
        print(f"  lr_schedule: {lr_schedule} (base_lr={base_lr:.2e})")
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

        # Regression note: this was hardcoded to 8, silently ignoring the
        # n_envs parameter train() actually accepts (default 4) -- found
        # while reducing parallelism after ppo_lstm_v6_time_features was
        # killed under sustained near-zero free memory. Now respects the
        # parameter so a caller can actually control it.
        env_fns = [make_env(**env_kwargs) for _ in range(n_envs)]
        train_env = SubprocVecEnv(env_fns)

        # Phase 2R fix (2): eval across ALL dates (training + held-out crash),
        # not the single trending-up date multi_dates[0]. See
        # make_multi_date_eval_env for the rationale and the held-out-in-selection
        # caveat.
        if mode == "counterfactual" and multi_dates:
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
            jump_diffusion=jump_diffusion, no_time_features=no_time_features,
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
        # Applies whether the model was just constructed or resumed from a
        # checkpoint -- load_model(device="mps") loads the LSTM weights onto
        # MPS in plain nn.LSTM form (unwrapped keys), so this patch step is
        # still needed after loading, same as after fresh construction.
        if device == "mps":
            _patch_mps_lstm(model.policy)

        # eval enough episodes that every date gets a fair sample (>=5 each).
        n_eval_episodes = max(50, len(eval_dates) * 5)
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
    p.add_argument("--jump-diffusion", action="store_true",
                   help="Enable Poisson jump diffusion in GARCH price model.")
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
        n_envs=args.n_envs,
        resume_from=args.resume_from,
        resume_mlflow_run_id=args.resume_mlflow_run_id,
        force_device=args.device,
        base_lr=args.base_lr,
    )