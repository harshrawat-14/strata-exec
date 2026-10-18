"""
SB3 training callbacks: MLflow metrics, memory, adaptive entropy, resume-safe eval.
"""

import os

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
import mlflow


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
        # Protocol v3: rewind each eval env's start cycle before an evaluation
        # round, so every round scores checkpoints on the identical episodes
        # (SB3's evaluate_policy does an extra reset per round, which would
        # otherwise shift the cycle by one position each time).
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            for env in getattr(self.eval_env, "envs", []):
                inner = getattr(env, "unwrapped", env)
                if hasattr(inner, "restart_start_cycle"):
                    inner.restart_start_cycle()
        prev_best = self.best_mean_reward
        result = super()._on_step()
        if (self._best_reward_path is not None
                and self.best_mean_reward > prev_best):
            import json
            with open(self._best_reward_path, "w") as f:
                json.dump({"best_mean_reward": self.best_mean_reward}, f)
        return result
