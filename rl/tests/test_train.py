"""
Tests for rl/train.py -- previously zero test coverage (Phase -1 audit
finding), which is directly why three real bugs sat undetected: train.py
could not even be imported (a module-level assertion on
cosine_with_restarts failed with its original 0.67/0.34 test inputs);
separately could not construct an environment (no_regime_impact passed to
StrataExecEnv.__init__, which never accepted it -- dead scaffolding from a
never-implemented feature, removed in the same Phase 2 commit that fixed
this file's importability); and get_lr_schedule was called but never
defined anywhere in the file (a bare NameError on the first line of
train() that needed a learning rate).
"""
from __future__ import annotations

import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import (  # noqa: E402
    cosine_with_restarts,
    get_lr_schedule,
    make_env,
    train,
    MLflowMetricsCallback,
    MemoryLoggingCallback,
)


class TestCosineWithRestarts:
    """
    Regression coverage for the boundary bug: the function itself is
    correct (matches its docstring), but naive test inputs landing
    fractionally on the wrong side of a cycle boundary give a wildly wrong
    answer (~0 instead of the cycle's peak) -- easy to get wrong again if
    this schedule is ever tuned, since the failure is silent (a bad
    learning-rate multiplier, not a crash) unless checked explicitly.
    """

    def test_start_is_full_peak(self):
        assert abs(cosine_with_restarts(1.0) - 1.0) < 0.01

    def test_end_is_near_zero(self):
        assert cosine_with_restarts(0.0) < 0.01

    def test_cycle_1_peak(self):
        # progress_remaining=0.66 -> progress=0.34, safely inside cycle 1
        # (33%-66% elapsed), near its start where the peak (0.5) occurs.
        assert abs(cosine_with_restarts(0.66) - 0.5) < 0.05

    def test_cycle_2_peak(self):
        # progress_remaining=0.32 -> progress=0.68, safely inside cycle 2
        # (66%-100% elapsed), near its start where the peak (0.25) occurs.
        assert abs(cosine_with_restarts(0.32) - 0.25) < 0.05

    def test_values_just_before_a_boundary_are_near_zero_not_the_next_peak(self):
        # The exact bug: progress_remaining=0.67 (progress=0.33, a hair
        # before the exact 1/3 boundary) is still the TAIL of cycle 0, where
        # cosine decay has nearly reached zero -- not the start of cycle 1's
        # peak. This is the value the original (broken) assertion used.
        assert cosine_with_restarts(0.67) < 0.01
        assert cosine_with_restarts(0.34) < 0.01

    def test_monotonic_decay_within_a_cycle(self):
        # Within cycle 0 (progress 0 -> 1/3), the multiplier should decay
        # monotonically from 1.0 toward 0.
        progresses_remaining = [1.0 - p for p in [0.0, 0.1, 0.2, 0.3]]
        values = [cosine_with_restarts(pr) for pr in progresses_remaining]
        assert values == sorted(values, reverse=True)


class TestMakeEnvDoesNotRaise:
    """
    Regression coverage for the no_regime_impact TypeError: make_env's
    returned initializer must actually be callable without raising, for
    every mode this project trains in. This doesn't spin up a real env for
    synthetic mode's random-choice path issues; it targets exactly the
    kwarg-mismatch class of bug (StrataExecEnv.__init__ silently gaining/
    losing a parameter that make_env still passes).
    """

    def test_make_env_kwargs_match_strataexecenv_signature(self):
        import inspect
        from environment import StrataExecEnv

        valid_params = set(inspect.signature(StrataExecEnv.__init__).parameters) - {"self"}

        # The exact kwargs train.py's make_env() passes through to
        # StrataExecEnv (mirrors env_kwargs construction in train()).
        passed_kwargs = {
            "mode", "adversarial", "lob_date", "agg_date", "btc_target",
            "n_state_dims", "seed", "fixed_steps", "fixed_size",
            "jump_diffusion", "no_time_features",
        }
        unknown = passed_kwargs - valid_params
        assert not unknown, (
            f"make_env passes kwargs StrataExecEnv.__init__ does not accept: "
            f"{unknown} -- this is exactly the class of bug that made "
            f"train.py unable to construct any environment (no_regime_impact)"
        )

    def test_make_env_returns_callable_initializer(self):
        init_fn = make_env(mode="synthetic", n_state_dims=8)
        assert callable(init_fn)


class TestGetLrSchedule:
    """
    Regression coverage: get_lr_schedule was called in train() but never
    defined anywhere in this file -- a bare NameError on the first training
    run. Covers all three --lr-schedule CLI choices.
    """

    def test_constant_returns_the_base_lr_unchanged(self):
        result = get_lr_schedule("constant", base_lr=3e-4, total_timesteps=1000)
        assert result == 3e-4

    def test_linear_is_callable_and_decays_to_zero(self):
        schedule = get_lr_schedule("linear", base_lr=3e-4, total_timesteps=1000)
        assert callable(schedule)
        assert schedule(1.0) == 3e-4
        assert schedule(0.0) == 0.0
        assert schedule(0.5) == 3e-4 * 0.5

    def test_cosine_restarts_is_callable_and_scaled_by_base_lr(self):
        schedule = get_lr_schedule("cosine_restarts", base_lr=3e-4, total_timesteps=1000)
        assert callable(schedule)
        assert abs(schedule(1.0) - 3e-4) < 1e-6
        assert schedule(1.0) > schedule(0.0)

    def test_unknown_schedule_raises(self):
        import pytest
        with pytest.raises(ValueError):
            get_lr_schedule("not-a-real-schedule", base_lr=3e-4, total_timesteps=1000)


def test_train_module_level_import_succeeds():
    # The module-level cosine_with_restarts self-test assertions run at
    # import time -- this test exists purely so "import train" itself is
    # covered by the test suite and a future regression fails loudly in CI,
    # not silently at the next person's first real training run.
    import importlib
    import train as train_module
    importlib.reload(train_module)
    assert hasattr(train_module, "train")


class TestMLflowMetricsCallback:
    """
    Regression coverage for the Phase 3 gap: train.py had mlflow.start_run/
    log_params/log_artifact but ZERO log_metric calls anywhere, so no
    training run's loss curves/entropy/episode-length ever reached mlflow.
    Tests the callback in isolation (bypassing full SB3 lifecycle, which
    needs a real env) against a real temp mlflow tracking store.
    """

    def test_on_rollout_end_logs_numeric_metrics_to_mlflow(self, tmp_path):
        import mlflow

        class _FakeLogger:
            name_to_value = {
                "rollout/ep_len_mean": 42.0,
                "train/entropy_loss": -1.5,
                "train/some_string_value": "not-a-number",  # must be skipped
            }

        class _FakeModel:
            logger = _FakeLogger()

        original_uri = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
        mlflow.set_experiment("test_callback")
        try:
            with mlflow.start_run(run_name="callback_test") as run:
                callback = MLflowMetricsCallback()
                callback.model = _FakeModel()
                callback.num_timesteps = 1000
                callback._on_rollout_end()

            client = mlflow.tracking.MlflowClient()
            metrics = client.get_run(run.info.run_id).data.metrics
            assert metrics["rollout_ep_len_mean"] == 42.0
            assert metrics["train_entropy_loss"] == -1.5
            assert "train_some_string_value" not in metrics
        finally:
            mlflow.set_tracking_uri(original_uri)


class TestMemoryLoggingCallback:
    """
    Regression coverage for the memory-visibility gap found during the
    ppo_lstm_v6_time_features incident: the training loop had no memory
    telemetry at all, so a multi-hour run being killed by something
    external (most likely, though not confirmed, macOS's low-memory
    killer, given free system memory sat near zero for most of the run)
    left nothing in the logs to diagnose it with after the fact -- only ad
    hoc `ps`/`vm_stat` snapshots taken manually during the incident.
    """

    def test_on_rollout_end_logs_main_and_children_rss(self, tmp_path):
        import mlflow

        original_uri = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
        mlflow.set_experiment("test_memory_callback")
        try:
            with mlflow.start_run(run_name="memory_test") as run:
                callback = MemoryLoggingCallback()
                callback._on_training_start()  # sets up psutil.Process()
                callback.num_timesteps = 5000
                callback._on_rollout_end()

            client = mlflow.tracking.MlflowClient()
            metrics = client.get_run(run.info.run_id).data.metrics
            # This process is genuinely running, so main RSS must be a real,
            # positive number -- not asserting an exact value (that would be
            # flaky), just that real data landed.
            assert metrics["mem_main_rss_mb"] > 0
            assert metrics["mem_children_rss_mb"] >= 0
            assert metrics["mem_total_rss_mb"] == pytest.approx(
                metrics["mem_main_rss_mb"] + metrics["mem_children_rss_mb"]
            )
        finally:
            mlflow.set_tracking_uri(original_uri)
