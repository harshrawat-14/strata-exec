"""
Tests for Phase 3's mlflow integration (log_eval_results_to_mlflow).

Regression context: train.py has mlflow.start_run/log_params/log_artifact
but zero log_metric calls, and evaluate.py had no mlflow integration at
all until this function existed -- meaning a training run's final eval
numbers (mean IS, CVaR-95 per date, the numbers that actually matter) were
never attached to that run's tracked history, and
experiment_tracker.py's get_best_run() always returned {} as a direct
consequence.

Uses a real local mlflow tracking store in a temp directory (not a mock)
so this actually exercises mlflow's start_run(run_id=...) reopen semantics,
not an assumption about how the mlflow client behaves.
"""
from __future__ import annotations

import tempfile

import mlflow
import pytest

from evaluate import log_eval_results_to_mlflow


@pytest.fixture
def temp_mlflow_tracking():
    with tempfile.TemporaryDirectory() as tmpdir:
        original_uri = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(f"file://{tmpdir}/mlruns")
        mlflow.set_experiment("test_experiment")
        yield
        mlflow.set_tracking_uri(original_uri)


def _make_unified_results():
    return {
        "2024-01-15": {
            "rl_agent": {
                "mean_IS": -0.53,
                "cvar_95": -0.54,
                "cvar_95_ci_lower": -0.55,
                "cvar_95_ci_upper": -0.53,
                "forced_liquidation_rate": 0.0,
            },
            "twap": {
                "mean_IS": -0.44,
                "cvar_95": -0.45,
                "forced_liquidation_rate": 0.0,
            },
        },
        "2024-06-10": {
            "rl_agent": {
                "mean_IS": -0.56,
                "cvar_95": -0.57,
                "cvar_95_ci_lower": -0.58,
                "cvar_95_ci_upper": -0.56,
                "forced_liquidation_rate": 0.02,
            },
        },
    }


class TestLogEvalResultsToMlflow:
    def test_logs_metrics_to_an_existing_run_not_a_new_one(self, temp_mlflow_tracking):
        with mlflow.start_run(run_name="pre_existing_run") as run:
            run_id = run.info.run_id
            mlflow.log_param("mode", "counterfactual")

        run_count_before = len(mlflow.search_runs(search_all_experiments=True))

        log_eval_results_to_mlflow(run_id, _make_unified_results())

        run_count_after = len(mlflow.search_runs(search_all_experiments=True))
        assert run_count_after == run_count_before, (
            "must reopen the existing run, not create a new one"
        )

    def test_logs_per_date_metrics_for_default_strategy(self, temp_mlflow_tracking):
        with mlflow.start_run(run_name="run_a") as run:
            run_id = run.info.run_id

        log_eval_results_to_mlflow(run_id, _make_unified_results())

        client = mlflow.tracking.MlflowClient()
        logged_run = client.get_run(run_id)
        metrics = logged_run.data.metrics

        assert metrics["eval_mean_is_20240115_rl_agent"] == -0.53
        assert metrics["eval_cvar95_20240115_rl_agent"] == -0.54
        assert metrics["eval_cvar95_ci_lower_20240115_rl_agent"] == -0.55
        assert metrics["eval_mean_is_20240610_rl_agent"] == -0.56

    def test_default_strategy_is_rl_agent_only(self, temp_mlflow_tracking):
        with mlflow.start_run(run_name="run_b") as run:
            run_id = run.info.run_id

        log_eval_results_to_mlflow(run_id, _make_unified_results())

        client = mlflow.tracking.MlflowClient()
        metrics = client.get_run(run_id).data.metrics
        # twap wasn't requested (default strategies=["rl_agent"]) -- must
        # not be logged.
        assert "eval_mean_is_20240115_twap" not in metrics

    def test_logs_requested_strategies(self, temp_mlflow_tracking):
        with mlflow.start_run(run_name="run_c") as run:
            run_id = run.info.run_id

        log_eval_results_to_mlflow(run_id, _make_unified_results(), strategies=["rl_agent", "twap"])

        client = mlflow.tracking.MlflowClient()
        metrics = client.get_run(run_id).data.metrics
        assert "eval_mean_is_20240115_twap" in metrics
        assert metrics["eval_mean_is_20240115_twap"] == -0.44

    def test_logs_aggregate_summary_across_dates(self, temp_mlflow_tracking):
        with mlflow.start_run(run_name="run_d") as run:
            run_id = run.info.run_id

        log_eval_results_to_mlflow(run_id, _make_unified_results())

        client = mlflow.tracking.MlflowClient()
        metrics = client.get_run(run_id).data.metrics
        expected_avg = (-0.53 + -0.56) / 2
        assert metrics["eval_mean_is_avg_rl_agent"] == pytest.approx(expected_avg)
