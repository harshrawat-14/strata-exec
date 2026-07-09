"""
rl/experiment_tracker.py

Thin query layer over MLflow's tracking store for comparing training runs
logged by rl/train.py (see mlflow.start_run in train()). Read-only — does
not start or modify runs.
"""

from __future__ import annotations

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient


def get_best_run(metric: str, higher_is_better: bool = True, experiment_name: str = "Default") -> dict:
    """
    Return the run with the best value of `metric` across all logged runs.

    Args:
        metric: metric name as logged via mlflow.log_metric (e.g. "mean_IS").
        higher_is_better: True to take the max, False to take the min.
        experiment_name: MLflow experiment to search within.

    Returns:
        dict with run_id, params, metrics, and artifact_uri for the best run.
        Empty dict if no runs with that metric exist.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return {}

    order = "DESC" if higher_is_better else "ASC"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.`{metric}` {order}"],
        max_results=1,
    )
    if not runs:
        return {}

    best = runs[0]
    return {
        "run_id": best.info.run_id,
        "run_name": best.data.tags.get("mlflow.runName"),
        "params": dict(best.data.params),
        "metrics": dict(best.data.metrics),
        "artifact_uri": best.info.artifact_uri,
    }


def compare_runs_table(experiment_name: str = "Default") -> pd.DataFrame:
    """
    Return a DataFrame with one row per run: run_name, key hyperparameters,
    and logged metrics, sorted by start time descending (most recent first).
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return pd.DataFrame()

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attribute.start_time DESC"],
    )

    rows = []
    for run in runs:
        row = {
            "run_name": run.data.tags.get("mlflow.runName"),
            "run_id": run.info.run_id,
            "status": run.info.status,
        }
        row.update(run.data.params)
        row.update(run.data.metrics)
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--metric", type=str, default="mean_IS")
    p.add_argument("--lower-is-better", action="store_true")
    args = p.parse_args()

    print(compare_runs_table().to_string(index=False))
    best = get_best_run(args.metric, higher_is_better=not args.lower_is_better)
    print(f"\nBest run by {args.metric}: {best}")
