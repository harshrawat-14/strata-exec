"""
Evaluation protocol v3 (adopted 2026-10-18; docs/methodology/evaluation-protocol-v3.md).

Every episode is EPISODE_STEPS long (16 hours of 30-second snapshots) and
starts anywhere in the first part of the day that still leaves that much
data, so an episode always ends on real order-book data. Evaluation spreads
N_EVAL_EPISODES starts evenly across that window instead of the legacy
protocol's 20 near-identical starts in the first 9.5 minutes.

Model selection uses VALIDATION_DATES_V3; the held-out dates are evaluated
once, at the final comparison.
"""

EPISODE_STEPS = 1920
N_EVAL_EPISODES = 20


def eval_start_fracs(n_episodes: int = N_EVAL_EPISODES) -> list:
    """Evenly spaced start positions in [0, 1] across the valid start window."""
    if n_episodes == 1:
        return [0.0]
    return [i / (n_episodes - 1) for i in range(n_episodes)]


def fit_start_fracs(n_fit: int) -> list:
    """Start positions for fitting a baseline: midway between eval positions'
    spread, so fit and eval episodes cover the same window without coinciding."""
    return [(i + 0.5) / n_fit for i in range(n_fit)]


# Honest risk aversion under v3, selected on TRAINING_DATES_V3 only
# (rl/experiments/v3_lambda_selection.py -> rl/results/v3_lambda_grid.csv).
AC_LAMBDA = 1e-6
ADAPTIVE_AC_LAMBDA = 1e-9
