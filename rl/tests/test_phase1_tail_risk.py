"""
Phase 1 tests: the AC risk-aversion (lambda) grid sweep.

Integration-level (spawns the real rl-env subprocess against real
2024-01-15 data), kept fast via a small n_episodes and a tiny lambda grid.
"""
from __future__ import annotations

import os

import pytest

from evaluate import evaluate_ac_lambda_grid, AC_LAMBDA_GRID, AC_DEFAULT_LAMBDA

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HAS_REAL_DATA = os.path.exists(
    os.path.join(REPO_ROOT, "target", "release", "rl-env")
) and os.path.exists(
    os.path.join(REPO_ROOT, "TradeData", "BookDepth", "BTCUSDT-bookDepth-2024-01-15.csv")
)

requires_real_env = pytest.mark.skipif(
    not _HAS_REAL_DATA,
    reason="requires target/release/rl-env binary and TradeData/ fixtures",
)


def test_ac_default_lambda_is_in_the_default_grid():
    # The whole point of the grid is to show where AC_DEFAULT_LAMBDA sits
    # relative to the sweep -- it must actually be one of the sweep points,
    # not just nearby.
    assert AC_DEFAULT_LAMBDA in AC_LAMBDA_GRID


@requires_real_env
def test_ac_lambda_grid_returns_both_ac_family_policies_per_lambda():
    lambdas = [1e-5, 1e-2]
    results = evaluate_ac_lambda_grid(
        date="2024-01-15", lambdas=lambdas, n_episodes=3,
    )
    assert set(results.keys()) == set(lambdas)
    for lam in lambdas:
        assert set(results[lam].keys()) == {"ac_optimal", "adaptive_ac"}
        for strategy_key in ("ac_optimal", "adaptive_ac"):
            r = results[lam][strategy_key]
            assert r["lambda"] == lam
            assert r["n_episodes"] == 3
            assert "cvar_95_ci_lower" in r and "cvar_95_ci_upper" in r


@requires_real_env
def test_higher_lambda_front_loads_execution_and_worsens_mean_is():
    # AC theory: higher risk-aversion -> more urgent (front-loaded)
    # execution -> more market impact cost -> worse (more negative) mean IS,
    # in exchange for less market-timing risk. A very patient (low lambda)
    # vs very urgent (high lambda) comparison on the same episodes should
    # show this direction, not just "some numbers came out."
    results = evaluate_ac_lambda_grid(
        date="2024-01-15", lambdas=[1e-6, 1e-1], n_episodes=10,
    )
    patient_is = results[1e-6]["ac_optimal"]["mean_IS"]
    urgent_is = results[1e-1]["ac_optimal"]["mean_IS"]
    assert urgent_is < patient_is, (
        f"expected urgent (lambda=0.1) mean IS ({urgent_is}) to be worse "
        f"(more negative) than patient (lambda=1e-6) mean IS ({patient_is})"
    )
