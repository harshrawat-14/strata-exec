"""
Unit and regression tests for rl/evaluate.py's metrics and unified
comparison harness (Phase 0 / Phase 1 of the StrataExec measurement work).
"""
from __future__ import annotations

import numpy as np
import pytest

from evaluate import (
    compute_cvar_95,
    ACOptimalPolicy,
    AdaptiveACPolicy,
    AC_DEFAULT_LAMBDA,
)


class TestComputeCvar95:
    """
    Regression coverage for the CVaR sign/tail bug: several call sites in
    this file used to report np.percentile(arr, 5) (a single point, VaR-5)
    mislabeled as "cvar95", and the Rust/parallel_sim.py siblings of this
    metric had an outright sign inversion (averaging the best tail instead
    of the worst). compute_cvar_95 is the one implementation everything
    should share; these tests hand-construct distributions where the true
    worst 5% is unambiguous by eye.
    """

    def test_worst_tail_not_best_tail(self):
        # 95 normal days at a mild -0.5% shortfall, 5 crash days at
        # catastrophic shortfalls. True worst 5% = {-50,-40,-30,-20,-10},
        # mean = -30.0. A best-tail bug would instead return -0.5.
        arr = np.array([-0.5] * 95 + [-50.0, -40.0, -30.0, -20.0, -10.0])
        result = compute_cvar_95(arr)
        assert abs(result - (-30.0)) < 1e-9, result
        assert result < -0.5, "must be worse than the normal-day mean"

    def test_is_a_tail_mean_not_a_single_percentile(self):
        # 100 samples: one true outlier (-100.0), rest clustered at -1.0.
        # VaR-5 (np.percentile(arr, 5)) on this array is ~-1.0 (the outlier
        # is only 1/100 of the mass, below the 5th percentile threshold),
        # so a VaR-5-mislabeled-as-CVaR95 implementation would report -1.0
        # here even though a true 5-episode-wide tail mean must include the
        # -100 outlier and come out well below -1.0.
        arr = np.array([-1.0] * 99 + [-100.0])
        result = compute_cvar_95(arr)
        var_5 = float(np.percentile(arr, 5))
        assert result < var_5 - 1.0, (
            f"cvar_95 ({result}) should be materially worse than VaR-5 "
            f"({var_5}) once the tail mean captures the outlier"
        )

    def test_empty_array_returns_nan(self):
        assert np.isnan(compute_cvar_95(np.array([])))

    def test_single_sample(self):
        assert compute_cvar_95(np.array([-3.0])) == -3.0


class TestAcLambdaOverride:
    """
    Regression coverage for the AC risk-aversion config gap: ACOptimalPolicy
    and AdaptiveACPolicy previously hardcoded lam=0.001 with no way for a
    caller to override it (Rust research-sim's own AC default is 1e-4, an
    order of magnitude different, and neither reconciled with the other).
    A baseline whose risk-aversion can't be set is either an invisible
    strawman or an invisible wall -- these tests confirm the override
    actually reaches the schedule, not just the constructor argument.
    """

    def test_ac_optimal_default_matches_documented_constant(self):
        assert ACOptimalPolicy().kappa == pytest.approx(
            ACOptimalPolicy(lam=AC_DEFAULT_LAMBDA).kappa
        )

    def test_ac_optimal_lambda_override_changes_schedule(self):
        # Higher lambda (more risk-averse) should front-load execution:
        # kappa increases with lambda, which sharpens the sinh() decay and
        # shifts probability mass toward earlier steps.
        low = ACOptimalPolicy(lam=1e-6)
        high = ACOptimalPolicy(lam=0.1)
        assert low.kappa != high.kappa
        # Front-loading: the high-lambda schedule should sell a larger
        # fraction of inventory in the first 10% of steps than the
        # low-lambda (patient) schedule.
        n = len(low._schedule)
        head = max(1, n // 10)
        assert high._schedule[:head].sum() > low._schedule[:head].sum()

    def test_adaptive_ac_lambda_override_reaches_schedule_builder(self):
        low = AdaptiveACPolicy(lam=1e-6)
        high = AdaptiveACPolicy(lam=0.1)
        sched_low = low._get_schedule(0.20)
        sched_high = high._get_schedule(0.20)
        assert not np.allclose(sched_low, sched_high)
