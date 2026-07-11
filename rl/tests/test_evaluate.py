"""
Unit and regression tests for rl/evaluate.py's metrics and unified
comparison harness (Phase 0 / Phase 1 of the StrataExec measurement work).
"""
from __future__ import annotations

import numpy as np
import pytest

from evaluate import (
    compute_cvar_95,
    bootstrap_cvar_95_ci,
    ACOptimalPolicy,
    AdaptiveACPolicy,
    AC_DEFAULT_LAMBDA,
    _stable_sinh_ratio,
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


class TestBootstrapCvar95Ci:
    """
    Global Rule 4: CVaR-95 must carry its own uncertainty, not be reported as
    a clean point estimate. On a small sample the tail is only a couple of
    points, so the CI should be visibly wide -- these tests check the CI
    behaves like a real confidence interval (contains the point estimate,
    narrows with more data, is reproducible for a fixed seed) rather than
    just existing as a field.
    """

    def test_ci_contains_point_estimate(self):
        rng = np.random.default_rng(1)
        arr = rng.normal(loc=-0.5, scale=0.1, size=50)
        result = bootstrap_cvar_95_ci(arr, n_bootstrap=2000, seed=1)
        assert result["ci_lower"] <= result["point"] <= result["ci_upper"]

    def test_ci_narrows_with_more_episodes(self):
        # Same underlying distribution, more samples -> tighter CI. This is
        # the concrete version of Global Rule 4's "if the episode count makes
        # the tail estimate unstable, raise it" guidance -- show it actually
        # helps, not just assert it should.
        rng = np.random.default_rng(2)
        small = rng.normal(loc=-0.5, scale=0.1, size=30)
        large = rng.normal(loc=-0.5, scale=0.1, size=500)
        small_result = bootstrap_cvar_95_ci(small, n_bootstrap=3000, seed=2)
        large_result = bootstrap_cvar_95_ci(large, n_bootstrap=3000, seed=2)
        assert large_result["width"] < small_result["width"], (
            f"500-episode CI ({large_result['width']:.4f}) should be narrower "
            f"than 30-episode CI ({small_result['width']:.4f})"
        )

    def test_reproducible_for_fixed_seed(self):
        arr = np.array([-0.5, -0.4, -0.6, -5.0, -0.55, -0.45] * 10)
        r1 = bootstrap_cvar_95_ci(arr, n_bootstrap=1000, seed=42)
        r2 = bootstrap_cvar_95_ci(arr, n_bootstrap=1000, seed=42)
        assert r1 == r2

    def test_empty_array(self):
        result = bootstrap_cvar_95_ci(np.array([]))
        assert np.isnan(result["point"])
        assert result["n_episodes"] == 0

    def test_n_tail_matches_compute_cvar_95(self):
        # n_tail should reflect the same tail-size rule compute_cvar_95 uses
        # internally (n // 20), so a reader can see how few points the point
        # estimate is actually built from.
        arr = np.zeros(50)
        result = bootstrap_cvar_95_ci(arr, n_bootstrap=100)
        assert result["n_tail"] == 2  # 50 // 20 = 2


class TestStableSinhRatio:
    """
    Regression coverage for a real numerical bug found via Phase 1's AC
    lambda-grid sweep: at high lambda combined with AdaptiveACPolicy's
    live-volatility clamp (current_sigma up to 2.0 in predict()), kappa*T
    reached into the thousands, sinh() overflowed to inf, and inf/inf
    produced NaN -- silently corrupting the AC schedule. The bug was easy to
    miss: the trajectory-level "higher lambda -> worse mean IS" assertion
    still passed on average despite the warnings, exactly the "config that
    hides the interesting failure" pattern (a passing test isn't evidence of
    correctness if it doesn't check the thing that can go wrong).
    """

    def test_matches_direct_sinh_for_moderate_kappa_t(self):
        # Sanity: for kappa*T well within float64's safe range, the stable
        # rewrite must agree with the naive formula to high precision.
        kappa, T = 0.01, 100
        t = np.arange(T)
        naive = np.sinh(kappa * (T - t)) / (np.sinh(kappa * T) + 1e-10)
        stable = _stable_sinh_ratio(kappa, T, t)
        assert np.allclose(naive, stable, atol=1e-9)

    def test_no_overflow_or_nan_at_extreme_kappa_t(self):
        # The exact scenario that triggered the bug: lambda=0.1, sigma
        # clamped to 2.0, eta=0.10 -> kappa=2.0, T=2880 -> kappa*T=5760,
        # far beyond where naive sinh(kappa*T) overflows to inf.
        kappa, T = 2.0, 2880
        t = np.arange(T)
        with np.errstate(over="raise", invalid="raise"):
            result = _stable_sinh_ratio(kappa, T, t)
        assert np.all(np.isfinite(result)), "must never produce inf/NaN"
        assert np.all(result >= 0.0) and np.all(result <= 1.0 + 1e-9), (
            "sinh(kappa*(T-t))/sinh(kappa*T) is a ratio of same-signed "
            "quantities and must stay in [0, 1]"
        )
        # At t=0 the ratio must be ~1.0 (full remaining inventory); it must
        # decay monotonically to ~0 as t -> T.
        assert result[0] == pytest.approx(1.0, abs=1e-6)
        assert result[-1] < 1e-6
        assert np.all(np.diff(result) <= 1e-12), "must be monotonically non-increasing"

    def test_ac_optimal_schedule_is_finite_at_extreme_lambda(self):
        # End-to-end: constructing ACOptimalPolicy at the lambda that used
        # to trigger the overflow must produce a finite, valid schedule.
        policy = ACOptimalPolicy(lam=0.1, sigma=2.0, eta=0.10, total_steps=2880)
        assert np.all(np.isfinite(policy._schedule))
        assert policy._schedule.sum() == pytest.approx(1.0, abs=1e-6)

    def test_adaptive_ac_schedule_is_finite_at_extreme_sigma(self):
        policy = AdaptiveACPolicy(lam=0.1, sigma_ref=0.20, eta=0.10, total_steps=2880)
        schedule = policy._get_schedule(2.0)  # the clamp's max current_sigma
        assert np.all(np.isfinite(schedule))
        assert schedule.sum() == pytest.approx(1.0, abs=1e-6)


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
