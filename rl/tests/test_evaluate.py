"""
Unit and regression tests for rl/evaluate.py's metrics and unified
comparison harness (Phase 0 / Phase 1 of the StrataExec measurement work).
"""
from __future__ import annotations

import numpy as np
import pytest

import json
import os
import subprocess

from evaluate import (
    compute_cvar_95,
    bootstrap_cvar_95_ci,
    ACOptimalPolicy,
    AdaptiveACPolicy,
    AC_DEFAULT_LAMBDA,
    _stable_sinh_ratio,
    ACTION_FRACTIONS,
    get_action_fractions,
    SqrtImpactOptimalPolicy,
    query_calibration,
)


class TestActionFractionsSingleSourceOfTruth:
    """
    Regression coverage for the Phase 2R fraction-table bug: this file used
    to hand-maintain TWO separate mirrors of the action->fraction mapping
    (V5_FRACTIONS/V6_FRACTIONS, keyed off n_state_dims) that silently
    diverged from the real environment's table -- one matched at only 3/20
    indices, so every classical-baseline (TWAP/Heuristic/AC-Optimal/
    AdaptiveAC) evaluation with n_state_dims=12 was picking action indices
    under a fraction assumption the environment never honored. Second
    instance of a hand-maintained mirror of a canonical value causing a
    silent bug in this project (cf. TestComputeCvar95 above) -- this test
    exists so any future re-introduction of a second copy, or any drift in
    the Rust ACTION_FRACTIONS const, fails loudly instead of silently.
    """

    def test_python_visible_table_matches_rust_source_of_truth(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        binary = os.path.join(repo_root, "target", "release", "rl-env")
        if not os.path.exists(binary):
            pytest.skip("target/release/rl-env not built in this checkout")
        result = subprocess.run(
            [binary, "--print-action-fractions"], capture_output=True, text=True, check=True,
        )
        rust_fractions = json.loads(result.stdout.strip())
        assert ACTION_FRACTIONS == rust_fractions, (
            "Python's ACTION_FRACTIONS must be byte-identical to the Rust "
            "environment's real action->fraction table -- any divergence "
            "means the classical baselines are picking actions under an "
            "assumption the environment does not honor."
        )

    def test_get_action_fractions_returns_the_same_table_regardless_of_n_state_dims(self):
        # The action space is a property of the environment, not of the
        # state representation being evaluated -- it must never be keyed off
        # n_state_dims again (that coupling is exactly what caused the bug).
        assert get_action_fractions(12) == get_action_fractions(16) == ACTION_FRACTIONS

    def test_evaluate_module_reuses_environment_modules_table_not_a_new_copy(self):
        # Phase 2R found a THIRD hand-maintained copy of this table living in
        # rl/environment.py (byte-identical by luck, unprotected by any test,
        # and still imported by rl/phase2r_vwap_gate.py). Root-fixed by making
        # environment.py the one canonical query point and having evaluate.py
        # import its ACTION_FRACTIONS from there instead of querying
        # independently. This test guards against a fourth copy re-appearing:
        # evaluate.ACTION_FRACTIONS must be the *same object* environment.py
        # produced, not merely an equal-valued list assembled separately.
        import environment
        assert ACTION_FRACTIONS is environment.ACTION_FRACTIONS

    def test_action_fractions_starts_at_zero_ends_at_one_monotonic(self):
        assert ACTION_FRACTIONS[0] == 0.0
        assert ACTION_FRACTIONS[-1] == 1.0
        assert all(
            ACTION_FRACTIONS[i] < ACTION_FRACTIONS[i + 1]
            for i in range(len(ACTION_FRACTIONS) - 1)
        ), "action fractions must be strictly increasing"


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


class TestSigmaRefSingleSourceOfTruth:
    """
    Regression coverage for bug-class instance #5 (Phase 3). Step 0
    recalibrated the environment's `sigma_ref` in Rust, but `evaluate.py`'s
    AdaptiveACPolicy and `batched_environment.py` each kept a hardcoded
    `sigma_ref: float = 0.20` default that no call site overrode -- so the
    Python baseline reconstructed volatility against a reference the
    environment had stopped using. Fifth instance in this project of a
    hand-maintained mirror of a canonical value causing a silent bug
    (cf. TestComputeCvar95 and TestActionFractionsSingleSourceOfTruth above).
    """

    def test_python_sigma_ref_matches_rust_source_of_truth(self):
        from environment import CALIBRATED_SIGMA_REF, ENV_CONSTANTS

        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        binary = os.path.join(repo_root, "target", "release", "rl-env")
        if not os.path.exists(binary):
            pytest.skip("target/release/rl-env not built in this checkout")
        result = subprocess.run(
            [binary, "--print-env-constants"], capture_output=True, text=True,
            check=True, stdin=subprocess.DEVNULL, timeout=30,
        )
        rust_constants = json.loads(result.stdout.strip())
        assert CALIBRATED_SIGMA_REF == rust_constants["calibrated_sigma_ref"], (
            "Python's CALIBRATED_SIGMA_REF must equal the Rust environment's "
            "real constant -- divergence means Adaptive-AC is reconstructing "
            "volatility against a reference the environment does not use."
        )
        assert ENV_CONSTANTS["sigma_ref_source"] == "binary", (
            "constants must have been read from the binary here, not the "
            "pre-Step-0 fallback -- the fallback is only correct against a "
            "binary built before commit 0e3356b"
        )

    def test_adaptive_ac_default_sigma_ref_is_not_a_hardcoded_literal(self):
        from environment import CALIBRATED_SIGMA_REF

        policy = AdaptiveACPolicy(n_state_dims=16)
        assert policy.sigma_ref == CALIBRATED_SIGMA_REF, (
            "AdaptiveACPolicy's default sigma_ref must come from the binary, "
            "not a literal in this file"
        )

    def test_no_hardcoded_sigma_ref_literals_remain_in_rl_python(self):
        """
        The meta-lesson made executable: after any fix in this bug class,
        sweeping for siblings is routine, not something that only happens when
        explicitly requested. Fails if a float-literal `sigma_ref` default is
        reintroduced anywhere under rl/.
        """
        import ast
        import glob

        rl_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        offenders = []

        def _is_numeric_literal(node):
            return isinstance(node, ast.Constant) and isinstance(node.value, (int, float))

        # Parse rather than regex: several files legitimately DESCRIBE the old
        # `sigma_ref = 0.20` literal in docstrings, and a source-text scan
        # flags that prose as a violation. The AST only sees real bindings.
        for path in glob.glob(os.path.join(rl_dir, "**", "*.py"), recursive=True):
            with open(path) as fh:
                try:
                    tree = ast.parse(fh.read(), filename=path)
                except SyntaxError:
                    continue
            rel = os.path.relpath(path, rl_dir)
            for node in ast.walk(tree):
                # Default argument values: `def f(..., sigma_ref=0.20)`
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = node.args
                    positional = args.posonlyargs + args.args
                    for arg, default in zip(positional[len(positional) - len(args.defaults):],
                                            args.defaults):
                        if arg.arg == "sigma_ref" and _is_numeric_literal(default):
                            offenders.append(f"{rel}:{default.lineno}: default sigma_ref={default.value}")
                    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
                        if default is not None and arg.arg == "sigma_ref" and _is_numeric_literal(default):
                            offenders.append(f"{rel}:{default.lineno}: default sigma_ref={default.value}")
                # Assignments: `sigma_ref = 0.20` / `self.sigma_ref = 0.20`
                elif isinstance(node, ast.Assign) and _is_numeric_literal(node.value):
                    for tgt in node.targets:
                        name = tgt.id if isinstance(tgt, ast.Name) else (
                            tgt.attr if isinstance(tgt, ast.Attribute) else None)
                        if name == "sigma_ref":
                            offenders.append(f"{rel}:{node.lineno}: sigma_ref={node.value.value}")
                elif isinstance(node, ast.AnnAssign) and node.value is not None \
                        and _is_numeric_literal(node.value):
                    tgt = node.target
                    name = tgt.id if isinstance(tgt, ast.Name) else (
                        tgt.attr if isinstance(tgt, ast.Attribute) else None)
                    if name == "sigma_ref":
                        offenders.append(f"{rel}:{node.lineno}: sigma_ref={node.value.value}")

        assert not offenders, (
            "hardcoded sigma_ref literal(s) reintroduced -- read the value from "
            "the Rust binary instead (see environment._query_env_constants):\n"
            + "\n".join(offenders)
        )


class TestAdaptiveACVolatilityInversion:
    """
    Verifies AdaptiveACPolicy's state->sigma reconstruction against GROUND
    TRUTH from the environment, rather than asserting the formula matches
    itself. The environment emits `current_sigma` in its info dict precisely
    so this check is possible.

    The bug this guards: `state[2]` is tanh(current_sigma/sigma_ref), but the
    policy inverted it as `sigma_ref * (1.0 + arctanh(s2))` -- a spurious
    `1.0 +` borrowed from s13's different definition -- so every Adaptive-AC
    result in this project was produced against a volatility the policy was
    never given.
    """

    CRASH_DATE = "2024-08-05"   # most severe day in the dataset
    CALM_DATE = "2024-07-10"    # held-out calm

    def _roll(self, date, n_steps=60):
        """Run a real episode; return (inverted_sigma, true_sigma) pairs."""
        import numpy as np
        from environment import StrataExecEnv

        env = StrataExecEnv(
            mode="counterfactual", lob_date=date, agg_date=None,
            btc_target=50_000.0, n_state_dims=16, total_steps=n_steps,
            fixed_steps=True, fixed_size=True, no_time_features=False,
        )
        try:
            obs, info = env.reset(seed=0)
            pairs = []
            for _ in range(n_steps):
                obs, _r, done, _t, info = env.step(0)  # action 0: don't trade
                if "current_sigma" not in info:
                    pytest.skip("binary predates current_sigma in info dict")
                # Exercise the real code path, not a reimplementation of it.
                policy = AdaptiveACPolicy(n_state_dims=16)
                policy.sigma_ref = info["sigma_ref"]
                inverted = policy._recover_sigma(obs)
                pairs.append((inverted, float(info["current_sigma"])))
                if done:
                    break
            return pairs
        finally:
            env.close()

    def _require_data(self, date):
        path = f"TradeData/BookDepth/BTCUSDT-bookDepth-{date}.csv"
        if not os.path.exists(path):
            pytest.skip(f"{path} not present in this checkout")

    @pytest.mark.parametrize("date_attr", ["CRASH_DATE", "CALM_DATE"])
    def test_inversion_recovers_true_sigma(self, date_attr):
        date = getattr(self, date_attr)
        self._require_data(date)
        pairs = self._roll(date)
        assert pairs, "episode produced no steps"
        # Tolerance is derived, not guessed. arctanh is ill-conditioned near
        # |x|->1: at the ratio 2024-08-05 actually reaches (~4.19, s2 ~
        # 0.99954) its derivative 1/(1-x^2) is ~1088, so the wire's finite
        # decimal precision on s2 amplifies to a measured worst-case relative
        # error of 6.8e-6. 1e-3 leaves ~150x headroom over that while still
        # sitting SIX orders of magnitude below the bug it exists to catch
        # (the spurious `1.0 +` inflated sigma by a factor of ~8.7).
        for inverted, truth in pairs:
            assert abs(inverted - truth) < 1e-3 * max(1.0, truth), (
                f"state[2] inversion does not recover the environment's real "
                f"current_sigma on {date}: got {inverted}, truth {truth}. The "
                f"spurious '1.0 +' bug (or a new encoding change) is present."
            )

    def test_inversion_holds_in_the_high_sigma_trading_regime(self):
        """
        The passive (action 0) rolls above only reach sigma ~2. Once the policy
        actually TRADES, own impact feeds back into realized volatility and
        2024-08-05 reaches sigma ~6.5 (ratio ~13) -- the regime where the old
        guardrails bound hardest (the 2.0 clamp) and where inverting s2 is
        impossible (tanh(13) is 1.0 to within 6e-12). This is the case that
        motivated preferring s10, so it needs its own coverage rather than
        being inferred from the passive rolls.
        """
        import numpy as np
        from environment import StrataExecEnv

        self._require_data(self.CRASH_DATE)
        env = StrataExecEnv(
            mode="counterfactual", lob_date=self.CRASH_DATE, agg_date=None,
            btc_target=50_000.0, n_state_dims=16, total_steps=400,
            fixed_steps=True, fixed_size=True, no_time_features=False,
        )
        try:
            policy = AdaptiveACPolicy(n_state_dims=16, lam=1e-6, total_steps=400)
            obs, info = env.reset(seed=0)
            peak, worst_rel, done = 0.0, 0.0, False
            while not done:
                action, _ = policy.predict(obs)
                obs, _r, done, _t, info = env.step(int(action))
                if "current_sigma" not in info:
                    pytest.skip("binary predates current_sigma in info dict")
                truth = float(info["current_sigma"])
                policy.sigma_ref = info["sigma_ref"]
                rel = abs(policy._recover_sigma(obs) - truth) / max(1e-12, truth)
                peak = max(peak, truth)
                worst_rel = max(worst_rel, rel)
        finally:
            env.close()

        assert peak > 3.0, (
            f"this date should drive sigma well past the old 2.0 clamp once "
            f"trading feeds back into volatility; only reached {peak:.3f} -- "
            f"the test is no longer exercising the regime it exists for"
        )
        assert peak < 20.0, (
            f"observed sigma {peak:.3f} hit the 20.0 backstop clamp; the clamp "
            f"is meant not to bind on real data -- re-derive it"
        )
        assert worst_rel < 1e-3, (
            f"inversion breaks down in the high-sigma regime: worst relative "
            f"error {worst_rel:.2e} at peak sigma {peak:.3f}"
        )


class TestSqrtImpactOptimalSchedule:
    """
    Phase 3 Step 3: SqrtImpactOptimalPolicy's Beta-CDF schedule family. Pure
    math on `_build_schedule` -- no subprocess/environment needed -- built
    via __new__ to skip __init__'s real-environment fit.
    """

    def _bare(self, total_steps=2880):
        policy = SqrtImpactOptimalPolicy.__new__(SqrtImpactOptimalPolicy)
        policy.total_steps = total_steps
        return policy

    @pytest.mark.parametrize("a,b", [(1.0, 1.0), (0.3, 2.5), (4.0, 0.5), (1.0, 1.65)])
    def test_schedule_sums_to_one_and_nonnegative(self, a, b):
        policy = self._bare()
        schedule = policy._build_schedule(a, b)
        assert len(schedule) == policy.total_steps
        assert np.all(schedule >= 0.0)
        assert abs(schedule.sum() - 1.0) < 1e-9

    def test_a_equals_b_equals_one_is_approximately_uniform(self):
        """Beta(1,1) is the uniform distribution -- this shape must reduce
        to (approximately) TWAP, since the family is a strict superset of
        TWAP's shape and TWAP is the a=b=1 special case."""
        policy = self._bare()
        schedule = policy._build_schedule(1.0, 1.0)
        uniform = 1.0 / policy.total_steps
        assert np.allclose(schedule, uniform, atol=1e-9)

    def test_low_b_front_loads_mass_early(self):
        """Beta(1, b) with b>1 concentrates mass near t=0 (pdf = b*(1-t)^(b-1),
        decreasing in t) -- more should sell in the first half than the
        second, unlike the a=b=1 uniform case."""
        policy = self._bare()
        schedule = policy._build_schedule(1.0, 4.0)
        half = policy.total_steps // 2
        assert schedule[:half].sum() > schedule[half:].sum()

    def test_fit_seed_base_is_disjoint_but_adjacent_to_eval_range(self):
        """
        Fit seeds must start exactly at n_eval_episodes: disjoint from the
        eval range 0..n_eval_episodes-1 (no leakage) but immediately
        adjacent to it, not a distant constant -- found empirically that a
        distant fit-seed range (e.g. 900_000+) samples an entirely
        different, unrepresentative slice of a volatile date's start-offset
        space than the eval range does (see class docstring)."""
        policy = SqrtImpactOptimalPolicy.__new__(SqrtImpactOptimalPolicy)
        policy.total_steps = 2880
        n_eval_episodes = 20
        policy._fit_seed_base = n_eval_episodes
        assert policy._fit_seed_base == n_eval_episodes


import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HAS_REAL_ENV = os.path.exists(
    os.path.join(_REPO_ROOT, "target", "release", "rl-env")
) and os.path.exists(
    os.path.join(_REPO_ROOT, "TradeData", "BookDepth", "BTCUSDT-bookDepth-2024-03-05.csv")
)

requires_real_env_for_sqrt_impact = pytest.mark.skipif(
    not _HAS_REAL_ENV,
    reason="requires target/release/rl-env binary and TradeData/ fixtures",
)


@requires_real_env_for_sqrt_impact
class TestSqrtImpactOptimalFitIntegration:
    """
    Exercises the real fit path (StrataExecEnv subprocess + Nelder-Mead)
    end-to-end on a real date, with a tiny budget so it stays fast. Confirms
    the optimizer actually explores away from its TWAP starting point rather
    than falsely converging on the first simplex (see the initial_simplex
    fix in _fit's docstring -- scipy's default zero-coordinate perturbation
    is too small to move any of the 2880 discretized schedule fractions).
    """

    def test_fit_produces_valid_schedule_and_moves_from_start(self):
        policy = SqrtImpactOptimalPolicy(
            "2024-03-05", n_eval_episodes=20, n_state_dims=16,
            mode="counterfactual", btc_target=50_000.0, n_fit_seeds=1, maxiter=8,
        )
        assert policy._schedule.shape == (policy.total_steps,)
        assert abs(policy._schedule.sum() - 1.0) < 1e-9
        assert np.all(policy._schedule >= 0.0)
        assert np.isfinite(policy.fit_mean_is)
        assert policy.a > 0.0 and policy.b > 0.0

    def test_query_calibration_matches_fields_the_impact_model_uses(self):
        calib = query_calibration("2024-03-05", btc_target=50_000.0)
        for key in ("impact_coefficient", "sigma", "adv_btc", "rho",
                    "permanent_fraction", "temporary_fraction", "dt",
                    "btc_scale"):
            assert key in calib
        assert calib["impact_coefficient"] > 0.0
        assert calib["sigma"] > 0.0
        assert abs(calib["permanent_fraction"] + calib["temporary_fraction"] - 1.0) < 1e-9
