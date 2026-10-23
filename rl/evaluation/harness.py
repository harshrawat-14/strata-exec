"""
Evaluation harness: run RL and every baseline through the same environment.
"""

from collections import Counter

import numpy as np

from environment import StrataExecEnv
from .metrics import bootstrap_cvar_95_ci, compute_cvar_95
from .policies.baselines import HeuristicPolicy, TWAPPolicy
from .policies.almgren_chriss import ACOptimalPolicy, AC_DEFAULT_LAMBDA, AdaptiveACPolicy
from .policies.sqrt_impact import SqrtImpactOptimalPolicy
from .episode import _run_episode
from .protocol import eval_start_fracs


def evaluate_model_on_date(
    model,
    date: str,
    adversarial: bool = False,
    n_episodes: int = 50,
    n_state_dims: int = 8,
) -> dict:

    env = StrataExecEnv(
        mode="historical",
        adversarial=adversarial,
        lob_date=date,
        agg_date=date,
        n_state_dims=n_state_dims,
        fixed_steps=True,
        fixed_size=True,
        no_time_features=n_state_dims < 16,
    )

    all_is = []
    all_rewards = []
    all_actions = []
    forced_liquidations = 0

    for ep in range(n_episodes):
        is_pct, ep_reward, forced, actions = _run_episode(
            model, env, seed=ep)
        if forced:
            forced_liquidations += 1
        if is_pct is not None:
            all_is.append(is_pct)
        all_rewards.append(ep_reward)
        all_actions.extend(actions)

    env.close()

    arr = np.array(all_is)

    action_counts = Counter(all_actions)
    total_actions = sum(action_counts.values())
    if total_actions > 0:
        action_dist = {
            a: action_counts[a] / total_actions
            for a in sorted(action_counts.keys())
        }
        mean_action = sum(a * f for a, f in action_dist.items())
        action_entropy = -sum(
            f * np.log(f + 1e-10) for f in action_dist.values()
        )
    else:
        action_dist = {}
        mean_action = float("nan")
        action_entropy = 0.0

    return {
        "date": date,
        "mean_IS": float(np.mean(arr)),
        "std_IS": float(np.std(arr)),
        "var_IS": float(np.var(arr)),
        "cvar95": compute_cvar_95(arr),
        "mean_reward": float(np.mean(all_rewards)),
        "forced_liquidation_rate": forced_liquidations / n_episodes,
        "n_episodes": n_episodes,
        "is_samples": all_is,
        "action_distribution": action_dist,
        "mean_action": float(mean_action),
        "action_entropy": float(action_entropy),
    }


def evaluate_model_counterfactual(
    model,
    date: str,
    n_episodes: int = 50,
    btc_target: float = 50000.0,
    n_state_dims: int = 12,
) -> dict:
    env = StrataExecEnv(
        mode="counterfactual",
        lob_date=date,
        agg_date=date,
        btc_target=btc_target,
        n_state_dims=n_state_dims,
        fixed_steps=True,
        fixed_size=True,
        no_time_features=n_state_dims < 16,
    )

    all_is = []
    all_rewards = []
    forced_liquidations = 0

    for ep in range(n_episodes):
        is_pct, ep_reward, forced, _ = _run_episode(model, env, seed=ep)
        if forced:
            forced_liquidations += 1
        if is_pct is not None:
            all_is.append(is_pct)
        all_rewards.append(ep_reward)

    env.close()
    arr = np.array(all_is)
    return {
        "mean_IS": float(np.mean(arr)),
        "std_IS":  float(np.std(arr)),
        "var_IS":  float(np.var(arr)),
        "cvar95":  compute_cvar_95(arr),
        "forced_liquidation_rate": forced_liquidations / n_episodes,
    }


def evaluate_on_synthetic(model, n_episodes=200, n_state_dims=8) -> dict:
    env = StrataExecEnv(
        mode="synthetic",
        n_state_dims=n_state_dims,
        fixed_steps=True,
        fixed_size=True,
        no_time_features=n_state_dims < 16,
    )
    all_is = []
    all_rewards = []

    for ep in range(n_episodes):
        is_pct, ep_reward, _, _ = _run_episode(
            model, env, seed=ep + 10000)
        if is_pct is not None:
            all_is.append(is_pct)
        all_rewards.append(ep_reward)

    env.close()
    arr = np.array(all_is)
    return {
        "mean_IS": float(np.mean(arr)),
        "var_IS": float(np.var(arr)),
        "cvar95": compute_cvar_95(arr),
    }


def evaluate_all_strategies_on_date(
    rl_model,
    date: str,
    n_episodes: int = 50,
    n_state_dims: int = 12,
    adversarial: bool = False,
    fixed_steps: bool = True,
    fixed_size: bool = True,
    ac_lambda: float = AC_DEFAULT_LAMBDA,
    n_bootstrap: int = 10000,
    bootstrap_seed: int = 0,
    mode: str = "historical",
    btc_target: float = 50_000.0,
    include_sqrt_impact_optimal: bool = False,
    sqrt_impact_n_fit_seeds: int = 3,
    sqrt_impact_maxiter: int = 25,
    episode_steps: int = None,
    adaptive_ac_lambda: float = None,
    order_share_adv: float = None,
) -> dict:
    """
    Evaluate all 5 (or 6, with include_sqrt_impact_optimal) strategies on
    the same LOB environment using the same
    transient-impact model (OW-style permanent/temporary decay structure,
    square-root instantaneous magnitude -- see README "Market Impact" for
    the Phase 2R naming correction), same real Binance price path, and same
    arrival price.

    Replaces the mixed-framework comparison where TWAP/Heuristic
    used research-sim (Monte Carlo + square-root impact) and RL
    used evaluate.py (real LOB + this transient-impact model).

    mode: "historical" (default, preserves every existing caller's behavior
    -- pure replay, no causal own-impact feedback, matches Phase 0/1's
    established harness) or "counterfactual" (real LOB + accumulated own
    impact -- what RL training actually uses). Phase 2R Task 4's 16-dim vs
    12-dim ablation uses "counterfactual" specifically: historical mode has
    no cf_lob, so state dims 8/9 (own-impact fractions) are structurally
    dead there, which would test the 16-dim arm with two of its own
    distinguishing features disabled -- not a fair test of the
    representation. See docs/experiments/02-reward-objective-audit.md Gate 3 and
    rl/results/PROVENANCE.md for why this default stays "historical" (all
    pre-Phase-2R results/comparability are historical-mode) while Task 4
    explicitly opts into "counterfactual".

    ac_lambda: risk-aversion parameter shared by both AC-family policies
    (AC Optimal and Adaptive AC) in this comparison -- exposed here rather
    than left as a hardcoded default so the AC baseline isn't a strawman
    (lambda too low) or an unbeatable wall (lambda too high) chosen
    invisibly. See AC_DEFAULT_LAMBDA for the Rust/Python default mismatch
    this parameter exists to let a caller work around.

    n_bootstrap/bootstrap_seed: passed to bootstrap_cvar_95_ci (Global Rule
    4) -- CVaR-95 on n_episodes episodes is a high-variance estimator (the
    mean of only n_episodes//20 tail points), so every cvar_95 reported here
    carries its own bootstrap CI rather than being presented as a clean
    point estimate.

    order_share_adv: protocol v3.1 -- order sized as this share of the date's
    20-day trailing ADV (evaluation/liquidity.py), overriding btc_target.
    Not supported together with include_sqrt_impact_optimal.

    adaptive_ac_lambda: Adaptive-AC's own risk aversion; defaults to ac_lambda.
    Under protocol v3 the honest values differ (AC 1e-6, Adaptive-AC 1e-9;
    rl/results/v3_lambda_grid.csv).

    episode_steps: evaluation protocol v3 (evaluation/protocol.py). Every
    episode is this long and episode i starts at eval_start_fracs(n)[i] of the
    day's valid window; schedule baselines adopt the env-reported horizon.
    None (default) keeps the legacy protocol byte-for-byte.

    include_sqrt_impact_optimal: adds Step 3's numerically-optimized
    sqrt-impact baseline (SqrtImpactOptimalPolicy, docs/proposals/research-roadmap.md).
    Defaults False -- fitting it runs real episodes through this exact env
    before evaluation even starts, which is real wall-clock cost every
    existing caller of this function would otherwise pay silently.
    """
    policies = {
        "twap": TWAPPolicy(n_state_dims),
        "heuristic": HeuristicPolicy(n_state_dims),
        "ac_optimal": ACOptimalPolicy(n_state_dims, lam=ac_lambda,
                                      track_schedule=bool(episode_steps)),
        "adaptive_ac": AdaptiveACPolicy(n_state_dims,
                                        lam=ac_lambda if adaptive_ac_lambda is None else adaptive_ac_lambda,
                                        track_schedule=bool(episode_steps)),
        "rl_agent": rl_model,
    }

    strategy_labels = {
        "twap":        "TWAP",
        "heuristic":   "Heuristic",
        "ac_optimal":  "AC Optimal",
        "adaptive_ac": "Adaptive AC",
        "rl_agent":    "RL Agent",
    }

    if include_sqrt_impact_optimal and order_share_adv is not None:
        raise ValueError("sqrt-impact baseline does not support order_share_adv yet")
    if include_sqrt_impact_optimal:
        sqrt_policy = SqrtImpactOptimalPolicy(
            date, n_eval_episodes=n_episodes,
            n_state_dims=n_state_dims, mode=mode, btc_target=btc_target,
            fixed_steps=fixed_steps, fixed_size=fixed_size,
            n_fit_seeds=sqrt_impact_n_fit_seeds, maxiter=sqrt_impact_maxiter,
            episode_steps=episode_steps,
        )
        print(f"    [SqrtImpactOptimal fit] a={sqrt_policy.a:.3f} "
              f"b={sqrt_policy.b:.3f} fit_mean_IS={sqrt_policy.fit_mean_is:.4f}")
        policies["sqrt_impact_optimal"] = sqrt_policy
        strategy_labels["sqrt_impact_optimal"] = "Sqrt-Impact Optimal"

    results = {}

    for strategy_key, policy in policies.items():
        print(f"    [{strategy_labels[strategy_key]}]", end=" ", flush=True)

        env_kwargs = dict(
            mode=mode,
            adversarial=adversarial,
            lob_date=date,
            agg_date=date,
            n_state_dims=n_state_dims,
            fixed_steps=fixed_steps,
            fixed_size=fixed_size,
            no_time_features=n_state_dims < 16,
            episode_steps=episode_steps,
            order_share_adv=order_share_adv,
        )
        if mode == "counterfactual":
            env_kwargs["btc_target"] = btc_target
        env = StrataExecEnv(**env_kwargs)
        # Protocol v3: starts spread evenly across the day's valid window.
        start_fracs = eval_start_fracs(n_episodes) if episode_steps else [None] * n_episodes

        all_is = []
        all_rewards = []
        forced_liq = 0
        all_actions = []

        for ep in range(n_episodes):
            is_pct, ep_reward, forced, actions = _run_episode(
                policy, env, seed=ep, start_frac=start_fracs[ep]
            )
            if forced:
                forced_liq += 1
            if is_pct is not None:
                all_is.append(is_pct)
            all_rewards.append(ep_reward)
            all_actions.extend(actions)

        env.close()

        arr = np.array(all_is) if all_is else np.array([0.0])

        action_counts = Counter(all_actions)
        total_actions = sum(action_counts.values()) or 1
        action_dist = {
            a: action_counts[a] / total_actions
            for a in sorted(action_counts.keys())
        }

        cvar_95 = compute_cvar_95(arr)
        cvar_ci = bootstrap_cvar_95_ci(arr, n_bootstrap=n_bootstrap, seed=bootstrap_seed)

        results[strategy_key] = {
            "strategy": strategy_labels[strategy_key],
            "date": date,
            "mean_IS": float(np.mean(arr)),
            "std_IS": float(np.std(arr)),
            "var_IS": float(np.var(arr)),
            "cvar_95": cvar_95,
            "cvar_95_ci_lower": cvar_ci["ci_lower"],
            "cvar_95_ci_upper": cvar_ci["ci_upper"],
            "cvar_95_ci_width": cvar_ci["width"],
            "cvar_95_n_tail": cvar_ci["n_tail"],
            "is_99th_pct": float(np.percentile(arr, 1)),
            "is_sharpe": float(np.mean(arr) / (np.std(arr) + 1e-10)),
            "mean_reward": float(np.mean(all_rewards)),
            "forced_liquidation_rate": forced_liq / n_episodes,
            "n_episodes": n_episodes,
            "is_samples": all_is,
            "action_distribution": action_dist,
            "mean_action": float(np.mean(all_actions)) if all_actions else 0.0,
        }

        print(f"IS={results[strategy_key]['mean_IS']:.4f}%  "
              f"CVaR95={cvar_95:.4f}% [{cvar_ci['ci_lower']:.4f}%, {cvar_ci['ci_upper']:.4f}%] "
              f"(tail n={cvar_ci['n_tail']})")

    return results


# Default risk-aversion grid for the AC baseline sweep (Phase 1 task 3a):
# spans patient (low lambda, trades slowly, more market risk) to urgent
# (high lambda, front-loads execution, more impact cost). Deliberately wider
# than a token 2-point grid so the reader can see AC_DEFAULT_LAMBDA's actual
# position on the curve, not just two endpoints.
AC_LAMBDA_GRID = [1e-5, 1e-4, AC_DEFAULT_LAMBDA, 1e-2, 1e-1]


def evaluate_ac_lambda_grid(
    date: str,
    lambdas: list = None,
    n_episodes: int = 50,
    n_state_dims: int = 12,
    fixed_steps: bool = True,
    fixed_size: bool = True,
    n_bootstrap: int = 10000,
    bootstrap_seed: int = 0,
    mode: str = "historical",
    btc_target: float = 50_000.0,
) -> dict:
    """
    Phase 1 task 3a: the AC baseline is not unique -- it depends on a
    risk-aversion parameter (see AC_DEFAULT_LAMBDA's Rust/Python mismatch
    note), and every p-value/comparison in this project is measured against
    it. Sweep AC Optimal and Adaptive AC across a lambda grid on the same
    counterfactual LOB episodes (same seeds) so a reader can see whether
    AC_DEFAULT_LAMBDA sits on a strawman (too patient), an unbeatable wall
    (too urgent), or somewhere reasonable in between -- rather than trusting
    a single hidden value.

    Only runs the two lambda-dependent policies (TWAP/Heuristic/RL don't
    depend on lambda and are already covered once by
    evaluate_all_strategies_on_date -- no need to re-run them per grid point).

    mode/btc_target: see evaluate_all_strategies_on_date's docstring for why
    Phase 2R's ablation uses "counterfactual" specifically. Default stays
    "historical" so every existing caller is unaffected.

    Returns {lambda: {"ac_optimal": {...}, "adaptive_ac": {...}}}, same
    per-strategy metrics shape as evaluate_all_strategies_on_date.
    """
    if lambdas is None:
        lambdas = AC_LAMBDA_GRID

    grid_results = {}

    for lam in lambdas:
        policies = {
            "ac_optimal": ACOptimalPolicy(n_state_dims, lam=lam),
            "adaptive_ac": AdaptiveACPolicy(n_state_dims, lam=lam),
        }
        strategy_labels = {"ac_optimal": "AC Optimal", "adaptive_ac": "Adaptive AC"}
        lam_result = {}

        for strategy_key, policy in policies.items():
            print(f"    [lambda={lam:.0e} {strategy_labels[strategy_key]}]", end=" ", flush=True)

            env_kwargs = dict(
                mode=mode,
                lob_date=date,
                agg_date=date,
                n_state_dims=n_state_dims,
                fixed_steps=fixed_steps,
                fixed_size=fixed_size,
                no_time_features=n_state_dims < 16,
            )
            if mode == "counterfactual":
                env_kwargs["btc_target"] = btc_target
            env = StrataExecEnv(**env_kwargs)

            all_is = []
            for ep in range(n_episodes):
                is_pct, _ep_reward, _forced, _actions = _run_episode(policy, env, seed=ep)
                if is_pct is not None:
                    all_is.append(is_pct)
            env.close()

            arr = np.array(all_is) if all_is else np.array([0.0])
            cvar_95 = compute_cvar_95(arr)
            cvar_ci = bootstrap_cvar_95_ci(arr, n_bootstrap=n_bootstrap, seed=bootstrap_seed)

            lam_result[strategy_key] = {
                "strategy": strategy_labels[strategy_key],
                "lambda": lam,
                "date": date,
                "mean_IS": float(np.mean(arr)),
                "std_IS": float(np.std(arr)),
                "cvar_95": cvar_95,
                "cvar_95_ci_lower": cvar_ci["ci_lower"],
                "cvar_95_ci_upper": cvar_ci["ci_upper"],
                "is_sharpe": float(np.mean(arr) / (np.std(arr) + 1e-10)),
                "n_episodes": n_episodes,
                "is_samples": all_is,
            }
            print(f"IS={lam_result[strategy_key]['mean_IS']:.4f}%  CVaR95={cvar_95:.4f}%")

        grid_results[lam] = lam_result

    return grid_results
