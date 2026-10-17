"""
Tail-risk and significance statistics on per-episode IS samples.
"""

from scipy import stats
import numpy as np


def compute_cvar_95(arr: np.ndarray) -> float:
    """
    CVaR-95: mean of the worst 5% of IS values (the tail mean, not a single
    percentile). Sign convention: more negative IS = worse execution cost
    (see _run_episode). Previously several call sites used
    np.percentile(arr, 5) here, which is VaR-5 (a single point on the
    distribution), not CVaR-95 (a tail mean) -- directionally correct but
    statistically the wrong quantity. This is the one implementation all
    evaluation functions should share.
    """
    if len(arr) == 0:
        return float("nan")
    n_tail = max(1, len(arr) // 20)
    sorted_asc = np.sort(arr)
    return float(np.mean(sorted_asc[:n_tail]))


def bootstrap_cvar_95_ci(
    arr: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """
    Bootstrap confidence interval on the CVaR-95 estimate itself (Global
    Rule 4), not just on the mean. CVaR-95 on N episodes is the mean of only
    N//20 tail observations -- e.g. ~2-3 points on 50 episodes -- a very
    high-variance estimator whose point estimate alone is easy to
    over-trust. Resample the episode array with replacement n_bootstrap
    times, recompute compute_cvar_95 on each resample, and report the
    percentile interval of that distribution.

    Uses a local np.random.Generator (seeded, reproducible per Global Rule 2)
    rather than the global numpy RNG, so this never interacts with any other
    code's random state (e.g. test_vs_baseline's bootstrap, which still uses
    the legacy global np.random API).
    """
    n = len(arr)
    if n == 0:
        return {"point": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"),
                "n_episodes": 0, "n_tail": 0, "width": float("nan")}

    point = compute_cvar_95(arr)
    rng = np.random.default_rng(seed)
    resample_cvars = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        resample = rng.choice(arr, size=n, replace=True)
        resample_cvars[i] = compute_cvar_95(resample)

    alpha = 1.0 - ci
    ci_lower = float(np.percentile(resample_cvars, 100 * alpha / 2))
    ci_upper = float(np.percentile(resample_cvars, 100 * (1 - alpha / 2)))
    n_tail = max(1, n // 20)

    return {
        "point": point,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_episodes": n,
        "n_tail": n_tail,
        "width": ci_upper - ci_lower,
    }


def test_vs_baseline(rl_is_samples: list, baseline_is: float) -> dict:
    """
    Test whether RL IS is significantly better than the static AC baseline.
    Returns t-statistic, one-tailed p-value, and 95% bootstrap CI.
    """
    arr = np.array(rl_is_samples)
    n = len(arr)

    if n == 0 or np.std(arr) < 1e-10:
        mean_val = float(arr.mean()) if n > 0 else float("nan")
        return {
            "t_stat": None,
            "p_value": None,
            "ci_lower": mean_val,
            "ci_upper": mean_val,
            "significantly_better": None,
            "note": "zero variance - deterministic policy",
        }

    t_stat, p_value = stats.ttest_1samp(arr, baseline_is)
    # One-tailed (RL better = higher IS than baseline).
    p_one_tailed = p_value / 2 if t_stat > 0 else 1 - p_value / 2

    bootstrap_means = [
        np.mean(np.random.choice(arr, size=n, replace=True))
        for _ in range(10000)
    ]
    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_one_tailed),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "significantly_better": bool(p_one_tailed < 0.05),
    }
