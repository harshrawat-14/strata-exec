"""
THE HONEST FAILURE ANALYSIS.
This is the core research deliverable of Phase 4.
"""

# Public API re-exported from the `evaluation` package, so existing
# `from evaluate import X` callers and the CLI below keep working.
# New code should import from `evaluation.<module>` directly.
import argparse

from environment import (  # noqa: F401
    ACTION_FRACTIONS,
    CALIBRATED_SIGMA_REF,
    REGIME_DIMS_ARE_CONTINUOUS,
    RL_ENV_BIN,
    StrataExecEnv,
)
from evaluation.dates import (  # noqa: F401
    TEST_DATES,
    DATES,
    HELD_OUT_CRASH_DATES,
    TRAINING_DATES,
    SEVERITY_RANK_V2,
    TRAINING_DATES_V2,
    HELD_OUT_DATES_V2,
    STATIC_OPTIMAL_IS,
    ADAPTIVE_OPTIMAL_IS,
    TWAP_IS,
    HEURISTIC_IS,
)
from evaluation.actions import (  # noqa: F401
    get_action_fractions,
    nearest_action,
)
from evaluation.metrics import (  # noqa: F401
    compute_cvar_95,
    bootstrap_cvar_95_ci,
    test_vs_baseline,
)
from evaluation.policies.baselines import (  # noqa: F401
    TWAPPolicy,
    HeuristicPolicy,
)
from evaluation.policies.almgren_chriss import (  # noqa: F401
    AC_DEFAULT_LAMBDA,
    _stable_sinh_ratio,
    ACOptimalPolicy,
    AdaptiveACPolicy,
)
from evaluation.policies.sqrt_impact import (  # noqa: F401
    query_calibration,
    SqrtImpactOptimalPolicy,
)
from evaluation.episode import (  # noqa: F401
    load_model,
    _run_episode,
)
from evaluation.harness import (  # noqa: F401
    evaluate_model_on_date,
    evaluate_model_counterfactual,
    evaluate_on_synthetic,
    evaluate_all_strategies_on_date,
    AC_LAMBDA_GRID,
    evaluate_ac_lambda_grid,
)
from evaluation.reporting import (  # noqa: F401
    print_unified_comparison,
    log_eval_results_to_mlflow,
)
from evaluation.analysis import (  # noqa: F401
    run_full_analysis,
)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--passive-model", required=True)
    p.add_argument("--adversarial-model", default=None)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--n-state-dims", type=int, default=8)
    p.add_argument("--eval-counterfactual", action="store_true",
        help="Also evaluate in counterfactual LOB (training env)")
    p.add_argument("--btc-target", type=float, default=50000.0)
    p.add_argument("--include-test-dates", action="store_true",
        help="Also evaluate on held-out test dates")
    p.add_argument("--unified-comparison", action=argparse.BooleanOptionalAction,
        default=True,
        help="Run all 5 strategies in same counterfactual LOB for "
             "methodologically valid comparison (default: on). Use "
             "--no-unified-comparison to skip it.")
    p.add_argument("--legacy-mixed-comparison", action="store_true",
        help="Also print/save the OLD strategy-comparison table that mixes "
             "Monte-Carlo-sourced TWAP/Heuristic/AC constants with live "
             "LOB-replay RL numbers. NOT a valid comparison -- historical "
             "reference only, off by default.")
    p.add_argument("--ac-lambda", type=float, default=AC_DEFAULT_LAMBDA,
        help=f"Risk-aversion for the AC-family baselines in the unified "
             f"comparison (default {AC_DEFAULT_LAMBDA}; Rust research-sim's "
             f"default is 1e-4 -- the two were never reconciled, override "
             f"explicitly if you need to match one).")
    args = p.parse_args()

    run_full_analysis(
        args.passive_model,
        args.adversarial_model,
        args.episodes,
        n_state_dims=args.n_state_dims,
        eval_counterfactual=args.eval_counterfactual,
        btc_target=args.btc_target,
        include_test_dates=args.include_test_dates,
        unified_comparison=args.unified_comparison,
        legacy_mixed_comparison=args.legacy_mixed_comparison,
        ac_lambda=args.ac_lambda,
    )
