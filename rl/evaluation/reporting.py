"""
Console tables and MLflow logging for evaluation results.
"""

import mlflow
import numpy as np

from .dates import DATES, TEST_DATES


def print_unified_comparison(all_results: dict):
    """
    Print full strategy comparison table from unified evaluation.
    all_results: {date: {strategy_key: metrics_dict}}
    """
    STRATEGY_ORDER = [
        "twap", "heuristic", "ac_optimal", "adaptive_ac", "rl_agent"
    ]
    STRATEGY_LABELS = {
        "twap":        "TWAP",
        "heuristic":   "Heuristic",
        "ac_optimal":  "AC Optimal",
        "adaptive_ac": "Adaptive AC",
        "rl_agent":    "RL Agent",
    }

    print("\n" + "=" * 90)
    print("UNIFIED COMPARISON - ALL STRATEGIES IN COUNTERFACTUAL LOB")
    print("Same OW impact model | Same real Binance price path | Same arrival price")
    print("More negative IS = higher execution cost | CVaR-95 = avg of worst 5% episodes")
    print("=" * 90)

    for date, date_results in all_results.items():
        if not date_results:
            continue

        regime = DATES.get(date) or TEST_DATES.get(date, "unknown")
        print(f"\n{'-' * 70}")
        print(f"  {date}  ({regime})")
        print(f"{'-' * 70}")

        header = (f"  {'Strategy':<18} {'Mean IS':>9} {'CVaR-95':>9} "
                  f"{'IS Sharpe':>10} {'Forced Liq':>11}")
        print(header)
        print(f"  {'-'*18} {'-'*9} {'-'*9} {'-'*10} {'-'*11}")

        rl_is = date_results.get("rl_agent", {}).get("mean_IS")

        for key in STRATEGY_ORDER:
            r = date_results.get(key)
            if r is None:
                continue

            mean_is = r["mean_IS"]
            cvar    = r["cvar_95"]
            sharpe  = r["is_sharpe"]
            fl_rate = r["forced_liquidation_rate"]

            if rl_is is not None and key != "rl_agent":
                vs_rl = "RL better" if rl_is > mean_is else "RL worse"
                vs_rl_str = f"  <- {vs_rl}"
            else:
                vs_rl_str = ""

            label = STRATEGY_LABELS[key]
            if key == "rl_agent":
                label = f"* {label}"

            print(f"  {label:<18} {mean_is:>+9.4f}% {cvar:>+9.4f}% "
                  f"{sharpe:>+10.4f} {fl_rate:>10.1%}"
                  f"{vs_rl_str}")

    print("\n" + "=" * 90)
    print("CVAR-95 COMPARISON (lower = worse tail risk)")
    print("=" * 90)

    dates_list = list(all_results.keys())
    if dates_list:
        print(f"\n  {'Strategy':<18}", end="")
        for date in dates_list:
            print(f"  {date}", end="")
        print()
        print(f"  {'-'*18}", end="")
        for _ in dates_list:
            print(f"  {'-'*10}", end="")
        print()

        for key in STRATEGY_ORDER:
            label = STRATEGY_LABELS[key]
            print(f"  {label:<18}", end="")
            for date in dates_list:
                r = all_results[date].get(key)
                if r:
                    print(f"  {r['cvar_95']:>+9.4f}%", end="")
                else:
                    print(f"  {'N/A':>10}", end="")
            print()


def log_eval_results_to_mlflow(
    run_id: str,
    unified_results: dict,
    strategies: list = None,
) -> None:
    """
    Phase 3 (concurrent with Phase 2): log final per-date evaluation metrics
    to an EXISTING mlflow run by run_id, rather than starting a fresh
    untracked run. This is the piece that was missing entirely -- train.py
    has mlflow.start_run/log_params/log_artifact but zero log_metric calls,
    and until this function existed, evaluate.py had no mlflow integration
    at all, so a training run's final eval numbers (the ones that actually
    matter -- mean IS, CVaR-95 per date) were never attached to that run's
    tracked history. experiment_tracker.py's get_best_run() has always
    returned {} as a direct consequence: there was nothing to query.

    unified_results: {date: {strategy_key: metrics_dict}}, the return shape
    of evaluate_all_strategies_on_date collected across multiple dates (see
    run_full_analysis's --unified-comparison path, or
    rl/experiments/phase1_tail_risk_report.py).

    strategies: which strategy_keys to log (default: just "rl_agent", since
    for a training run's own tracked history the model's own performance is
    what's relevant -- the classical baselines don't change run to run and
    don't need re-logging against every model's run_id).

    Uses mlflow.start_run(run_id=..., nested=False) to REOPEN the existing
    run rather than create a new one -- this is the standard mlflow pattern
    for attaching metrics after the original `with mlflow.start_run():`
    block has already exited (e.g. because this is called from a separate
    process/script after training finished).
    """
    if strategies is None:
        strategies = ["rl_agent"]

    with mlflow.start_run(run_id=run_id):
        for date, date_results in unified_results.items():
            for strategy_key in strategies:
                r = date_results.get(strategy_key)
                if r is None:
                    continue
                safe_date = date.replace("-", "")
                mlflow.log_metric(f"eval_mean_is_{safe_date}_{strategy_key}", r["mean_IS"])
                mlflow.log_metric(f"eval_cvar95_{safe_date}_{strategy_key}", r["cvar_95"])
                if "cvar_95_ci_lower" in r:
                    mlflow.log_metric(f"eval_cvar95_ci_lower_{safe_date}_{strategy_key}", r["cvar_95_ci_lower"])
                    mlflow.log_metric(f"eval_cvar95_ci_upper_{safe_date}_{strategy_key}", r["cvar_95_ci_upper"])
                mlflow.log_metric(f"eval_forced_liq_{safe_date}_{strategy_key}", r["forced_liquidation_rate"])

        # Aggregate summary metrics across all logged dates, so
        # get_best_run(metric="eval_mean_is_avg", ...) has something to
        # compare across runs without needing to know every date's name.
        for strategy_key in strategies:
            means = [
                date_results[strategy_key]["mean_IS"]
                for date_results in unified_results.values()
                if strategy_key in date_results
            ]
            cvars = [
                date_results[strategy_key]["cvar_95"]
                for date_results in unified_results.values()
                if strategy_key in date_results
            ]
            if means:
                mlflow.log_metric(f"eval_mean_is_avg_{strategy_key}", float(np.mean(means)))
            if cvars:
                mlflow.log_metric(f"eval_cvar95_avg_{strategy_key}", float(np.mean(cvars)))
