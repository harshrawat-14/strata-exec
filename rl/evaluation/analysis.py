"""
End-to-end analysis driver behind `python rl/evaluate.py`.
"""

from market_data import book_path

import os

import pandas as pd

from .dates import ADAPTIVE_OPTIMAL_IS, DATES, HEURISTIC_IS, STATIC_OPTIMAL_IS, TEST_DATES, TWAP_IS
from .metrics import test_vs_baseline
from .policies.almgren_chriss import AC_DEFAULT_LAMBDA
from .episode import load_model
from .harness import evaluate_all_strategies_on_date, evaluate_model_counterfactual, evaluate_model_on_date, evaluate_on_synthetic
from .reporting import print_unified_comparison


def run_full_analysis(
    passive_model_path: str,
    adversarial_model_path: str = None,
    n_episodes: int = 50,
    output_dir: str = "rl/results/",
    n_state_dims: int = 8,
    eval_counterfactual: bool = False,
    btc_target: float = 50000.0,
    include_test_dates: bool = False,
    unified_comparison: bool = True,
    legacy_mixed_comparison: bool = False,
    ac_lambda: float = AC_DEFAULT_LAMBDA,
):
    """
    unified_comparison (default True): run every strategy (TWAP, Heuristic,
        AC Optimal, Adaptive AC, RL) through the same counterfactual LOB
        path via evaluate_all_strategies_on_date -- the only methodologically
        valid cross-strategy comparison in this file.
    legacy_mixed_comparison (default False): also print/save the OLD
        strategy-comparison table that puts Monte-Carlo-sourced TWAP/
        Heuristic/AC constants (STATIC_OPTIMAL_IS etc.) next to live
        LOB-replay RL numbers. This mixes two different simulators and is
        not a valid comparison -- only enable it for historical/debugging
        reference, never to support a beats-AC claim.
    ac_lambda: risk-aversion for the AC-family baselines in the unified
        comparison (see AC_DEFAULT_LAMBDA).
    """
    os.makedirs(output_dir, exist_ok=True)

    passive_model = load_model(passive_model_path)
    adv_model = load_model(adversarial_model_path) \
                if adversarial_model_path else None

    print("\n" + "=" * 70)
    print("PHASE 4 - HONEST FAILURE ANALYSIS")
    print("RL Agent vs Static Optimal AC across 5 Market Regimes")
    print("=" * 70)

    print("\nEvaluating on synthetic (training environment)...")
    synth = evaluate_on_synthetic(passive_model, n_episodes=200,
                                  n_state_dims=n_state_dims)
    print(f"  Passive RL synthetic IS: {synth['mean_IS']:.4f}%")
    print(f"  Variance: {synth['var_IS']:.4f}")

    results = []

    for date, regime in DATES.items():
        print(f"\nEvaluating {date} ({regime})...")

        pr = evaluate_model_on_date(
            passive_model, date,
            adversarial=False,
            n_episodes=n_episodes,
            n_state_dims=n_state_dims)

        row = {
            "date": date,
            "regime": regime,
            "rl_passive_IS": pr["mean_IS"],
            "rl_passive_var": pr["var_IS"],
            "rl_passive_cvar95": pr["cvar95"],
            "rl_passive_forced_liq": pr["forced_liquidation_rate"],
            "rl_passive_degradation":
                pr["mean_IS"] - synth["mean_IS"],
            "mean_action": pr["mean_action"],
            "action_entropy": pr["action_entropy"],
            "action_distribution": pr["action_distribution"],
        }

        print(f"  Mean action: {pr['mean_action']:.2f}")
        print(f"  Action entropy: {pr['action_entropy']:.3f}")
        top3 = sorted(
            pr["action_distribution"].items(),
            key=lambda x: -x[1],
        )[:3]
        print(f"  Top 3 actions: {top3}")
        print(f"  RL IS (LOB replay): {pr['mean_IS']:.4f}%")

        if legacy_mixed_comparison:
            # WARNING: STATIC_OPTIMAL_IS is a Monte Carlo point estimate,
            # pr["is_samples"] are counterfactual-LOB-replay samples. This
            # t-test compares across two different simulators and is not a
            # methodologically valid significance test -- see
            # evaluate_all_strategies_on_date for the valid comparison.
            stat_test = test_vs_baseline(
                pr["is_samples"],
                STATIC_OPTIMAL_IS[date],
            )
            row.update({
                "twap_IS_MONTE_CARLO_DO_NOT_COMPARE": TWAP_IS.get(date),
                "heuristic_IS_MONTE_CARLO_DO_NOT_COMPARE": HEURISTIC_IS.get(date),
                "static_optimal_IS_MONTE_CARLO_DO_NOT_COMPARE": STATIC_OPTIMAL_IS[date],
                "adaptive_optimal_IS_MONTE_CARLO_DO_NOT_COMPARE": ADAPTIVE_OPTIMAL_IS[date],
                "p_value_vs_ac_INVALID_MIXED_SIMULATOR": stat_test["p_value"],
                "ci_lower": stat_test["ci_lower"],
                "ci_upper": stat_test["ci_upper"],
                "significantly_better_INVALID_MIXED_SIMULATOR":
                    stat_test["significantly_better"],
            })
            baseline = STATIC_OPTIMAL_IS[date]
            print(
                f"  [LEGACY, MIXED SIMULATORS -- NOT A VALID COMPARISON] "
                f"RL 95% CI [{stat_test['ci_lower']:.4f}%, "
                f"{stat_test['ci_upper']:.4f}%]"
            )
            if stat_test["p_value"] is None:
                print(
                    f"  [LEGACY] vs AC Monte Carlo ({baseline:.4f}%): "
                    f"p=N/A (zero variance — deterministic policy)"
                )
            else:
                sig = stat_test["significantly_better"]
                label = "SIGNIFICANT" if sig else "not significant"
                print(
                    f"  [LEGACY] vs AC Monte Carlo ({baseline:.4f}%): "
                    f"p={stat_test['p_value']:.4f} {label}"
                )

        if adv_model:
            ar = evaluate_model_on_date(
                adv_model, date,
                adversarial=False,
                n_episodes=n_episodes,
                n_state_dims=n_state_dims)
            row.update({
                "rl_adv_IS": ar["mean_IS"],
                "rl_adv_var": ar["var_IS"],
                "rl_adv_cvar95": ar["cvar95"],
                "rl_adv_forced_liq": ar["forced_liquidation_rate"],
                "rl_adv_degradation":
                    ar["mean_IS"] - synth["mean_IS"],
            })

        results.append(row)

    df = pd.DataFrame(results)

    if legacy_mixed_comparison:
        print("\n" + "=" * 90)
        print("[LEGACY, MIXED SIMULATORS -- NOT A VALID COMPARISON]")
        print("STRATEGY COMPARISON - IS% across 5 dates")
        print("TWAP/Heuristic/StatOpt/AdaptOpt are Monte Carlo point estimates;")
        print("RL-Pass is a live counterfactual-LOB-replay mean. Different")
        print("simulators -- see evaluate_all_strategies_on_date for the valid")
        print("apples-to-apples comparison instead.")
        print("=" * 90)

        header = (
            f"{'Date':<12} {'Regime':<22} "
            f"{'TWAP':>8} {'Heuristic':>10} "
            f"{'StatOpt':>8} {'AdaptOpt':>9} {'RL-Pass':>8}"
        )
        if adv_model:
            header += f" {'RL-Adv':>8}"
        print(header)
        print("-" * len(header))

        for _, row in df.iterrows():
            twap = row.get("twap_IS_MONTE_CARLO_DO_NOT_COMPARE")
            heur = row.get("heuristic_IS_MONTE_CARLO_DO_NOT_COMPARE")
            twap_s = f"{twap:>7.3f}%" if twap is not None else f"{'N/A':>8}"
            heur_s = (
                f"{heur:>9.3f}%" if heur is not None else f"{'N/A':>10}"
            )
            line = (
                f"{row['date']:<12} {row['regime']:<22} "
                f"{twap_s} {heur_s} "
                f"{row['static_optimal_IS_MONTE_CARLO_DO_NOT_COMPARE']:>7.3f}% "
                f"{row['adaptive_optimal_IS_MONTE_CARLO_DO_NOT_COMPARE']:>8.3f}% "
                f"{row['rl_passive_IS']:>7.3f}%"
            )
            if adv_model:
                line += f" {row['rl_adv_IS']:>7.3f}%"
            print(line)

    print("\n" + "=" * 70)
    print("SIM-TO-REAL DEGRADATION - RL (passive training)")
    print(f"Synthetic baseline IS: {synth['mean_IS']:.4f}%")
    print("=" * 70)

    for _, row in df.iterrows():
        deg = row['rl_passive_degradation']
        fl  = row['rl_passive_forced_liq']
        print(f"{row['date']} ({row['regime']}):")
        print(f"  IS degradation: {deg:+.4f}pp vs synthetic")
        print(f"  Forced liquidations: {fl:.1%} of episodes")
        print(f"  CVaR95: {row['rl_passive_cvar95']:.4f}%")

    print("\n" + "=" * 70)
    print("VARIANCE ANALYSIS - is RL reducing variance vs TWAP?")
    print("=" * 70)
    print("(Compare rl_passive_var against TWAP IS variance ~930M")
    print(" and Static Optimal IS variance from Phase 2 results)")

    for _, row in df.iterrows():
        print(f"{row['date']}: RL var = {row['rl_passive_var']:.2f}")

    csv_path = f"{output_dir}failure_analysis.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")

    if eval_counterfactual:
        print("\n" + "=" * 70)
        print("COUNTERFACTUAL EVALUATION (training environment)")
        print("=" * 70)

        cf_results = []
        for date, regime in DATES.items():
            r = evaluate_model_counterfactual(
                passive_model, date,
                n_episodes=n_episodes,
                btc_target=btc_target,
                n_state_dims=n_state_dims,
            )
            cf_results.append({
                "date": date,
                "regime": regime,
                "cf_IS": r["mean_IS"],
                "cf_var": r["var_IS"],
                "cf_std": r["std_IS"],
            })
            print(f"{date} ({regime}):")
            print(f"  IS: {r['mean_IS']:.4f}% +/- {r['std_IS']:.4f}%")
            print(f"  Variance: {r['var_IS']:.6f}")
            print(f"  CVaR95: {r['cvar95']:.4f}%")

        pd.DataFrame(cf_results).to_csv(
            f"{output_dir}counterfactual_eval.csv", index=False)

    if include_test_dates:
        print("\n" + "=" * 70)
        print("OUT-OF-SAMPLE TEST DATES (never used in training)")
        print("=" * 70)

        test_results = []
        for date, regime in TEST_DATES.items():
            lob_path = book_path(date)
            if not os.path.exists(lob_path):
                print(f"  SKIPPING {date} -- file not found")
                continue

            r = evaluate_model_on_date(
                passive_model, date,
                n_episodes=n_episodes,
                n_state_dims=n_state_dims,
            )
            test_results.append({
                "date": date,
                "regime": regime,
                "RL_IS": r["mean_IS"],
                "RL_var": r["var_IS"],
            })
            print(f"{date} ({regime}): IS={r['mean_IS']:.4f}%")

        if test_results:
            pd.DataFrame(test_results).to_csv(
                f"{output_dir}test_dates_eval.csv", index=False)

    avg_degradation = df['rl_passive_degradation'].mean()
    worst_date = df.loc[
        df['rl_passive_degradation'].idxmax(), 'date']
    worst_regime = df.loc[
        df['rl_passive_degradation'].idxmax(), 'regime']

    print("\n" + "=" * 70)
    print("FINDING SUMMARY")
    print("=" * 70)
    print(f"Mean sim-to-real degradation: {avg_degradation:+.4f}pp")
    print(f"Worst date: {worst_date} ({worst_regime})")
    print(f"  This regime differs most from GARCH training data.")
    print(f"  Likely cause: "
          f"{'jump events' if 'crash' in worst_regime else 'vol regime mismatch'}")

    if adv_model:
        avg_adv_deg = df['rl_adv_degradation'].mean()
        improvement = avg_degradation - avg_adv_deg
        print(f"\nAdversarial training effect: {improvement:+.4f}pp")
        if improvement > 0:
            print("  Adversarial training reduces sim-to-real gap.")
        else:
            print("  Adversarial training did not reduce gap on passive book.")
            print("  Likely cause: adversarial agents don't match real market.")

    if unified_comparison:
        print("\n" + "=" * 70)
        print("RUNNING UNIFIED COMPARISON")
        print("Evaluating all 5 strategies in same counterfactual LOB...")
        print("=" * 70)

        unified_results = {}
        dates_to_evaluate = list(DATES.keys())
        if include_test_dates:
            dates_to_evaluate += [
                d for d in TEST_DATES
                if os.path.exists(
                    book_path(d)
                )
            ]

        for date in dates_to_evaluate:
            lob_path = book_path(date)
            if not os.path.exists(lob_path):
                print(f"  SKIPPING {date} — LOB file not found")
                continue

            print(f"\n  Evaluating {date}...")
            date_results = evaluate_all_strategies_on_date(
                rl_model=passive_model,
                date=date,
                n_episodes=n_episodes,
                n_state_dims=n_state_dims,
                ac_lambda=ac_lambda,
            )
            unified_results[date] = date_results

        print_unified_comparison(unified_results)

        unified_rows = []
        for date, date_results in unified_results.items():
            for strategy_key, r in date_results.items():
                unified_rows.append({
                    "date": date,
                    "regime": DATES.get(date) or TEST_DATES.get(date, ""),
                    "strategy": r["strategy"],
                    "mean_IS": r["mean_IS"],
                    "std_IS": r["std_IS"],
                    "cvar_95": r["cvar_95"],
                    "is_99th_pct": r["is_99th_pct"],
                    "is_sharpe": r["is_sharpe"],
                    "forced_liquidation_rate": r["forced_liquidation_rate"],
                })

        unified_df = pd.DataFrame(unified_rows)
        unified_csv = f"{output_dir}unified_comparison.csv"
        unified_df.to_csv(unified_csv, index=False)
        print(f"\nUnified results saved: {unified_csv}")
