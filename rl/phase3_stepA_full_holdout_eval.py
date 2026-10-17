"""
Phase 3 Step A: run the existing Step 3/4 sqrt-impact comparison harness across
ALL 9 HELD_OUT_DATES_V2, not just the 4 dates prior headlines reported.

Predictions were pre-registered in docs/experiments/10-full-holdout-evaluation-predictions.md before this
was run. Single-variable change from phase3_step4_sqrt_impact.csv: same
checkpoint, same honest lambda, same n_episodes, same policy set -- only the
date coverage changes.

Usage:
    rl/.venv/bin/python rl/phase3_stepA_full_holdout_eval.py \
        --model rl/models/phase3_step4_capacity_entropy_best/best_model
"""
from __future__ import annotations

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from evaluate import (
    load_model,
    evaluate_all_strategies_on_date,
    query_calibration,
    HELD_OUT_DATES_V2,
    SEVERITY_RANK_V2,
    TRAINING_DATES_V2,
)

HONEST_LAMBDA = 1e-6
BTC_TARGET = 50_000.0
N_STATE_DIMS = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-episodes", type=int, default=20)
    ap.add_argument("--sqrt-fit-seeds", type=int, default=3)
    ap.add_argument("--sqrt-fit-maxiter", type=int, default=25)
    ap.add_argument("--out", default="rl/results/phase3_stepA_full_holdout.csv")
    args = ap.parse_args()

    assert not (set(TRAINING_DATES_V2) & set(HELD_OUT_DATES_V2)), (
        "TRAINING_DATES_V2 / HELD_OUT_DATES_V2 overlap -- refusing to run, "
        "this would invalidate every held-out comparison"
    )

    rl_model = load_model(args.model)

    rows = []
    for date in HELD_OUT_DATES_V2:
        regime = SEVERITY_RANK_V2[date][1]
        calib = query_calibration(date, btc_target=BTC_TARGET)
        print(f"\n=== {date} ({regime}) -- calibrated Y={calib['impact_coefficient']:.4f} "
              f"sigma={calib['sigma']:.4f} adv_btc={calib['adv_btc']:.0f} ===")

        results = evaluate_all_strategies_on_date(
            rl_model=rl_model,
            date=date,
            n_episodes=args.n_episodes,
            n_state_dims=N_STATE_DIMS,
            mode="counterfactual",
            btc_target=BTC_TARGET,
            ac_lambda=HONEST_LAMBDA,
            include_sqrt_impact_optimal=True,
            sqrt_impact_n_fit_seeds=args.sqrt_fit_seeds,
            sqrt_impact_maxiter=args.sqrt_fit_maxiter,
        )

        for strategy_key, r in results.items():
            rows.append({
                "date": date,
                "regime": regime,
                "strategy": r["strategy"],
                "mean_IS": r["mean_IS"],
                "std_IS": r["std_IS"],
                "cvar_95": r["cvar_95"],
                "cvar_95_ci_lower": r["cvar_95_ci_lower"],
                "cvar_95_ci_upper": r["cvar_95_ci_upper"],
                "n_episodes": r["n_episodes"],
                "impact_coefficient": calib["impact_coefficient"],
                "sigma": calib["sigma"],
                "adv_btc": calib["adv_btc"],
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df)} rows to {args.out}")

    pivot = df.pivot(index=["date", "regime"], columns="strategy", values="mean_IS")
    print("\nmean_IS (all 9 HELD_OUT_DATES_V2):")
    print(pivot.round(4).to_string())

    if "RL Agent" in df["strategy"].unique() and "AC Optimal" in df["strategy"].unique():
        rl = df[df.strategy == "RL Agent"].set_index("date")["mean_IS"]
        ac = df[df.strategy == "AC Optimal"].set_index("date")["mean_IS"]
        delta = (rl - ac).sort_values()
        print("\nRL - AC-Optimal (positive = RL better):")
        for date, d in delta.items():
            regime = SEVERITY_RANK_V2[date][1]
            print(f"    {date} ({regime:6}) {d:+.4f}")


if __name__ == "__main__":
    main()
