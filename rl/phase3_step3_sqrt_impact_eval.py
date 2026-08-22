"""
Phase 3 Step 3: numerically-optimized sqrt-impact baseline vs the existing
comparison table (docs/phase3_plan.md Step 3, promoted to critical path
after Step 2b).

Step 2b showed RL beats AC-Optimal on both held-out crash dates while pacing
*slower* and *less* urgently than AC overall -- so the crash-date advantage
does not come from timing. That leaves two live explanations this baseline
is built to separate: (a) AC's LINEAR-impact assumption is simply a poor fit
to the real, empirically square-root/transient-decay cost structure this
project's counterfactual environment actually runs, so AC is a weak
strawman independent of RL; or (b) both AC and RL are far from what is
actually achievable here.

SqrtImpactOptimalPolicy (rl/evaluate.py) fits a 2-parameter Beta-CDF
selling-curve family by running real episodes through the exact simulator
under evaluation (not a hand-derived closed form) and freezing the result --
the closest thing to "the best a schedule-only policy can do under the true
cost model" that this project can produce without a full trajectory
optimizer over the real order book.

Usage:
    rl/.venv/bin/python rl/phase3_step3_sqrt_impact_eval.py \
        --model rl/models/phase3_step2b_allcoverage_best/best_model
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from evaluate import (
    load_model,
    evaluate_all_strategies_on_date,
    query_calibration,
)

HONEST_LAMBDA = 1e-6
BTC_TARGET = 50_000.0
N_STATE_DIMS = 16

# The four dates docs/phase3_step2b_result.md reports (2 held-out crash,
# 1 held-out calm, 1 held-out normal) -- reusing exactly these keeps this
# table directly comparable to the Step 2b numbers already on record.
DATES = {
    "2024-03-05": "crash",
    "2024-08-05": "crash",
    "2024-07-10": "calm",
    "2024-12-10": "normal",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-episodes", type=int, default=20)
    ap.add_argument("--sqrt-fit-seeds", type=int, default=3)
    ap.add_argument("--sqrt-fit-maxiter", type=int, default=25)
    ap.add_argument("--out", default="rl/results/phase3_step3_sqrt_impact.csv")
    args = ap.parse_args()

    rl_model = load_model(args.model)

    rows = []
    for date, regime in DATES.items():
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
    print("\nmean_IS (RL_agent should be read as the model in --model):")
    print(pivot.round(4).to_string())


if __name__ == "__main__":
    main()
