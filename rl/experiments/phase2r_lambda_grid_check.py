"""
Phase 2R gate: is AC_DEFAULT_LAMBDA a strawman? Sweep AC-Optimal/Adaptive-AC
across a wide lambda grid on the 4 dates that anchor the Task 4 anomaly
(2 held-out crash, 2 held-out calm), post-correction simulator, counterfactual
mode, and compare against the already-trained RL arms at each lambda.

Answers directly: does AC at its BEST lambda still lose to RL on calm dates?
Does a better lambda make AC beat RL more decisively on crash dates (as the
Phase 1 grid's direction suggested it might)?

Usage: rl/.venv/bin/python rl/experiments/phase2r_lambda_grid_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/, for bare imports like `import evaluate`

import numpy as np

from evaluate import load_model, evaluate_ac_lambda_grid, evaluate_all_strategies_on_date

MODEL_16DIM = "rl/models/ppo_ablation_16dim_cf_best/best_model"
MODEL_12DIM = "rl/models/ppo_ablation_12dim_cf_best/best_model"
BTC_TARGET = 50_000.0
N_EPISODES = 50
# Wide grid: current default (0.001) plus much more urgent and much more
# patient extremes, so we're not just re-sampling near the same point.
LAMBDA_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

DATES = {
    "2024-03-05": "held-out CRASH",
    "2024-08-05": "held-out CRASH",
    "2024-07-10": "held-out CALM",
    "2024-12-10": "held-out CALM",
}


def main():
    print("Loading RL models (already trained, reused)...")
    v16 = load_model(MODEL_16DIM)
    v12 = load_model(MODEL_12DIM)

    # RL numbers don't depend on lambda -- get them once per date.
    rl_by_date = {}
    for date in DATES:
        r16 = evaluate_all_strategies_on_date(
            rl_model=v16, date=date, n_episodes=N_EPISODES, n_state_dims=16,
            mode="counterfactual", btc_target=BTC_TARGET,
        )
        r12 = evaluate_all_strategies_on_date(
            rl_model=v12, date=date, n_episodes=N_EPISODES, n_state_dims=12,
            mode="counterfactual", btc_target=BTC_TARGET,
        )
        rl_by_date[date] = (r16["rl_agent"]["mean_IS"], r12["rl_agent"]["mean_IS"])

    for date, label in DATES.items():
        rl16_is, rl12_is = rl_by_date[date]
        print(f"\n{'=' * 90}")
        print(f"{date}  ({label})   RL-16dim={rl16_is:+.4f}%   RL-12dim={rl12_is:+.4f}%")
        print(f"{'=' * 90}")
        grid = evaluate_ac_lambda_grid(
            date=date, lambdas=LAMBDA_GRID, n_episodes=N_EPISODES, n_state_dims=16,
            mode="counterfactual", btc_target=BTC_TARGET,
        )
        best_ac_lam, best_ac_is = None, -1e18
        best_aac_lam, best_aac_is = None, -1e18
        print(f"{'lambda':>10s} {'AC_mean_IS':>12s} {'vs RL16':>10s} {'vs RL12':>10s}   "
              f"{'AAC_mean_IS':>12s} {'vs RL16':>10s} {'vs RL12':>10s}")
        for lam in LAMBDA_GRID:
            ac_is = grid[lam]["ac_optimal"]["mean_IS"]
            aac_is = grid[lam]["adaptive_ac"]["mean_IS"]
            if ac_is > best_ac_is:
                best_ac_is, best_ac_lam = ac_is, lam
            if aac_is > best_aac_is:
                best_aac_is, best_aac_lam = aac_is, lam
            print(f"{lam:>10.0e} {ac_is:>+12.4f} {rl16_is-ac_is:>+10.4f} {rl12_is-ac_is:>+10.4f}   "
                  f"{aac_is:>+12.4f} {rl16_is-aac_is:>+10.4f} {rl12_is-aac_is:>+10.4f}")
        print(f"\nBest lambda for AC-Optimal:  {best_ac_lam:.0e}  (mean_IS={best_ac_is:+.4f}%)  "
              f"RL16 {'BEATS' if rl16_is>best_ac_is else 'LOSES TO'} best-tuned AC "
              f"(delta={rl16_is-best_ac_is:+.4f}pp)")
        print(f"Best lambda for Adaptive-AC: {best_aac_lam:.0e}  (mean_IS={best_aac_is:+.4f}%)  "
              f"RL16 {'BEATS' if rl16_is>best_aac_is else 'LOSES TO'} best-tuned Adaptive-AC "
              f"(delta={rl16_is-best_aac_is:+.4f}pp)")
        print(f"(default AC_DEFAULT_LAMBDA=0.001 mean_IS was: "
              f"{grid[1e-3]['ac_optimal']['mean_IS']:+.4f}% AC / "
              f"{grid[1e-3]['adaptive_ac']['mean_IS']:+.4f}% AdaptiveAC)")


if __name__ == "__main__":
    main()
