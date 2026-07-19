"""
Phase 2R close-out gate, item 1 (greenlit): full 11-date AC/Adaptive-AC lambda-grid
sweep, extended downward past the prior grid's edge (1e-6, which was still the best
point on 3/4 anchor dates -- meaning the true AC optimum wasn't yet bracketed).

RL-16dim / RL-12dim mean_IS per date is NOT re-evaluated -- reused from the existing
post-correction ablation CSV (rl/results/phase2r_ablation_postcorrection.csv), which
was produced by the same models/config this grid compares against, so re-running RL
would just reproduce the same numbers at extra cost.

Usage: rl/.venv/bin/python rl/phase2r_lambda_grid_full.py
"""
from __future__ import annotations

import pandas as pd

from evaluate import evaluate_ac_lambda_grid

BTC_TARGET = 50_000.0
N_EPISODES = 50

# Extended downward from the prior grid ([1e-6..1.0]) since 1e-6 was still the best
# point on 3/4 anchor dates -- bracket the optimum instead of hitting the wall again.
LAMBDA_GRID = [1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

DATE_REGIMES = {
    "2024-01-15": "calm bull",
    "2024-02-20": "moderate volatility",
    "2024-03-05": "crash - flash liquidation cascade",
    "2024-04-15": "BTC correction (moderate stress)",
    "2024-06-10": "quiet consolidation",
    "2024-07-10": "mid-summer sideways",
    "2024-08-05": "crash - Yen unwind",
    "2024-09-15": "moderate volatility",
    "2024-10-15": "moderate volatility",
    "2024-11-06": "post-election surge",
    "2024-12-10": "December bull run",
}


def main():
    rl_df = pd.read_csv("rl/results/phase2r_ablation_postcorrection.csv").set_index("date")

    rows = []
    for date, regime in DATE_REGIMES.items():
        rl16_is = float(rl_df.loc[date, "rl16_mean_IS"])
        rl12_is = float(rl_df.loc[date, "rl12_mean_IS"])
        print(f"\n{'=' * 90}")
        print(f"{date}  ({regime})   RL-16dim={rl16_is:+.4f}%   RL-12dim={rl12_is:+.4f}%")
        print(f"{'=' * 90}")
        grid = evaluate_ac_lambda_grid(
            date=date, lambdas=LAMBDA_GRID, n_episodes=N_EPISODES, n_state_dims=16,
            mode="counterfactual", btc_target=BTC_TARGET,
        )
        best_ac_lam, best_ac_is = None, -1e18
        best_aac_lam, best_aac_is = None, -1e18
        for lam in LAMBDA_GRID:
            ac_is = grid[lam]["ac_optimal"]["mean_IS"]
            aac_is = grid[lam]["adaptive_ac"]["mean_IS"]
            if ac_is > best_ac_is:
                best_ac_is, best_ac_lam = ac_is, lam
            if aac_is > best_aac_is:
                best_aac_is, best_aac_lam = aac_is, lam
            rows.append({
                "date": date, "regime": regime, "lambda": lam,
                "ac_mean_IS": ac_is, "aac_mean_IS": aac_is,
                "rl16_mean_IS": rl16_is, "rl12_mean_IS": rl12_is,
            })
        interior_ac = LAMBDA_GRID[0] < best_ac_lam < LAMBDA_GRID[-1]
        interior_aac = LAMBDA_GRID[0] < best_aac_lam < LAMBDA_GRID[-1]
        print(f"Best AC lambda:  {best_ac_lam:.0e} (mean_IS={best_ac_is:+.4f}%)  "
              f"{'INTERIOR' if interior_ac else 'AT GRID EDGE -- still not bracketed'}  "
              f"RL16 {'beats' if rl16_is>best_ac_is else 'loses to'} best AC "
              f"(delta={rl16_is-best_ac_is:+.4f}pp)")
        print(f"Best AAC lambda: {best_aac_lam:.0e} (mean_IS={best_aac_is:+.4f}%)  "
              f"{'INTERIOR' if interior_aac else 'AT GRID EDGE -- still not bracketed'}  "
              f"RL16 {'beats' if rl16_is>best_aac_is else 'loses to'} best AAC "
              f"(delta={rl16_is-best_aac_is:+.4f}pp)")

    out = pd.DataFrame(rows)
    out.to_csv("rl/results/phase2r_lambda_grid_full.csv", index=False)
    print(f"\nSaved rl/results/phase2r_lambda_grid_full.csv ({len(out)} rows)")

    # Summary across all 11 dates.
    print(f"\n{'=' * 90}\nSUMMARY (best lambda per date)\n{'=' * 90}")
    print(f"{'date':>12s} {'regime':>28s} {'RL16':>9s} {'best AC':>9s} {'@lam':>8s} "
          f"{'best AAC':>9s} {'@lam':>8s} {'RL16 vs best':>13s}")
    wins, losses = 0, 0
    for date, regime in DATE_REGIMES.items():
        d = out[out["date"] == date]
        rl16_is = d["rl16_mean_IS"].iloc[0]
        best_ac_row = d.loc[d["ac_mean_IS"].idxmax()]
        best_aac_row = d.loc[d["aac_mean_IS"].idxmax()]
        best_overall = max(best_ac_row["ac_mean_IS"], best_aac_row["aac_mean_IS"])
        result = "RL16 wins" if rl16_is > best_overall else "RL16 loses"
        wins += rl16_is > best_overall
        losses += rl16_is <= best_overall
        print(f"{date:>12s} {regime:>28s} {rl16_is:>+9.4f} "
              f"{best_ac_row['ac_mean_IS']:>+9.4f} {best_ac_row['lambda']:>8.0e} "
              f"{best_aac_row['aac_mean_IS']:>+9.4f} {best_aac_row['lambda']:>8.0e} "
              f"{result:>13s}")
    print(f"\nRL16 vs best-tuned (AC or AAC) across all 11 dates: {wins} wins / {losses} losses")


if __name__ == "__main__":
    main()
