"""
Honest-lambda re-selection under evaluation protocol v3.

Rule (fixed before running, same as the original honest-lambda protocol in
docs/experiments/02-reward-objective-audit.md): ONE lambda for all dates,
chosen to maximise AC-Optimal's mean IS averaged over TRAINING_DATES_V3 only.
Reported alongside: the median across dates, and Adaptive-AC's choice, as
consistency checks. Validation and held-out dates are never touched.

Protocol v3: 1,920-step episodes, 20 evenly spaced starts per date, fixed
order size, closed-loop schedule tracking (docs/methodology/evaluation-protocol-v3.md).

Grid: the original 13-point grid plus 3e-6, 3e-5, 3e-4 for resolution near
the previous optimum (1e-6).

Run from the repo root:
    python rl/experiments/v3_lambda_selection.py
"""
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/, for bare imports like `import evaluate`

import numpy as np
import pandas as pd

LAMBDA_GRID = sorted([1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5,
                      1e-4, 3e-4, 1e-3, 1e-2, 1e-1, 1.0])
OUT = "rl/results/v3_lambda_grid.csv"


def run_date(date):
    from environment import StrataExecEnv
    from evaluation.dates import SEVERITY_RANK_V2
    from evaluation.episode import _run_episode
    from evaluation.policies.almgren_chriss import ACOptimalPolicy, AdaptiveACPolicy
    from evaluation.protocol import EPISODE_STEPS, N_EVAL_EPISODES, eval_start_fracs

    env = StrataExecEnv(mode="counterfactual", lob_date=date, agg_date=date,
                        btc_target=50_000.0, n_state_dims=16, fixed_steps=True,
                        fixed_size=True, episode_steps=EPISODE_STEPS)
    fracs = eval_start_fracs(N_EVAL_EPISODES)
    rows = []
    try:
        for lam in LAMBDA_GRID:
            for name, policy in (
                ("AC Optimal", ACOptimalPolicy(16, lam=lam, track_schedule=True)),
                ("Adaptive AC", AdaptiveACPolicy(16, lam=lam, track_schedule=True)),
            ):
                out = [_run_episode(policy, env, seed=ep, start_frac=fracs[ep])
                       for ep in range(N_EVAL_EPISODES)]
                is_vals = [o[0] for o in out if o[0] is not None]
                rows.append(dict(date=date, regime=SEVERITY_RANK_V2[date][1], strategy=name,
                                 lam=lam, mean_IS=float(np.mean(is_vals)),
                                 forced_rate=float(np.mean([o[2] for o in out]))))
    finally:
        env.close()
    print(f"  {date} done", flush=True)
    return rows


def main():
    from evaluation.dates import TRAINING_DATES_V3

    with Pool(6) as pool:
        rows = [r for rs in pool.map(run_date, TRAINING_DATES_V3) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    agg = df.pivot_table(index="lam", columns="strategy", values="mean_IS", aggfunc=["mean", "median"])
    forced = df.pivot_table(index="lam", columns="strategy", values="forced_rate", aggfunc="mean")
    print("\nmean_IS across the 12 TRAINING_DATES_V3 (higher = better):")
    print(agg.round(4).to_string())
    print("\nforced-liquidation rate:")
    print(forced.round(3).to_string())
    best = {s: agg[("mean", s)].idxmax() for s in ("AC Optimal", "Adaptive AC")}
    best_med = agg[("median", "AC Optimal")].idxmax()
    print(f"\nSELECTED (AC mean): lambda = {best['AC Optimal']:.0e}")
    print(f"  checks: AC median -> {best_med:.0e}; Adaptive-AC mean -> {best['Adaptive AC']:.0e}")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
