"""docs/experiments/12-forced-liquidation-at-data-exhaustion.md.

Run twice, once per binary, selected via STRATAEXEC_RL_ENV_BIN (the pre-fix
binary is HEAD~1 of the fix commit). Issue 1 measurement: how eval episodes end in counterfactual fixed-horizon
mode. TRAINING_DATES_V2 only. Writes one row per (date, policy, seed)."""
import sys, json, argparse
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/
from environment import StrataExecEnv
from evaluation.dates import TRAINING_DATES_V2
from evaluation.episode import load_model
from evaluation.policies.baselines import TWAPPolicy, HeuristicPolicy
from evaluation.policies.almgren_chriss import ACOptimalPolicy, AdaptiveACPolicy

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
ap.add_argument("--seeds", type=int, default=20); args = ap.parse_args()
TOTAL = 1_000_000.0  # default total_inventory (fixed_size)
policies = {
    "TWAP": TWAPPolicy(16),
    "AC": ACOptimalPolicy(16, lam=1e-6),
    "Heuristic": HeuristicPolicy(16),
    "AdaptiveAC": AdaptiveACPolicy(16, lam=1e-6),
    "RL": load_model("rl/models/phase3_step4_capacity_entropy_best/best_model"),
}
rows = []
for date in TRAINING_DATES_V2:
    env = StrataExecEnv(mode="counterfactual", lob_date=date, agg_date=date, btc_target=50_000.0,
                        n_state_dims=16, fixed_steps=True, fixed_size=True)
    for name, pol in policies.items():
        for seed in range(args.seeds):
            obs, info = env.reset(seed=seed)
            state, start, done, steps = None, np.ones((1,), dtype=bool), False, 0
            while not done:
                a, state = pol.predict(obs, state=state, episode_start=start, deterministic=True)
                start = np.zeros((1,), dtype=bool)
                obs, r, done, _, info = env.step(int(np.asarray(a).item())); steps += 1
            fq = info.get("forced_liquidation_qty", 0.0)
            rows.append(dict(date=date, policy=name, seed=seed, steps=steps,
                             exhausted=info.get("adjusted_book_depth", 1.0) <= 0.0,
                             forced_frac=fq / TOTAL,
                             forced_filled_frac=info.get("forced_liq_filled_qty", 0.0) / TOTAL,
                             punitive_notional=info.get("forced_liq_punitive_notional", 0.0),
                             IS_pct=-info["total_IS_pct"]))
    env.close()
    print(date, flush=True)
pd.DataFrame(rows).to_csv(args.out, index=False)
