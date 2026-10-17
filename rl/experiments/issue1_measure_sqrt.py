"""docs/experiments/12-forced-liquidation-at-data-exhaustion.md.

Run twice, once per binary, selected via STRATAEXEC_RL_ENV_BIN (the pre-fix
binary is HEAD~1 of the fix commit). Issue 1, sqrt-impact baseline: refit + evaluate on TRAINING_DATES_V2 with
the binary given by STRATAEXEC_RL_ENV_BIN. Step A settings."""
import sys, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/
from environment import StrataExecEnv
from evaluation.dates import TRAINING_DATES_V2
from evaluation.policies.sqrt_impact import SqrtImpactOptimalPolicy
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
rows = []
for date in TRAINING_DATES_V2:
    pol = SqrtImpactOptimalPolicy(date=date, n_eval_episodes=20, n_fit_seeds=3, maxiter=25)
    env = StrataExecEnv(mode="counterfactual", lob_date=date, agg_date=date, btc_target=50_000.0,
                        n_state_dims=16, fixed_steps=True, fixed_size=True)
    for seed in range(20):
        obs, info = env.reset(seed=seed); done = False
        while not done:
            act, _ = pol.predict(obs)
            obs, r, done, _, info = env.step(int(np.asarray(act).item()))
        rows.append(dict(date=date, seed=seed, a=pol.a, b=pol.b, fit_IS=pol.fit_mean_is,
                         IS_pct=-info["total_IS_pct"],
                         forced_frac=info.get("forced_liquidation_qty", 0.0) / 1e6,
                         empty_book=info.get("adjusted_book_depth", 1.0) <= 0.0))
    env.close()
    print(date, f"a={pol.a:.3f} b={pol.b:.3f} eval_IS={np.mean([r['IS_pct'] for r in rows if r['date']==date]):.4f}", flush=True)
pd.DataFrame(rows).to_csv(a.out, index=False)
