"""
Constant schedule offsets around AC (no learning), VALIDATION_DATES_V3, protocol v3.
Bounds what a fixed pace shift can achieve with the M2 residual action space
(docs/experiments/15-*). delta = 0 is AC itself.

    python rl/experiments/residual_constant_offsets.py rl/results/residual_constant_offsets.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/, for bare imports like `import evaluate`
from multiprocessing import Pool
import numpy as np, pandas as pd
def run(args):
    date, delta = args
    from environment import StrataExecEnv
    from evaluation.episode import _run_episode
    from evaluation.protocol import EPISODE_STEPS, eval_start_fracs
    from evaluation.residual import DELTAS, ResidualACAgent
    class Fixed:
        def predict(self, obs, state=None, episode_start=None, deterministic=True):
            return np.array([DELTAS.index(delta)]), state
    env = StrataExecEnv(mode="counterfactual", lob_date=date, agg_date=date, btc_target=50_000.0,
                        n_state_dims=16, fixed_steps=True, fixed_size=True, episode_steps=EPISODE_STEPS)
    out = [_run_episode(ResidualACAgent(Fixed()), env, seed=ep, start_frac=f) for ep, f in enumerate(eval_start_fracs(20))]
    env.close()
    return dict(date=date, delta=delta, IS=float(np.mean([o[0] for o in out])), forced=float(np.mean([o[2] for o in out])))
if __name__ == "__main__":
    from evaluation.dates import VALIDATION_DATES_V3
    from evaluation.residual import DELTAS
    with Pool(9) as p:
        df = pd.DataFrame(p.map(run, [(d, x) for d in VALIDATION_DATES_V3 for x in DELTAS]))
    df.to_csv(sys.argv[1], index=False)
    piv = df.pivot(index="delta", columns="date", values="IS")
    print(piv.round(4).to_string())
    print("\nforced rate by delta:", df.groupby("delta").forced.mean().round(2).to_dict())
    ac = piv.loc[0.0]
    print("\nbest constant delta per date (hindsight) and gain vs AC:")
    for d in piv.columns: print(f"  {d}: delta={piv[d].idxmax():+.2f}  gain={piv[d].max()-ac[d]:+.3f} pp")
    print(f"best single delta across dates: {piv.mean(axis=1).idxmax():+.2f} (mean gain {piv.mean(axis=1).max()-ac.mean():+.3f} pp)")
