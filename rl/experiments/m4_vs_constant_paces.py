"""M4 (200k) vs constant 1x / 1.5x paces on VALIDATION_DATES_V3, protocol v3 (doc 19 result)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/
from multiprocessing import Pool
import numpy as np, collections
def run(args):
    date, who = args
    from environment import StrataExecEnv
    from evaluation.decoding import MeanValueDecoder
    from evaluation.episode import load_model
    from evaluation.protocol import eval_start_fracs
    from evaluation.rate import RATES, RateAgent
    class Fixed:
        def __init__(self, r): self.i = RATES.index(r)
        def predict(self, obs, state=None, episode_start=None, deterministic=True): return np.array([self.i]), state
    dec = MeanValueDecoder(load_model("rl/models/checkpoints/m4_v3_rate/ckpt_199992_steps"), RATES) if who == "M4" else Fixed(float(who[:-1]))
    env = StrataExecEnv(mode="counterfactual", lob_date=date, agg_date=date, btc_target=50_000.0, n_state_dims=16, fixed_steps=True, fixed_size=True, episode_steps=1920)
    IS, paces, big = [], collections.Counter(), 0.0
    for ep, f in enumerate(eval_start_fracs(20)):
        agent = RateAgent(dec); obs, info = env.reset(seed=ep, options={"start_frac": f}); agent.set_horizon(info["episode_horizon"])
        st=None; s=np.ones((1,),bool); done=False
        while not done:
            a, st = agent.predict(obs, state=st, episode_start=s); s=np.zeros((1,),bool); paces[agent.last_rate]+=1
            obs, r, done, _, info = env.step(int(a[0])); big=max(big, info.get("fill_qty",0)/1e6)
        IS.append(-info["total_IS_pct"])
    env.close()
    n=sum(paces.values()); mean_pace=sum(k*c for k,c in paces.items())/n
    top=", ".join(f"{k:g}x:{c/n:.0%}" for k,c in paces.most_common(3))
    return (date, who, float(np.mean(IS)), mean_pace, top, big)
if __name__ == "__main__":
    from evaluation.dates import VALIDATION_DATES_V3
    with Pool(8) as p:
        res = p.map(run, [(d, w) for d in VALIDATION_DATES_V3 for w in ("M4", "1.0x", "1.5x")])
    for d, w, IS, mp, top, big in res:
        print(f"{d} {w:5s} IS={IS:+.4f}  mean pace={mp:.2f}x  [{top}]  largest trade={big:.2%}")
    import pandas as pd
    df = pd.DataFrame(res, columns=["date","who","IS","pace","top","big"])
    print(df.groupby("who").IS.mean().round(4).to_string())
