"""Collect era-2 market states + undisturbed mids (action 0 = no own impact)."""
import sys; sys.path.insert(0,'rl')
import numpy as np, pandas as pd
from multiprocessing import Pool
def run(d):
    from environment import StrataExecEnv
    env=StrataExecEnv(mode="counterfactual",lob_date=d,agg_date=d,n_state_dims=16,seed=9999,fixed_steps=True,fixed_size=True,episode_steps=1920,start_cycle=5,order_share_adv=0.05)
    out=[]
    for ep in range(5):
        o,_=env.reset(); done=False; k=0
        while not done:
            o2,r,t,tr,i=env.step(0); done=t or tr
            out.append(dict(date=d,ep=ep,k=k,mid=i["book_mid"],**{f"s{j}":float(o[j]) for j in range(16)}))
            o=o2; k+=1
    env.close(); return out
if __name__=="__main__":
    from evaluation.dates import TRAINING_DATES_V3, VALIDATION_DATES_V3
    with Pool(8) as p: rows=[r for rs in p.map(run, list(TRAINING_DATES_V3)+list(VALIDATION_DATES_V3)) for r in rs]
    pd.DataFrame(rows).to_parquet(sys.argv[1]); print(len(rows))
