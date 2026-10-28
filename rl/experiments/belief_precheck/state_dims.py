import sys; sys.path.insert(0,'rl')
import numpy as np
from environment import StrataExecEnv
names=["inventory_left","time_left","vol(tanh)","half_spread","depth_1pct","OFI","drift_10step","last_slippage","perm_impact","temp_impact","vol_regime","liq_regime","time_of_day","intraday_vol","sin_tod","cos_tod"]
rows={}
for d,reg in [("2024-01-15","calm"),("2024-04-15","crash"),("2024-02-20","normal")]:
    env=StrataExecEnv(mode="counterfactual",lob_date=d,agg_date=d,n_state_dims=16,seed=9999,fixed_steps=True,fixed_size=True,episode_steps=1920,start_cycle=20,order_share_adv=0.05)
    S=[]
    for ep in range(3):
        o,_=env.reset(); done=False
        while not done:
            S.append(o); o,r,t,tr,_=env.step(1); done=t or tr
    env.close(); rows[reg]=np.array(S)
print(f"{'dim':<16}"+"".join(f"{r+' mean':>12}{'sd':>8}" for r in rows))
for i,n in enumerate(names):
    print(f"{i:>2} {n:<13}"+"".join(f"{rows[r][:,i].mean():>12.3f}{rows[r][:,i].std():>8.3f}" for r in rows))
