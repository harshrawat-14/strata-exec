import sys; sys.path.insert(0,'rl')
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import r2_score, balanced_accuracy_score
from evaluation.dates import SEVERITY_RANK_V2, VALIDATION_DATES_V3
df=pd.read_parquet(sys.argv[1])
g=df.groupby(["date","ep"])
# forward drift over the rest of the episode: mean future mid / mid now - 1 (%)
def fwd(m):
    m=m.values; c=np.cumsum(m[::-1])[::-1]; n=np.arange(len(m),0,-1)
    out=np.full(len(m),np.nan); out[:-1]=(c[1:]/n[1:])/m[:-1]-1; return out*100
df["y"]=g.mid.transform(fwd)
for L in (10,60,240):
    df[f"r{L}"]=g.mid.transform(lambda m: m/m.shift(L)-1).fillna(0)*100
df["regime"]=df.date.map(lambda d: SEVERITY_RANK_V2[d][1])
df=df.dropna(subset=["y"])
va=df.date.isin(VALIDATION_DATES_V3); tr=~va
sets={"market dims (2-6,10,11,13)":["s2","s3","s4","s5","s6","s10","s11","s13"],
      "+ time of day (12,14,15)":["s2","s3","s4","s5","s6","s10","s11","s13","s12","s14","s15"],
      "market + past returns 10/60/240":["s2","s3","s4","s5","s6","s10","s11","s13","r10","r60","r240"]}
print("forward drift over rest of episode, R^2 on validation dates (train dates in parens)")
for n,f in sets.items():
    for m in (RidgeCV(alphas=np.logspace(-3,3,13)), HistGradientBoostingRegressor(max_iter=200)):
        m.fit(df.loc[tr,f],df.loc[tr,"y"])
        print(f"  {n:<34}{type(m).__name__[:5]}: {r2_score(df.loc[va,'y'],m.predict(df.loc[va,f])):+.3f} ({r2_score(df.loc[tr,'y'],m.predict(df.loc[tr,f])):+.3f})")
print("regime class, balanced accuracy on validation (chance 0.33)")
for n,f in sets.items():
    m=HistGradientBoostingClassifier(max_iter=200).fit(df.loc[tr,f],df.loc[tr,"regime"])
    print(f"  {n:<34} {balanced_accuracy_score(df.loc[va,'regime'],m.predict(df.loc[va,f])):.3f}  (train {balanced_accuracy_score(df.loc[tr,'regime'],m.predict(df.loc[tr,f])):.3f})")
print("sd of forward drift (pp):",df.y.std().round(3))
from scipy.stats import pearsonr
f=sets["market + past returns 10/60/240"]
m=RidgeCV(alphas=np.logspace(-3,3,13)).fit(df.loc[tr,f],df.loc[tr,"y"]); p=m.predict(df.loc[va,f])
print("R^2 of predicting 0 on validation:", round(r2_score(df.loc[va,'y'],np.zeros(va.sum())),3), " ridge corr:", round(pearsonr(p,df.loc[va,'y'])[0],3))
for d in VALIDATION_DATES_V3:
    s=df.date==d; print(f"  {d} corr {pearsonr(m.predict(df.loc[s,f]),df.loc[s,'y'])[0]:+.3f}")
# crash vs not, from vol alone
df["crash"]=df.regime=="crash"
c=LogisticRegression().fit(df.loc[tr,["s10"]],df.loc[tr,"crash"])
pr=c.predict_proba(df.loc[va,["s10"]])[:,1]
print("P(crash) from vol_regime, mean by validation date:", df.loc[va].assign(p=pr).groupby("date").p.mean().round(2).to_dict())
print("train dates:", df.loc[tr].assign(p=c.predict_proba(df.loc[tr,['s10']])[:,1]).groupby(['date','regime']).p.mean().round(2).to_dict())
