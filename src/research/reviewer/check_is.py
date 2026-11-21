import numpy as np, pandas as pd
d=pd.read_csv("data/processed/research_implied_storage/implied_weekly_main.csv",parse_dates=["week","origin","target","obs_date"])
assert d.obs_date.max()<=pd.Timestamp("2021-12-31") and d.origin.max()<=pd.Timestamp("2021-12-31")
print(len(d),"rows; informative",d.informative.sum())
print("target month/day:",d.target.dt.strftime("%m-%d").value_counts().to_dict())
d["path5"]=d.avg5+d.x_info
print("benchmark vs avg5+x_info: mean diff",round((d.benchmark-d.path5).mean(),1),"sd",round((d.benchmark-d.path5).std(),1))
inf=d[d.informative]; acc=inf[inf.actual.notna()]
print("acc n",len(acc))
for nm,c in (("implied","L_hat"),("M2","p50"),("5y bench","benchmark"),("5y avg5+x_info","path5")):
    e=acc[c]-acc.actual; print(f"{nm:15s} MAE {e.abs().mean():.0f} bias {e.mean():.0f}")
print("by winter MAE/bias:");g=acc.assign(eL=acc.L_hat-acc.actual,eM=acc.p50-acc.actual,eB=acc.benchmark-acc.actual)
print(g.groupby("winter")[["eL","eM","eB"]].agg(lambda s:f"{s.abs().mean():.0f}/{s.mean():.0f}").assign(n=g.groupby("winter").size()).to_string())
# all 70 weeks with actual: M2 vs 5y
a=d[d.actual.notna()]; print("ALL rows with actual n",len(a),"M2 MAE",round((a.p50-a.actual).abs().mean()),"5y MAE",round((a.benchmark-a.actual).abs().mean()))
# debiased M2: subtract trailing? just shift by its mean bias on these weeks
print("M2 median - 5y bench on informative:",round((inf.p50-inf.benchmark).mean()),"on all:",round((d.p50-d.benchmark).mean()))
print("gap G informative mean",round(inf.G.mean()),"; L_hat - bench",round((inf.L_hat-inf.benchmark).mean()))
# selection: spread level in informative vs not
print("mean spread/april informative vs not:",round(inf.y.mean(),4),round(d[~d.informative].y.mean(),4))
print("noninformative implied x_hat (where finite):",d[~d.informative].x_hat.describe().round(0).to_dict())
print("floored rows",(inf.x_hat+inf.avg5<500).sum())
# who moves: informative at t+h selection effect
w=pd.read_csv("data/processed/research_implied_storage/who_moves_h1.csv")
print("who_moves h1 n",len(w))
# rows informative at t whose t+1 row is non-informative
d=d.sort_values("week"); nx=d[["week","informative","y"]].copy(); nx["week"]-=pd.Timedelta(days=7)
m=d.merge(nx,on="week",suffixes=("","_n")); m=m[m.informative]
print("informative at t:",len(m),"dropped because t+1 non-informative:",(~m.informative_n).sum(),
      "their mean dy",round((m.y_n-m.y)[~m.informative_n].mean(),4),"kept mean dy",round((m.y_n-m.y)[m.informative_n].mean(),4))
