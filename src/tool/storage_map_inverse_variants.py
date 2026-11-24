"""Inverse-regression variants x ~ asinh(S/c) (agent storage_map): c grid, heteroskedastic, % target, ln(H/J).
Run: python -m src.tool.storage_map_inverse_variants  (~1.5 min) -> folds_inverse_variants.csv"""
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, t as tdist
from src.tool import _map_data as MD
from src.tool.storage_map_eval import score_grid, XG, DX, WF_MIN
t = MD.winter_table()
folds = [("lowo", Y) for Y in t.index] + [("wf", Y) for Y in t.index if (t.index < Y).sum() >= WF_MIN]
def fit_h(z, y, hetero):
    X = np.column_stack([np.ones_like(z), z]); b,*_ = np.linalg.lstsq(X, y, rcond=None)
    s = np.std(y - X@b)
    if not hetero: return np.r_[b, np.log(s), 0.0]
    nll = lambda p: np.sum(p[2]+p[3]*z + 0.5*((y-p[0]-p[1]*z)/np.exp(p[2]+p[3]*z))**2)
    return minimize(nll, np.r_[b, np.log(s), 0.0], method="Nelder-Mead", options={"maxiter":4000,"xatol":1e-6,"fatol":1e-8}).x
rows=[]
def run(name, z, yv, scale, hetero=False, B=100):
    for mode, Y in folds:
        tr = (t.index != Y) if mode=="lowo" else (t.index < Y)
        zt, yt = z[tr], yv[tr]; n=len(zt); rng=np.random.default_rng(Y)
        draws=[fit_h(zt, yt, hetero)]
        for _ in range(B):
            i=rng.integers(0,n,n); draws.append(fit_h(zt[i], yt[i], hetero))
        zo = z[t.index==Y][0]; sc = scale[t.index==Y][0]
        p = np.zeros_like(XG)
        for d in draws:
            mu = (d[0]+d[1]*zo)*sc; sd = np.exp(d[2]+d[3]*zo)*sc*np.sqrt(n/(n-2))
            p += norm.pdf(XG, mu, sd)
        p/=p.sum(); xo=t.x[Y]
        rows.append({"spec":name,"mode":mode,"Y":Y,"x":xo,**score_grid(p,xo)})
one = np.ones(len(t))
for c in (0.02,0.05,0.1,0.2):
    run(f"inv_c{c}", np.arcsinh(t.S.values/c), t.x.values, one)
run("inv_c0.05_het", np.arcsinh(t.S.values/0.05), t.x.values, one, hetero=True)
run("inv_c0.05_pct", np.arcsinh(t.S.values/0.05), t.xpct.values, t.avg5_m.values/100)
run("inv_rel_c0.01", np.arcsinh(t.lnHJ.values/0.01), t.x.values, one)
d=pd.DataFrame(rows); d.to_csv(MD.CACHE/"folds_inverse_variants.csv",index=False)
out=[]
for (s,m),g in d.groupby(["spec","mode"]):
  for w,gg in (("all",g),("2010+",g[g.Y>=2010])):
    out.append({"spec":s,"mode":m,"win":w,"n":len(gg),"crps":gg.crps.mean(),"logs":gg.logs.mean(),"sd":gg.sd.mean(),
      "cov50":((gg.pit>.25)&(gg.pit<.75)).mean(),"cov80":((gg.pit>.1)&(gg.pit<.9)).mean(),"cov90":((gg.pit>.05)&(gg.pit<.95)).mean(),
      **{f"b{-x0}":((gg[f"P{-x0}"]-(gg.x<x0))**2).mean() for x0 in (-200,-400,-600)}})
pd.set_option("display.width",200); print(pd.DataFrame(out).round(3).to_string())
