import numpy as np, pandas as pd, sys
from src.research.v4_premium_shocks.common import build, nw_ols, CHOOSE, CHECK
p = build(max_h=30)
p.to_pickle("data/processed/research_reviewer_v4/ps_build30.pkl")
rng = np.random.default_rng(1)
def boot(d, y, xs, B=2000):
    g=[x for _,x in d.groupby("W")]; out=[]
    for _ in range(B):
        s=pd.concat([g[i] for i in rng.integers(0,len(g),len(g))])
        A=np.column_stack([np.ones(len(s)), s[xs].to_numpy(float)])
        out.append(np.linalg.lstsq(A, s[y].to_numpy(float), rcond=None)[0][1:])
    return np.array(out)
for name in ["second","hj"]:
    w = p[p[f"wkprem_{name}"].notna() & p.month.isin([11,12,1,2])]
    for h in [5,10,15,20,25,30]:
        y=f"fut{h}_{name}"; d=w.dropna(subset=[y])
        b,se = nw_ols(d[y], d[[f"wknews_{name}",f"wkprem_{name}"]], h//5+1)
        bn,sn = nw_ols(d[y], d[[f"wkmove_{name}"]], h//5+1)
        bs = boot(d,y,[f"wknews_{name}",f"wkprem_{name}"],1000)
        diff = bs[:,0]-bs[:,1]
        # share of var
        vs = d[f"wknews_{name}"].var()/d[f"wkmove_{name}"].var()
        print(f"{name} h={h:2d} n={len(d)} prem {b[2]:+.2f} (t {b[2]/se[2]:+.1f}) news {b[1]:+.2f} (t {b[1]/se[1]:+.1f}) raw {bn[1]:+.2f} (t {bn[1]/sn[1]:+.1f}) | newsminusprem {np.mean(b[1]-b[2]):+.2f} CI [{np.percentile(diff,2.5):+.2f},{np.percentile(diff,97.5):+.2f}] P(diff<=0)={np.mean(diff<=0):.3f} newsvarshare {vs:.2f}")
    # lowo for h=20 second
    if name=="second":
        d=w.dropna(subset=["fut20_second"])
        for W_ in sorted(d.W.unique()):
            dd=d[d.W!=W_]; b,se=nw_ols(dd.fut20_second, dd[["wknews_second","wkprem_second"]],5)
            print("  ex", W_, f"{b[2]:+.2f} (t {b[2]/se[2]:+.1f})", end=";")
        print()
        for W_ in sorted(d.W.unique()):
            dd=d[d.W==W_]
            if len(dd)>8:
                b,se=nw_ols(dd.fut20_second, dd[["wknews_second","wkprem_second"]],5); print("  only",W_,len(dd),f"{b[2]:+.2f}",end=";")
        print()
