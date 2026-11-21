import numpy as np, pandas as pd
CUT=pd.Timestamp("2022-01-01")
fl=pd.read_pickle("data/processed/research_curve_consistency/flies.pkl")
assert fl.date.max()<CUT
print("flies date range", fl.date.min().date(), fl.date.max().date(), len(fl))
IS=fl[fl.date.between("2011-01-01","2017-12-31")]
thr=IS.fA.abs().quantile(0.99); print("q99 thr bp", thr*1e4)
# independent: index by (di,k)
B=fl.set_index(["di","k"]).B
sig=fl[fl.fA.abs()>thr].copy()
key=list(zip(sig.di+20,sig.k))
sig["B_h"]=B.reindex(pd.MultiIndex.from_tuples(key)).values
sig=sig.dropna(subset=["B_h"])
sig["gross"]=-np.sign(sig.fA)*(sig.B_h-sig.B)*(-2)*(-1)*10000  # placeholder fixed below
# explicit legs: P&L of short centre fly = 10000*(dF_lo -2 dF_c + dF_hi) = -2*10000*dB
sig["gross"]=-np.sign(sig.fA)*2*(sig.B_h-sig.B)*10000
for per,(a,b) in {"IS":("2011-01-01","2017-12-31"),"OOS":("2018-01-01","2021-12-31")}.items():
    j=sig[sig.date.between(a,b)].sort_values("date")
    one=j.gross-4*4*10; two=j.gross-8*4*10
    print(per,"n",len(j),"gross",round(j.gross.mean()),"net(4tk,one side $160)",round(one.mean()),
          "net(4tk per side, $320 rt)",round(two.mean()),"net 10tk per side",round((j.gross-800).mean()))
    print("  Mar/Apr centre share",round(j.cm.isin([3,4]).mean(),2),"sign mix",j.fA.gt(0).mean().round(2))
    d=np.sort(j.di.unique()); ep=np.cumsum(np.r_[1,np.diff(d)>20]); m=dict(zip(d,ep))
    j["ep"]=j.di.map(m)
    e=j.groupby("ep").agg(start=("date","min"),end=("date","max"),n=("gross","size"),
        net=("gross",lambda g:(g-160).mean()),net320=("gross",lambda g:(g-320).mean()),
        cm=("cm",lambda s:s.mode().iat[0]), kink_bp=("fA",lambda s:1e4*s.abs().mean()),
        Bmove=("B",lambda s: 0))
    e["sum_net"]=j.groupby("ep").gross.apply(lambda g:(g-160).sum())
    print(e.round(0).to_string())
    # episode-level mean (equal weight per episode)
    print("  episode-weighted mean net($160)",round(e.net.mean()),"excl best 2:",
          round(j[~j.ep.isin(e.sum_net.nlargest(2).index)].gross.sub(160).mean()))
    # dollar sanity: implied |dB| vs log kink
    print("  mean |B| $",round(j.B.abs().mean(),3),"mean F",round(j.F.mean(),2),"mean |fA|*F $",round((j.fA.abs()*j.F).mean(),3))
