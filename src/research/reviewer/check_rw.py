import numpy as np, pandas as pd, json, statsmodels.api as sm
from scipy.stats import norm
d=pd.read_csv("data/processed/research_regimes_weather/panel_with_regimes.csv",parse_dates=["F","obs_date","h_last"])
assert d.obs_date.max()<=pd.Timestamp("2021-12-31")
TR=["2010-11","2011-12","2012-13","2013-14","2014-15","2015-16","2016-17"]
# 1. independent filter from frozen params -> verify pS2_pre uses data to t-1
p=json.load(open("data/processed/research_regimes_weather/hmm2_params_frozen.json"))
mu,sd,P,pi=np.array(p["mean"]),np.array(p["sd"]),np.array(p["transmat"]),np.array(p["startprob"])
d=d.sort_values("F").reset_index(drop=True); pre=np.zeros(len(d))
for w,g in d.groupby("winter",sort=False):
    prior=pi.copy()
    for i in g.index:
        pre[i]=prior[1]; post=prior*norm.pdf(d.dS[i],mu,sd); post/=post.sum(); prior=post@P
print("max |pre - pS2_pre|",np.abs(pre-d.pS2_pre).max())
# perturb dS_t and confirm pre_t unchanged (no use of t)
def slope(s,y="fwd_4",x="dS"):
    m=sm.OLS(s[y],sm.add_constant(s[x])).fit(cov_type="HAC",cov_kwds={"maxlags":4}); return m.params[x],m.tvalues[x],len(s)
for samp,sub in (("all",d),("2017-21",d[~d.winter.isin(TR)]),("2010-17",d[d.winter.isin(TR)])):
    for s in (1,0):
        q=sub[sub.stress_pre==s].dropna(subset=["dS","fwd_4"])
        print(samp,"stress_pre",s,"slope,t,n",[round(v,3) for v in slope(q)])
q=d[d.stress_pre==1].dropna(subset=["dS","fwd_4"])
# influence: drop largest |dS| weeks
for k in (1,2,3):
    qq=q.drop(q.dS.abs().nlargest(k).index); print("drop top",k,"|dS|:",[round(v,3) for v in slope(qq)])
print(q.loc[q.dS.abs().nlargest(3).index,["F","dS","fwd_4"]].round(3).to_string())
# winters in stressed sample
print(q.groupby("winter").size().to_dict())
# Permutation-style null for 'best of 32': shuffle regime label in blocks of winters? simple: random labels with same count within winters
rng=np.random.default_rng(0); d4=d.dropna(subset=["dS","fwd_4"]).copy(); obs=slope(q)[0]
mins=[]
for b in range(2000):
    lab=d4.groupby("winter").stress_pre.transform(lambda s: rng.permutation(s.values))
    a=d4[lab==1]; c=d4[lab==0]
    mins.append(np.polyfit(a.dS,a.fwd_4,1)[0])
print("P(random same-size subset slope <= obs)",np.mean(np.array(mins)<=obs))
# trades
for f in ("test","train"):
    t=pd.read_csv(f"data/processed/research_regimes_weather/rule_trades_{f}.csv",parse_dates=["F"])
    print(f,"all trades",len(t),round(t.pnl.mean()))
# re-run stressed rule on test
def run(sub,X=0.05,reg=None):
    out=[];busy=pd.Timestamp("1900-01-01")
    for _,r in sub.sort_values("F").iterrows():
        if r.F<=busy or not r.dS>X: continue
        if reg is not None and r.stress_pre!=reg: continue
        h=next((h for h in (4,3,2,1) if pd.notna(r[f"fwd_{h}"])),0)
        if h==0: continue
        out.append(dict(F=r.F.date(),dS=round(r.dS,3),h=h,pS_pre=round(r.pS2_pre,2),ch=round(r[f"fwd_{h}"],3),pnl=round((-r[f"fwd_{h}"]-0.008)*1e4)))
        busy=r.F+pd.Timedelta(days=7*h)
    return pd.DataFrame(out)
te=d[~d.winter.isin(TR)]; tr=d[d.winter.isin(TR)]
for nm,sub,reg in (("test stressed",te,1),("train stressed",tr,1),("test calm",te,0)):
    t=run(sub,reg=reg); print(nm,len(t),"mean",t.pnl.mean().round(),"median",t.pnl.median()); print(t.to_string())
    print("  per-side cost ($160/spread):",(t.pnl-80).mean().round())
