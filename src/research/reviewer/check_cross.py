import numpy as np, pandas as pd, statsmodels.api as sm
P=pd.read_pickle("data/processed/research_kalman_scarcity/panel_with_signals.pkl")
assert P.date.max()<=pd.Timestamp("2021-12-31")
R=pd.read_csv("data/processed/research_regimes_weather/panel_with_regimes.csv",parse_dates=["F"])
W=P.season.astype(float)
X=pd.DataFrame({"sur":P.surprise_pub,"hdd":P.hdd_anom,"sur_w":P.surprise_pub*W,"hdd_w":P.hdd_anom*W})
d=P.d_s_rw; e=P.est&d.notna()&X.notna().all(1)
r=sm.OLS(d[e],sm.add_constant(X[e])).fit(); u=d-r.predict(sm.add_constant(X)); thr=u[e].abs().quantile(.9)
same=P.winter.shift(-4)==P.winter
fma=(P.MarApr.shift(-4)-P.MarApr).where(same)
D=pd.DataFrame({"u":u,"fma":fma,"jump":u.abs()>thr,"month":P.index.month,"est":P.est},index=P.index)
D=D.join(R.set_index("F")[["stress_pre"]],how="left")
def sl(s):
    s=s.dropna(subset=["u","fma"]); 
    if len(s)<4: return (len(s),)
    m=sm.OLS(s.fma,sm.add_constant(s.u)).fit(cov_type="HAC",cov_kwds={"maxlags":4}); return len(s),round(m.params.u,2),round(m.tvalues.u,2)
c=D[~D.est & D.jump]
print("check-window KF jumps, all months:",sl(c))
print("  Nov-Feb only:",sl(c[c.month.isin([11,12,1,2])]))
print("  Mar-Oct only:",sl(c[~c.month.isin([11,12,1,2])]))
print("  Nov-Feb & stress_pre=1:",sl(c[c.stress_pre==1]), " stress_pre=0:",sl(c[c.stress_pre==0]))
print("  upward jumps only, Nov-Feb stressed:",sl(c[(c.stress_pre==1)&(c.u>0)]))
# regimes stressed-pre sample: upward vs downward dS
q=R[(R.stress_pre==1)].dropna(subset=["dS","fwd_4"])
for nm,s in (("up",q[q.dS>0]),("down",q[q.dS<0])):
    m=sm.OLS(s.fwd_4,sm.add_constant(s.dS)).fit(); print("regimes stressed",nm,len(s),round(m.params.dS,2))
# overlap of curve_consistency & regimes OOS profit episodes
print("regimes stressed test trades dates: 2017-12-01, 2018-01-12, 2018-11-16, 2019-01-18, 2021-11-19")
