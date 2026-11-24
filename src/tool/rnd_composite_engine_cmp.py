"""Composite vs shipped direct engine (sln): selection/blend rules, tail comparison, calibration. Writes t_blend_engine.csv, t_compare_engine.csv."""
import pandas as pd, numpy as np
from src.tool import rnd_composite_analyse as A, rnd_composite_eval as E
v=E.vs_engine(); v=v[v.Y.isin(E.SCORED)]
v['n_oi']=(v.nq*(1-v.zero_oi_share.fillna(1))).round()
v['win']=np.where(v.Y.between(2014,2016),'choose',np.where(v.Y.between(2017,2021),'check','2013'))
print('usable share', v.usable.mean(), 'zero_oi', v.zero_oi.mean(), 'default', v.default_settle.mean())
print(v.groupby('win').agg(n=('date','size'),usable=('usable','mean'),zero_oi=('zero_oi','mean'),n_oi_med=('n_oi','median'),nq=('nq','median')))
def blend_crps(w): return w*v.crps+(1-w)*v.crps_comp-w*(1-w)*v.cross
rules={}
rules['direct (sln) always']=v.crps
rules['composite always']=v.crps_comp
rules['50/50']=blend_crps(0.5)
for N in (1,3,5):
    sel=(v.n_oi>=N)&v.usable.astype(bool)&~v.default_settle.astype(bool)
    rules[f'direct if usable & n_oi>={N} else composite']=np.where(sel,v.crps,v.crps_comp)
    print(N,'fallback share', (~sel).groupby(v.win).mean().round(3).to_dict())
for nf in (4,8):
    w=np.clip(v.n_oi/nf,0,1)*v.usable.astype(float)
    rules[f'weight w=min(1,n_oi/{nf})']=blend_crps(w)
rng=np.random.default_rng(0)
def boot(z,mask):
    g=pd.DataFrame({'z':z[mask],'Y':v.Y[mask]}).groupby('Y').z.agg(['sum','count'])
    s,c=g['sum'].values,g['count'].values; idx=rng.integers(0,len(s),(4000,len(s)))
    m=s[idx].sum(1)/c[idx].sum(1); return f"{s.sum()/c.sum():+.4f} [{np.percentile(m,2.5):+.4f}, {np.percentile(m,97.5):+.4f}]"
rows=[]
for k,z in rules.items():
    z=pd.Series(np.asarray(z),index=v.index)
    rows.append({'rule':k,'CRPS choose':round(z[v.win=='choose'].mean(),4),'CRPS check':round(z[v.win=='check'].mean(),4),
     'vs direct, choose':boot(z-v.crps,v.win=='choose'),'vs direct, check':boot(z-v.crps,v.win=='check')})
t=pd.DataFrame(rows); print(t.to_string()); t.to_csv(E.OUT/'t_blend_engine.csv',index=False)
# tail comparison vs sln engine, composite gauss/auto & cso & mix
d=A.daily()
m=d.merge(v[['date','Y','P+0.10','P+0.25','P+0.50','q05','q50','q95','nq','n_oi','usable','win']],on=['date','Y'],suffixes=('','_e'))
m=m[m.usable.astype(bool)]
rows=[]
for (win,fam,mode),x in m[m.win!='2013'].groupby(['win','fam','mode']):
    if mode not in ('auto','hist','fixed'): continue
    r={'win':win,'fam':fam,'mode':mode,'n':len(x)}
    for a,b in (('pa10','P+0.10'),('pa25','P+0.25'),('pa50','P+0.50')):
        r[a+'_bias']=(x[a]-x[b]).mean(); r[a+'_mae']=(x[a]-x[b]).abs().mean()
    r['q95_bias_med']=(x.q95-x.q95_e).median(); r['q5_bias_med']=(x.q5-x.q05).median()
    r['iqr_ratio']=((x.q75-x.q25)/(x.q95_e-x.q05)*0+ (x.q75-x.q25)).median()
    rows.append(r)
tc=pd.DataFrame(rows).drop(columns='iqr_ratio'); print(tc.round(4).to_string()); tc.to_csv(E.OUT/'t_compare_engine.csv',index=False)
# calibration of engine
ev={}
for t,c in ((10,'P+0.10'),(25,'P+0.25'),(50,'P+0.50')):
    e=(v.ST>v.F+t/100).astype(float); ev[t]=(e.mean(), v[c].mean(), ((v[c]-e)**2).mean())
print('engine events freq/p/brier', {k:tuple(round(x,4) for x in val) for k,val in ev.items()})
print('engine crps', v.crps.mean().round(4), 'pit mean', v.pit.mean().round(3), 'out90', ((v.pit<.05)|(v.pit>.95)).mean().round(3))
print('by winter', v.groupby('Y')[['crps','crps_comp']].mean().round(4).T.to_string())
