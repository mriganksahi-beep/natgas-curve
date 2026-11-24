import pandas as pd, numpy as np
V='data/processed/v5/'
s=pd.read_csv(V+'surface.csv',parse_dates=['date']); dh=pd.read_csv(V+'dh_straddle.csv',parse_dates=['date','date1'])
sp=pd.read_csv(V+'spreads.csv',parse_dates=['date']); wk=pd.read_csv(V+'weekly.csv',parse_dates=['date','expiry'])
for n,d in [('surface',s),('dh',dh),('spreads',sp),('weekly',wk)]:
    print(n,d.shape,d.date.min().date(),d.date.max().date())
print('dh date1 max',dh.date1.max())
# stale ATM iv in surface: unchanged atm_iv day to day for same cid
s=s.sort_values(['cid','date']); s['d_iv']=s.groupby('cid').atm_iv.diff(); s['dF']=s.groupby('cid').F.diff()
s['tb']=pd.cut(s['T']*252,[0,21,63,126,252,400])
print('share |d atm_iv|<1e-6 by tenor:'); print(s.groupby('tb').apply(lambda g:(g.d_iv.abs()<1e-6).mean()))
print('share stale by year (T>126td):'); print(s[s['T']*252>126].groupby(s.date.dt.year).apply(lambda g:(g.d_iv.abs()<1e-6).mean()))
# dh: P1==P0 with F moved
dh['stale']=(dh.P1==dh.P0)&(dh.F1!=dh.F0)
dh['tb']=pd.cut(dh['T']*252,[0,21,63,126,252,400])
print('dh stale share by tenor');print(dh.groupby('tb').stale.mean())
print('dh stale share by year');print(dh.groupby(dh.date.dt.year).stale.mean().round(3))
# weekday pattern in resid
dh['wd']=dh.date1.dt.dayofweek; dh['rr']=dh.resid/dh.P0; dh['cal']=(dh.date1-dh.date).dt.days
print(dh.groupby('cal')[['rr','theta_pnl','vega_pnl','resid']].mean().assign(n=dh.groupby('cal').size()))
print('vega_pnl/P0 by cal'); print((dh.vega_pnl/dh.P0).groupby(dh.cal).mean())
