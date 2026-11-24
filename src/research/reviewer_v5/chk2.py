import pandas as pd, numpy as np
V='data/processed/v5/'
s=pd.read_csv(V+'surface.csv',parse_dates=['date'])
s['td']=s['T']*252
s['flat']=(s['skew'].abs()<1e-4)&(s.curv.abs()<1e-4)
s['tb']=pd.cut(s.td,[0,21,63,126,252,400])
print('flat smile share by tenor');print(s.groupby('tb').flat.mean())
print('flat share by year, td>126');print(s[s.td>126].groupby(s.date.dt.year).flat.mean().round(3))
# zero volume share near-money
s['zv']=(s.call_vol.fillna(0)+s.put_vol.fillna(0))==0
print('zero vol share by tenor');print(s.groupby('tb').zv.mean())
print('zero vol by year td<=63');print(s[s.td<=63].groupby(s.date.dt.year).zv.mean().round(3))
print('zero vol by year all');print(s.groupby(s.date.dt.year).zv.mean().round(3))
# atm_iv rounded repetition: same atm across consecutive days
dh=pd.read_csv(V+'dh_straddle.csv',parse_dates=['date','date1'])
dh['dF']=(dh.F1-dh.F0).abs()
big=dh[dh.dF>0.05]; print('P1==P0 when |dF|>5c:',(big.P1==big.P0).mean(), len(big))
# iv change zero but F moved: implied by settle; compute straddle iv change dist
dh['div']=dh.iv1-dh.iv0
print('dh |div|<1e-4 share by td bucket'); dh['tb']=pd.cut(dh['T']*252,[0,21,63,126,252,400]); print(dh.groupby('tb').div.apply(lambda x:(x.abs()<1e-4).mean()))
