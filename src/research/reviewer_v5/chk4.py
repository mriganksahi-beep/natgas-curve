import pandas as pd, numpy as np
sp=pd.read_csv('data/processed/v5/spreads.csv',parse_dates=['date']).sort_values(['cid','date'])
sp['band']=sp.nv.between(1.15,1.21); sp['td']=sp['T']*252
sp['oi']=sp.cso_call_oi.fillna(0)+sp.cso_put_oi.fillna(0)
sp['doi']=sp.groupby('cid').oi.diff()
for nm,g in [('band',sp[sp.band]),('nonband',sp[~sp.band])]:
    print(nm,len(g),'oi==0',(g.oi==0).mean().round(3),'doi==0',(g.doi==0).mean().round(3),'median oi',g.oi.median())
# near-band genuine? rows with nv in [1.0,1.15) or (1.21,1.4]
nb=sp[(sp.nv.between(1.0,1.4))&~sp.band]; print('adjacent-band rows',len(nb), nb.groupby(nb.date.dt.year).size().to_dict())
# relation nv vs |S| and F level for non-band td<=126
o=sp[~sp.band&(sp.td.between(5,126))]
print(o.groupby(pd.qcut(o.td,4)).nv.median())
# raw: check G4X last settle date vs ON option expiry
