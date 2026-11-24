import pandas as pd, numpy as np
sp=pd.read_csv('data/processed/v5/spreads.csv',parse_dates=['date'])
sp['td']=sp['T']*252; sp['band']=sp.nv.between(1.15,1.21)
sp['tb']=pd.cut(sp.td,[0,5,21,42,63,126,252,400])
print(sp.groupby('tb').agg(n=('band','size'),band=('band','mean')))
print(sp.groupby(sp.date.dt.year).agg(n=('band','size'),band=('band','mean')).round(3))
# rows in band with td<=126: are they stuck (nv unchanged day to day)?
sp=sp.sort_values(['cid','date']); sp['dnv']=sp.groupby('cid').nv.diff()
b=sp[sp.band&(sp.td<=126)]
print('in-band rows td<=126:',len(b),' share with |dnv|<1e-3:',(b.dnv.abs()<1e-3).mean())
print(b.groupby(b.date.dt.year).size())
print(b[['td','nv','S','iv_near','n_strikes']].describe().round(3))
# nv quantiles outside band, td<=126
o=sp[~sp.band&(sp.td<=126)]
print('nv quantiles td<=126 non-band', o.nv.quantile([.5,.9,.95,.99,.999]).round(3).to_dict(), 'share nv>1.0',(o.nv>1.0).mean())
print('band rows with dnv nonzero examples'); print(b[b.dnv.abs()>0.01][['date','cid','td','nv','dnv']].head(15))
# pinned nv exact value
print(sp[sp.band].nv.round(4).value_counts().head(8))
