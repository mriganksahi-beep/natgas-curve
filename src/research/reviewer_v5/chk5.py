import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
from src.research.v5_implied_corr.common import boot_mean, gas_year
s=pd.read_pickle('data/processed/research_v5_implied_corr/rows.pkl')
s['oi']=s.cso_call_oi.fillna(0)+s.cso_put_oi.fillna(0)
s['blk']=gas_year(s.date)
base=s.tdays.between(5,126)&(s.rho_R0>=-1)
s['rc']=s.rho_R0.clip(upper=1); s['crp']=s.rho_r_life-s.rc
for nm,m in [('band (agent)',~s.stale),('OI>0',s.oi>0),('band&OI>0',(~s.stale)&(s.oi>0)),('none',s.stale|~s.stale)]:
    d=s[base&m].dropna(subset=['crp'])
    mu,lo,hi,nb=boot_mean(d,'crp')
    print(f'{nm:14s} n={len(d):5d} mean={mu:.3f} [{lo:.3f},{hi:.3f}] med={d.crp.median():.3f} blocks={nb}')
    print('   by year n:',d.groupby(d.date.dt.year).size().to_dict())
    hj=d[d.cm==3]; print('   HJ mean',round(hj.crp.mean(),3), len(hj))
# winter-equal weighting
d=s[base&~s.stale].dropna(subset=['crp'])
gm=d.groupby('blk').crp.mean(); print('gas-year means',gm.round(3).to_dict(),' equal-wt mean',round(gm.mean(),3), 'n pos',(gm>0).sum(),'/',len(gm))
# excluding 2014 & 2019 gas years
for ex in [[2014],[2019],[2014,2019],[2021]]:
    print('ex',ex, round(d[~d.blk.isin(ex)].crp.mean(),3))
# share of crp driven by rows with low rho
print('crp mean when rc>0.95:',round(d[d.rc>0.95].crp.mean(),3),' n',(d.rc>0.95).sum(),'; rc<=0.95',round(d[d.rc<=0.95].crp.mean(),3),(d.rc<=0.95).sum())
