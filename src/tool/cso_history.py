"""Mid-November March-April spread option: implied normal vol vs the vol the spread realised to expiry, by winter.
Writes data/processed/tool_live/cso_midnov_history.csv (licensed-data derived; git-ignored)."""
import numpy as np, pandas as pd
sp=pd.concat([pd.read_csv('data/processed/v5/spreads.csv',parse_dates=['date']),pd.read_csv('data/processed/v5_locked/spreads.csv',parse_dates=['date'])]).drop_duplicates(['date','cid'],keep='last')
sp=sp[sp.cid%100==3]
c=pd.read_csv('data/raw/ng_curve_daily.csv',parse_dates=['date'],usecols=['date','contract_year','contract_month','settle'])
c=c[c.settle>0]; c['cid']=c.contract_year*100+c.contract_month
px=c.pivot_table(index='date',columns='cid',values='settle').sort_index(); days=px.index
rows=[]
for Y in range(2011,2027):
    cid=Y*100+3; g=sp[(sp.cid==cid)&(sp.date.between(f'{Y-1}-11-10',f'{Y-1}-11-21'))].sort_values('date')
    if g.empty or cid not in px or cid+1 not in px: continue
    r=g.iloc[(g.date-pd.Timestamp(f'{Y-1}-11-17')).abs().argsort().iloc[0]]
    S=(px[cid]-px[cid+1]).dropna(); S=S[S.index>=r.date]
    e=days[days.searchsorted(pd.Timestamp(Y,3,1))-4]; S=S[S.index<=e]
    dS=np.diff(S.values); rv=np.sqrt(np.sum(dS**2)/(len(dS)/252))
    oi=r.cso_call_oi+r.cso_put_oi
    rows.append(dict(Y=Y,date=r.date.date(),nv=r.nv,rv=rv,lr=np.log(r.nv/rv),oi=oi))
h=pd.DataFrame(rows); print(h.round(3).to_string()); print('mean lr',h.lr.mean().round(3),'median',h.lr.median().round(3),'share>0',(h.lr>0).mean().round(2))
print('mean ratio',np.exp(h.lr).mean().round(2)); h.to_csv('data/processed/tool_live/cso_midnov_history.csv',index=False)
