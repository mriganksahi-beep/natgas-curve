import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
from src.research.v5_greeks_signals import common as C
tr=pd.read_csv('data/processed/research_v5_greeks_signals/part4_trades.csv',parse_dates=['date'])
c=pd.read_csv('data/raw/ng_curve_daily.csv',parse_dates=['date'],usecols=['date','contract_year','contract_month','settle'])
c=c[(c.date<='2022-01-31')&(c.settle>0)]; c['cid']=c.contract_year*100+c.contract_month
px=c.pivot_table(index='date',columns='cid',values='settle'); days=px.index
w=pd.read_csv('data/raw/options/ng_weekly_options_v2.csv.gz',parse_dates=['date'],usecols=['date','root','cy','cm','cp','k_raw','settle'])
w=w[(w.date<='2021-12-31')&(w.settle>0)].copy(); w['K']=(w.k_raw/1000).round(4); w['series']=w.root+'_'+(w.cy*100+w.cm).astype(str)
last=w.groupby('series').date.max()
L=w[w.date==w.series.map(last)]
pv=L.pivot_table(index=['series','K'],columns='cp',values='settle')
out=[]
for r in tr.itertuples():
    e=last[r.series]; K=round(r.K,4)
    try: x=pv.loc[(r.series,K)]; Pl=x.get('C',np.nan)+x.get('P',np.nan)
    except KeyError: Pl=np.nan
    # match futures contract by F at entry
    row=px.loc[r.date].dropna(); row=row[row.index>=r.date.year*100+r.date.month][:3]
    cid=row.index[np.argmin(np.abs(row.values-r.F))]
    Fe=px.at[e,cid]; i=days.searchsorted(e); Fn=px.iat[i+1,px.columns.get_loc(cid)] if i+1<len(days) else np.nan
    tv=Pl-abs(Fe-K) if np.isfinite(Pl) else np.nan
    true_exp_next = np.isfinite(tv) and tv>0.005
    pay=abs(Fn-K) if true_exp_next else abs(Fe-K)
    out.append((tv,true_exp_next,(r.prem-pay)*1e4-(0.03*r.prem+(0.002 if pay>0 else 0))*1e4))
o=pd.DataFrame(out,columns=['tv','shifted','net_fix'],index=tr.index); tr=tr.join(o)
tr['win']=np.where(tr.date<'2017-01-01','2014-16','2017-21')
g=tr.groupby(['kind','win'])
print(g.agg(n=('net','size'),net=('net','mean'),shifted=('shifted','mean'),net_fix=('net_fix','mean')).round(2))
rep=tr[(tr.kind=='report')&(tr.win=='2014-16')]
print('report 2014-16 net_fix CI', C.block_boot_ci(rep.net_fix.values,4))
print('unknown tv share', tr.tv.isna().mean())
