import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from src.research.v5_greeks_signals import common as C
from src.research.v5_greeks_signals import frozen_test as FT
end=pd.Timestamp('2021-12-31')
r,px=FT.front_returns(end)
d=pd.read_csv(C.V5/'dh_straddle.csv',parse_dates=['date','date1'])
d=d[d.date1<=end].sort_values(['date','T']); d['rank']=d.groupby('date').cumcount()+1; d=d[d['rank']<=3]
d['cal']=(d.date1-d.date).dt.days
d['resid_td']=d.pnl-d.gamma_pnl-d.theta0*365.25/252-d.vega_pnl
g=d.groupby('date1'); n=g.size()
s=pd.DataFrame({'resid_rel123':g.resid.sum()/g.P0.sum(),'resid_td':g.resid_td.sum()/g.P0.sum(),'vega':g.vega_pnl.sum(),'cal':g.cal.first()})[n==3]
f=pd.DataFrame({'r_front':r}).join(s,how='left'); f['r5']=f.r_front.rolling(5).sum()
z={k:C.wf_z(f[k]) for k in ['resid_rel123','resid_td','vega','r_front','r5']}
y=f.r_front.shift(-1)
f['wkend']=(f.cal>1).astype(float)
print('mean resid_rel123 by cal', f.groupby('cal').resid_rel123.mean().round(4).to_dict())
print('mean next-day return by wkend-signal day', y.groupby(f.wkend).mean().round(3).to_dict())
for lab,a,b in [('all','2010-01-01','2021-12-31'),('choose','2010-01-01','2016-12-31'),('check','2017-01-01','2021-12-31')]:
    sel=(f.index>=a)&(f.index<=b)
    for sig in ['resid_rel123','resid_td']:
        for ctl in [False,True]:
            X=pd.DataFrame({'z':z[sig],'r1':z['r_front'],'r5':z['r5']})
            if ctl: X['wk']=f.wkend
            X=X[sel]; yy=y[sel]
            bb,tt,nn=C.hac_ols(yy,X,5)
            Xw=X.assign(z=X.z.clip(-2.33,2.33)); yw=yy.clip(yy.quantile(.01),yy.quantile(.99))
            bw,tw,_=C.hac_ols(yw,Xw,5)
            print(lab,sig,'wk-ctl' if ctl else '      ', 't=%.2f tw=%.2f n=%d'%(tt[1],tw[1],nn))
    # ex-episodes
    for ex in [('2014-01-01','2014-03-31'),('2018-10-15','2018-12-31')]:
        pass
sel=(f.index>='2010-01-01')
m=~(((f.index>='2014-01-01')&(f.index<='2014-03-31'))|((f.index>='2018-10-15')&(f.index<='2018-12-31')))
for sig in ['resid_rel123','vega']:
    X=pd.DataFrame({'z':z[sig],'r1':z['r_front'],'r5':z['r5']})[sel&m]; bb,tt,nn=C.hac_ols(y[sel&m],X,5)
    print('ex Q1-2014 & Nov-Dec2018',sig,'t=%.2f'%tt[1])
