# Applies panel.pre2010_curve's mapping logic to POST-2010-06-07 EIA rows only (dev, <=2021-12-31)
import numpy as np, pandas as pd, csv
from src.v4.panel import addm, price_block, dev_curve
rows=[]
with open("data/raw/eia_futures_c1_c4.csv") as f:
    r=csv.reader(f); hdr=next(r)
    for line in r:
        if "2010-06-07" <= line[0] <= "2021-12-31": rows.append(line)
e=pd.DataFrame(rows,columns=hdr); e["date"]=pd.to_datetime(e.date); e=e.set_index("date")
for k in range(1,5): e[f"c{k}"]=pd.to_numeric(e[f"c{k}"],errors="coerce")
out={}
for d,rw in e.iterrows():
    for j in range(3):
        cid=addm(d.year*100+d.month,j); y,m=divmod(cid,100)
        if pd.Timestamp(y,m,1)-pd.offsets.BDay(3)>=d: c1=cid; break
    for k in range(1,5): out.setdefault(d,{})[addm(c1,k-1)]=rw[f"c{k}"]
pxe=pd.DataFrame(out).T.sort_index()
expe=pd.Series({cid: pd.Timestamp(cid//100,cid%100,1)-pd.offsets.BDay(3) for cid in pxe.columns})
pxd,expd=dev_curve(); pxd=pxd[(pxd.index>="2010-06-07")&(pxd.index<="2021-12-31")]
common=pxe.index.intersection(pxd.index)
a=pxe.loc[common]; b=pxd.loc[common,[c for c in pxe.columns if c in pxd.columns]].reindex(columns=pxe.columns)
diff=(a-b).abs().stack()
print("cells",len(diff),"exact(<=0.0005)",(diff<=0.0005).mean().round(4),">0.01",(diff>0.01).sum(),">0.05",(diff>0.05).sum())
bad=diff[diff>0.05]
print(bad.reset_index().head(15))
# build locked-style price block on dev EIA and compare r_second / d_hj with Databento panel
pb=price_block(pxe,expe,with_strip=False)
pan=pd.read_csv("data/processed/v4/panel_dev.csv",parse_dates=["date"]); pan=pan[pan.date<="2021-12-31"]
m=pb.merge(pan,on="date",suffixes=("_e","_d"))
for c in ["r_front","r_second","d_hj"]:
    dd=(m[c+"_e"]-m[c+"_d"]).abs()
    print(c,"n",dd.notna().sum(),"corr",m[[c+"_e",c+"_d"]].corr().iloc[0,1].round(4),"p99 absdiff",dd.quantile(.99).round(4),"max",dd.max().round(3), "contract mismatch", (m["second_e"]!=m["second_d"]).sum() if c=="r_second" else "")
mm=m[(m.r_second_e-m.r_second_d).abs()>2][["date","front_e","front_d","second_e","second_d","r_second_e","r_second_d"]]
print(mm.head(12))
mh=m[(m.d_hj_e-m.d_hj_d).abs()>0.05][["date","hj_e","hj_d","d_hj_e","d_hj_d"]]; print(len(mh)); print(mh.head(8))
m["pd_mis"]=m.prev_date_e!=m.prev_date_d
dd=(m.r_second_e-m.r_second_d).abs()
print("days |diff|>1pp:",(dd>1).sum(),"of which prev_date mismatch:",(m.pd_mis&(dd>1)).sum(), "total prev_date mismatches", m.pd_mis.sum())
print("winter Nov-Feb |diff|>1:", ((dd>1)&m.date.dt.month.isin([11,12,1,2])).sum())
x=m[(dd>1)].date.dt.strftime("%Y-%m-%d").tolist(); print(x[:40])
# weekly (Fri-Fri) second-month sum comparison
m["wk"]=m.date.dt.to_period("W-FRI"); g=m.groupby("wk")[["r_second_e","r_second_d"]].sum()
print("weekly corr", g.corr().iloc[0,1].round(4), "p99 |diff|", (g.r_second_e-g.r_second_d).abs().quantile(.99).round(3))
