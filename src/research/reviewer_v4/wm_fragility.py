import numpy as np, pandas as pd
from src.research.v4_widowmaker.data import load_px, signals, YEARS, SIDE
from src.research.v4_widowmaker.rules import pnl, _lag
px, sg = load_px(), signals()
def frames(t0fn):
    out={}
    for Y in YEARS:
        S=(px[(Y,3)]-px[(Y,4)]).dropna(); S=S[S.index>=pd.Timestamp(Y-1,10,1)]
        ent=S.index[S.index.searchsorted(t0fn(Y))]
        d=pd.DataFrame({"S":S}).join(sg,how="left"); out[Y]=d[d.index>=ent].copy()
    return out
def stats(W, posfn, years=range(2011,2022)):
    fin, dd = [], []
    for Y in years:
        w=W[Y]; pos=posfn(w); x=pnl(w,pos,SIDE[2]); c=np.cumsum(x)
        fin.append(c[-1]); dd.append(np.min(c - np.maximum.accumulate(np.r_[0,c])[1:]) if len(c) else 0)
        dd[-1]=min(dd[-1], c.min(), 0)
    return np.array(fin), np.array(dd)
def delay(col, th, lag=0):
    def f(w):
        z=np.nan_to_num(w[col].values.astype(float),nan=-9); ok=np.where(z<=th)[0]; t=np.zeros(len(w))
        if len(ok): t[ok[0]:]=1.0
        return _lag(t,lag)
    return f
base=lambda w: np.ones(len(w))
def R(W, fn, years=range(2011,2022)):
    fb,db=stats(W,base,years); fr,dr=stats(W,fn,years)
    return (fr.mean()/fb.mean())/(dr.min()/db.min()), fr, fb, dr, db
Wd=frames(lambda Y: pd.Timestamp(Y-1,12,1))
fb,db=stats(Wd,base); print("base dec mean",fb.mean().round(0),"worstDD",db.min())
print("E grid (Dec 1 entry), R over 11 winters:")
for col in ["zL23_1","zL23_3","zL23_5","zL123_5"]:
    print(col, " ".join(f"th{th}:{R(Wd,delay(col,th))[0]:.2f}" for th in [0,0.25,0.5,0.75,1.0,1.25,1.5,2.0]))
r,fr,fb,dr,db=R(Wd,delay("zL23_5",1.0))
print("per-winter changed (rule-base final, DD):"); 
for i,Y in enumerate(range(2011,2022)):
    if abs(fr[i]-fb[i])>1 or abs(dr[i]-db[i])>1: print(" ",Y, int(fr[i]), int(fb[i]), int(dr[i]), int(db[i]))
print("LOWO R (th1.0):", {Y: round(R(Wd,delay("zL23_5",1.0),[y for y in range(2011,2022) if y!=Y])[0],2) for Y in range(2011,2022)})
print("entry-date shift, R of E th1.0 / th0.5:")
for off in [-7,-4,-2,0,2,4,7,10]:
    Wx=frames(lambda Y,o=off: pd.Timestamp(Y-1,12,1)+pd.Timedelta(days=o))
    print(" ",off, round(R(Wx,delay("zL23_5",1.0))[0],2), round(R(Wx,delay("zL23_5",0.5))[0],2))
# Nov entry
Wn=frames(lambda Y: (lambda n: n+pd.Timedelta(days=(4-n.weekday())%7))(pd.Timestamp(Y-1,11,1)))
print("Nov entry E:", " ".join(f"th{th}:{R(Wn,delay('zL23_5',th))[0]:.2f}" for th in [0,0.5,1.0,1.5]))
