"""Check 2: headline premium arithmetic, entry/expiry conventions, cross-agent agreement."""
import numpy as np
import pandas as pd

from src.research.reviewer_v3.common import load_hj, wboot, MULT, RT, CUT

sp, px = load_hj()
days = px.index
rows = []
for Y in range(2011, 2022):
    S = sp[Y]
    ltd = S.index[-1]
    # official LTD: 3 business (trading) days before 1 March
    first = pd.Timestamp(Y, 3, 1)
    off = days[days.searchsorted(first) - 3]
    nov1 = pd.Timestamp(Y - 1, 11, 1)
    e1 = S.index[S.index.searchsorted(nov1)]
    fri = nov1 + pd.Timedelta(days=(4 - nov1.weekday()) % 7)
    e2 = S.index[S.index.searchsorted(fri)]
    path = S[S.index >= e1]
    path2 = S[S.index >= e2]
    mtm2 = -(path2 - path2.iloc[0]) * MULT
    dd2 = (mtm2 - mtm2.cummax()).min()
    nov = S[(S.index >= nov1) & (S.index < pd.Timestamp(Y - 1, 12, 1))]
    rows.append(dict(Y=Y, ltd=ltd.date(), ltd_ok=ltd == off, S_exp=S.iloc[-1],
                     nov1_entry=e1.date(), pnl_nov1=-(S.iloc[-1] - S[e1]) * MULT - RT,
                     fri_entry=e2.date(), pnl_fri=-(S.iloc[-1] - S[e2]) * MULT - RT,
                     worst_dd_fri=dd2, novavg_minus_exp=nov.mean() - S.iloc[-1]))
t = pd.DataFrame(rows).set_index("Y")
pd.set_option("display.width", 200)
print(t.round(3).to_string())
for c in ("pnl_nov1", "pnl_fri"):
    lo, hi = wboot({y: v for y, v in t[c].items()})
    print(f"{c}: mean {t[c].mean():.0f} CI [{lo:.0f}, {hi:.0f}]  ex-2014 {t[c].drop(2014).mean():.0f}  "
          f"pos {int((t[c] > 0).sum())}/11 ; 2016-21 mean {t.loc[2016:2021, c].mean():.0f}; 2017-21 {t.loc[2018:2021, c].mean():.0f}")
print("Nov-avg minus expiry: mean %.3f, >0 in %d/11" % (t.novavg_minus_exp.mean(), (t.novavg_minus_exp > 0).sum()))
# entry-date sensitivity: P&L for entries on each trading day 1-14 Nov
sens = []
for k in range(0, 8):
    v = []
    for Y in range(2011, 2022):
        S = sp[Y]
        i = S.index.searchsorted(pd.Timestamp(Y - 1, 11, 1)) + k
        v.append(-(S.iloc[-1] - S.iloc[i]) * MULT - RT)
    sens.append(np.mean(v))
print("mean P&L by entry trading-day offset from 1 Nov (0..7):", np.round(sens, 0))

# premium_map L9 cell, independently: Fridays with round(days to H expiry / 30.4375) == 9
cell = {}
for Y in range(2011, 2022):
    S = sp[Y]
    T = S.index[-1]
    fr = S[(S.index.weekday == 4)]
    L = np.round((T - fr.index).days / 30.4375).astype(int)
    sel = fr[L == 9]
    if len(sel):
        cell[Y] = (sel - S.iloc[-1]).mean() * 100
c = pd.Series(cell)
print("Mar-Apr L9 drift (c) by year:", c.round(0).to_dict())
lo, hi = wboot({y: v for y, v in c.items()})
print(f"L9 mean {c.mean():.1f}c CI [{lo:.1f},{hi:.1f}], >0 {int((c > 0).sum())}/{len(c)}; median {c.median():.1f}")
lo, hi = wboot({y: v for y, v in c.drop(2014).items()})
print(f"L9 ex-2014 mean {c.drop(2014).mean():.1f}")
# what L9 short would earn after costs, per year
print("L9 short net $/spread per year: mean %.0f" % (c.mean() / 100 * MULT - RT))
# premium_map L4 benchmark total 2016-21 vs Nov-entry totals
print("Sum of Nov1->expiry net, expiries 2016-2021: %.0f" % t.loc[2016:2021, "pnl_nov1"].sum())
