"""Step 4: IV term structure vs storage tightness, and whether it predicts realised vol / straddle P&L.

Daily, at the settle:
  WS  = atm_iv(Jan of the coming winter) - atm_iv(Apr of the same winter)  ["winter vs shoulder"]
        coming winter = nearest Jan contract with tdays >= 21 (switches after the Dec expiry).
  FB  = atm_iv(front, tdays >= 10) - atm_iv(contract with tdays nearest 252)  ["front vs 12m"]
Outcomes over the next 20 trading days (same contracts):
  dRV = log RV20(leg1) - log RV20(leg2)
  PNL = sum of next-20-day delta-hedged ATM straddle P&L per unit vega (dh_straddle, re-struck
        daily, gross) for leg1 minus leg2 [vol points]: the vega-neutral long leg1 / short leg2 pair.
Sampled weekly (last trading day of each week); OLS with Newey-West (6 lags, overlap is 4 weeks)
and gas-year block bootstrap CIs on the slope.  Descriptive within each window (no fitting used
for trading).  sdev = storage minus prior-years same-week mean (bcf; defined from 2012).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import data as D
from .vrp import prep_base
from .stats import RNG


def legs(b):
    rows = []
    for d, g in b.groupby("date"):
        g = g.set_index("cid")
        jan = g[(g.dm == 1) & (g.tdays >= 21)]
        r = {"date": d}
        if len(jan):
            cj = jan.index.min()
            ca = cj + 3
            if ca in g.index:
                r.update(ws_a=cj, ws_b=ca)
        fr = g[g.tdays >= 10]
        if len(fr):
            cf = fr.tdays.idxmin()
            cb = (g.tdays - 252).abs().idxmin()
            if abs(g.at[cb, "tdays"] - 252) <= 15:
                r.update(fb_a=cf, fb_b=cb)
        rows.append(r)
    return pd.DataFrame(rows).set_index("date")


def build(force=False):
    f = D.CACHE / "s4_term.csv"
    if f.exists() and not force:
        return pd.read_csv(f, parse_dates=["date"]).set_index("date")
    b = prep_base()
    L = legs(b)
    bx = b.set_index(["date", "cid"])
    dh = D.dh()
    dh["pv"] = dh.pnl / dh.vega0 * 100
    pv = dh.pivot_table(index="date", columns="cid", values="pv")
    days = pv.index
    fwd = pv[::-1].rolling(20, min_periods=15).sum()[::-1]          # sum over [d, d+19]: P&L of positions opened d..d+19
    out = []
    for d, r in L.iterrows():
        row = {"date": d}
        for tag in ("ws", "fb"):
            a, c = r.get(f"{tag}_a"), r.get(f"{tag}_b")
            if pd.isna(a) or pd.isna(c):
                continue
            a, c = int(a), int(c)
            ia, ic = bx.loc[(d, a)], bx.loc[(d, c)]
            row[tag] = (ia.atm_iv - ic.atm_iv) * 100
            row[f"{tag}_drv"] = np.log(ia.rv20f / ic.rv20f)
            if d in fwd.index and a in fwd.columns and c in fwd.columns:
                row[f"{tag}_pnl"] = fwd.at[d, a] - fwd.at[d, c]
        row["sdev"] = bx.loc[d].sdev.iloc[0]
        out.append(row)
    t = pd.DataFrame(out).set_index("date")
    t.to_csv(f)
    return t


def ols(t, y, x, lags=6):
    q = t[[y] + x].dropna()
    q = q.groupby(q.index.to_period("W")).tail(1)
    if len(q) < 30:
        return None
    X = sm.add_constant(q[x])
    m = sm.OLS(q[y], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    # gas-year block bootstrap of the slope
    gy = D.gas_year(q.index)
    grp = [q.index[gy == g] for g in np.unique(gy)]
    bs = []
    for _ in range(1000):
        idx = np.concatenate([grp[i] for i in RNG.integers(0, len(grp), len(grp))])
        qq = q.loc[idx]
        try:
            bs.append(sm.OLS(qq[y], sm.add_constant(qq[x])).fit().params[x[0]])
        except Exception:
            pass
    return {"y": y, "x": "+".join(x), "slope": m.params[x[0]], "t_hac": m.tvalues[x[0]],
            "bs_lo": np.percentile(bs, 2.5), "bs_hi": np.percentile(bs, 97.5), "r2": m.rsquared, "n": len(q)}


def run(force=False):
    t = build(force)
    rows = []
    for win, sel in (("choose", t.index.year <= 2016), ("check", t.index.year >= 2017), ("all", slice(None))):
        tt = t[sel]
        for spec in [("ws", ["sdev"]), ("fb", ["sdev"]),
                     ("ws_drv", ["ws"]), ("ws_pnl", ["ws"]), ("fb_drv", ["fb"]), ("fb_pnl", ["fb"]),
                     ("ws_pnl", ["sdev"]), ("fb_pnl", ["sdev"])]:
            r = ols(tt, *spec)
            if r:
                r["win"] = win
                rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(D.CACHE / "s4_results.csv", index=False)
    desc = t.groupby(np.where(t.index.year <= 2016, "choose", "check"))[["ws", "fb", "ws_pnl", "fb_pnl"]].agg(["mean", "std"])
    return t, res, desc


if __name__ == "__main__":
    import sys
    pd.set_option("display.width", 220)
    t, res, desc = run(force="--force" in sys.argv)
    t2 = t.copy()
    t2["s"] = pd.cut(t2.sdev, [-1e9, -150, 150, 1e9], labels=["tight", "normal", "loose"])
    print(t2.groupby("s", observed=True)[["ws", "fb", "ws_pnl", "fb_pnl"]].mean().round(2))
    print(desc.round(2))
    print(res.round(3).to_string(index=False))
