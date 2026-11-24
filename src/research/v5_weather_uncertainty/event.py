"""Q2: does front implied vol react to rising ensemble disagreement - same day, or with a lag?

Instrument: for each date d, the 'front' = nearest ON contract with > 15 trading days to option expiry
(fixed over the event window d-5..d+10 so no roll inside the window).  Outcome: change in its atm_iv
(vol pts) from the d-1 settle; also the futures log return over the same days.

A) Regressions (all Oct-Apr dates, HAC lags = max(5, k)):
     same day : dIV(d-1 -> d)      ~ dU_d + r_d + |r_d|
     lagged   : dIV(d -> d+k), k=1,5,10 ~ dU_d + r_d + |r_d|  (tradable: everything on the right known at d)
   dU = 1-run change in log normalised disagreement (w2, w3, and mean of w2/w3), standardised.
   Also: the LEVEL (log 3-day mean) instead of the change, for the lagged regressions.
B) Event study: one-run increases in log unc_n_w23 above the 90th percentile of the full development
   sample (descriptive threshold), declustered (>= 5 trading days apart).  Mean path minus the
   same-calendar-month unconditional mean path; gas-year bootstrap CI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import data as D

PRE, POST = 5, 10


def daily(force=False):
    f = D.CACHE / "q2_daily.pkl"
    if f.exists() and not force:
        return pd.read_pickle(f)
    w = D.weather_unc()
    s = D.surface()
    px, days, _ = D.futures()
    days = days[days <= D.CUT]
    iv = s.pivot_table(index="date", columns="cid", values="atm_iv").reindex(days)
    lp = np.log(px.reindex(days))
    td = s.pivot_table(index="date", columns="cid", values="tdays").reindex(days)
    rows = []
    for d in w.index:
        i = days.get_loc(d)
        if i - PRE - 1 < 0 or i + POST >= len(days):
            continue
        t = td.loc[d].dropna()
        t = t[t > 15]
        if t.empty:
            continue
        c = t.index.min()
        path = days[i - PRE - 1: i + POST + 1]
        v = iv.loc[path, c].values * 100
        p = lp.loc[path, c].values * 100
        rows.append({"date": d, "cid": c, "iv": v - v[PRE], "lp": p - p[PRE], "iv_lvl": v[PRE + 1]})
    out = pd.DataFrame(rows)
    out.to_pickle(f)
    return out


def regressions(Q, w):
    X = w.loc[Q.date].reset_index(drop=True)
    iv = np.vstack(Q.iv.values)   # columns: offsets -PRE-1 .. +POST relative index; PRE -> d-1 (zero)
    lp = np.vstack(Q.lp.values)
    o = PRE + 1                   # column of day d
    r_d = lp[:, o]                # d-1 -> d
    out = []
    for name in ("unc_n_w2", "unc_n_w3", "unc_n_w23"):
        z = X["dl_" + name]
        z = (z - z.mean()) / z.std()
        zl = np.log(X[name + "_m3"])
        zl = (zl - zl.mean()) / zl.std()
        targets = {"same day": iv[:, o]}
        for k in (1, 5, 10):
            targets[f"d->d+{k}"] = iv[:, o + k] - iv[:, o]
        for tname, y in targets.items():
            for xname, xv in (("change", z), ("level", zl)):
                if tname == "same day" and xname == "level":
                    continue
                df = pd.DataFrame({"y": y, "x": xv, "r": r_d, "ar": np.abs(r_d),
                                   "win": D.window(Q.date)}).dropna()
                k = 1 if tname == "same day" else int(tname.split("+")[1])
                for win in ("choose", "check", "all"):
                    g = df if win == "all" else df[df.win == win]
                    m = sm.OLS(g.y, sm.add_constant(g[["x", "r", "ar"]])).fit(
                        cov_type="HAC", cov_kwds={"maxlags": max(5, k)})
                    out.append({"unc": name, "x": xname, "target": tname, "win": win, "n": len(g),
                                "b_volpts_per_sd": m.params["x"], "t": m.tvalues["x"]})
    return pd.DataFrame(out)


def event_study(Q, w, var="unc_n_w23", q=0.90, gap=5):
    x = w.loc[Q.date, "dl_" + var].values
    thr = np.nanquantile(x, q)
    cand = np.where(x >= thr)[0]
    ev, last = [], -99
    di = {d: i for i, d in enumerate(Q.date)}
    for j in cand:
        if j - last >= gap:
            ev.append(j)
            last = j
    iv = np.vstack(Q.iv.values)
    lp = np.vstack(Q.lp.values)
    mon = Q.date.dt.month.values
    # unconditional same-calendar-month mean path
    base_iv = pd.DataFrame(iv).groupby(mon).mean()
    base_lp = pd.DataFrame(lp).groupby(mon).mean()
    E = iv[ev] - base_iv.loc[mon[ev]].values
    L = lp[ev] - base_lp.loc[mon[ev]].values
    gy = D.gas_year(Q.date.values[ev])
    rng = np.random.default_rng(11)
    ug = np.unique(gy)
    bs = []
    for _ in range(2000):
        pick = rng.choice(ug, len(ug))
        idx = np.concatenate([np.where(gy == g)[0] for g in pick])
        bs.append(E[idx].mean(0))
    bs = np.array(bs)
    res = pd.DataFrame({"offset": np.arange(-PRE - 1, POST + 1), "iv_excess": E.mean(0),
                        "lo": np.percentile(bs, 2.5, 0), "hi": np.percentile(bs, 97.5, 0),
                        "lp_excess": L.mean(0)})
    evd = pd.DataFrame({"date": Q.date.values[ev], "dl": x[ev], "iv_d": E[:, PRE + 1], "iv_d10": E[:, -1]})
    # by window
    win = D.window(Q.date.values[ev])
    byw = pd.DataFrame({"win": win, "same": E[:, PRE + 1], "p5": E[:, PRE + 6] - E[:, PRE + 1],
                        "p10": E[:, -1] - E[:, PRE + 1]}).groupby("win").agg(["mean", "count"])
    return res, evd, byw, thr


def run(force=False):
    Q = daily(force)
    w = D.weather_unc()
    R = regressions(Q, w)
    R.to_csv(D.CACHE / "q2_regs.csv", index=False)
    out = {}
    for var in ("unc_n_w2", "unc_n_w3", "unc_n_w23"):
        res, evd, byw, thr = event_study(Q, w, var)
        res.to_csv(D.CACHE / f"q2_event_{var}.csv", index=False)
        evd.to_csv(D.CACHE / f"q2_events_{var}.csv", index=False)
        out[var] = (res, evd, byw, thr)
    return R, out


if __name__ == "__main__":
    import sys
    pd.set_option("display.width", 250)
    R, out = run(force="--force" in sys.argv)
    print(R.round(3).to_string())
    for k, (res, evd, byw, thr) in out.items():
        print(k, "thr", round(thr, 3), "events", len(evd))
        print(res.round(2).T.to_string())
        print(byw.round(2))
