"""Base table: one row per surface (date, cid) with backward features and forward outcomes,
plus GARCH filtering.  Cached to CACHE/base.parquet-like CSV."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter

from . import data as D


def returns(px):
    lp = np.log(px)
    r = lp.diff()
    # a return only where the contract settled on both consecutive trading days
    return r


def rv_back(r2, n):
    return np.sqrt(252 * r2.rolling(n, min_periods=max(3, int(0.8 * n))).mean())


def rv_fwd(r2, n):
    # mean of r2 over (t, t+n]; NaN unless all n exist (outcome cut honoured by px end)
    s = r2[::-1].rolling(n, min_periods=n).mean()[::-1].shift(-1)
    return np.sqrt(252 * s)


def front_series(r, oexp):
    """Front-contract daily return (same contract both days): min cid with option expiry >= d."""
    cids = np.array(sorted(r.columns))
    oe = oexp.reindex(cids).values
    out = pd.Series(np.nan, index=r.index)
    assert np.all(np.diff(oe.astype("int64")) >= 0)
    for i, d in enumerate(r.index):
        out.iat[i] = r.at[d, cids[np.searchsorted(oe, d)]]
    return out


# ----------------------------------------------------------------------------- GARCH(1,1)
def garch_nll(p, x):
    w, a, b = p
    if w <= 0 or a < 0 or b < 0 or a + b >= 0.999:
        return 1e10
    u = np.empty_like(x)
    u[0] = w + (a + b) * x.var()
    u[1:] = w + a * x[:-1] ** 2
    h = lfilter([1.0], [1.0, -b], u)
    return 0.5 * np.sum(np.log(h) + x ** 2 / h)


def garch_fit(x, p0=(0.05, 0.08, 0.90)):
    r = minimize(garch_nll, p0, args=(x,), method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-6})
    return r.x


def garch_filter(x, p):
    """Conditional variance for t+1 given returns through t (x in % units)."""
    w, a, b = p
    V = w / (1 - a - b)
    x2 = np.where(np.isfinite(x), x ** 2, V)
    u = w + a * x2
    u[0] += b * V
    return lfilter([1.0], [1.0, -b], u)


def garch_avg(h1, p, H):
    w, a, b = p
    V = w / (1 - a - b)
    k = np.arange(H)
    return V + (h1 - V) * np.mean((a + b) ** k)


def build(force=False):
    f = D.CACHE / "base.csv"
    if f.exists() and not force:
        return pd.read_csv(f, parse_dates=["date", "oexp"])
    px, days, oexp = D.futures()
    r = returns(px)
    r2 = r ** 2
    s = D.surface()
    feats = {}
    feats["rv5"] = rv_back(r2, 5)
    feats["rv21"] = rv_back(r2, 21)
    feats["rv63"] = rv_back(r2, 63)
    feats["ewma"] = np.sqrt(252 * r2.ewm(alpha=0.06, min_periods=20).mean())   # RiskMetrics lambda 0.94
    out = {"rv20f": rv_fwd(r2, 20)}
    # life RV: cum r2 to option expiry
    cum = r2.fillna(0).cumsum()
    cnt = r2.notna().cumsum()
    s = s[s.date <= D.CUT].copy()
    s["oexp"] = s.cid.map(oexp)
    for k, v in {**feats, **out}.items():
        st = v.stack()
        st.index.names = ["date", "cid"]
        s = s.join(st.rename(k), on=["date", "cid"])
    # features must come only from <= CUT (true: date <= CUT and backward windows)
    life = []
    for cid, g in s.groupby("cid"):
        e = oexp.get(cid)
        if cid not in cum.columns or e is None or e > D.OUTCOME_CUT or e not in cum.index:
            life.append(pd.Series(np.nan, index=g.index))
            continue
        c0 = cum[cid].reindex(g.date).values
        n0 = cnt[cid].reindex(g.date).values
        n = cnt.at[e, cid] - n0
        v = (cum.at[e, cid] - c0) / np.where(n > 0, n, np.nan)
        life.append(pd.Series(np.sqrt(252 * v), index=g.index))
    s["rv_life"] = pd.concat(life)
    s.loc[s.tdays < 20, "rv20f"] = np.nan
    # front continuous returns for GARCH; monthly expanding refit using returns < month start
    fr = front_series(r[r.index <= D.CUT], oexp) * 100
    months = pd.date_range("2011-01-01", "2022-01-01", freq="MS")
    params = {}
    for m in months:
        x = fr[fr.index < m].dropna().values
        if len(x) < 120:
            continue
        fits = [garch_fit(x, q) for q in ((0.05, 0.08, 0.90), (0.3, 0.1, 0.85), (1.0, 0.15, 0.7))]
        params[m] = min(fits, key=lambda q: garch_nll(q, x))
    par = pd.DataFrame(params, index=["w", "a", "b"]).T
    par.to_csv(D.CACHE / "garch_params.csv")
    # per-contract filtered variance using the params in force at each date (refit month start)
    s["garch20"] = np.nan
    s["garch_life"] = np.nan
    s["mstart"] = s.date.values.astype("datetime64[M]")
    rr = (r[r.index <= D.CUT] * 100)
    for m, p in par.iterrows():
        idx = s.index[s.mstart == m]
        if not len(idx):
            continue
        sub = s.loc[idx]
        for cid, g in sub.groupby("cid"):
            x = rr[cid].loc[:g.date.max()]
            first = x.first_valid_index()
            if first is None:
                continue
            x = x.loc[first:]
            # variance targeting: persistence (a, b) from the front fit, long-run level = the
            # contract's own mean squared return over the 252 days before the refit month
            prior = x[x.index < m].tail(252).dropna()
            if len(prior) < 40:
                continue
            V = float((prior ** 2).mean())
            pc = np.array([V * (1 - p.a - p.b), p.a, p.b])
            h = pd.Series(garch_filter(x.values, pc), index=x.index)
            h1 = h.reindex(g.date).values
            s.loc[g.index, "garch20"] = np.sqrt(252 * garch_avg(h1, pc, 20)) / 100
            s.loc[g.index, "garch_life"] = [np.sqrt(252 * garch_avg(a, pc, max(int(t), 1))) / 100
                                            for a, t in zip(h1, g.tdays)]
    # regimes / exogenous
    s["sdev"] = D.storage_dev(s.date).values
    pan = D.panel()
    s = s.join(pan[["lvl_w1", "lvl_w2", "hj"]], on="date")
    s["dm"] = s.cid % 100
    s["dclass"] = s.dm.map(D.dclass)
    s["gy"] = D.gas_year(s.date)
    s.drop(columns=["mstart"]).to_csv(f, index=False)
    return pd.read_csv(f, parse_dates=["date", "oexp"])


if __name__ == "__main__":
    b = build(force=True)
    print(b.shape)
    print(b[["atm_iv", "rv5", "rv21", "ewma", "garch20", "rv20f", "rv_life", "garch_life", "sdev"]].describe().T)
