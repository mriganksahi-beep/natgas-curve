"""Q1: does GEFS ensemble disagreement predict forward realised vol beyond HAR+seasonality, IV,
forecast level and storage?  Walk-forward: pooled OLS on log RV, expanding window, refitted at each
month start on rows whose forward window had ended (resolved) before that month start.

Feature blocks (all known at the d settle):
  HAR  : log rv5, log rv21, log rv63 (own contract), log tdays, calendar-month dummies (Oct..Apr)
  IV   : log atm_iv
  UL   : log of 3-day mean unc_n_w1, unc_n_w2, unc_n_w3          (disagreement LEVELS)
  UD   : 5-day change of log 3-day-mean unc_n_w2, unc_n_w3       (disagreement CHANGES)
  LVL  : lvl_w1, lvl_w2, lvl_w3, |lvl_w1+lvl_w2+lvl_w3|            (forecast level, cold > 0)
  STO  : sdev / 100                                               (storage tightness, from 2012)
Nested comparisons (one-sided Clark-West, HAC lags = h; gas-year bootstrap CI of the OOS R2 gain):
  A  HAR        -> HAR+UL
  B  HAR        -> HAR+UD
  C  HAR+IV     -> HAR+IV+UL+UD
  D  HAR+IV+LVL+STO -> +UL+UD        (rows with sdev; from winter 2012-13)
  E  HAR+IV     -> HAR+IV+UL[w2,w3 only]   (the brief's emphasis on w2/w3)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import data as D

MONTHS = [10, 11, 12, 1, 2, 3, 4]


def design(b):
    X = pd.DataFrame(index=b.index)
    X["c"] = 1.0
    for k in (5, 21, 63):
        X[f"lrv{k}"] = np.log(b[f"rv{k}"].clip(lower=0.05))
    X["ltd"] = np.log(b.tdays)
    for m in MONTHS[1:]:
        X[f"m{m}"] = (b.month == m).astype(float)
    X["liv"] = np.log(b.atm_iv)
    for c in D.UNC:
        X["ul_" + c[-2:]] = np.log(b[c + "_m3"])
    X["ud_w2"] = b["d5l_unc_n_w2"]
    X["ud_w3"] = b["d5l_unc_n_w3"]
    for c in ("lvl_w1", "lvl_w2", "lvl_w3"):
        X[c] = b[c] / 10.0
    X["lvl_abs"] = (b.lvl_w1 + b.lvl_w2 + b.lvl_w3).abs() / 10.0
    X["sto"] = b.sdev / 100.0
    return X


HAR = ["c", "lrv5", "lrv21", "lrv63", "ltd"] + [f"m{m}" for m in MONTHS[1:]]
UL = ["ul_w1", "ul_w2", "ul_w3"]
UD = ["ud_w2", "ud_w3"]
LVL = ["lvl_w1", "lvl_w2", "lvl_w3", "lvl_abs"]
MODELS = {
    "HAR": HAR, "HAR+UL": HAR + UL, "HAR+UD": HAR + UD,
    "HAR+IV": HAR + ["liv"], "HAR+IV+U": HAR + ["liv"] + UL + UD, "HAR+IV+U23": HAR + ["liv", "ul_w2", "ul_w3"],
    "FULL0": HAR + ["liv"] + LVL + ["sto"], "FULL": HAR + ["liv"] + LVL + ["sto"] + UL + UD,
}
COMPS = [("A", "HAR", "HAR+UL", False), ("B", "HAR", "HAR+UD", False), ("C", "HAR+IV", "HAR+IV+U", False),
         ("D", "FULL0", "FULL", True), ("E", "HAR+IV", "HAR+IV+U23", False)]


def walk_forward(b, X, y, cols, days):
    """Expanding-window OLS refitted at each month start; returns OOS predictions (NaN before first fit)."""
    pred = pd.Series(np.nan, index=b.index)
    ybar = pd.Series(np.nan, index=b.index)
    # resolution date of each row's target = trading day index of date + h
    di = days.get_indexer(b.date)
    res_idx = di + b.h.values
    ms = b.date.dt.to_period("M")
    ok = X[cols].notna().all(1) & y.notna()
    for m in sorted(ms.unique()):
        start = m.to_timestamp()
        si = days.searchsorted(start)
        tr = ok & (res_idx < si) & (b.date < start)
        te = (ms == m) & X[cols].notna().all(1)
        if tr.sum() < 250 or b.loc[tr, "gy"].nunique() < 1 or not te.any():
            continue
        Xt = X.loc[tr, cols]
        keep = [c for c in cols if Xt[c].std() > 0 or c == "c"]
        beta = np.linalg.lstsq(Xt[keep].values, y[tr].values, rcond=None)[0]
        pred[te] = X.loc[te, keep].values @ beta
        ybar[te] = y[tr].mean()
    return pred, ybar


def hac_mean_t(x, lags):
    x = pd.Series(x).dropna()
    r = sm.OLS(x.values, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return r.params[0], r.tvalues[0]


def run(force=False):
    f = D.CACHE / "q1_preds.csv"
    b = D.base()
    _, days, _ = D.futures()
    if f.exists() and not force:
        P = pd.read_csv(f, parse_dates=["date"])
    else:
        out = []
        for (h, leg), g in b.groupby(["h", "leg"]):
            g = g.sort_values("date").reset_index(drop=True)
            X = design(g)
            y = np.log(g.rvf)
            res = g[["date", "h", "leg", "gy", "win", "sdev"]].copy()
            res["y"] = y
            for name, cols in MODELS.items():
                res["p_" + name], res["ybar"] = walk_forward(g, X, y, cols, days)
            out.append(res)
        P = pd.concat(out, ignore_index=True)
        P.to_csv(f, index=False)
    rng = np.random.default_rng(7)
    rows = []
    for (h, leg), g in P.groupby(["h", "leg"]):
        for win in ("choose", "check", "all"):
            gw = g if win == "all" else g[g.win == win]
            for cid, m0, m1, needs in COMPS:
                q = gw.dropna(subset=["y", "p_" + m0, "p_" + m1, "ybar"]).sort_values("date")
                if needs:
                    q = q[q.sdev.notna()]
                e0 = q.y - q["p_" + m0]
                e1 = q.y - q["p_" + m1]
                eb = q.y - q.ybar
                r2_0 = 1 - (e0 ** 2).sum() / (eb ** 2).sum()
                r2_1 = 1 - (e1 ** 2).sum() / (eb ** 2).sum()
                cw = e0 ** 2 - (e1 ** 2 - (q["p_" + m0] - q["p_" + m1]) ** 2)  # Clark-West adjusted
                _, t = hac_mean_t(cw.values, lags=int(h))
                # gas-year bootstrap of the R2 gain
                G = q.assign(s0=e0 ** 2, s1=e1 ** 2, sb=eb ** 2).groupby("gy")[["s0", "s1", "sb"]].sum()
                k = len(G)
                idx = rng.integers(0, k, size=(2000, k))
                S = G.values[idx].sum(1)
                gain = (S[:, 0] - S[:, 1]) / S[:, 2]
                rows.append({"h": h, "leg": leg, "win": win, "comp": cid, "base": m0, "alt": m1, "n": len(q),
                             "r2_base": r2_0, "r2_alt": r2_1, "gain": r2_1 - r2_0,
                             "g_lo": np.percentile(gain, 2.5), "g_hi": np.percentile(gain, 97.5),
                             "cw_t": t})
    T = pd.DataFrame(rows)
    from scipy.stats import norm
    T["p"] = 1 - norm.cdf(T.cw_t)
    # Holm over the 30 choose-window tests
    c = T.win == "choose"
    pv = T.loc[c, "p"].sort_values()
    m = len(pv)
    holm = pd.Series(np.minimum(1, np.maximum.accumulate(pv.values * (m - np.arange(m)))), index=pv.index)
    T["p_holm_choose"] = holm
    T.to_csv(D.CACHE / "q1_results.csv", index=False)
    return T, P


def describe_coefs():
    """Descriptive (full-sample, not walk-forward) pooled coefficients of the FULL model, HAC lags = h,
    plus leave-one-winter-out range of the unc coefficients."""
    b = D.base()
    rows = []
    for (h, leg), g in b.groupby(["h", "leg"]):
        g = g.sort_values("date").reset_index(drop=True)
        X = design(g)
        y = np.log(g.rvf)
        for name in ("HAR+IV+U", "FULL"):
            cols = MODELS[name]
            ok = X[cols].notna().all(1) & y.notna()
            r = sm.OLS(y[ok], X.loc[ok, cols]).fit(cov_type="HAC", cov_kwds={"maxlags": int(h)})
            for c in UL + UD + (LVL + ["sto"] if name == "FULL" else []) + ["liv"]:
                lo = []
                for gy in g.gy.unique():
                    o2 = ok & (g.gy != gy)
                    lo.append(np.linalg.lstsq(X.loc[o2, cols].values, y[o2].values, rcond=None)[0][cols.index(c)])
                rows.append({"h": h, "leg": leg, "model": name, "var": c, "coef": r.params[c], "t": r.tvalues[c],
                             "lowo_min": min(lo), "lowo_max": max(lo)})
    T = pd.DataFrame(rows)
    T.to_csv(D.CACHE / "q1_coefs.csv", index=False)
    return T


if __name__ == "__main__":
    import sys
    T, _ = run(force="--force" in sys.argv)
    pd.set_option("display.width", 250)
    print(T.round(3).to_string())
    C = describe_coefs()
    print(C[C["var"].str.startswith("u")].round(3).to_string())
