"""Frozen tests for the locked 2022-2026 options sample (v5_greeks_signals). Run ONCE on the locked panel.

    python -m src.research.v5_greeks_signals.frozen_test --dev      # development reference (<= 2021)
    python -m src.research.v5_greeks_signals.frozen_test --locked   # final exam: rows 2022-01-01 .. end

Self-contained: front returns are rebuilt from ng_curve_daily.csv (nearest contract with >= 2 trading
days to futures expiry, expiry = 3rd trading day before the delivery month), options from
<v5dir>/dh_straddle.csv and <v5dir>/weekly.csv, weekly settles from the raw weekly options file,
EIA release days from data/raw/storage.csv. Walk-forward z-scores use all history from 2010
(expanding, monthly refit, data before the month only), so the locked run needs the full 2010-2026 file.

P1 (primary): resid_rel123(t) = sum over the 3 nearest ON expiries of the attribution residual
   (pnl - gamma_pnl - theta_pnl - vega_pnl) of the straddle held t-1 -> t, divided by the sum of their
   premiums P0. Regression r_front(t+1) ~ 1 + z(resid_rel123) + z(r_front(t)) + z(r_front(t-4..t)),
   Newey-West 5 lags, test rows. PASS if t(z) >= +1.645 in the raw regression AND in the winsorised one
   (z clipped at +-2.33, y clipped at its test-sample 1st/99th percentiles).
   Logged: thr1 rule (long 1 front contract when z>1, short when z<-1, 1-day hold): mean next-day return.
S1: same with vega_pnl123 (sum over the 3 nearest expiries of vega0*(iv1-iv0)); expected sign negative;
   PASS if t(z) <= -1.645 raw AND winsorised.
S2: sell, at the settle before each EIA report day, the shortest weekly ATM straddle (<= 2 trading days)
   whose life spans the report, hold to expiry; net of 3% of premium (+$0.002 if exercised).
   PASS if mean net P&L > 0 and 4-trade moving-block bootstrap 2.5% bound > 0.
   Logged: same trade on spans with no report ('none'), and report-minus-none difference.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.research.v5_greeks_signals import common as C
from src.research.v5_greeks_signals import part4_weekly as P4


def front_returns(end):
    c = pd.read_csv(C.ROOT / "data/raw/ng_curve_daily.csv", parse_dates=["date"],
                    usecols=["date", "contract_year", "contract_month", "settle"]).dropna(subset=["settle"])
    c = c[(c.date <= end) & (c.settle > 0)]
    c["cid"] = c.contract_year * 100 + c.contract_month
    px = c.pivot_table(index="date", columns="cid", values="settle").sort_index()
    days = px.index
    fexp = {}
    for cid in px.columns:
        y, m = divmod(int(cid), 100)
        p = days.searchsorted(pd.Timestamp(y, m, 1))
        fexp[cid] = p - 3 if 3 <= p < len(days) else 10 ** 9   # index of expiry day
    fexp = pd.Series(fexp)
    out = {}
    lp = np.log(px)
    for i in range(1, len(days)):
        live = fexp[(fexp - i >= 2)].index
        live = [x for x in sorted(live) if np.isfinite(px.iat[i, px.columns.get_loc(x)])]
        if not live:
            continue
        fr = live[0]
        out[days[i]] = 100 * (lp.iat[i, px.columns.get_loc(fr)] - lp.iat[i - 1, px.columns.get_loc(fr)])
    return pd.Series(out, name="r_front"), px


def greek_signals(v5dir, end):
    d = pd.read_csv(v5dir / "dh_straddle.csv", parse_dates=["date", "date1"])
    d = d[d.date1 <= end].sort_values(["date", "T"])
    d["rank"] = d.groupby("date").cumcount() + 1
    d = d[d["rank"] <= 3]
    g = d.groupby("date1")
    n = g.size()
    s = pd.DataFrame({"vega_pnl123": g.vega_pnl.sum(), "resid_rel123": g.resid.sum() / g.P0.sum()})
    return s[n == 3], d


def hac(y, X):
    b, t, n = C.hac_ols(y, X, 5)
    return b[1], t[1], n


def run(v5dir, start, end, label):
    r, px = front_returns(end)
    sig, dfull = greek_signals(v5dir, end)
    f = pd.DataFrame({"r_front": r}).join(sig, how="left")
    f["r_front_5"] = f.r_front.rolling(5).sum()
    z = {k: C.wf_z(f[k]) for k in ("vega_pnl123", "resid_rel123", "r_front", "r_front_5")}
    y = f.r_front.shift(-1)
    sel = (f.index >= start) & (f.index <= end)
    res = {"label": label, "start": str(start.date()), "end": str(end.date())}
    for key, s, sign in (("P1", "resid_rel123", +1), ("S1", "vega_pnl123", -1)):
        X = pd.DataFrame({"z": z[s], "r1": z["r_front"], "r5": z["r_front_5"]})[sel]
        yy = y[sel]
        b, t, n = hac(yy, X)
        Xw = X.assign(z=X.z.clip(-2.33, 2.33))
        yw = yy.clip(yy.quantile(0.01), yy.quantile(0.99))
        bw, tw, _ = hac(yw, Xw)
        # tradable log: thr1, 1-day hold, direction frozen = sign
        pos = (sign * np.sign(z[s])).where(z[s].abs() > 1, 0.0).fillna(0)
        res[key] = {"signal": s, "beta": b, "t": t, "n": n, "beta_w": bw, "t_w": tw,
                    "pass": bool(sign * t >= 1.645 and sign * tw >= 1.645)}
        # $ P&L needs the front $ change: rebuild from returns and price level of the front
        res[key]["thr1_days_active"] = int((pos[sel] != 0).sum())
        res[key]["thr1_mean_r_next_pct"] = float((pos * y)[sel][pos[sel] != 0].mean())
    # S2 weekly report straddle
    days = px.index
    R = P4.report_days(days, cut=end)
    isR = pd.Series(days.isin(R), index=days)
    w = pd.read_csv(v5dir / "weekly.csv", parse_dates=["date", "expiry"])
    w = w[w.date <= end]
    cid = []
    for d_, F in zip(w.date, w.F):
        row = px.loc[d_].dropna()
        row = row[row.index >= d_.year * 100 + d_.month][:3]
        cid.append(row.index[np.argmin(np.abs(row.values - F))] if len(row) else np.nan)
    w["cid"] = cid
    w["match_err"] = [abs(px.at[d_, c] - F) if np.isfinite(c) else np.nan for d_, c, F in zip(w.date, w.cid, w.F)]
    w = w[w.match_err < 0.02].copy()
    pos_ = {d_: i for i, d_ in enumerate(days)}
    w["n"] = w.tdays
    w["nE"] = [int(isR.iloc[pos_[d_] + 1: pos_[d_] + 1 + n].sum()) for d_, n in zip(w.date, w.n)]
    w = w[(w.atm_iv > 0.05) & (w.atm_iv < 3)]
    dh_ = dfull
    tr = P4.trade(w, px, days, pos_, isR, cut=end, dh_df=dh_)
    tr = tr[(tr.date >= max(start, pd.Timestamp("2014-01-01"))) & (tr.date <= end)]
    rep, non = tr[tr.kind == "report"], tr[tr.kind == "none"]
    lo, hi = C.block_boot_ci(rep.net.values, 4) if len(rep) >= 10 else (np.nan, np.nan)
    res["S2"] = {"n": len(rep), "mean_net": rep.net.mean(), "ci": [lo, hi], "hit": (rep.net > 0).mean(),
                 "none_n": len(non), "none_mean_net": non.net.mean(),
                 "pass": bool(len(rep) >= 10 and rep.net.mean() > 0 and lo > 0)}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--locked", action="store_true")
    a = ap.parse_args()
    if a.locked:
        out = run(C.ROOT / "data/processed/v5_locked", pd.Timestamp("2022-01-01"), pd.Timestamp("2026-12-31"), "locked")
    else:
        out = {}
        for lab, s, e in (("dev_all", "2010-01-01", "2021-12-31"), ("dev_choose", "2010-01-01", "2016-12-31"),
                          ("dev_check", "2017-01-01", "2021-12-31")):
            out[lab] = run(C.V5, pd.Timestamp(s), pd.Timestamp(e), lab)
    print(json.dumps(out, indent=1, default=float))
    if not a.locked:
        (C.CACHE / "frozen_dev_reference.json").write_text(json.dumps(out, indent=1, default=float))
