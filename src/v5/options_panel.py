"""Study v5 shared OPTIONS panel (built once by the orchestrator; agents read it, never rebuild it).

Licensed CME data (Databento settlements): every output is git-ignored.
  data/processed/v5/        development rows only (date <= 2021-12-31)   <- agents read these
  data/processed/v5_locked/ full 2010-2026 rows (2022-01-01 onward is the locked options test sample)

Conventions
  * Black-76 / Bachelier with r = 0 (settlement prices, no discounting; bias < ~1% of premium).
  * Monthly options (ON) are American; treated as European (early-exercise premium is negligible
    near the money for futures options at these rates).
  * Option expiry: the trading day before the futures' last trading day (3 trading days before the
    delivery month). Weekly options: expiry = the series' last settlement date.
  * Moneyness k = ln(K/F). Implied vols are taken from OUT-of-the-money options (calls K>=F, puts K<F).
  * Underlying: ON options on contract cid use that futures settle F. Weekly options and spread options
    use the parity-implied underlying F = K + C - P at the strike where |C - P| is smallest.

Files
  surface.csv   one row per (date, cid) for ON options with T >= 2 trading days and <= 18 months:
      F, T (years), atm_iv (straddle at the two strikes around F, interpolated to F),
      iv_m20, iv_m10, iv_p10, iv_p20 (iv at k = -0.2, -0.1, +0.1, +0.2; linear in k between OTM strikes),
      skew = iv_p10 - iv_m10 (calls minus puts: >0 = upside fear), curv = (iv_p10 + iv_m10)/2 - atm_iv,
      atm straddle Greeks per 1 contract unit ($/MMBtu): gamma, vega (per 1.00 vol), theta (per calendar day),
      call_oi, put_oi, call_vol, put_vol (near-the-money strikes), gex = sum_K OI_K * gamma_K * F^2 / 100
      (gamma notional per 1% move, unsigned: dealer side unknown), n_strikes
  dh_straddle.csv   delta-hedged ATM straddle held one trading day, per (date, cid):
      K (strike nearest F at d), P0, P1 (straddle settles d and next day, same strike), F0, F1,
      delta0, gamma0, vega0, theta0, iv0, iv1 (straddle-implied at that strike),
      pnl = (P1 - P0) - delta0 (F1 - F0), and attribution:
      gamma_pnl = 0.5 gamma0 (F1-F0)^2, theta_pnl = theta0 * calendar days, vega_pnl = vega0 (iv1 - iv0),
      resid = pnl - gamma_pnl - theta_pnl - vega_pnl
  spreads.csv   one row per (date, cid) for 1-month calendar spread options (G4X; cid = near leg month):
      S (futures spread near-far), S_parity, T, nv (Bachelier ATM normal vol, $/MMBtu/sqrt(yr)),
      iv_near, iv_far (ON ATM vols of the legs), rho_impl = (sn^2 + sf^2 - nv^2)/(2 sn sf) with
      sn = F_near*iv_near, sf = F_far*iv_far (normal approx; far-leg vol measured to its own expiry -
      a known approximation), rho_valid flag, n_strikes, cso_call_oi, cso_put_oi
  weekly.csv    per (date, series) for weekly options: root, expiry, T, F (parity), atm_iv, n_strikes
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/options"
OUT_DEV = ROOT / "data/processed/v5"
OUT_LOCK = ROOT / "data/processed/v5_locked"
CUT = pd.Timestamp("2021-12-31")


# ------------------------------------------------------------------ pricing (vectorised)
def black_price(F, K, T, s, cp):
    v = s * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * v * v) / v
    d2 = d1 - v
    call = F * ndtr(d1) - K * ndtr(d2)
    return np.where(cp == "C", call, call - (F - K))


def black_iv(price, F, K, T, cp, lo=1e-3, hi=6.0, n=60):
    price, F, K, T = map(np.asarray, (price, F, K, T))
    a, b = np.full(price.shape, lo), np.full(price.shape, hi)
    ok = (T > 0) & (F > 0) & (K > 0) & np.isfinite(price)
    intrinsic = np.where(cp == "C", np.maximum(F - K, 0), np.maximum(K - F, 0))
    ok &= price > intrinsic + 1e-6
    for _ in range(n):
        m = 0.5 * (a + b)
        pm = black_price(F, K, np.maximum(T, 1e-9), m, cp)
        up = pm < price
        a = np.where(up, m, a)
        b = np.where(up, b, m)
    out = 0.5 * (a + b)
    out[~ok | (out > hi * 0.999)] = np.nan
    return out


def black_greeks(F, K, T, s):
    v = s * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * v * v) / v
    pdf = norm.pdf(d1)
    gamma = pdf / (F * v)                       # per option
    vega = F * pdf * np.sqrt(T)                 # per 1.00 vol
    theta = -F * pdf * s / (2 * np.sqrt(T)) / 365.25   # per calendar day
    return d1, gamma, vega, theta


def bach_straddle(F, K, T, s):
    v = s * np.sqrt(T)
    d = (F - K) / v
    return (F - K) * (2 * ndtr(d) - 1) + 2 * v * norm.pdf(d)


def bach_iv_straddle(price, F, K, T, lo=1e-5, hi=30.0, n=60):
    a, b = np.full(np.shape(price), lo), np.full(np.shape(price), hi)
    for _ in range(n):
        m = 0.5 * (a + b)
        up = bach_straddle(F, K, T, m) < price
        a = np.where(up, m, a)
        b = np.where(up, b, m)
    out = 0.5 * (a + b)
    out[(out > hi * 0.999) | ~np.isfinite(price) | (price <= np.abs(F - K) + 1e-6)] = np.nan
    return out


# ------------------------------------------------------------------ inputs
def futures():
    c = pd.read_csv(ROOT / "data/raw/ng_curve_daily.csv", parse_dates=["date"],
                    usecols=["date", "contract_year", "contract_month", "settle"]).dropna(subset=["settle"])
    c = c[c.settle > 0]
    c["cid"] = c.contract_year * 100 + c.contract_month
    px = c.pivot_table(index="date", columns="cid", values="settle").sort_index()
    days = px.index
    fexp, oexp = {}, {}
    for cid in px.columns:
        y, m = divmod(int(cid), 100)
        p = days.searchsorted(pd.Timestamp(y, m, 1))
        if 4 <= p < len(days):
            fexp[cid], oexp[cid] = days[p - 3], days[p - 4]
        else:
            e = pd.Timestamp(y, m, 1) - pd.offsets.BDay(3)
            fexp[cid], oexp[cid] = e, e - pd.offsets.BDay(1)
    return px, pd.Series(fexp), pd.Series(oexp), days


def load_monthly():
    o = pd.read_csv(RAW / "ng_options_near_money_v2.csv.gz", parse_dates=["date"])
    o = o[o.root.isin(["ON", "G4X"]) & (o.settle > 0)].copy()
    o["cid"] = (o.cy * 100 + o.cm).astype(int)
    o["K"] = o.k_raw / 1000.0
    return o


def trading_years(d0, d1, days):
    """Trading-day count / 252 between two dates (expiry inclusive)."""
    i0, i1 = days.searchsorted(d0), days.searchsorted(d1)
    return (i1 - i0) / 252.0


# ------------------------------------------------------------------ surface + greeks + flows
def build_surface(o, px, oexp, days):
    on = o[o.root == "ON"].copy()
    Fm = px.stack().rename("F")
    Fm.index.names = ["date", "cid"]
    on = on.join(Fm, on=["date", "cid"]).dropna(subset=["F"])
    on["expiry"] = on.cid.map(oexp)
    pos = pd.Series(np.arange(len(days)), index=days)
    on["tdays"] = on.expiry.map(lambda e: days.searchsorted(e)) - on.date.map(pos)
    on = on[(on.tdays >= 2) & (on.tdays <= 380)]
    on["T"] = on.tdays / 252.0
    otm = ((on.cp == "C") & (on.K >= on.F)) | ((on.cp == "P") & (on.K < on.F))
    on["iv"] = np.nan
    on.loc[otm, "iv"] = black_iv(on.loc[otm, "settle"].values, on.loc[otm, "F"].values,
                                 on.loc[otm, "K"].values, on.loc[otm, "T"].values, on.loc[otm, "cp"].values)
    on["k"] = np.log(on.K / on.F)
    # greeks for every strike at its own iv (OTM iv applied to both C and P at that strike)
    ivk = on[otm][["date", "cid", "K", "iv"]].dropna()
    on = on.drop(columns="iv").merge(ivk, on=["date", "cid", "K"], how="left")
    _, g, _, _ = black_greeks(on.F.values, on.K.values, on["T"].values, on.iv.fillna(0.5).values)
    on["gamma_k"] = np.where(on.iv.notna(), g, np.nan)
    rows = []
    for (d, cid), g in on.groupby(["date", "cid"], sort=False):
        F, T = g.F.iat[0], g["T"].iat[0]
        sm = g.dropna(subset=["iv"]).drop_duplicates("K").sort_values("k")
        if len(sm) < 3:
            continue
        kk, vv = sm.k.values, sm.iv.values
        def at(x):
            return np.interp(x, kk, vv) if kk[0] <= x <= kk[-1] else np.nan
        atm = at(0.0)
        r = {"date": d, "cid": cid, "F": F, "T": T, "atm_iv": atm,
             "iv_m20": at(-0.2), "iv_m10": at(-0.1), "iv_p10": at(0.1), "iv_p20": at(0.2), "n_strikes": len(sm)}
        if np.isfinite(atm):
            _, gam, veg, th = black_greeks(F, F, T, atm)
            r.update(gamma=2 * gam, vega=2 * veg, theta=2 * th)
        c, p = g[g.cp == "C"], g[g.cp == "P"]
        r.update(call_oi=c.oi.sum(min_count=1), put_oi=p.oi.sum(min_count=1),
                 call_vol=c.volume.sum(min_count=1), put_vol=p.volume.sum(min_count=1),
                 gex=float((g.oi.fillna(0) * g.gamma_k.fillna(0)).sum() * F * F / 100))
        rows.append(r)
    s = pd.DataFrame(rows)
    s["skew"] = s.iv_p10 - s.iv_m10
    s["curv"] = (s.iv_p10 + s.iv_m10) / 2 - s.atm_iv
    return s, on


def build_dh(on, px, days):
    """Delta-hedged ATM straddle, one trading day, same strike."""
    st = on.pivot_table(index=["date", "cid", "K"], columns="cp", values="settle").dropna()
    st["P"] = st["C"] + st["P"]
    st = st["P"]
    nxt = pd.Series(days[1:], index=days[:-1])
    base = on.drop_duplicates(["date", "cid"])[["date", "cid", "F", "T"]].copy()
    base["d1"] = base.date.map(nxt)
    out = []
    for r in base.itertuples(index=False):
        if pd.isna(r.d1):
            continue
        try:
            ks = st.loc[(r.date, r.cid)]
        except KeyError:
            continue
        K = ks.index.values[np.argmin(np.abs(ks.index.values - r.F))]
        try:
            P1 = st.loc[(r.d1, r.cid, K)]
        except KeyError:
            continue
        F1 = px.at[r.d1, r.cid] if r.cid in px.columns and r.d1 in px.index else np.nan
        out.append((r.date, r.d1, r.cid, K, ks.loc[K], P1, r.F, F1, r.T))
    d = pd.DataFrame(out, columns=["date", "date1", "cid", "K", "P0", "P1", "F0", "F1", "T"])
    d = d.dropna()
    tdays = np.maximum(d["T"].values * 252 - 1, 0.5)
    d["iv0"] = _straddle_iv(d.P0.values, d.F0.values, d.K.values, d["T"].values)
    d["iv1"] = _straddle_iv(d.P1.values, d.F1.values, d.K.values, tdays / 252.0)
    d1, g, v, th = black_greeks(d.F0.values, d.K.values, d["T"].values, d.iv0.values)
    d["delta0"] = 2 * ndtr(d1) - 1
    d["gamma0"], d["vega0"], d["theta0"] = 2 * g, 2 * v, 2 * th
    dF = d.F1 - d.F0
    cal = (d.date1 - d.date).dt.days
    d["pnl"] = (d.P1 - d.P0) - d.delta0 * dF
    d["gamma_pnl"] = 0.5 * d.gamma0 * dF ** 2
    d["theta_pnl"] = d.theta0 * cal
    d["vega_pnl"] = d.vega0 * (d.iv1 - d.iv0)
    d["resid"] = d.pnl - d.gamma_pnl - d.theta_pnl - d.vega_pnl
    return d


def _straddle_iv(P, F, K, T):
    a, b = np.full(P.shape, 1e-3), np.full(P.shape, 6.0)
    for _ in range(60):
        m = 0.5 * (a + b)
        c = black_price(F, K, T, m, np.array(["C"] * len(P)))
        val = c + (c - (F - K))
        up = val < P
        a, b = np.where(up, m, a), np.where(up, b, m)
    out = 0.5 * (a + b)
    out[(out > 5.99) | (P <= np.abs(F - K) + 1e-6)] = np.nan
    return out


# ------------------------------------------------------------------ calendar spread options
def build_spreads(o, px, oexp, days, surface):
    g4 = o[o.root == "G4X"].copy()
    st = g4.pivot_table(index=["date", "cid", "K"], columns="cp", values="settle").dropna().reset_index()
    st["d"] = (st.C - st.P).abs()
    oi = g4.groupby(["date", "cid", "cp"]).oi.sum(min_count=1).unstack("cp")
    atm_iv = surface.set_index(["date", "cid"]).atm_iv
    pos = pd.Series(np.arange(len(days)), index=days)
    rows = []
    for (d, cid), g in st.groupby(["date", "cid"], sort=False):
        far = (cid // 100) * 100 + cid % 100 + 1 if cid % 100 < 12 else (cid // 100 + 1) * 100 + 1
        e = oexp.get(cid)
        if e is None or d not in pos.index:
            continue
        tdays = days.searchsorted(e) - pos[d]
        if tdays < 2:
            continue
        T = tdays / 252.0
        Fn = px.at[d, cid] if cid in px.columns and d in px.index else np.nan
        Ff = px.at[d, far] if far in px.columns and d in px.index else np.nan
        S = Fn - Ff
        j = g.d.values.argmin()
        Spar = g.K.values[j] + g.C.values[j] - g.P.values[j]
        F = S if np.isfinite(S) else Spar
        ks = g.K.values
        lo, hi = ks[ks <= F], ks[ks >= F]
        if not len(lo) or not len(hi):
            continue
        vals = []
        for K in sorted({lo.max(), hi.min()}):
            P = float((g.C + g.P)[g.K == K].iat[0])
            vals.append((K, bach_iv_straddle(np.array([P]), F, K, T)[0]))
        vals = [(k, v) for k, v in vals if np.isfinite(v)]
        if not vals:
            continue
        nv = vals[0][1] if len(vals) == 1 else np.interp(F, [vals[0][0], vals[1][0]], [vals[0][1], vals[1][1]])
        ivn, ivf = atm_iv.get((d, cid), np.nan), atm_iv.get((d, far), np.nan)
        sn, sf = Fn * ivn, Ff * ivf
        rho = (sn ** 2 + sf ** 2 - nv ** 2) / (2 * sn * sf) if np.isfinite(sn * sf) and sn * sf > 0 else np.nan
        r = {"date": d, "cid": cid, "S": S, "S_parity": Spar, "T": T, "nv": nv, "iv_near": ivn, "iv_far": ivf,
             "rho_impl": rho, "rho_valid": bool(np.isfinite(rho) and -1 <= rho <= 1), "n_strikes": len(ks)}
        if (d, cid) in oi.index:
            r["cso_call_oi"], r["cso_put_oi"] = oi.loc[(d, cid)].get("C", np.nan), oi.loc[(d, cid)].get("P", np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ weekly options
def build_weekly(days):
    w = pd.read_csv(RAW / "ng_weekly_options_v2.csv.gz", parse_dates=["date"])
    w = w[w.settle > 0].copy()
    w["K"] = w.k_raw / 1000.0
    w["series"] = w.root + "_" + (w.cy * 100 + w.cm).astype(str)
    expiry = w.groupby("series").date.max()
    st = w.pivot_table(index=["date", "series", "K"], columns="cp", values="settle").dropna().reset_index()
    st["d"] = (st.C - st.P).abs()
    pos = pd.Series(np.arange(len(days)), index=days)
    rows = []
    for (d, s), g in st.groupby(["date", "series"], sort=False):
        e = expiry[s]
        if d not in pos.index or e <= d:
            continue
        tdays = days.searchsorted(e) - pos[d]
        if tdays < 1:
            continue
        j = g.d.values.argmin()
        F = g.K.values[j] + g.C.values[j] - g.P.values[j]
        K = g.K.values[j]
        P = g.C.values[j] + g.P.values[j]
        iv = _straddle_iv(np.array([P]), np.array([F]), np.array([K]), np.array([tdays / 252.0]))[0]
        rows.append({"date": d, "series": s, "root": s.split("_")[0], "expiry": e, "tdays": tdays,
                     "F": F, "atm_iv": iv, "n_strikes": len(g)})
    return pd.DataFrame(rows)


def build():
    OUT_DEV.mkdir(parents=True, exist_ok=True)
    OUT_LOCK.mkdir(parents=True, exist_ok=True)
    px, fexp, oexp, days = futures()
    o = load_monthly()
    surface, on = build_surface(o, px, oexp, days)
    print("surface", surface.shape, flush=True)
    dh = build_dh(on, px, days)
    print("dh", dh.shape, flush=True)
    spreads = build_spreads(o, px, oexp, days, surface)
    print("spreads", spreads.shape, flush=True)
    weekly = build_weekly(days)
    print("weekly", weekly.shape, flush=True)
    for name, df in (("surface", surface), ("dh_straddle", dh), ("spreads", spreads), ("weekly", weekly)):
        df.to_csv(OUT_LOCK / f"{name}.csv", index=False)
        df[df.date <= CUT].to_csv(OUT_DEV / f"{name}.csv", index=False)
    return surface, dh, spreads, weekly


if __name__ == "__main__":
    build()
