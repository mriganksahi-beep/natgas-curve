"""Build the daily signal / target panel (cached). All signals are known at the settle of day t.

Signals
  Greek attribution (dh_straddle rows with date1 = t, i.e. the straddle held t-1 -> t; ranks 1-3 by T):
    vshock1, vshock123 : implied-vol change at fixed strike (iv1 - iv0) minus the part explained by the
                         same-day futures log move, beta fitted walk-forward (expanding, monthly), per rank;
                         vshock123 = mean over ranks 1-3. Vol points (0.01 = 1 vol).
    vshock123_5        : 5-day sum of vshock123
    vega_pnl123        : sum of vega_pnl over ranks 1-3 ($/MMBtu per straddle)
    resid_rel123       : sum resid / sum P0 (unexplained straddle P&L as a fraction of premium)
    gt_rel123          : (gamma_pnl + theta_pnl) / P0 summed - realised-minus-implied variance; a pure
                         FUTURES-move quantity, used as a control/benchmark
  Surface (same-contract changes t-1 -> t; rank fixed at t-1):
    d_level, d_term (atm1 - atm3), d_skew1, d_skew2, d_curv1 ; *_res = residual on the same-day front
    return (walk-forward beta); skew1/skew2/term/curv1 levels; skew1_ch5 (5-day change);
    pc1..pc3 = walk-forward PCA scores of daily surface changes.
  Flows (LAGGED one trading day: CME open interest / final volume for day t are published on t+1):
    pc_oi (log put OI / call OI, ranks 1-2), d_pc_oi5, pc_vol (log put vol / call vol),
    call_spike, put_spike (log volume vs prior 20-day mean, ranks 1-2), spike_diff = call - put.
Targets / P&L inputs
  dF_front, dF_second (rolled, $), d_hj ($), r_front, rv5 (sum of next 5 r_front^2),
  straddle: rank-1 dh_straddle row with date = t (pnl t -> t+1), P0, gamma0, dF, switch flag.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.v5_greeks_signals import common as C

OUT = C.CACHE / "features.pkl"


def greek_block(d: pd.DataFrame) -> pd.DataFrame:
    d = d[d["rank"] <= 3].copy()
    d["dv"] = d.iv1 - d.iv0
    d["r"] = np.log(d.F1 / d.F0)
    rows = []
    for k in (1, 2, 3):
        x = d[d["rank"] == k].set_index("date1").sort_index()
        x = x[~x.index.duplicated()]
        res = C.wf_beta_resid(x.dv, x[["r"]])
        rows.append(pd.DataFrame({f"vshock_r{k}": res, f"vega_pnl_r{k}": x.vega_pnl,
                                  f"resid_r{k}": x.resid, f"P0_r{k}": x.P0,
                                  f"gt_r{k}": x.gamma_pnl + x.theta_pnl}))
    g = pd.concat(rows, axis=1)
    out = pd.DataFrame(index=g.index)
    out["vshock1"] = g.vshock_r1
    out["vshock123"] = g[[f"vshock_r{k}" for k in (1, 2, 3)]].mean(axis=1)
    out["vshock123_5"] = out.vshock123.rolling(5).sum()
    out["vega_pnl123"] = g[[f"vega_pnl_r{k}" for k in (1, 2, 3)]].sum(axis=1, min_count=3)
    P0 = g[[f"P0_r{k}" for k in (1, 2, 3)]].sum(axis=1, min_count=3)
    out["resid_rel123"] = g[[f"resid_r{k}" for k in (1, 2, 3)]].sum(axis=1, min_count=3) / P0
    out["gt_rel123"] = g[[f"gt_r{k}" for k in (1, 2, 3)]].sum(axis=1, min_count=3) / P0
    out.index.name = "date"
    return out


def straddle_block(d: pd.DataFrame) -> pd.DataFrame:
    x = d[d["rank"] == 1].set_index("date").sort_index()
    x = x[~x.index.duplicated()]
    st = pd.DataFrame({"st_pnl": x.pnl, "st_P0": x.P0, "st_gamma0": x.gamma0,
                       "st_dF": x.F1 - x.F0, "st_cid": x.cid, "st_iv0": x.iv0,
                       "st_vega_pnl": x.vega_pnl})
    st["st_switch"] = (st.st_cid != st.st_cid.shift()).astype(int)
    return st


def surface_block(s: pd.DataFrame, r_front: pd.Series) -> pd.DataFrame:
    dates = np.sort(s.date.unique())
    prev = pd.Series(dates[:-1], index=dates[1:])
    piv = s.set_index(["date", "cid"])
    cols = ["atm_iv", "iv_m20", "iv_m10", "iv_p10", "iv_p20", "skew", "curv",
            "call_oi", "put_oi", "call_vol", "put_vol"]
    rk = s.set_index("date")[["cid", "rank"]]
    rows = {}
    for d, dp in prev.items():
        r0 = rk.loc[dp]
        c = r0.sort_values("rank").cid.values[:6]
        try:
            a = piv.loc[dp].reindex(c)[cols]
            b = piv.loc[d].reindex(c)[cols]
        except KeyError:
            continue
        rec = {}
        for i in range(min(6, len(c))):
            rec[f"datm{i+1}"] = b.atm_iv.iloc[i] - a.atm_iv.iloc[i]
        for kcol in ("iv_m20", "iv_m10", "iv_p10", "iv_p20"):
            rec[f"d{kcol}_1"] = b[kcol].iloc[0] - a[kcol].iloc[0]
        rec["d_skew1"] = b["skew"].iloc[0] - a["skew"].iloc[0]
        rec["d_skew2"] = b["skew"].iloc[1] - a["skew"].iloc[1] if len(c) > 1 else np.nan
        rec["d_curv1"] = b.curv.iloc[0] - a.curv.iloc[0]
        rows[d] = rec
    ch = pd.DataFrame.from_dict(rows, orient="index")
    ch["d_level"] = ch.datm1
    ch["d_term"] = ch.datm1 - ch.datm3
    # levels on day t (current ranks)
    lv = {}
    for d, g in s.groupby("date"):
        g = g.sort_values("rank")
        oi = g.head(2)
        lv[d] = {"skew1": g["skew"].iloc[0], "skew2": g["skew"].iloc[1] if len(g) > 1 else np.nan,
                 "curv1": g.curv.iloc[0], "atm1": g.atm_iv.iloc[0],
                 "term": g.atm_iv.iloc[0] - (g.atm_iv.iloc[2] if len(g) > 2 else np.nan),
                 "call_oi": oi.call_oi.sum(), "put_oi": oi.put_oi.sum(),
                 "call_vol": oi.call_vol.sum(), "put_vol": oi.put_vol.sum()}
    lv = pd.DataFrame.from_dict(lv, orient="index")
    out = ch.join(lv, how="outer").sort_index()
    out.index.name = "date"
    # residualise surface changes on the same-day front return (walk-forward)
    rf = r_front.reindex(out.index)
    X = pd.DataFrame({"r": rf})
    for k in ("d_level", "d_term", "d_skew1", "d_skew2", "d_curv1"):
        out[f"{k}_res"] = C.wf_beta_resid(out[k], X)
    out["skew1_ch5"] = out.skew1 - out.skew1.shift(5)
    # flows (lag one day)
    fl = pd.DataFrame(index=out.index)
    fl["pc_oi"] = np.log(out.put_oi / out.call_oi)
    fl["d_pc_oi5"] = fl.pc_oi - fl.pc_oi.shift(5)
    fl["pc_vol"] = np.log((out.put_vol + 1) / (out.call_vol + 1))
    fl["call_spike"] = np.log((out.call_vol + 1) / (out.call_vol.shift(1).rolling(20).mean() + 1))
    fl["put_spike"] = np.log((out.put_vol + 1) / (out.put_vol.shift(1).rolling(20).mean() + 1))
    fl["spike_diff"] = fl.call_spike - fl.put_spike
    out = out.join(fl.shift(1))
    return out


def wf_pca(ch: pd.DataFrame, cols, n=3, min_obs=250) -> pd.DataFrame:
    X = ch[cols]
    per = X.index.to_period("M")
    out = pd.DataFrame(np.nan, index=X.index, columns=[f"pc{i+1}" for i in range(n)])
    load_hist = {}
    for m in per.unique():
        hist = X[per < m].dropna()
        if len(hist) < min_obs:
            continue
        mu, sd = hist.mean(), hist.std()
        Z = (hist - mu) / sd
        w, v = np.linalg.eigh(np.cov(Z.values.T))
        v = v[:, ::-1][:, :n]
        # sign convention: PC1 loads + on ATM; PC2 + on front-minus-back ATM; PC3 + on call wing
        ref = [cols.index("datm1"), cols.index("datm1"), cols.index("div_p20_1")]
        for i in range(n):
            if i == 1:
                s_ = v[cols.index("datm1"), i] - v[cols.index("datm6"), i]
            else:
                s_ = v[ref[i], i]
            if s_ < 0:
                v[:, i] *= -1
        load_hist[str(m)] = v
        cur = X[per == m]
        out.loc[cur.index] = (((cur - mu) / sd).values @ v)
    return out, load_hist


def build(force=False) -> pd.DataFrame:
    if OUT.exists() and not force:
        return pd.read_pickle(OUT)
    p = C.panel()
    d = C.dh()
    s = C.surface()
    g = greek_block(d)
    st = straddle_block(d)
    sf = surface_block(s, p.r_front)
    pcs, loads = wf_pca(sf, ["datm1", "datm2", "datm3", "datm4", "datm5", "datm6",
                             "div_m20_1", "div_m10_1", "div_p10_1", "div_p20_1"])
    pd.to_pickle(loads, C.CACHE / "pca_loadings.pkl")
    keep = ["front", "second", "px_front", "px_second", "r_front", "r_second", "dF_front", "dF_second",
            "roll_front", "roll_second", "hj", "d_hj", "hj_year", "news_tot", "news_w1", "news_w2",
            "news_w3", "news_w0", "lvl_w1", "lvl_w2", "lvl_w3", "storage_surprise_5y", "storage_change",
            "iv_front", "front_days_left"]
    f = p[keep].join(g, how="left").join(sf, how="left").join(pcs, how="left").join(st, how="left")
    f["roll_hj"] = (f.hj_year != f.hj_year.shift()).astype(int)
    f["rv5"] = C.fwd_sum(f.r_front ** 2, 5)
    f.to_pickle(OUT)
    return f


if __name__ == "__main__":
    f = build(force=True)
    print(f.shape)
    print(f.describe().T[["count", "mean", "std"]].to_string())
