"""Part 3: option positioning near expiry - does open-interest gamma damp realised vol or pin settlement?

Per monthly ON contract (front at its expiry), day E = option expiry, day A = E - 5 trading days:
  gex_A    : unsigned OI-weighted gamma notional at A (surface.csv); also 'conc' = gex_A divided by
             (near-money OI x ATM gamma x F^2/100) = how concentrated near-money OI is at the money
  vr       : realised vol of the futures over (A, E] (5 daily log changes, annualised) / atm_iv_A
  pin      : K* = strike with the largest call+put OI at A (raw ON file, <= 2021, within +-15% of F_A).
             hit = |F_E - K*| <= $0.025 ; placebo = same rate for K* +- $0.10 / +- $0.15
  dh short : P&L of shorting the ATM straddle delta-hedged daily from A to E-2 (dh_straddle rows),
             cost 1.5% of premium per side + hedge cost; compared across walk-forward gex terciles
             (percentile among previous contracts' gex_A, >= 12 contracts history).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.v5_greeks_signals import common as C


def expiries(px: pd.DataFrame) -> pd.Series:
    days = px.index
    oexp = {}
    for cid in px.columns:
        y, m = divmod(int(cid), 100)
        p = days.searchsorted(pd.Timestamp(y, m, 1))
        if 4 <= p < len(days):
            oexp[cid] = days[p - 4]
    return pd.Series(oexp)


def raw_oi(cids, dates):
    use = ["date", "root", "cy", "cm", "cp", "k_raw", "oi"]
    parts = []
    for ch in pd.read_csv(C.ROOT / "data/raw/options/ng_options_near_money_v2.csv.gz", usecols=use,
                          parse_dates=["date"], chunksize=1_000_000):
        ch = ch[(ch.date <= C.CUT) & (ch.root == "ON")]
        ch["cid"] = ch.cy * 100 + ch.cm
        key = set(zip(dates, cids))
        ch = ch[[k in key for k in zip(ch.date, ch.cid)]]
        parts.append(ch)
    return pd.concat(parts)


def main():
    px = C.curve()
    oexp = expiries(px)
    s = C.surface()
    d = C.dh()
    days = px.index
    recs = []
    for cid, E in oexp.items():
        if E < pd.Timestamp("2010-07-01") or E > C.CUT:
            continue
        pE = days.get_loc(E)
        A = days[pE - 5]
        row = s[(s.date == A) & (s.cid == cid)]
        if row.empty or not np.isfinite(px.at[E, cid]):
            continue
        row = row.iloc[0]
        r = np.diff(np.log(px[cid].iloc[pE - 5:pE + 1].values))
        rv = np.sqrt(252 / 5 * np.nansum(r ** 2))
        oi = row.call_oi + row.put_oi
        conc = row.gex / (oi * row.gamma * row.F ** 2 / 100) if oi > 0 else np.nan
        dd = d[(d.cid == cid) & (d.date >= A) & (d.date < E)]
        short = -(dd.pnl.sum()) * C.UNIT
        cost = (2 * C.OPT_HALF * dd.P0.iloc[0] + C.FUT_COST * (dd.gamma0 * (dd.F1 - dd.F0)).abs().sum()) * C.UNIT \
            if len(dd) else np.nan
        recs.append({"cid": cid, "E": E, "A": A, "F_A": row.F, "F_E": px.at[E, cid], "iv_A": row.atm_iv,
                     "gex_A": row.gex, "conc": conc, "oi_A": oi, "rv": rv, "vr": rv / row.atm_iv,
                     "logvr": np.log(rv / row.atm_iv), "short_gross": short, "short_net": short - cost,
                     "n_dh": len(dd)})
    ev = pd.DataFrame(recs).sort_values("E").reset_index(drop=True)
    ev["gex_q"] = C.wf_quantile(ev.gex_A)
    ev["conc_q"] = C.wf_quantile(ev.conc)
    # pin test from raw strike-level OI at A
    o = raw_oi(ev.cid.tolist(), ev.A.tolist())
    o["K"] = o.k_raw / 1000
    g = o.groupby(["date", "cid", "K"]).oi.sum().reset_index()
    kstar = {}
    for (dA, cid), gg in g.groupby(["date", "cid"]):
        F = ev.loc[(ev.A == dA) & (ev.cid == cid), "F_A"].iloc[0]
        gg = gg[(gg.K / F > 0.85) & (gg.K / F < 1.15)]
        if gg.oi.sum() > 0:
            kstar[(dA, cid)] = gg.loc[gg.oi.idxmax(), "K"]
    ev["Kstar"] = [kstar.get((a, c), np.nan) for a, c in zip(ev.A, ev.cid)]
    ev["pin_hit"] = (ev.F_E - ev.Kstar).abs() <= 0.025
    for dlt in (0.10, 0.15):
        ev[f"plac_{dlt}"] = (((ev.F_E - ev.Kstar - dlt).abs() <= 0.025).astype(float) +
                             ((ev.F_E - ev.Kstar + dlt).abs() <= 0.025).astype(float)) / 2
    ev["dist_A"] = (ev.F_A - ev.Kstar).abs()
    ev["dist_E"] = (ev.F_E - ev.Kstar).abs()
    ev.to_csv(C.CACHE / "part3_expiry_events.csv", index=False)
    # summaries
    rows = []
    for wn, w in (("choose", C.CHOOSE), ("check", C.CHECK), ("all", (C.CHOOSE[0], C.CHECK[1]))):
        e = ev[(ev.E >= w[0]) & (ev.E <= w[1])]
        for lab, sel in (("all", e.gex_q.notna()), ("gex top", e.gex_q >= 2 / 3), ("gex bottom", e.gex_q < 1 / 3),
                         ("conc top", e.conc_q >= 2 / 3), ("conc bottom", e.conc_q < 1 / 3)):
            x = e[sel]
            rows.append({"window": wn, "group": lab, "n": len(x), "mean_logvr": x.logvr.mean(),
                         "median_vr": x.vr.median(), "pin_hit": x.pin_hit.mean(),
                         "placebo_0.10": x["plac_0.1"].mean(), "placebo_0.15": x["plac_0.15"].mean(),
                         "short_net_mean": x.short_net.mean(), "short_net_sd": x.short_net.std()})
    summ = pd.DataFrame(rows)
    # regression logvr on gex_q (rank) - expiry-level, HAC lag 1
    reg = {}
    for wn, w in (("choose", C.CHOOSE), ("check", C.CHECK)):
        e = ev[(ev.E >= w[0]) & (ev.E <= w[1])]
        b, t, n = C.hac_ols(e.logvr.values, e[["gex_q"]].values, 1)
        b2, t2, n2 = C.hac_ols(e.logvr.values, e[["conc_q"]].values, 1)
        reg[wn] = {"b_gexq": b[1], "t_gexq": t[1], "b_concq": b2[1], "t_concq": t2[1], "n": n}
    reg = pd.DataFrame(reg).T
    summ.to_csv(C.CACHE / "part3_summary.csv", index=False)
    reg.to_csv(C.CACHE / "part3_reg.csv")
    return ev, summ, reg


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    ev, summ, reg = main()
    print(ev.describe().T.round(3).to_string())
    print(summ.round(3).to_string())
    print(reg.round(3).to_string())
    from scipy.stats import binomtest
    e = ev.dropna(subset=["Kstar"])
    print("pin hits", e.pin_hit.sum(), "of", len(e), "placebo rate", e["plac_0.1"].mean(), e["plac_0.15"].mean())
    print(binomtest(int(e.pin_hit.sum()), len(e), max(e["plac_0.1"].mean(), 1e-3), alternative="greater"))
