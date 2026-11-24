"""Step 3: tradable correlation trade, held to the CSO expiry, delta-hedged daily with futures.

Legs (per 1 CSO straddle, $/MMBtu; x 10,000 = $ per contract):
  CSO : SHORT 1 G4X straddle at the listed strike nearest the futures spread S (Bachelier, nv at that strike),
        held to expiry, settled at intrinsic |S_T - K| (CSO and ON options share the expiry day).
  NEAR: LONG w_n ON straddles on the near leg at the strike nearest F_n, held to the same expiry (intrinsic).
  FAR : LONG w_f ON straddles on the far leg at the strike nearest F_f, CLOSED at the near expiry at its settle
        (raw settle at that strike; Black with that day's ATM vol if the strike is missing).
Weights:
  G (primary, chosen a priori): dollar-gamma matched at entry, w = Gamma_CSO / Gamma_leg. Then
      P&L ~ Gamma (sum dFn dFf - rho_i sn sf dt): a covariance swap = the textbook correlation trade.
  V (variant): rho-neutral vega hedge, w_n = Vega_S * d nv/d sn / Vega_n(normal), same for far
      (d nv/d sn = (sn - rho sf)/nv). Holds rho fixed; exposed only to rho (to first order).
Delta hedge: each option's delta at its ENTRY implied vol (no marks needed), rebalanced at every settle;
  futures cost 2 ticks ($0.002) per unit traded per leg, incl. the final unwind.
Option costs: CSO 4% of premium per side (3/5/8% sensitivity), ON max(1.5% x price, $0.002) per option per side.
Benchmarks: A = short CSO straddle alone (delta-hedged); B = short leg straddles alone (G weights);
  C = correlation trade = A + long legs (G); D = same with V weights.
"""
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import norm

from src.v5.options_panel import bach_iv_straddle, bach_straddle, black_price, _straddle_iv

from .common import CACHE, CSO_HS, CUT, ON_HS, ON_MIN, TICK2, futures, raw_options, spreads, surface

ENTRY_T = (21, 42, 63, 105)


def _straddle_tab(o):
    st = o.pivot_table(index=["root", "cid", "date", "K"], columns="cp", values="settle").dropna()
    s = (st["C"] + st["P"]).rename("P").sort_index()
    return s, None


def black_straddle(F, K, T, s):
    c = black_price(np.atleast_1d(F), np.atleast_1d(K), np.atleast_1d(T), np.atleast_1d(s), np.array(["C"]))
    return float((2 * c - (F - K))[0])


def black_delta(F, K, T, s):
    v = s * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * v * v) / v
    return 2 * ndtr(d1) - 1


def black_gamma(F, K, T, s):
    v = s * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * v * v) / v
    return 2 * norm.pdf(d1) / (F * v)


def black_vega(F, K, T, s):
    v = s * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * v * v) / v
    return 2 * F * norm.pdf(d1) * np.sqrt(T)


def bach_delta(S, K, T, nv):
    return 2 * ndtr((S - K) / (nv * np.sqrt(T))) - 1


def build_trades(cso_hs=CSO_HS):
    p = CACHE / f"trades_{int(cso_hs * 1000)}.pkl"
    if p.exists():
        return pd.read_pickle(p)
    px, days, oexp = futures()
    sp = spreads()
    sp = sp[~sp.stale]
    sf = surface().set_index(["date", "cid"]).atm_iv
    o = raw_options()
    o = o[o.date >= "2012-01-01"]
    P, _ = _straddle_tab(o)
    del o
    pos = pd.Series(np.arange(len(days)), index=days)
    trades, daily = [], []
    for cid, g in sp.groupby("cid"):
        far = int(g.far.iat[0])
        en = oexp.get(cid)
        if en is None or en > CUT or cid not in px.columns or far not in px.columns:
            continue
        iE = days.searchsorted(en)
        for E in ENTRY_T:
            cand = g[(g.tdays <= E) & (g.tdays >= E - 5)].sort_values("date")
            done = False
            for d in cand.date:
                i0 = pos[d]
                Fn, Ff = px.at[d, cid], px.at[d, far]
                S = Fn - Ff
                try:
                    cs = P.loc[("G4X", cid, d)]
                    ns = P.loc[("ON", cid, d)]
                    fs = P.loc[("ON", far, d)]
                except KeyError:
                    continue
                Ks = cs.index.values[np.argmin(np.abs(cs.index.values - S))]
                Kn = ns.index.values[np.argmin(np.abs(ns.index.values - Fn))]
                Kf = fs.index.values[np.argmin(np.abs(fs.index.values - Ff))]
                Tn = (iE - i0) / 252.0
                ef = oexp[far]
                Tf = Tn + np.busday_count(en.date(), ef.date()) / 252.0
                PS, PN, PF = cs.loc[Ks], ns.loc[Kn], fs.loc[Kf]
                nv0 = bach_iv_straddle(np.array([PS]), S, Ks, Tn)[0]
                ivn = _straddle_iv(np.array([PN]), np.array([Fn]), np.array([Kn]), np.array([Tn]))[0]
                ivf = _straddle_iv(np.array([PF]), np.array([Ff]), np.array([Kf]), np.array([Tf]))[0]
                if not np.isfinite(nv0 * ivn * ivf) or abs(Ks - S) > 0.1:
                    continue
                done = True
                break
            if not done:
                continue
            # weights
            GS = 2 * norm.pdf((S - Ks) / (nv0 * np.sqrt(Tn))) / (nv0 * np.sqrt(Tn))
            Gn, Gf = black_gamma(Fn, Kn, Tn, ivn), black_gamma(Ff, Kf, Tf, ivf)
            wG_n, wG_f = GS / Gn, GS / Gf
            sn, sfv = Fn * ivn, Ff * ivf
            rho_i = (sn ** 2 + sfv ** 2 - nv0 ** 2) / (2 * sn * sfv)
            VS = 2 * norm.pdf((S - Ks) / (nv0 * np.sqrt(Tn))) * np.sqrt(Tn)       # per normal vol
            Vn, Vf = black_vega(Fn, Kn, Tn, ivn) / Fn, black_vega(Ff, Kf, Tf, ivf) / Ff
            wV_n = VS * (sn - rho_i * sfv) / nv0 / Vn
            wV_f = VS * (sfv - rho_i * sn) / nv0 / Vf
            # daily path
            idx = days[i0:iE + 1]
            fn, ff = px[cid].reindex(idx).ffill().values, px[far].reindex(idx).ffill().values
            s_ = fn - ff
            tau = np.maximum((iE - np.arange(i0, iE + 1)) / 252.0, 1e-6)
            tauf = tau + (Tf - Tn)
            dS = bach_delta(s_, Ks, tau, nv0)
            dN = black_delta(fn, Kn, tau, ivn)
            dFr = black_delta(ff, Kf, tauf, ivf)
            dS[-1], dN[-1] = 0.0, 0.0                       # expired: hedge unwound
            # marks (for drawdowns): raw settle at the strike, else model at today's vol
            mS, mN, mF = np.empty(len(idx)), np.empty(len(idx)), np.empty(len(idx))
            for j, dd in enumerate(idx):
                v = P.get(("G4X", cid, dd, Ks), np.nan)
                if not np.isfinite(v):
                    nvt = g.nv[g.date == dd]
                    v = bach_straddle(s_[j], Ks, tau[j], nvt.iat[0] if len(nvt) else nv0)
                mS[j] = v
                v = P.get(("ON", cid, dd, Kn), np.nan)
                if not np.isfinite(v):
                    a = sf.get((dd, cid), np.nan)
                    v = black_straddle(fn[j], Kn, tau[j], a if np.isfinite(a) else ivn)
                mN[j] = v
                v = P.get(("ON", far, dd, Kf), np.nan)
                if not np.isfinite(v):
                    a = sf.get((dd, far), np.nan)
                    v = black_straddle(ff[j], Kf, tauf[j], a if np.isfinite(a) else ivf)
                mF[j] = v
            mS[-1], mN[-1] = abs(s_[-1] - Ks), abs(fn[-1] - Kn)       # expiry intrinsic
            dfn, dff = np.diff(fn), np.diff(ff)

            def leg_path(m, dvec, dF_list, sign, band=0.0, unit_cost=TICK2):
                """daily P&L of 1 (sign) straddle, delta hedged with optional no-trade band (in delta units).
                dF_list: (dF, k) per futures leg; the hedge is one 'unit' traded as k-weighted legs;
                unit_cost: cost per unit of hedge traded (per-leg ticks summed, or one listed spread)."""
                opt = sign * np.diff(m)
                tgt = -sign * dvec
                h = np.empty(len(tgt))
                cur = 0.0
                for j, x in enumerate(tgt):
                    if j == len(tgt) - 1 or abs(x - cur) > band or j == 0:
                        cur = x
                    h[j] = cur
                h[-1] = 0.0
                hedge = np.zeros(len(opt))
                for dF, k in dF_list:
                    hedge += k * h[:-1] * dF
                cost = unit_cost * np.abs(np.diff(np.r_[0.0, h[:-1], 0.0])).sum()
                return opt + hedge, cost

            # CSO: delta in S -> near +, far - ; short => sign -1
            dFh = dFr.copy(); dFh[-1] = 0.0
            HEDGE = {"daily_leg": (0.0, 2 * TICK2, TICK2), "daily_spread": (0.0, TICK2, TICK2),
                     "band": (0.2, TICK2, TICK2)}
            res = {}
            for hm, (bd, ucS, ucL) in HEDGE.items():
                a = leg_path(mS, dS, [(dfn, 1.0), (dff, -1.0)], -1, bd, ucS)
                b = leg_path(mN, dN, [(dfn, 1.0)], +1, bd, ucL)
                c = leg_path(mF, dFh, [(dff, 1.0)], +1, bd, ucL)
                res[hm] = (a, b, c)
            (pS, cS), (pN1, cN1), (pF1, cF1) = res["daily_leg"]
            # option transaction costs (entry; exit for far leg; CSO / near settle at intrinsic)
            oc_S = cso_hs * PS
            Cn0, Cf0 = black_price(np.array([Fn]), np.array([Kn]), np.array([Tn]), np.array([ivn]), np.array(["C"]))[0], \
                black_price(np.array([Ff]), np.array([Kf]), np.array([Tf]), np.array([ivf]), np.array(["C"]))[0]
            Pn0, Pf0 = Cn0 - (Fn - Kn), Cf0 - (Ff - Kf)
            oc_N = max(ON_HS * Cn0, ON_MIN) + max(ON_HS * Pn0, ON_MIN)
            oc_F = 2 * (max(ON_HS * Cf0, ON_MIN) + max(ON_HS * Pf0, ON_MIN))        # in and out (approx.)
            ex_S = TICK2 * 2 * (abs(s_[-1] - Ks) > 0)                                  # exercise offset
            ex_N = TICK2 * (abs(fn[-1] - Kn) > 0)
            rec = {"cid": cid, "cm": cid % 100, "far": far, "E": E, "entry": d, "expiry": en, "tdays": iE - i0,
                   "S0": S, "Ks": Ks, "PS": PS, "PN": PN, "PF": PF, "Fn0": Fn, "Ff0": Ff, "nv0": nv0, "ivn": ivn, "ivf": ivf, "rho_i0": rho_i,
                   "wG_n": wG_n, "wG_f": wG_f, "wV_n": wV_n, "wV_f": wV_f,
                   "S_T": s_[-1], "far_exit_raw": np.isfinite(P.get(("ON", far, idx[-1], Kf), np.nan)),
                   "cso_raw_share": np.mean([np.isfinite(P.get(("G4X", cid, dd, Ks), np.nan)) for dd in idx[:-1]]),
                   # gross P&L per unit position
                   "gS": pS.sum(), "gN": pN1.sum(), "gF": pF1.sum(),
                   "cS": cS + oc_S + ex_S, "cN": cN1 + oc_N + ex_N, "cF": cF1 + oc_F,
                   "ocS": oc_S + ex_S, "ocN": oc_N + ex_N, "ocF": oc_F}
            for hm, ((a, ca), (b, cb), (c, cc)) in res.items():
                rec.update({f"gS_{hm}": a.sum(), f"gN_{hm}": b.sum(), f"gF_{hm}": c.sum(),
                            f"hS_{hm}": ca, f"hN_{hm}": cb, f"hF_{hm}": cc})
            # realised vs implied: covariance check
            rec["rv_S"] = np.sqrt(np.sum(np.diff(s_) ** 2) / Tn)
            trades.append(rec)
            daily.append(pd.DataFrame({"date": idx[1:], "cid": cid, "E": E, "pS": pS, "pN": pN1, "pF": pF1,
                                       "pS_band": res["band"][0][0], "pN_band": res["band"][1][0],
                                       "pF_band": res["band"][2][0]}))
    T = pd.DataFrame(trades)
    D = pd.concat(daily, ignore_index=True)
    pd.to_pickle((T, D), p)
    return T, D


def combine(T, D):
    """Strategy P&L per trade (net, $/MMBtu per CSO straddle) and daily MTM."""
    T = T.copy()
    T["A_short_cso"] = T.gS - T.cS
    T["B_short_legs"] = -(T.wG_n * T.gN + T.wG_f * T.gF) - (T.wG_n * T.cN + T.wG_f * T.cF)
    T["C_corr_G"] = T.A_short_cso + T.wG_n * (T.gN - T.cN) + T.wG_f * (T.gF - T.cF)
    T["D_corr_V"] = T.A_short_cso + T.wV_n * T.gN - abs(T.wV_n) * T.cN + T.wV_f * T.gF - abs(T.wV_f) * T.cF
    T["C_gross"] = T.gS + T.wG_n * T.gN + T.wG_f * T.gF
    w = T.set_index(["cid", "E"])[["wG_n", "wG_f", "wV_n", "wV_f"]]
    D = D.join(w, on=["cid", "E"])
    D["A_short_cso"] = D.pS
    D["B_short_legs"] = -(D.wG_n * D.pN + D.wG_f * D.pF)
    D["C_corr_G"] = D.pS + D.wG_n * D.pN + D.wG_f * D.pF
    D["D_corr_V"] = D.pS + D.wV_n * D.pN + D.wV_f * D.pF
    return T, D
