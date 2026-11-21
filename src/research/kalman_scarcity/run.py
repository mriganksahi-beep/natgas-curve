"""Kalman scarcity study: fit, filter, benchmarks, evaluation (a)-(c), Markov switching, figures.

    cd /home/claude/natgas-curve && python -m src.research.kalman_scarcity.run

Everything uses data <= 2021-12-31 (cut inside data.py loaders).
Estimation window: target winters 2010-11 .. 2016-17 (weeks up to the Feb-2017 roll).
Check window: target winters 2017-18 .. 2021-22 (weeks Mar 2017 .. Dec 2021), parameters frozen.
"""
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .data import CACHE, NAMES, ROOT, build_panel
from .kf import OneFactorKF

warnings.filterwarnings("ignore")
FIG = ROOT / "reports/agents/figures"
OUT = CACHE
EST_LAST_WINTER = 2017
RNG = np.random.default_rng(20261121)
NBOOT = 2000
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85"
# The three Kalman specifications tried (all reported). rw is primary (fixed before results).
# rw_floor: each measurement-noise sd >= 1 cent, added after the rw fit put FebApr noise at ~0.
SPECS = {"rw": (False, 0.0), "ar1": (True, 0.0), "rw_floor": (False, 1e-4)}
EPISODES = {"Jan-Feb 2014 polar vortex": ("2014-01-03", "2014-03-07"),
            "Nov 2018 squeeze": ("2018-11-02", "2018-12-07"),
            "Feb 2021 Uri": ("2021-02-05", "2021-02-26")}


# ---------------------------------------------------------------- model fitting
def fit_models(P, force=False):
    f = OUT / "kf_params.json"
    res = json.loads(f.read_text()) if (f.exists() and not force) else {}
    Y = P[NAMES].values
    T = P[["tau_" + n for n in NAMES]].values
    roll = P["roll"].values
    est = (P["winter"] <= EST_LAST_WINTER).values
    for name, (ar, hf) in SPECS.items():
        if name in res:
            continue
        m = OneFactorKF(ar1=ar, h_floor=hf)
        r = m.fit(Y[est], T[est], roll[est])
        res[name] = {"theta": list(map(float, m.theta)), "nll": float(r.fun), "k": len(m.theta)}
        f.write_text(json.dumps(res, indent=1))
    return res


def run_filters(P, params):
    Y = P[NAMES].values
    T = P[["tau_" + n for n in NAMES]].values
    roll = P["roll"].values
    out = {}
    for name, (ar, hf) in SPECS.items():
        m = OneFactorKF(ar1=ar, h_floor=hf)
        out[name] = m.filter(np.array(params[name]["theta"]), Y, T, roll)
    return out


def pc1(P):
    """First PC of standardised spreads, loadings from complete estimation-window weeks.
    With missing legs the score is the least-squares projection on the live loadings."""
    est = (P["winter"] <= EST_LAST_WINTER).values
    X = P[NAMES].values
    mu, sd = np.nanmean(X[est], 0), np.nanstd(X[est], 0)
    Z = (X - mu) / sd
    full = est & np.isfinite(Z).all(1)
    w, V = np.linalg.eigh(np.cov(Z[full].T))
    v = V[:, -1] * np.sign(V[NAMES.index("MarApr"), -1])
    share = w[-1] / w.sum()
    sc = np.full(len(P), np.nan)
    for t in range(len(P)):
        m = np.isfinite(Z[t])
        if m.any():
            sc[t] = Z[t, m] @ v[m] / (v[m] @ v[m])
    return sc, v, share


def changes(x, P):
    d = pd.Series(x, index=P.index).diff()
    d[P["roll"].values] = np.nan
    return d


# ---------------------------------------------------------------- statistics helpers
def winter_boot(df, stat, groups="winter", n=NBOOT):
    g = df[groups].values
    ug = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in ug}
    pt = stat(df)
    bs = []
    for _ in range(n):
        pick = RNG.choice(ug, len(ug), replace=True)
        rows = np.concatenate([idx[u] for u in pick])
        try:
            bs.append(stat(df.iloc[rows]))
        except Exception:
            pass
    bs = np.array(bs, float)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
    return pt, lo, hi


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[m], b[m])[0, 1]


def partial_corr(df, x, y, z):
    d = df[[x, y, z]].dropna()
    rx = d[x] - np.polyval(np.polyfit(d[z], d[x], 1), d[z])
    ry = d[y] - np.polyval(np.polyfit(d[z], d[y], 1), d[z])
    return np.corrcoef(rx, ry)[0, 1]


def hac_ols(y, X, lags):
    d = pd.concat([y, X], axis=1).dropna()
    r = sm.OLS(d.iloc[:, 0], sm.add_constant(d.iloc[:, 1:])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return r


# ---------------------------------------------------------------- main
def main(force=False):
    P = build_panel()
    params = fit_models(P, force=force)
    F = run_filters(P, params)
    P["s_rw"], P["s_ar1"], P["s_rwf"] = F["rw"]["s"], F["ar1"]["s"], F["rw_floor"]["s"]
    P["sd_rw"], P["sd_rwf"] = np.sqrt(F["rw"]["P"]), np.sqrt(F["rw_floor"]["P"])
    P["pc1"], v_pc, pc_share = pc1(P)
    sig = {"KF state (RW)": "s_rw", "KF state (AR1)": "s_ar1", "KF state (RW, noise floor)": "s_rwf", "Mar-Apr spread": "MarApr", "PC1": "pc1"}
    for k, c in sig.items():
        P["d_" + c] = changes(P[c].values, P)
    P["est"] = P["winter"] <= EST_LAST_WINTER
    P["horizon_wk"] = (pd.to_datetime(P["winter"].astype(str) + "-03-31") - P["date"]).dt.days / 7
    P["season"] = P["date"].dt.month.isin([11, 12, 1, 2, 3])
    P.to_pickle(OUT / "panel_with_signals.pkl")
    R = {"params": {k: {kk: (np.round(vv, 5).tolist() if np.ndim(vv) else round(float(vv), 5))
                        for kk, vv in F[k]["params"].items()} for k in F},
         "nll": {k: params[k]["nll"] for k in params},
         "pc1_loadings": dict(zip(NAMES, np.round(v_pc, 3).tolist())), "pc1_share": round(pc_share, 3)}
    # LR test AR1 vs RW on the estimation window
    R["LR_ar1_vs_rw"] = 2 * (params["rw"]["nll"] - params["ar1"]["nll"])

    # ---- stability / noise diagnostics (all weeks)
    stab = []
    for k, c in sig.items():
        x, d = P[c], P["d_" + c]
        z = (x - x[P.est].mean()) / x[P.est].std()
        dz = d / x[P.est].std()
        stab.append({"signal": k, "sd_weekly_change_over_sd_level": round(dz.std(), 3),
                     "ac1_weekly_change": round(d.autocorr(1), 3),
                     "corr_with_MarApr_level": round(corr(x.values, P["MarApr"].values), 3)})
    R["stability"] = stab

    # ---- (a) eventual end-March surplus (ex post), by horizon bucket
    buckets = [("5-8 wk (Feb)", 4.5, 8.5), ("9-17 wk (Dec-Jan)", 8.5, 17.5), ("18-26 wk (Oct-Nov)", 17.5, 26.5),
               ("27-56 wk (Mar-Sep)", 26.5, 60)]
    A = P[P["endmar_surplus"].notna()].copy()
    rows_a = []
    for bname, lo, hi in buckets:
        B = A[(A.horizon_wk > lo) & (A.horizon_wk <= hi)]
        for k, c in sig.items():
            for sample, S in (("all 2012-2021", B), ("est 2012-2017", B[B.est]), ("check 2018-2021", B[~B.est])):
                if S["winter"].nunique() < 3:
                    continue
                pt, l, h = winter_boot(S, lambda d: corr(d[c].values, d["endmar_surplus"].values))
                rows_a.append({"bucket": bname, "signal": k, "sample": sample, "n_winters": S.winter.nunique(),
                               "n": len(S), "corr": pt, "lo": l, "hi": h})
            # incremental over storage known at t
            pt, l, h = winter_boot(B, lambda d: partial_corr(d, c, "endmar_surplus", "surplus_known"))
            rows_a.append({"bucket": bname, "signal": k, "sample": "all, partial | storage known", "n_winters":
                           B.winter.nunique(), "n": len(B), "corr": pt, "lo": l, "hi": h})
        # storage-known benchmark itself
        pt, l, h = winter_boot(B, lambda d: corr(d["surplus_known"].values, d["endmar_surplus"].values))
        rows_a.append({"bucket": bname, "signal": "storage surplus known at t", "sample": "all 2012-2021",
                       "n_winters": B.winter.nunique(), "n": len(B), "corr": pt, "lo": l, "hi": h})
    # paired difference KF(RW) - MarApr, all, by bucket
    for bname, lo, hi in buckets:
        B = A[(A.horizon_wk > lo) & (A.horizon_wk <= hi)]
        for a_, b_ in (("s_rw", "MarApr"), ("s_rw", "pc1"), ("s_rwf", "MarApr")):
            pt, l, h = winter_boot(B, lambda d: corr(d[a_].values, d["endmar_surplus"].values)
                                   - corr(d[b_].values, d["endmar_surplus"].values))
            rows_a.append({"bucket": bname, "signal": f"DIFF {a_} - {b_}", "sample": "all 2012-2021",
                           "n_winters": B.winter.nunique(), "n": len(B), "corr": pt, "lo": l, "hi": h})
    TA = pd.DataFrame(rows_a)
    TA.to_csv(OUT / "eval_a.csv", index=False)

    # ---- (b) levels vs storage known; changes vs surprises (lead/lag)
    rows_b = []
    Bv = P[P["surplus_known"].notna()]
    for k, c in sig.items():
        for sample, S in (("all", Bv), ("Nov-Mar", Bv[Bv.season])):
            pt, l, h = winter_boot(S, lambda d: corr(d[c].values, d["surplus_known"].values))
            rows_b.append({"what": "level vs storage surplus known", "signal": k, "sample": sample, "lag": 0,
                           "corr": pt, "lo": l, "hi": h, "n": len(S)})
    for a_, b_ in (("s_rw", "MarApr"), ("s_rw", "pc1")):
        for sample, S in (("all", Bv), ("Nov-Mar", Bv[Bv.season])):
            pt, l, h = winter_boot(S, lambda d: corr(d[a_].values, d["surplus_known"].values)
                                   - corr(d[b_].values, d["surplus_known"].values))
            rows_b.append({"what": "level vs storage surplus known", "signal": f"DIFF {a_} - {b_}",
                           "sample": sample, "lag": 0, "corr": pt, "lo": l, "hi": h, "n": len(S)})
    # changes: corr(dX_t, surprise published at week t+k); k=+1 is the surprise for the week ending t
    for k, c in sig.items():
        for lag in (-2, -1, 0, 1, 2, 3, 4, 5):
            D = pd.DataFrame({"d": P["d_" + c], "sur": P["surprise_pub"].shift(-lag), "winter": P["winter"],
                              "season": P["season"]}).dropna()
            for sample, S in (("all", D), ("Nov-Mar", D[D.season])):
                pt, l, h = winter_boot(S, lambda d: corr(d["d"].values, d["sur"].values))
                rows_b.append({"what": "dX_t vs surprise published t+lag", "signal": k, "sample": sample,
                               "lag": lag, "corr": pt, "lo": l, "hi": h, "n": len(S)})
    TB = pd.DataFrame(rows_b)
    TB.to_csv(OUT / "eval_b.csv", index=False)
    # does dX_t carry info about next week's surprise beyond this week's weather? (Nov-Mar, HAC 4)
    lead = []
    for k, c in sig.items():
        D = pd.DataFrame({"next_sur": P["surprise_pub"].shift(-1), "hdd_anom": P["hdd_anom"],
                          "dx": P["d_" + c] / P[c][P.est].std()})[P.season]
        for sample, S in (("all", D), ("est", D[P.est[P.season]]), ("check", D[~P.est[P.season]])):
            r = hac_ols(S["next_sur"], S[["hdd_anom", "dx"]], 4)
            lead.append({"signal": k, "sample": sample, "coef_dx_bcf_per_sd": r.params["dx"],
                         "se": r.bse["dx"], "t": r.tvalues["dx"], "coef_hdd": r.params["hdd_anom"],
                         "r2": r.rsquared, "n": int(r.nobs)})
        # surprise published at t+3 (= storage week ending t+14) on dX_t, with and without
        # the realised (ex post) HDD anomalies of weeks t+1, t+2: is the lead weather-forecast pricing?
        D = pd.DataFrame({"sur3": P["surprise_pub"].shift(-3), "h1": P["hdd_anom"].shift(-1),
                          "h2": P["hdd_anom"].shift(-2), "h0": P["hdd_anom"],
                          "dx": P["d_" + c] / P[c][P.est].std()})[P.season]
        for ctrl, cols in (("no weather", ["dx"]), ("+ realised HDD t..t+2", ["h0", "h1", "h2", "dx"])):
            r = hac_ols(D["sur3"], D[cols], 4)
            lead.append({"signal": k, "sample": "lead3 " + ctrl, "coef_dx_bcf_per_sd": r.params["dx"],
                         "se": r.bse["dx"], "t": r.tvalues["dx"], "coef_hdd": np.nan,
                         "r2": r.rsquared, "n": int(r.nobs)})
    TL = pd.DataFrame(lead)
    TL.to_csv(OUT / "eval_b_lead.csv", index=False)

    # ---- (c) unexplained jumps and reversal
    rows_c = []
    W = P["season"].astype(float)
    Xexp = pd.DataFrame({"sur": P["surprise_pub"], "hdd": P["hdd_anom"],
                         "sur_w": P["surprise_pub"] * W, "hdd_w": P["hdd_anom"] * W})
    for k, c in sig.items():
        d = P["d_" + c]
        e = P.est & d.notna() & Xexp.notna().all(1)
        r = sm.OLS(d[e], sm.add_constant(Xexp[e])).fit()
        u = d - r.predict(sm.add_constant(Xexp))
        thr = u[e].abs().quantile(0.9)
        jump = (u.abs() > thr).astype(float)
        for hz in (1, 2, 3, 4):
            fwd = pd.Series(P[c].values, index=P.index).shift(-hz) - P[c]
            same = P["winter"].shift(-hz) == P["winter"]
            fwd[~same] = np.nan
            fwd_ma = (P["MarApr"].shift(-hz) - P["MarApr"]).where(same)
            for sample, msk in (("est", P.est), ("check", ~P.est), ("all", P.est | ~P.est)):
                D = pd.DataFrame({"fwd": fwd, "fwd_ma": fwd_ma, "u": u, "uj": u * jump, "winter": P.winter})[msk]
                D = D.dropna()
                r1 = hac_ols(D["fwd"], D[["u"]], hz + 2)
                r2 = hac_ols(D["fwd"], D[["uj"]], hz + 2)
                r3 = hac_ols(D["fwd_ma"], D[["uj"]], hz + 2)
                bpt, blo, bhi = winter_boot(D, lambda x: np.cov(x["uj"], x["fwd"])[0, 1] / np.var(x["uj"], ddof=1)
                                            if (x["uj"] != 0).sum() >= 3 else np.nan)
                rows_c.append({"signal": k, "h": hz, "sample": sample, "n": len(D), "n_jumps": int((D.uj != 0).sum()),
                               "beta_all_u": r1.params["u"], "t_all_u": r1.tvalues["u"],
                               "beta_jumps": r2.params["uj"], "t_jumps": r2.tvalues["uj"],
                               "beta_jumps_boot_lo": blo, "beta_jumps_boot_hi": bhi,
                               "beta_MarApr_on_jumps": r3.params["uj"], "t_MarApr_on_jumps": r3.tvalues["uj"],
                               "jump_threshold": thr, "expl_r2_est": r.rsquared})
    TC = pd.DataFrame(rows_c)
    TC.to_csv(OUT / "eval_c.csv", index=False)

    # ---- (4) Markov switching on weekly changes of the RW state
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    ds = P["d_s_rw"].fillna(0.0)
    est_mask = P.est.values
    ms_est = MarkovRegression(ds[est_mask].values, k_regimes=2, trend="c", switching_trend=False,
                              switching_variance=True)
    best = None
    for seed in range(5):
        try:
            rr = ms_est.fit(search_reps=20, disp=False) if seed == 0 else ms_est.fit(
                start_params=ms_est.start_params * (1 + 0.2 * RNG.standard_normal(len(ms_est.start_params))),
                disp=False)
            if best is None or rr.llf > best.llf:
                best = rr
        except Exception:
            pass
    ms_full = MarkovRegression(ds.values, k_regimes=2, trend="c", switching_trend=False, switching_variance=True)
    fr = ms_full.filter(best.params)
    sig2 = best.params[-2:]
    stressed = int(np.argmax(sig2))
    P["p_stress"] = fr.filtered_marginal_probabilities[:, stressed]
    R["markov"] = {"params": dict(zip(ms_est.param_names, np.round(best.params, 6).tolist())),
                   "stressed_regime": stressed, "sd_calm": float(np.sqrt(sig2.min())),
                   "sd_stressed": float(np.sqrt(sig2.max())),
                   "share_weeks_p>0.5_est": float((P.p_stress[P.est] > 0.5).mean()),
                   "share_weeks_p>0.5_check": float((P.p_stress[~P.est] > 0.5).mean())}
    ep = {}
    for name, (a, b) in EPISODES.items():
        a0, b0 = pd.Timestamp(a) - pd.Timedelta(days=21), pd.Timestamp(b) + pd.Timedelta(days=21)
        S = P.loc[a0:b0, ["p_stress", "d_s_rw", "s_rw", "MarApr"]]
        ep[name] = {str(i.date()): [round(x, 3) for x in r] for i, r in S.iterrows()}
    R["markov_episodes"] = ep
    # does the stressed regime mark weeks where jumps reverse? (est vs check), h=4, KF
    P.to_pickle(OUT / "panel_with_signals.pkl")
    (OUT / "results.json").write_text(json.dumps(R, indent=1, default=float))
    figures(P, TA)
    return P, R, TA, TB, TL, TC


def figures(P, TA):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": GREY, "axes.labelcolor": "#333", "xtick.color": "#555",
                         "ytick.color": "#555"})
    # Fig 1: filtered state vs Mar-Apr
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    x = P["date"]
    s, sd = P["s_rw"], P["sd_rw"]
    ax[0].fill_between(x, s - 1.64 * sd, s + 1.64 * sd, color=BLUE, alpha=0.15, lw=0)
    ax[0].plot(x, P["s_rwf"], color=AQUA, lw=2.4, alpha=0.6, label="Variant: noise floor 1 cent")
    ax[0].plot(x, s, color=BLUE, lw=1.2, label="Filtered scarcity s_t (RW, $/MMBtu Mar-Apr equivalent at delivery)")
    ax[0].plot(x, P["MarApr"], color=ORANGE, lw=1.2, label="Mar-Apr spread of the target winter ($/MMBtu)")
    for r in P.index[P.roll]:
        ax[0].axvline(P.at[r, "date"], color=GREY, lw=0.5, ls=":")
    ax[0].axvline(pd.Timestamp("2017-02-24"), color="#333", lw=1)
    ax[0].text(pd.Timestamp("2017-03-10"), ax[0].get_ylim()[1] * 0.92, "parameters frozen ->", fontsize=8)
    for name, (a, b) in EPISODES.items():
        ax[0].axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#d0d0cc", alpha=0.5, lw=0)
        ax[0].text(pd.Timestamp(a), ax[0].get_ylim()[1] * 0.8, name, fontsize=7.5, color="#333")
    ax[0].axhline(0, color=GREY, lw=0.6)
    ax[0].set_ylabel("$/MMBtu")
    ax[0].legend(loc="upper left", frameon=False, fontsize=8)
    ax[0].set_title("Filtered market-expected scarcity vs the March-April spread, 2010-2021 "
                    "(dotted = target-winter roll)", fontsize=9.5, loc="left")
    ax[1].plot(x, P["p_stress"], color=AQUA, lw=1.2)
    ax[1].set_ylabel("P(stressed)")
    ax[1].set_ylim(0, 1)
    ax[1].set_title("Markov-switching filtered probability of the high-variance regime (weekly changes in s_t)",
                    fontsize=9, loc="left")
    for name, (a, b) in EPISODES.items():
        ax[1].axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#d0d0cc", alpha=0.5, lw=0)
    fig.tight_layout()
    fig.savefig(FIG / "kalman_scarcity_state.png", dpi=150)
    plt.close(fig)

    # Fig 2: (a) correlations with end-March surplus by horizon, all winters
    T = TA[(TA["sample"] == "all 2012-2021") & TA.signal.isin(["KF state (RW)", "Mar-Apr spread", "PC1",
                                                                  "storage surplus known at t"])]
    bk = list(dict.fromkeys(T.bucket))
    sigs = ["KF state (RW)", "Mar-Apr spread", "PC1", "storage surplus known at t"]
    cols = [BLUE, ORANGE, AQUA, GREY]
    fig, ax = plt.subplots(figsize=(9, 4))
    wdt = 0.2
    for j, (sname, col) in enumerate(zip(sigs, cols)):
        S = T[T.signal == sname].set_index("bucket").reindex(bk)
        xs = np.arange(len(bk)) + (j - 1.5) * wdt
        val = S["corr"].values if sname != "storage surplus known at t" else -S["corr"].values
        lo = S["lo"].values if sname != "storage surplus known at t" else -S["hi"].values
        hi = S["hi"].values if sname != "storage surplus known at t" else -S["lo"].values
        ax.bar(xs, val, width=wdt - 0.03, color=col, label=sname + (" (sign flipped)" if "storage" in sname else ""))
        ax.errorbar(xs, val, yerr=[val - lo, hi - val], fmt="none", ecolor="#333", lw=0.8, capsize=2)
    ax.axhline(0, color=GREY, lw=0.6)
    ax.set_xticks(np.arange(len(bk)))
    ax.set_xticklabels(bk, fontsize=8)
    ax.set_xlabel("weeks from observation to end-March of the target winter")
    ax.set_ylabel("correlation with end-March\nsurplus vs prior-yrs avg")
    ax.set_title("What the curve knew: correlation with the eventual end-March storage surplus "
                 "(winters 2011-12 to 2020-21, 95% winter-bootstrap CI)", fontsize=9, loc="left")
    ax.set_ylim(-1.05, 0.75)
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=4)
    fig.tight_layout()
    fig.savefig(FIG / "kalman_scarcity_endmarch.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    if "--figs" in sys.argv:
        figures(pd.read_pickle(OUT / "panel_with_signals.pkl"), pd.read_csv(OUT / "eval_a.csv"))
        sys.exit()
    P, R, TA, TB, TL, TC = main(force="--refit" in sys.argv)
    pd.set_option("display.width", 220, "display.max_rows", 400, "display.max_columns", 30)
    print(json.dumps({k: R[k] for k in ("params", "nll", "LR_ar1_vs_rw", "pc1_loadings", "pc1_share",
                                        "stability", "markov")}, indent=1, default=float))
    print(TA.round(3).to_string())
    print(TB.round(3).to_string())
    print(TL.round(3).to_string())
    print(TC.round(3).to_string())
    print(json.dumps(R["markov_episodes"], indent=0))
