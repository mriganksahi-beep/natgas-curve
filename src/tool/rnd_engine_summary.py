"""Tables and figures for reports/tool/rnd_engine.md, from the caches written by rnd_engine_eval.

    python -m src.tool.rnd_engine_summary      # -> data/processed/research_tool_rnd/summary_*.csv, figures

Windows: choose = winters Y in {2014, 2015, 2016} (2013-14 .. 2015-16); check = Y in 2017..2022
(2016-17 .. Nov-Dec 2021). Y = 2013 (2012-13) is reported for coverage only.
Selection rule (fixed before looking at choose results): among methods with choose coverage >= 90% and
no residual arbitrage, lowest mean rank over {LOSO RMSE (OI>0 strikes), thin-lattice holdout RMSE,
thin-lattice |dP| vs dense model-free, Roll noise of P(S_T > F+0.25)}. Calibration is descriptive.
"""
import json

import matplotlib
import numpy as np
import pandas as pd

from src.tool import rnd_direct as R

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C = R.CACHE
FIG = R.ROOT / "reports/tool/figures"
FIG.mkdir(parents=True, exist_ok=True)
METHODS = ["mixture", "quad", "svi", "convex", "sln"]
LABEL = {"mixture": "Normal mixture (baseline)", "quad": "Quadratic normal vol", "svi": "Normal-SVI",
         "convex": "Model-free convex BL", "sln": "Shifted-lognormal mixture"}
COL = {"mixture": "#2a78d6", "quad": "#eb6834", "svi": "#1baf7a", "convex": "#eda100", "sln": "#e87ba4"}
CHOOSE, CHECK = [2014, 2015, 2016], [2017, 2018, 2019, 2020, 2021, 2022]


def win(Y):
    return np.where(np.isin(Y, CHOOSE), "choose", np.where(np.isin(Y, CHECK), "check", "pre"))


def load(prefix, ys=range(2013, 2023)):
    parts = [pd.read_pickle(C / f"{prefix}_{y}.pkl") for y in ys if (C / f"{prefix}_{y}.pkl").exists()]
    d = pd.concat(parts, ignore_index=True)
    d["win"] = win(d.Y.values)
    return d


def rmse(e):
    e = np.asarray(e, float)
    e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan


def stability(d):
    """Day-to-day |dP| at F+delta on consecutive quote days, both usable, |dF| < 2c; Roll noise sd."""
    out = []
    for (m, Y), g in d[d.has_quotes].sort_values("date").groupby(["method", "Y"]):
        g = g.reset_index(drop=True)
        ok = g.usable.values
        for dl in ["+0.10", "+0.25", "+0.50"]:
            P = g[f"P{dl}"].values
            dP = np.diff(P)
            dF = np.diff(g.F.values)
            good = ok[1:] & ok[:-1] & (np.abs(dF) < 0.02)
            for v, gd in zip(dP, good):
                if gd:
                    out.append({"method": m, "Y": Y, "k": dl, "dP": v})
            # Roll estimator on consecutive good changes
            x = np.where(good, dP, np.nan)
            a, b = x[1:], x[:-1]
            mk = np.isfinite(a) & np.isfinite(b)
            if mk.sum() >= 5:
                out.append({"method": m, "Y": Y, "k": dl, "roll_cov": float(np.mean(a[mk] * b[mk])), "n_roll": int(mk.sum())})
    s = pd.DataFrame(out)
    s["win"] = win(s.Y.values)
    return s


def main():
    res = {}
    daily = load("daily")
    dq = daily[daily.has_quotes]
    # ---------------- (d) coverage, (b) arbitrage, fit quality
    cov = daily.groupby(["method", "win"]).usable.mean().unstack()
    covY = daily.groupby(["method", "Y"]).usable.mean().unstack()
    arb = dq.groupby(["method", "win"]).apply(lambda g: (g.neg_mass.fillna(0) > 1e-4).mean()).unstack()
    fit = dq.groupby(["method", "win"]).rmse.median().unstack()
    arb3 = dq.groupby(["method", "win"]).apply(lambda g: (g.neg_mass.fillna(0) > 1e-3).mean()).unstack()
    lowinfo = dq.groupby("Y").low_info.mean()
    # ---------------- (a) LOSO
    lo = load("loso")
    lo["e"] = lo.pred - lo.px
    lo["re"] = lo.e / (lo.px + 0.003)
    loso = lo.groupby(["method", "win"]).e.apply(rmse).unstack()
    loso_med = lo.groupby(["method", "win"]).e.apply(lambda e: np.nanmedian(np.abs(e))).unstack()
    loso_wing = lo.groupby(["method", "win", "wing"]).e.apply(rmse).unstack([1, 2])
    loso_rel = lo.groupby(["method", "win"]).re.apply(lambda e: np.nanmedian(np.abs(e))).unstack()
    # ---------------- thin-ification (dense days)
    th = load("thin", [2015, 2016, 2017])
    thin = th.groupby("method").agg(ho_rmse=("ho_rmse", lambda x: float(np.sqrt(np.mean(x ** 2)))),
                                    dP25_self=("dP_self25", lambda x: x.abs().mean()),
                                    dP10_ref=("dP_ref10", lambda x: x.abs().mean()),
                                    dP25_ref=("dP_ref25", lambda x: x.abs().mean()),
                                    dP50_ref=("dP_ref50", lambda x: x.abs().mean()),
                                    bias25_ref=("dP_ref25", "mean"), days=("date", "nunique"))
    thin_c = th[th.Y.isin(CHOOSE)].groupby("method").agg(
        ho_rmse=("ho_rmse", lambda x: float(np.sqrt(np.mean(x ** 2)))),
        dPref=("dP_ref25", lambda x: x.abs().mean()))
    # ---------------- (c) stability
    st = stability(daily)
    stab = st.dropna(subset=["dP"]).groupby(["method", "k", "win"]).dP.apply(lambda x: x.abs().median()).unstack([1, 2])
    stab90 = st.dropna(subset=["dP"]).groupby(["method", "k", "win"]).dP.apply(lambda x: x.abs().quantile(0.9)).unstack([1, 2])
    rc = st.dropna(subset=["roll_cov"])
    roll = rc.groupby(["method", "k", "win"]).apply(
        lambda g: float(np.sqrt(max(-np.average(g.roll_cov, weights=g.n_roll), 0)))).unstack([1, 2])
    # ---------------- (e) calibration (winters with an outcome, Y <= 2021, usable days)
    ca = dq[dq.usable & dq.pit.notna()]
    pitY = ca.groupby(["method", "Y"]).pit.mean().unstack()
    cal = ca.groupby(["method", "Y"]).agg(crps=("crps", "mean"), logs=("logpdf", "mean")).reset_index()
    cal["win"] = win(cal.Y.values)
    calw = cal.groupby(["method", "win"])[["crps", "logs"]].mean().unstack()
    # PIT at fixed points in the winter (first usable day on/after Dec 1, Jan 2, Feb 1)
    pts = []
    for (m, Y), g in ca.groupby(["method", "Y"]):
        for lab, d0 in [("Dec1", pd.Timestamp(Y - 1, 12, 1)), ("Jan2", pd.Timestamp(Y, 1, 2)), ("Feb1", pd.Timestamp(Y, 2, 1))]:
            gg = g[g.date >= d0]
            if len(gg):
                pts.append({"method": m, "Y": Y, "pt": lab, "pit": gg.pit.iloc[0]})
    pts = pd.DataFrame(pts)
    pit_pts = pts.pivot_table(index=["Y", "pt"], columns="method", values="pit")
    # ---------------- zero-OI weights
    ow = load("oiw", range(2014, 2023))
    ow["e"] = ow.pred - ow.px
    oiw = ow.groupby(["method", "zw", "win"]).e.apply(rmse).unstack([2])

    # ---------------- selection on choose window
    sel = pd.DataFrame({"cov": cov["choose"], "arb": arb["choose"], "loso": loso["choose"],
                        "thin_ho": thin_c.ho_rmse, "thin_dP": thin_c.dPref,
                        "noise25": roll[("+0.25", "choose")]})
    elig = (sel["cov"] >= 0.9) & (sel["arb"] <= 0.01)
    ranks = sel[["loso", "thin_ho", "thin_dP", "noise25"]].rank()
    sel["mean_rank"] = ranks.mean(axis=1)
    sel["eligible"] = elig
    chosen = sel[elig].mean_rank.idxmin() if elig.any() else sel.mean_rank.idxmin()
    sel_chk = pd.DataFrame({"cov": cov["check"], "arb": arb["check"], "loso": loso["check"],
                            "noise25": roll[("+0.25", "check")]})
    res["chosen"] = chosen

    tabs = {"coverage": cov, "coverage_by_Y": covY, "arb": arb, "arb_1e3": arb3, "fit_rmse": fit, "loso": loso, "loso_med": loso_med,
            "loso_wing": loso_wing, "loso_rel": loso_rel, "thin": thin, "stab_med": stab, "stab_p90": stab90,
            "roll_noise": roll, "pit_by_Y": pitY, "calib": calw, "pit_pts": pit_pts, "oiw": oiw,
            "select_choose": sel, "select_check": sel_chk, "low_info_by_Y": lowinfo.to_frame()}
    for k, v in tabs.items():
        v.to_csv(C / f"summary_{k}.csv")
    pd.set_option("display.width", 220)
    for k, v in tabs.items():
        print(f"\n=== {k}\n{v.round(4).to_string()}")
    flags = dq[dq.method == chosen].groupby("Y").agg(days=("date", "size"), nq=("nq", "median"),
                                                     default=("default_settle", "mean"), zero_oi=("zero_oi", "mean"),
                                                     zoi_share=("zero_oi_share", "mean"),
                                                     ext=("extrapolated_share", "median"),
                                                     dropped=("n_dropped", "sum"),
                                                     parity_gap=("parity_gap", lambda s: s.abs().max()))
    flags.to_csv(C / "summary_flags_chosen.csv")
    print("\n=== flags (chosen)\n", flags.round(3).to_string())
    final_tables(res)
    json.dump(res, open(C / "summary.json", "w"), indent=1, default=float)
    figures(daily, lo, st, ca, chosen)


def boot_ratio(lo, a, b, win_, nb=2000, seed=0):
    """Day-block bootstrap of RMSE_a / RMSE_b on paired LOSO rows (same date, strike)."""
    x = lo[lo.win == win_].pivot_table(index=["date", "K"], columns="method", values="e")[[a, b]].dropna()
    days = x.index.get_level_values(0)
    ud = days.unique()
    grp = {d: x.loc[d].values for d in ud}
    rng = np.random.default_rng(seed)
    pt = np.sqrt(np.mean(x[a] ** 2)) / np.sqrt(np.mean(x[b] ** 2))
    out = []
    for _ in range(nb):
        v = np.concatenate([grp[d] for d in rng.choice(ud, len(ud))])
        out.append(np.sqrt(np.mean(v[:, 0] ** 2)) / np.sqrt(np.mean(v[:, 1] ** 2)))
    return pt, np.percentile(out, 2.5), np.percentile(out, 97.5), len(ud)


def final_tables(res):
    """Shipped configuration (sln, marks dropped wing by wing) and the same quote rule for svi / mixture."""
    f = load("final")
    fq = f[f.has_quotes]
    out = {}
    out["final_coverage"] = f.groupby(["method", "win"]).usable.mean().unstack()
    out["final_coverage_by_Y"] = f.groupby(["method", "Y"]).usable.mean().unstack()
    out["final_arb"] = fq.groupby(["method", "win"]).apply(lambda g: (g.neg_mass.fillna(0) > 1e-4).mean()).unstack()
    st = stability(f)
    out["final_stab_med"] = st.dropna(subset=["dP"]).groupby(["method", "k", "win"]).dP.apply(
        lambda x: x.abs().median()).unstack([1, 2])
    out["final_stab_p90"] = st.dropna(subset=["dP"]).groupby(["method", "k", "win"]).dP.apply(
        lambda x: x.abs().quantile(0.9)).unstack([1, 2])
    rc = st.dropna(subset=["roll_cov"])
    out["final_roll"] = rc.groupby(["method", "k", "win"]).apply(
        lambda g: float(np.sqrt(max(-np.average(g.roll_cov, weights=g.n_roll), 0)))).unstack([1, 2])
    ca = fq[fq.usable & fq.pit.notna()]
    out["final_pit_by_Y"] = ca.groupby(["method", "Y"]).pit.mean().unstack()
    cal = ca.groupby(["method", "Y"]).agg(crps=("crps", "mean"), logs=("logpdf", "mean"),
                                         tail10=("pit", lambda p: ((p < 0.1) | (p > 0.9)).mean())).reset_index()
    cal["win"] = win(cal.Y.values)
    out["final_calib"] = cal.groupby(["method", "win"])[["crps", "logs", "tail10"]].mean().unstack()
    out["final_calib_by_Y"] = cal.pivot_table(index="Y", columns="method", values=["crps", "logs"])
    lf = load("losofinal")
    lf["e"] = lf.pred - lf.px
    out["final_loso"] = lf.groupby(["method", "win"]).e.apply(rmse).unstack()
    out["final_loso_med"] = lf.groupby(["method", "win"]).e.apply(lambda e: np.nanmedian(np.abs(e))).unstack()
    out["final_loso_wing"] = lf.groupby(["method", "win", "wing"]).e.apply(rmse).unstack([1, 2])
    bt = []
    for w in ["choose", "check"]:
        for b in ["mixture", "svi"]:
            pt, l, h, n = boot_ratio(lf, "sln", b, w)
            bt.append({"win": w, "vs": b, "rmse_ratio_sln_over": pt, "lo95": l, "hi95": h, "days": n})
    lo = load("loso")
    lo["e"] = lo.pred - lo.px
    for w in ["choose", "check"]:
        for b in ["mixture", "svi", "convex", "quad"]:
            pt, l, h, n = boot_ratio(lo, "sln", b, w)
            bt.append({"win": w, "vs": b + " (all quotes)", "rmse_ratio_sln_over": pt, "lo95": l, "hi95": h, "days": n})
    out["final_boot"] = pd.DataFrame(bt)
    d = fq.copy()
    d["atom_ratio"] = d.cdf_abs.apply(lambda c: np.max(np.diff(c)) if isinstance(c, np.ndarray) else np.nan) \
        * 2 * np.pi * d.atm_px / 0.01
    out["final_atoms"] = d.groupby(["method", "win"]).atom_ratio.apply(lambda a: (a > 6).mean()).unstack()
    s = fq[fq.method == "sln"]
    out["final_flags"] = s.groupby("Y").agg(days=("date", "size"), nq_med=("nq", "median"), nq_min=("nq", "min"),
                                            marks_kept=("marks_kept", "mean"), zero_oi_all=("zero_oi", "mean"),
                                            zoi_share=("zero_oi_share", "mean"), default=("default_settle", "mean"),
                                            ext_med=("extrapolated_share", "median"),
                                            low_info=("low_info", "mean"), dropped_med=("n_dropped", "median"),
                                            parity_gap_max=("parity_gap", lambda x: x.abs().max()))
    for k, v in out.items():
        v.to_csv(C / f"summary_{k}.csv")
        print(f"\n=== {k}\n{v.round(4).to_string()}")
    res["final_coverage_sln"] = out["final_coverage"].loc["sln"].to_dict()


def figures(daily, lo, st, ca, chosen):
    # Fig 1: method comparison bars (LOSO RMSE and Roll noise, choose vs check)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(METHODS))
    for j, (w, off) in enumerate([("choose", -0.2), ("check", 0.2)]):
        v = [rmse(lo[(lo.method == m) & (lo.win == w)].e) * 100 for m in METHODS]
        ax[0].bar(x + off, v, 0.38, color=[COL[m] for m in METHODS], alpha=1.0 if w == "choose" else 0.55,
                  edgecolor="white", linewidth=2)
        for xi, vi in zip(x + off, v):
            ax[0].text(xi, vi, f"{vi:.2f}", ha="center", va="bottom", fontsize=8, color="#333")
        rc = st.dropna(subset=["roll_cov"])
        rc = rc[(rc.k == "+0.25") & (rc.win == w)]
        nv = [np.sqrt(max(-np.average(rc[rc.method == m].roll_cov, weights=rc[rc.method == m].n_roll), 0)) * 100
              for m in METHODS]
        ax[1].bar(x + off, nv, 0.38, color=[COL[m] for m in METHODS], alpha=1.0 if w == "choose" else 0.55,
                  edgecolor="white", linewidth=2)
        for xi, vi in zip(x + off, nv):
            ax[1].text(xi, vi, f"{vi:.1f}", ha="center", va="bottom", fontsize=8, color="#333")
    for a, t in zip(ax, ["Leave-one-strike-out price RMSE (cents)", "Day-to-day noise in P(S_T > F+0.25), pp (Roll)"]):
        a.set_xticks(x)
        a.set_xticklabels([LABEL[m].replace(" ", "\n", 1) for m in METHODS], fontsize=8)
        a.set_title(t, fontsize=10)
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color="#e5e5e5")
        a.set_axisbelow(True)
    fig.text(0.5, 0.01, "Solid = choose window (2013-14..2015-16); faded = check window (2016-17..Nov-Dec 2021)",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "rnd_engine_compare.png", dpi=130)
    plt.close(fig)

    # Fig 2: time series of P(S_T > F+0.25) in a dense and a thin winter, all methods
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), sharey=False)
    for a, Y in zip(ax, [2015, 2019]):
        for m in METHODS:
            g = daily[(daily.Y == Y) & (daily.method == m) & daily.usable].sort_values("date")
            lw = 2.2 if m == chosen else 1.2
            a.plot(g.date, g["P+0.25"], color=COL[m], lw=lw, label=LABEL[m])
        a.set_title(f"P(S_T > F + $0.25), winter {Y - 1}-{str(Y)[2:]}", fontsize=10)
        a.spines[["top", "right"]].set_visible(False)
        a.grid(color="#eeeeee")
        a.tick_params(labelsize=8)
    ax[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "rnd_engine_stability.png", dpi=130)
    plt.close(fig)

    # Fig 3: example densities on hard days
    days = [("2015-01-07", 2015), ("2018-01-10", 2018), ("2018-11-14", 2019), ("2021-01-12", 2021)]
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    for a, (d, Y) in zip(ax, days):
        for m in METHODS:
            r = R.spread_density(d, Y, method=m)
            if not len(r["grid"]):
                continue
            F = r["F"]
            mk = (r["grid"] > F - 0.6) & (r["grid"] < F + 1.2)
            a.plot(r["grid"][mk], r["pdf"][mk], color=COL[m], lw=2.2 if m == chosen else 1.1, label=LABEL[m])
        q = r["quotes"]
        for k in q["K"]:
            if F - 0.6 < k < F + 1.2:
                a.axvline(k, color="#bbbbbb", lw=0.6, ls=":")
        a.set_title(f"{d} (n={r['n_quotes']}, F={F:.2f})", fontsize=9)
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=8)
        a.set_xlabel("S_T ($/MMBtu); dotted = quoted strikes", fontsize=8)
    ax[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "rnd_engine_densities.png", dpi=130)
    plt.close(fig)

    # Fig 4: PIT of realised S_T by winter (chosen method), daily
    fig, ax = plt.subplots(figsize=(7, 3.6))
    g = ca[ca.method == chosen]
    ys = sorted(g.Y.unique())
    ax.boxplot([g[g.Y == y].pit for y in ys], tick_labels=[f"{y - 1}-{str(y)[2:]}" for y in ys], widths=0.5,
               medianprops={"color": COL[chosen], "lw": 2})
    ax.axhline(0.5, color="#888", lw=0.8, ls="--")
    ax.set_ylim(0, 1)
    ax.set_title(f"PIT of realised S_T under the implied CDF, every usable day ({LABEL[chosen]})", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "rnd_engine_pit.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
