"""Tables and figures for reports/tool/rnd_composite.md (reads caches written by rnd_composite_eval)."""
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.tool import rnd_composite as C  # noqa: E402
from src.tool import rnd_composite_eval as E  # noqa: E402
from src.tool import rnd_direct as R  # noqa: E402

FIG = C.ROOT / "reports/tool/figures"
COL = {"direct": "#2a78d6", "gauss": "#eb6834", "t4": "#1baf7a", "mix": "#eda100", "hist": "#6250d6"}
WINTER = [11, 12, 1, 2]
rng = np.random.default_rng(0)


def daily():
    d = pd.concat([pd.read_pickle(E.OUT / f"daily_{y}.pkl") for y in range(2011, 2023)
                   if (E.OUT / f"daily_{y}.pkl").exists()], ignore_index=True)
    return d[d.date.dt.month.isin(WINTER)]


def boot_mean(df, col, by="Y", n=4000):
    g = df.groupby(by)[col].agg(["sum", "count"])
    ys = g.index.values
    s, c = g["sum"].values, g["count"].values
    idx = rng.integers(0, len(ys), (n, len(ys)))
    m = s[idx].sum(1) / c[idx].sum(1)
    return float(s.sum() / c.sum()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fmt(t):
    return f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"


def main():
    d = daily()
    dt = E.direct_table()
    dt = dt[dt.date.dt.month.isin(WINTER)]
    res = {}
    # ---------------------------------------------------------------- coverage
    g = d[(d.fam == "gauss") & (d["mode"] == "auto")].copy()
    g["usable"] = (g.nH >= 5) & ((g.nJ >= 5) | g.j_proxy) & (g.neg_mass < 0.05) & (g[["rmseH", "rmseJ"]].max(axis=1) < 0.05)
    fut_days = {Y: len([x for x in C._winter_days(Y) if x.month in WINTER]) for Y in range(2011, 2023)}
    cov = g.groupby("Y").agg(n=("date", "size"), usable=("usable", "mean"), cso_valid=("cso_valid", "mean"),
                             j_proxy=("j_proxy", "mean"), src_hist=("src", lambda s: (s == "hist").mean()),
                             nH=("nH", "median"), nJ=("nJ", "median"), rmseH=("rmseH", "median"),
                             bfly_ok=("bfly_ok", "mean"), neg_mass=("neg_mass", "max"), oi3=("n_oi_prev", lambda s: (s >= 3).mean()))
    cov["trading_days"] = pd.Series(fut_days)
    cov["direct_any"] = dt.groupby("Y").size() / cov.trading_days
    cov["direct_nondefault"] = dt[dt.default != True].groupby("Y").size() / cov.trading_days
    cov["composite_cov"] = cov.n * cov.usable / cov.trading_days
    cov.to_csv(E.OUT / "t_coverage.csv")
    print(cov.round(3).to_string())
    # ---------------------------------------------------------------- composite vs direct
    m = d.merge(dt, on=["date", "Y"], suffixes=("", "_dir"))
    m = m[m.default != True]
    m["win"] = np.where(m.Y.between(2014, 2016), "choose", np.where(m.Y.between(2017, 2021), "check", "other"))
    m["dense"] = (m.nq >= 15) & (m.n_oi_prev >= 3)
    rows = []
    for (win, fam, mode), x in m[m.win != "other"].groupby(["win", "fam", "mode"]):
        x = x[x.dense] if win == "choose" else x
        if len(x) == 0:
            continue
        r = {"win": win, "fam": fam, "mode": mode, "n": len(x), "winters": x.Y.nunique()}
        for k in ("pa10", "pa25", "pa50"):
            r[k + "_bias"] = (x[k] - x[k + "_dir"]).mean()
            r[k + "_mae"] = (x[k] - x[k + "_dir"]).abs().mean()
        for k in ("q5", "q50", "q95"):
            r[k + "_mad"] = (x[k] - x[k + "_dir"]).abs().median()
            r[k + "_bias"] = (x[k] - x[k + "_dir"]).median()
        r["sd_ratio_med"] = (x.sd / x.sd_dir).median()
        r["iqr_ratio_med"] = ((x.q75 - x.q25) / (x.q75_dir - x.q25_dir)).median()
        rows.append(r)
    cmp_ = pd.DataFrame(rows)
    cmp_.to_csv(E.OUT / "t_compare.csv", index=False)
    print(cmp_.round(4).to_string())
    # copula effect on upper tail (same mode)
    piv = d.pivot_table(index=["date", "Y", "mode"], columns="fam", values=["pa25", "pa50", "q95", "rho"])
    cop = {}
    for mode in ("cso", "hist", "fixed"):
        x = piv.xs(mode, level="mode")
        cop[mode] = {"n": len(x),
                     "pa50_gauss": x[("pa50", "gauss")].mean(), "pa50_t4": x[("pa50", "t4")].mean(),
                     "pa50_mix": x[("pa50", "mix")].mean(),
                     "pa25_gauss": x[("pa25", "gauss")].mean(), "pa25_t4": x[("pa25", "t4")].mean(),
                     "pa25_mix": x[("pa25", "mix")].mean(),
                     "q95_t4_minus_g": (x[("q95", "t4")] - x[("q95", "gauss")]).median(),
                     "q95_mix_minus_g": (x[("q95", "mix")] - x[("q95", "gauss")]).median(),
                     "rho_g": x[("rho", "gauss")].median(), "rho_t4": x[("rho", "t4")].median(),
                     "rho_mix": x[("rho", "mix")].median()}
    cop = pd.DataFrame(cop).T
    cop.to_csv(E.OUT / "t_copula.csv")
    print(cop.round(4).to_string())
    # mode effect at fixed family (cso days): hist/fixed vs cso
    pm = d[d.cso_valid].pivot_table(index=["date", "Y", "fam"], columns="mode", values=["rho", "pa25", "pa50"])
    modeeff = {}
    for mode in ("hist", "fixed"):
        modeeff[mode] = {"rho_err_med": (pm[("rho", mode)] - pm[("rho", "cso")]).median(),
                         "rho_err_mad": (pm[("rho", mode)] - pm[("rho", "cso")]).abs().median(),
                         "pa25_mae": (pm[("pa25", mode)] - pm[("pa25", "cso")]).abs().mean(),
                         "pa50_mae": (pm[("pa50", mode)] - pm[("pa50", "cso")]).abs().mean()}
    modeeff = pd.DataFrame(modeeff).T
    modeeff.to_csv(E.OUT / "t_modes.csv")
    print(modeeff.round(4).to_string())
    # ---------------------------------------------------------------- calibration at realised S_T
    sc = d[d.Y.isin(E.SCORED)].copy()
    cal = []
    for (fam, mode), x in sc.groupby(["fam", "mode"]):
        r = {"route": f"composite {fam}/{mode}", "days": len(x), "winters": x.Y.nunique(),
             "crps": boot_mean(x, "crps"), "pit_mean": x.pit.mean(),
             "out90": ((x.pit < 0.05) | (x.pit > 0.95)).mean(), "pit_gt95": (x.pit > 0.95).mean()}
        for t in (10, 25, 50):
            ev = (x.y > x.F + t / 100).astype(float)
            r[f"brier{t}"] = ((x[f"pa{t}"] - ev) ** 2).mean()
            r[f"freq{t}"] = ev.mean()
            r[f"p{t}"] = x[f"pa{t}"].mean()
        cal.append(r)
    dts = dt[dt.Y.isin(E.SCORED)].merge(sc[(sc.fam == "gauss") & (sc["mode"] == "auto")][["date", "Y", "y", "F"]],
                                        on=["date", "Y"])
    r = {"route": "direct (G4X, all fits)", "days": len(dts), "winters": dts.Y.nunique(), "crps": boot_mean(dts, "crps"),
         "pit_mean": dts.pit.mean(), "out90": ((dts.pit < 0.05) | (dts.pit > 0.95)).mean(), "pit_gt95": (dts.pit > 0.95).mean()}
    for t in (10, 25, 50):
        ev = (dts.y > dts.F + t / 100).astype(float)
        r[f"brier{t}"] = ((dts[f"pa{t}"] - ev) ** 2).mean()
        r[f"freq{t}"] = ev.mean()
        r[f"p{t}"] = dts[f"pa{t}"].mean()
    cal.append(r)
    cal = pd.DataFrame(cal)
    cal["crps_str"] = cal.crps.map(lambda t: f"{t[0]:.4f} [{t[1]:.4f}, {t[2]:.4f}]")
    cal.drop(columns="crps").to_csv(E.OUT / "t_calibration.csv", index=False)
    print(cal.drop(columns="crps").round(4).to_string())
    # same-day comparison, 2013-21: composite gauss/auto vs direct, per window
    same = dts.merge(sc[(sc.fam == "gauss") & (sc["mode"] == "auto")][["date", "Y", "crps", "pit"]], on=["date", "Y"],
                     suffixes=("_dir", "_comp"))
    same["d"] = same.crps_comp - same.crps_dir
    res["same_day"] = {w: fmt(boot_mean(same[same.Y.between(a, b)], "d")) + f" n={int(same.Y.between(a, b).sum())}"
                       for w, (a, b) in {"2013-21": (2013, 2021), "choose 2014-16": (2014, 2016),
                                         "check 2017-21": (2017, 2021)}.items()}
    res["same_day_by_winter"] = same.groupby("Y")[["crps_dir", "crps_comp"]].mean().round(4).to_dict()
    print(res["same_day"])
    # ---------------------------------------------------------------- blend
    if (E.OUT / "blendeval.pkl").exists():
        b = pd.read_pickle(E.OUT / "blendeval.pkl")
        b["win"] = np.where(b.Y.between(2014, 2016), "choose", np.where(b.Y.between(2017, 2021), "check", "2013"))
        rules = {}
        for nfull in (2, 4, 8, 12):
            w = np.clip(b.n_oi_prev / nfull, 0, 1)
            w = np.where(b.default == True, 0, w)
            # interpolate CRPS between evaluated weights (piecewise-linear in w; CRPS is convex in w -> upper bound)
            ws = np.array([0, .25, .5, .75, 1.0])
            cr = b[[f"crps_w{int(x*100)}" for x in ws]].values
            b[f"crps_rule_n{nfull}"] = [np.interp(wi, ws, ci) for wi, ci in zip(w, cr)]
        for N in (1, 3, 5):
            sel = (b.n_oi_prev >= N) & (b.default != True)
            b[f"crps_sel_N{N}"] = np.where(sel, b.crps_w100, b.crps_w0)
        b["crps_half"] = b.crps_w50
        cols = ["crps_w0", "crps_w100", "crps_half"] + [c for c in b.columns if c.startswith("crps_rule") or c.startswith("crps_sel")]
        tb = []
        for c in cols:
            tb.append({"rule": c, **{w: fmt(boot_mean(b[b.win == w].assign(z=b[c] - b.crps_w100), "z"))
                                     for w in ("choose", "check")},
                       "mean_choose": b[b.win == "choose"][c].mean(), "mean_check": b[b.win == "check"][c].mean()})
        tb = pd.DataFrame(tb)
        tb.to_csv(E.OUT / "t_blend.csv", index=False)
        print(tb.to_string())
    json.dump(res, open(E.OUT / "summary.json", "w"), indent=1, default=str)
    figures(d, dt, m)


def figures(d, dt, m):
    # 1. example densities
    days = [("2015-01-13", 2015, "dense day, Jan 2015"), ("2015-12-08", 2016, "dense day, Dec 2015"),
            ("2018-11-14", 2019, "squeeze, 14 Nov 2018"), ("2021-01-19", 2021, "thin CSO, Jan 2021")]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for a, (dd, Y, ttl) in zip(ax.flat, days):
        rd = pd.read_pickle(E.ISD / f"rnd_daily_{Y}.pkl")
        row = rd[rd.date == dd]
        F = None
        for fam in ("gauss", "t4", "mix"):
            o = C.spread_density_composite(dd, Y, "auto", fam)
            if o is None:
                continue
            F = o["F"]
            a.plot(o["grid"], o["pdf"], color=COL[fam], lw=1.6, label=f"composite {fam} (rho {o['rho']:.3f}, {o['rho_source']})")
        dr = R.spread_density(dd, Y)
        if len(dr["grid"]):
            a.plot(dr["grid"], dr["pdf"], color=COL["direct"], lw=2,
                   label=f"direct engine sln ({C.n_oi_quotes(dr)} OI quotes)")
        if F is not None:
            a.axvline(F, color="#52514e", lw=0.8, ls=":")
            sd = o["quantiles"][0.95] - o["quantiles"][0.05]
            a.set_xlim(o["quantiles"][0.01] - 0.2 * sd, o["quantiles"][0.99] + 0.2 * sd)
        a.set_title(ttl, fontsize=10)
        a.set_xlabel("S_T = H - J at March option expiry ($/MMBtu)", fontsize=8)
        a.legend(fontsize=7, frameon=False)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Composite (ON legs + copula) vs direct (G4X) risk-neutral densities", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "rnd_composite_examples.png", dpi=110)
    plt.close(fig)
    # 2. tail scatter
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    v = E.vs_engine()
    x = d[d["mode"] == "cso"].merge(v[["date", "Y", "P+0.10", "P+0.25", "P+0.50"]], on=["date", "Y"])
    x = x[x.Y.between(2014, 2021)]
    for a, k in zip(ax, ("pa10", "pa25", "pa50")):
        kd = "P+" + f"{int(k[2:]) / 100:.2f}"
        for fam in ("gauss", "t4", "mix"):
            z = x[x.fam == fam]
            a.scatter(z[kd], z[k], s=7, alpha=0.5, color=COL[fam], label=fam)
        x[k + "_dir"] = x[kd]
        lim = max(x[k].max(), x[k + "_dir"].max()) * 1.05
        a.plot([0, lim], [0, lim], color="#52514e", lw=0.8)
        a.set_xlabel(f"direct P(S_T > F + {int(k[2:]) / 100:.2f})", fontsize=8)
        a.set_xlim(0, lim); a.set_ylim(0, lim)
        a.set_ylabel("composite (rho from CSO ATM)", fontsize=8)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Upper-tail probabilities, CSO-calibrated composite vs direct engine, winters 2013-14 to 2020-21", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "rnd_composite_tail.png", dpi=110)
    plt.close(fig)
    # 3. PIT + CRPS by winter
    sc = d[d.Y.isin(E.SCORED)]
    g = sc[(sc.fam == "gauss") & (sc["mode"] == "auto")]
    dts = E.vs_engine()
    dts = dts[dts.Y.isin(E.SCORED)]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.linspace(0, 1, 11)
    ax[0].hist([g.pit, dts.pit], bins=bins, density=True, color=[COL["gauss"], COL["direct"]],
               label=["composite gauss/auto (winters 2010-11..2020-21)", "direct engine sln (2012-13..2020-21)"], rwidth=0.9)
    ax[0].axhline(1, color="#52514e", lw=0.8)
    ax[0].set_xlabel("PIT of realised S_T (daily, overlapping within winter)", fontsize=8)
    ax[0].legend(frameon=False, fontsize=8)
    cw = g.groupby("Y").crps.mean()
    cd = dts.groupby("Y").crps.mean()
    yy = np.arange(len(cw))
    ax[1].bar(yy - 0.2, cw.values, 0.38, color=COL["gauss"], label="composite")
    ax[1].bar([list(cw.index).index(y) + 0.2 for y in cd.index], cd.values, 0.38, color=COL["direct"], label="direct")
    ax[1].set_xticks(yy)
    ax[1].set_xticklabels([E.winter_label(y) if hasattr(E, "winter_label") else f"{y-1}-{str(y)[2:]}" for y in cw.index],
                          rotation=45, fontsize=7)
    ax[1].set_ylabel("mean CRPS ($/MMBtu)", fontsize=8)
    ax[1].legend(frameon=False, fontsize=8)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "rnd_composite_calibration.png", dpi=110)
    plt.close(fig)
    # 4. rho paths
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for a, Y in zip(ax, (2015, 2019, 2021)):
        z = d[(d.Y == Y) & (d.fam == "gauss")].pivot_table(index="date", columns="mode", values="rho")
        for mode, col in (("cso", COL["gauss"]), ("hist", COL["hist"]), ("fixed", "#52514e")):
            if mode in z:
                a.plot(z.index, z[mode], color=col, lw=1.4, label=mode, marker="o" if mode == "cso" else None, ms=2)
        e = d[(d.Y == Y) & (d.fam == "gauss") & (d["mode"] == "auto")].set_index("date").rho_ewma
        a.plot(e.index, e.values, color=COL["direct"], lw=1, ls="--", label="EWMA realised")
        a.set_title(f"winter {Y-1}-{str(Y)[2:]}", fontsize=10)
        a.tick_params(axis="x", labelsize=7, rotation=30)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].set_ylim(0.6, 1.005)
    ax[0].set_ylabel("Gaussian-copula rho", fontsize=8)
    ax[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "rnd_composite_rho.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
