"""Figures and resolution tables for reports/tool/storage_map.md (agent storage_map).

python -m src.tool.storage_map_figures      # < 1 min; needs the eval fold caches
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from src.tool import _map_data as MD, storage_map as SM

FIG = MD.ROOT / "reports/tool/figures"
FIG.mkdir(parents=True, exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # validated default categorical order
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6})
ash = lambda s: np.arcsinh(np.asarray(s) / SM.C_ASINH)


def fig_scatter(t, m, mf):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sg = np.linspace(-0.2, 3.2, 400)
    pm = SM.posterior(m, sg)
    cdf = np.cumsum(pm, axis=1)
    q = {k: np.array([np.interp(k, c, SM.X_GRID) for c in cdf]) for k in (0.1, 0.5, 0.9)}
    ax.fill_betweenx(ash(sg), q[0.1], q[0.9], color=C[0], alpha=0.15, lw=0, label="inv_asinh: 10-90% of x | S_T")
    ax.plot(q[0.5], ash(sg), color=C[0], lw=2, label="inv_asinh: median x | S_T (default)")
    xg = np.linspace(-1100, 1000, 300)
    f, _ = SM.forward(mf, xg)
    ax.plot(xg, ash(f), color=C[1], lw=2, label="hetero_exp forward curve f(x)")
    v5 = -0.083 + 0.034 * np.exp(-xg / 387)
    ax.plot(xg, ash(v5), color=C[3], lw=2, ls="--", label="v5 frozen 2022 map (fit 2011-21)")
    for era, mk, col in (("pre", t.index < 2010, C[2]), ("post", t.index >= 2010, INK)):
        ax.scatter(t.x[mk], ash(t.S[mk]), s=28, color=col, edgecolor="white", lw=1, zorder=3,
                   label="winters 1994-2009 (EIA)" if era == "pre" else "winters 2010-21")
    for Y in (1994, 1996, 2001, 2003, 2012, 2014, 2016, 2018, 2019, 2021):
        ax.annotate(str(Y), (t.x[Y], ash(t.S[Y])), xytext=(4, 3), textcoords="offset points", fontsize=7, color=MUTED)
    ticks = [-0.15, -0.05, 0, 0.05, 0.2, 0.5, 1, 3]
    ax.set_yticks(ash(ticks), [f"{v:g}" for v in ticks])
    ax.set_xlim(-1300, 1100)
    ax.set_xlabel("x = end-March storage minus prior 5-yr avg (Bcf, EIA monthly US)")
    ax.set_ylabel("H-J at March option expiry, $/MMBtu (asinh scale)")
    ax.set_title("Spread at expiry vs end-March storage, 28 winters 1994-2021", loc="left", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "storage_map_scatter.png", dpi=150)
    plt.close(fig)


def fig_resolution(t, m):
    sg = np.linspace(-0.2, 1.0, 241)
    pdraw = SM.posterior(m, sg, per_draw=True)
    cd = np.cumsum(pdraw, axis=2)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for i, x0 in enumerate(SM.THRESHOLDS):
        j = np.searchsorted(SM.X_GRID, x0) - 1
        pr = cd[:, :, j]
        ax.fill_between(sg, np.quantile(pr, .05, axis=0), np.quantile(pr, .95, axis=0), color=C[i], alpha=0.12, lw=0)
        ax.plot(sg, pr.mean(axis=0), color=C[i], lw=2, label=f"P(x < {x0})")
        ax.axhline((t.x < x0).mean(), color=C[i], lw=1, ls=":")
    ax.set_xlabel("S_T = H-J at option expiry ($/MMBtu)")
    ax.set_ylabel("probability")
    ax.set_title("Storage odds given the expiry spread (map fit 1994-2021; band = 5-95% bootstrap; dotted = base rate)",
                 loc="left", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG / "storage_map_resolution.png", dpi=150)
    plt.close(fig)


def fig_wf():
    d = pd.concat([pd.read_csv(MD.CACHE / "folds_inverse_variants.csv"), pd.read_csv(MD.CACHE / "folds_baselines.csv")])
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for k, (spec, lab, off) in enumerate((("climatology", "climatology", -0.18), ("inv_c0.05", "inv_asinh (S_T)", 0.0),
                                          ("ceil_asinhJK", "ceiling: J-K at expiry", 0.18))):
        g = d[(d.spec == spec) & (d["mode"] == "wf")].sort_values("Y")
        lo, hi = g.med - 1.2816 * g.sd, g.med + 1.2816 * g.sd     # approx 80% interval (normal)
        ax.vlines(g.Y + off, lo, hi, color=C[k], lw=2.2, label=f"{lab}: approx. 80% interval")
        ax.scatter(g.Y + off, g.med, color=C[k], s=12, zorder=3)
    g = d[(d.spec == "climatology") & (d["mode"] == "wf")]
    ax.scatter(g.Y, g.x, marker="D", color=INK, s=22, zorder=4, label="realised x")
    ax.axhline(-400, color=MUTED, lw=0.8, ls="--")
    ax.set_ylabel("x (Bcf)")
    ax.set_title("Walk-forward predictive of end-March x given the realised expiry spread, 2004-2021", loc="left",
                 fontsize=10)
    ax.set_xticks(range(2004, 2022, 2))
    ax.set_ylim(-1150, 1150)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    fig.savefig(FIG / "storage_map_wf.png", dpi=150)
    plt.close(fig)


def tables(t, m, mf):
    out = {}
    # resolution in S (inverse map) and discrimination (empirical AUC of S for each event)
    out["describe"] = SM.describe(m, (-0.15, -0.10, -0.05, 0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.5, 3.0)).round(3)
    auc = {}
    for x0 in (-200, -400, -600):
        ev = t.x < x0
        u = mannwhitneyu(t.S[ev], t.S[~ev]).statistic
        auc[x0] = {"n_events": int(ev.sum()), "AUC_S": round(u / (ev.sum() * (~ev).sum()), 3),
                   "AUC_JK": round(mannwhitneyu(t.JK[ev], t.JK[~ev]).statistic / (ev.sum() * (~ev).sum()), 3)}
    out["auc"] = auc
    # forward model separability: AUC of S | x=x1 vs S | x=0 under hetero_exp
    from scipy.stats import norm
    f0, s0 = SM.forward(mf, [0.0])
    sep = {}
    for x1 in (300, -200, -400, -600, -800, -1000):
        f1, s1 = SM.forward(mf, [x1])
        sep[x1] = round(float(norm.cdf((f1 - f0) / np.sqrt(s1 ** 2 + s0 ** 2))[0]), 3)
    out["fwd_auc_vs_x0"] = sep
    # monthly vs weekly
    w = t.dropna(subset=["em_w"])
    out["weekly_minus_monthly_level"] = {"n": len(w), "mean": round((w.em_w - w.em_m).mean(), 1),
                                         "sd": round((w.em_w - w.em_m).std(), 1)}
    w5 = t[t.index >= 2015]
    out["xw_minus_x_2015_21"] = {"n": len(w5), "mean": round((w5.xw - w5.x).mean(), 1),
                                 "sd": round((w5.xw - w5.x).std(), 1), "corr": round(np.corrcoef(w5.xw, w5.x)[0, 1], 4)}
    out["frozen_inv_2021"] = {"alpha": round(m["theta"][0], 1), "beta": round(m["theta"][1], 1),
                              "sigma": round(float(np.exp(m["theta"][2])), 1),
                              "beta_5_95": [round(v, 1) for v in np.quantile(m["draws"][:, 1], [.05, .95])],
                              "prior_mean_sd": [round(v, 1) for v in m["prior"]]}
    th = mf["theta"]
    out["frozen_fwd_2021"] = {"a": round(th[0], 4), "b": round(float(np.exp(th[1])), 4),
                              "tau": round(float(np.exp(th[2])), 1), "s0": round(float(np.exp(th[3])), 4),
                              "k": round(float(np.exp(th[4])), 3)}
    return out


if __name__ == "__main__":
    t = MD.winter_table()
    m = SM.fit_map(2021, B=200)
    mf = SM.fit_map(2021, spec="hetero_exp", B=40)
    fig_scatter(t, m, mf)
    fig_resolution(t, m)
    fig_wf()
    o = tables(t, m, mf)
    pd.set_option("display.width", 200)
    print(o.pop("describe").to_string())
    print(json.dumps(o, indent=1, default=float))
    json.dump(o, open(MD.CACHE / "resolution.json", "w"), indent=1, default=float)
