"""Run the full Q1 curve-consistency study (data < 2022-01-01 only).

    python -m src.research.curve_consistency.run_all

Writes tables to data/processed/research_curve_consistency/ and figures to
reports/agents/figures/curve_consistency_*.png. About 30 seconds.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import CACHE, FIG, CUTOFF, load_curve, storage_deviation
from . import carry as C
from .kinks import fit_all, add_flies
from .reversion import (IS, OOS, TICKS, regressions, oos_r2, grid_select, trades, summarize)

CACHE.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 220)
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def out(name, df):
    df.to_csv(CACHE / f"{name}.csv")
    print(f"\n=== {name} ===\n{df.round(4).to_string()}")


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)


def main():
    d, dates = load_curve()
    assert d.date.max() < CUTOFF
    sd = storage_deviation(dates)
    assert sd.index.max() < CUTOFF

    # ------------------------------------------------------------ Part 1: carry
    pr = C.adjacent_pairs(d)
    pr["dev"] = sd.dev.reindex(pr.date).values
    fixed, wf, by_year = C.estimate_c(pr)
    summer = pr[pr.near_m.between(4, 8) & pr.date.between(*C.IS)]
    c_summer = summer.s.quantile(0.95)
    out("c_estimates", pd.DataFrame({
        "autumn_front_2011_17": pd.Series(fixed),
        "summer_pairs_2011_17": summer.s.quantile([0.5, 0.9, 0.95, 0.99, 1.0]).values}))
    out("c_walkforward", wf.to_frame("c_q95"))
    ex = pd.concat([C.exceedance_table(pr, fixed[0.95], "autumn q95 (fixed)"),
                    C.exceedance_table(pr, wf, "autumn q95 (walk-fwd)"),
                    C.exceedance_table(pr, c_summer, "summer-pair q95")])
    out("carry_exceedance", ex)
    bp_a, byy_a, cy_a = C.exceed_by(pr, fixed[0.95])
    bp_s, byy_s, cy_s = C.exceed_by(pr, c_summer)
    out("exceed_by_pair", pd.concat({"autumn_c": bp_a, "summer_c": bp_s}, axis=1))
    out("exceed_by_year", pd.concat({"autumn_c": byy_a, "summer_c": byy_s}, axis=1))
    au = C.autumn_front(pr)
    ay = au.groupby(au.date.dt.year).agg(q95=("s", lambda s: s.quantile(.95)), med=("s", "median"),
                                         dev=("dev", "mean"))
    out("autumn_contango_vs_storage", ay)
    rho = ay.dropna()[["med", "dev"]].corr(method="spearman").iloc[0, 1]
    print("Spearman(autumn median contango, storage dev) =", round(rho, 2), "n =", ay.dev.notna().sum())

    # ------------------------------------------------------------ Part 2: fair curve
    fit, shapes, lam = fit_all(d)
    fl = add_flies(fit)
    fit.to_pickle(CACHE / "fit.pkl"); fl.to_pickle(CACHE / "flies.pkl")
    out("seasonal_shapes", shapes)
    out("fit_rmse_bp", pd.DataFrame({
        "contract kink sd (bp)": 1e4 * fit.groupby(fit.date.dt.year)[["kA", "kB"]].std().stack(),
    }).unstack())
    out("signal_corr", fl[["fA", "fB", "fbench"]].corr())

    # ------------------------------------------------------------ Part 3: reversion
    out("kink_regressions", regressions(fit, fl))
    out("kink_oos_r2", oos_r2(fit, fl))
    g = grid_select(fl)
    out("trade_grid", g)
    sel = (g[g.period == "2011-17"].sort_values("t", ascending=False).groupby("signal").head(1)
           .set_index("signal")[["q", "thr_bp", "h"]])
    out("trade_selected_on_2011_17", sel)

    rows, curves = [], {}
    for sig, r in sel.iterrows():
        thr, h = r.thr_bp / 1e4, int(r.h)
        j = trades(fl, sig, h, thr)
        curves[sig] = j
        for per, (a, b) in (("2011-17", IS), ("2018-21", OOS)):
            jj = j[j.date.between(a, b)]
            rows.append(dict(signal=sig, period=per, cost="gross", **summarize(jj, ticks=0)))
            for tk in TICKS:
                rows.append(dict(signal=sig, period=per, cost=f"{tk} ticks", **summarize(jj, ticks=tk)))
        # robustness: 1-day entry lag, ex March/April centres, liquidity filter
        lag = trades(fl, sig, h, thr)
        lagged = fl[["di", "k", "B"]].copy(); lagged["di"] -= 1
        lag = lag.merge(lagged.rename(columns={"B": "B1"}), on=["di", "k"])
        lag["gross"] = -np.sign(lag[sig]) * 2 * (lag.B_h - lag.B1) * 10_000
        for per, (a, b) in (("2011-17", IS), ("2018-21", OOS)):
            rows.append(dict(signal=sig, period=per, cost="4 ticks, enter t+1",
                             **summarize(lag[lag.date.between(a, b)], ticks=4)))
            jx = j[~j.cm.isin([3, 4]) & j.date.between(a, b)]
            rows.append(dict(signal=sig, period=per, cost="4 ticks, ex Mar/Apr centre", **summarize(jx, ticks=4)))
        jl = trades(fl, sig, h, thr, liquid=True)
        for per, (a, b) in (("2016-17", ("2016-01-01", "2017-12-31")), ("2018-21", OOS)):
            rows.append(dict(signal=sig, period=per, cost="4 ticks, all",
                             **summarize(j[j.date.between(a, b)], ticks=4)))
            rows.append(dict(signal=sig, period=per, cost="4 ticks, liquid legs",
                             **summarize(jl[jl.date.between(a, b)], ticks=4)))
        out(f"trade_months_{sig}", j.assign(net=j.gross - 160).groupby(j.date.dt.to_period("M"))
            .agg(n=("net", "size"), net_mean=("net", "mean"), top_cm=("cm", lambda s: s.mode().iat[0])))
    tr = pd.DataFrame(rows)
    out("trade_results", tr)

    # small-kink trade: a liquid, frequent version at the median threshold, h=20, liquid legs
    rows = []
    for sig in ("fA", "fbench"):
        for q in (0.5, 0.75, 0.9):
            thr = fl[fl.date.between(*IS)][sig].abs().quantile(q)
            jl = trades(fl, sig, 20, thr, liquid=True)
            for per, (a, b) in (("2016-17", ("2016-01-01", "2017-12-31")), ("2018-21", OOS)):
                rows.append(dict(signal=sig, q=q, period=per, **summarize(jl[jl.date.between(a, b)], 4)))
    out("trade_liquid_grid_h20", pd.DataFrame(rows))

    # ------------------------------------------------------------ Part 4: concentration
    fit["dev"] = sd.dev.reindex(fit.date).values
    fl["dev"] = sd.dev.reindex(fl.date).values
    fit["abs_bp"] = 1e4 * fit.kA.abs()
    fl["abs_bp"] = 1e4 * fl.fA.abs()
    fl["absb_bp"] = 1e4 * fl.fbench.abs()
    big = fl.abs_bp > fl[fl.date.between(*IS)].abs_bp.quantile(0.95)
    conc = pd.DataFrame({
        "mean |contract kink| bp": fit.groupby("cm").abs_bp.mean(),
        "mean signed contract kink bp": 1e4 * fit.groupby("cm").kA.mean(),
        "mean |fly kink| bp": fl.groupby("cm").abs_bp.mean(),
        "mean |bench fly| bp": fl.groupby("cm").absb_bp.mean(),
        "share of top-5% fly kinks %": 100 * fl[big].cm.value_counts(normalize=True).sort_index(),
    })
    out("kink_by_contract_month", conc)
    out("kink_by_tenor", fl.groupby("tau")[["abs_bp", "absb_bp"]].mean())
    f13 = fl[fl.date >= "2013-01-01"].dropna(subset=["dev"]).copy()
    f13["storage"] = pd.cut(f13.dev, [-1e9, -150, 150, 1e9], labels=["tight (<-150 Bcf)", "normal", "loose (>+150 Bcf)"])
    out("kink_by_storage", f13.groupby("storage", observed=True).agg(
        n=("abs_bp", "size"), fly_abs_bp=("abs_bp", "mean"), bench_abs_bp=("absb_bp", "mean"),
        top5_share=("abs_bp", lambda s: 100 * (s > fl[fl.date.between(*IS)].abs_bp.quantile(.95)).mean())))
    mh = f13[f13.cm.isin([3, 4])]
    out("marapr_kink_by_storage", mh.groupby(["storage", "cm"], observed=True).agg(
        fly_abs_bp=("abs_bp", "mean"), fly_signed_bp=("fA", lambda s: 1e4 * s.mean())).unstack())
    out("kink_by_obs_month_storage", f13.groupby([f13.date.dt.month, "storage"], observed=True)
        .abs_bp.mean().unstack())

    # ------------------------------------------------------------ figures
    # Fig 1: implied convenience yield by pair (summer c) + autumn contango vs storage
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    cy = cy_s
    q = cy.groupby("near_m").y.quantile([0.05, 0.5, 0.95]).unstack()
    lbl = [f"{MONTHS[m - 1]}→{MONTHS[m % 12]}" for m in q.index]
    xs = np.arange(12)
    ax[0].vlines(xs, 100 * q[0.05], 100 * q[0.95], color=BLUE, lw=2, alpha=0.45)
    ax[0].plot(xs, 100 * q[0.5], "o", color=BLUE, ms=8)
    ax[0].axhline(0, color=GREY, lw=1)
    ax[0].set_xticks(xs, lbl, rotation=60, fontsize=8)
    ax[0].set_ylabel("implied convenience yield, ¢/MMBtu per month")
    ax[0].set_title(f"Implied convenience yield by month pair\n(c = {100 * c_summer:.1f}¢, r = 2%; dot = median, bar = 5–95%)", fontsize=10)
    ax[0].text(8.6, 52, "below 0: contango\nabove summer carry", fontsize=8, color="#555")
    style(ax[0])
    a2 = ay.dropna()
    ax[1].scatter(a2.dev, 100 * a2.med, s=60, color=ORANGE, edgecolor="white", lw=1.5, zorder=3)
    for yr, r in a2.iterrows():
        ax[1].annotate(str(yr), (r.dev, 100 * r.med), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax[1].set_xlabel("storage minus 5-yr average, Bcf (mid-Sep to mid-Nov mean)")
    ax[1].set_ylabel("median front-pair contango, ¢/MMBtu per month")
    ax[1].set_title(f"Autumn front-month contango vs storage surplus\n(Spearman ρ = {rho:.2f}, 2013–2021)", fontsize=10)
    style(ax[1])
    fig.tight_layout(); fig.savefig(FIG / "curve_consistency_carry.png", dpi=150); plt.close(fig)

    # Fig 2: where kinks concentrate
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
    w = 0.38
    t = f13.groupby(["cm", "storage"], observed=True).abs_bp.mean().unstack()
    ax[0].bar(xs - w / 2, t["tight (<-150 Bcf)"], w - 0.04, color=ORANGE, label="storage tight (< −150 Bcf vs 5-yr)")
    ax[0].bar(xs + w / 2, t["loose (>+150 Bcf)"], w - 0.04, color=BLUE, label="storage loose (> +150 Bcf)")
    ax[0].set_xticks(xs, MONTHS)
    ax[0].set_ylabel("mean |fly kink|, bp of price")
    ax[0].set_title("Fitted-curve fly kink by centre contract month, 2013–2021", fontsize=10)
    ax[0].legend(frameon=False, fontsize=8)
    style(ax[0])
    tt = fl.groupby("tau")[["abs_bp", "absb_bp"]].mean()
    ax[1].plot(tt.index, tt.abs_bp, "-o", color=BLUE, lw=2, ms=5, label="fitted-curve fly kink")
    ax[1].plot(tt.index, tt.absb_bp, "-o", color=ORANGE, lw=2, ms=5, label="butterfly vs trailing norm")
    ax[1].set_xlabel("months to delivery of centre contract")
    ax[1].set_ylabel("mean |kink|, bp")
    ax[1].set_title("Kink size by tenor, 2011–2021", fontsize=10)
    ax[1].legend(frameon=False, fontsize=8)
    style(ax[1])
    fig.tight_layout(); fig.savefig(FIG / "curve_consistency_kinks.png", dpi=150); plt.close(fig)

    # Fig 3: cumulative net P&L of the selected rules
    fig, ax = plt.subplots(figsize=(9, 4))
    for sig, col, nm in (("fA", BLUE, "fitted curve (var A)"), ("fbench", ORANGE, "butterfly benchmark")):
        j = curves[sig]
        s = (j.gross - 160).groupby(j.date).sum().cumsum() / 1e3
        r = sel.loc[sig]
        ax.step(s.index, s.values, where="post", color=col, lw=2,
                label=f"{nm}: |kink| > {r.thr_bp:.0f} bp, hold {int(r.h)} d")
    ax.axvspan(pd.Timestamp("2018-01-01"), pd.Timestamp("2021-12-31"), color="#f1f0ec", zorder=0)
    ax.text(pd.Timestamp("2018-03-01"), ax.get_ylim()[1] * 0.9, "out of sample", fontsize=9, color="#555")
    ax.axhline(0, color=GREY, lw=1)
    ax.set_ylabel("cumulative net P&L, $000 (1 fly per signal-day, 4 ticks/leg)")
    ax.set_title("Selected butterfly rules (chosen on 2011–2017)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax)
    fig.tight_layout(); fig.savefig(FIG / "curve_consistency_trade.png", dpi=150); plt.close(fig)
    print("\nvariants in trade grid:", len(g[g.period == "2011-17"]))


if __name__ == "__main__":
    sys.exit(main())
