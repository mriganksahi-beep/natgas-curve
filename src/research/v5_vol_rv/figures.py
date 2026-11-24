"""Three figures for reports/agents/v5_vol_rv.md (statistics only, no raw prices)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from . import data as D, trade as T
from .vrp import prep_base, TEN_LAB
from .stats import table

SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DIV = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})


def fig_vrp():
    b = prep_base()
    m = (b.groupby(["dm", "ten"], observed=True).gap_life.mean() * 100).unstack()[TEN_LAB]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
    im = ax[0].imshow(m.values, cmap=DIV, vmin=-8, vmax=8, aspect="auto")
    lab = "FGHJKMNQUVXZ"
    ax[0].set_yticks(range(12), [f"{lab[i]} ({i + 1:02d})" for i in range(12)])
    ax[0].set_xticks(range(len(TEN_LAB)), TEN_LAB)
    for i in range(12):
        for j in range(len(TEN_LAB)):
            v = m.values[i, j]
            if np.isfinite(v):
                ax[0].text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7, color=INK)
    ax[0].set_xlabel("time to option expiry")
    ax[0].set_title("IV minus realised vol over the option's life (vol pts)\nblue = IV rich (short-vol pays), red = IV cheap",
                    fontsize=9, loc="left", color=INK)
    fig.colorbar(im, ax=ax[0], fraction=0.04)
    t = table(b, "gap_life", ["stor", "dclass"], scale=100)
    regs = ["tight(<-150)", "normal", "loose(>150)"]
    for k, c in enumerate(["winter", "shoulder", "summer"]):
        q = t[t.dclass == c].set_index("stor").reindex(regs)
        x = np.arange(3) + (k - 1) * 0.25
        ax[1].errorbar(x, q["mean"], yerr=[q["mean"] - q.lo, q.hi - q["mean"]], fmt="o", ms=7, color=SER[k],
                       capsize=3, lw=2, label=f"{c} delivery")
    ax[1].axhline(0, color=INK2, lw=1)
    ax[1].set_xticks(range(3), ["storage tight\n(< -150 bcf vs prior yrs)", "normal", "loose\n(> +150 bcf)"])
    ax[1].set_ylabel("IV - life RV (vol pts), 95% gas-year bootstrap CI")
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].grid(axis="y", color=GRID)
    ax[1].set_title("By storage regime (dates 2012-2021)", fontsize=9, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(D.FIG / "v5_vol_rv_vrp.png", dpi=150)


def fig_trade():
    p = T.add_signals(pd.read_pickle(D.CACHE / "s3_positions_m.pkl"))
    fig, ax = plt.subplots(figsize=(10, 4.4))
    rng = np.random.default_rng(3)
    curves = []
    for _ in range(100):
        pp, _ = T.evaluate(p, "random", rng=rng)
        curves.append(T.daily_series(pp).cumsum() * 100)
    R = pd.concat(curves, axis=1).ffill()
    ax.fill_between(R.index, R.quantile(0.05, axis=1), R.quantile(0.95, axis=1), color=GRID, label="random ranking 5-95%")
    names = {"S_ewma": "signal: IV - EWMA RV (rank, L/S)", "short_all": "short all (unconditional VRP)",
             "term": "term rule: short front 2 / long back 2", "storage_sign": "storage sign (post-hoc)"}
    for k, (r, lab) in enumerate(names.items()):
        pp, _ = T.evaluate(p, r)
        c = T.daily_series(pp).cumsum() * 100
        ax.plot(c.index, c.values, color=SER[k], lw=2, label=lab)
        ax.text(c.index[-1], c.values[-1], f" {c.values[-1]:.0f}", color=INK2, fontsize=8, va="center")
    pp, _ = T.evaluate(p, "S_ewma", c=0.0, fut=0.0)
    g = T.daily_series(pp).cumsum() * 100
    ax.plot(g.index, g.values, color=SER[0], lw=1.2, ls="--", label="signal, gross (no costs)")
    for m in ("2014-01", "2018-11", "2021-02"):
        a = pd.Timestamp(m + "-01")
        ax.axvspan(a, a + pd.offsets.MonthEnd(0), color="#e34948", alpha=0.15, lw=0)
    ax.axvline(pd.Timestamp("2017-01-01"), color=INK2, lw=1, ls=":")
    ax.text(pd.Timestamp("2017-01-15"), ax.get_ylim()[1] * 0.0 + 5, "choose | check", color=INK2, fontsize=8)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("cumulative net P&L, vol pts per unit vega per side")
    ax.set_title("Monthly vega-neutral delta-hedged straddle books, 1.5% option half-spread + 2-tick hedges "
                 "(shaded: Jan 2014, Nov 2018, Feb 2021)", fontsize=9, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.grid(axis="y", color=GRID)
    fig.tight_layout()
    fig.savefig(D.FIG / "v5_vol_rv_trade.png", dpi=150)


def fig_term():
    t = pd.read_csv(D.CACHE / "s4_term.csv", parse_dates=["date"]).set_index("date")
    w = t.groupby(t.index.to_period("W")).tail(1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for k, (win, sel) in enumerate((("2012-16", w.index.year <= 2016), ("2017-21", w.index.year >= 2017))):
        q = w[sel].dropna(subset=["ws", "ws_drv"])
        ax[0].scatter(q.ws, q.ws_drv, s=10, color=SER[k], alpha=0.6, label=win, edgecolor="none")
    ax[0].axhline(0, color=INK2, lw=0.8)
    ax[0].set_xlabel("Jan - Apr ATM IV spread, same winter (vol pts)")
    ax[0].set_ylabel("log RV20(Jan) - log RV20(Apr), next 20 days")
    ax[0].set_title("Term spread predicts relative realised vol", fontsize=9, loc="left", color=INK)
    ax[0].legend(frameon=False, fontsize=8)
    t["s"] = pd.cut(t.sdev, [-1e9, -150, 150, 1e9], labels=["tight", "normal", "loose"])
    m = w.join(t["s"], rsuffix="_").groupby("s", observed=True)[["ws_pnl", "fb_pnl"]].mean()
    x = np.arange(len(m))
    ax[1].bar(x - 0.18, m.ws_pnl, 0.34, color=SER[0], label="long Jan / short Apr")
    ax[1].bar(x + 0.18, m.fb_pnl, 0.34, color=SER[1], label="long front / short 12m")
    ax[1].axhline(0, color=INK2, lw=0.8)
    ax[1].set_xticks(x, m.index)
    ax[1].set_ylabel("next-20d gross dh P&L, vol pts (vega-neutral)")
    ax[1].set_title("...but not the vega-neutral pair P&L; storage regime", fontsize=9, loc="left", color=INK)
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(D.FIG / "v5_vol_rv_term.png", dpi=150)


if __name__ == "__main__":
    fig_vrp(); fig_trade(); fig_term()
