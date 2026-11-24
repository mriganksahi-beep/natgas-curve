"""Figures for v5_weather_uncertainty (static PNG; tables in the report are the table view)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import data as D

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig_forecast_event():
    T = pd.read_csv(D.CACHE / "q1_results.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURF)
    a = ax[0]
    _style(a)
    sub = T[(T.comp.isin(["A", "C", "D"])) & (T.win != "all")]
    labels, xs = [], 0
    comps = {"A": ("HAR -> +disagreement", C[0]), "C": ("HAR+IV -> +disagreement", C[1]),
             "D": ("HAR+IV+level+storage -> +disagreement", C[2])}
    for wi, win in enumerate(["choose", "check"]):
        for h in (5, 10, 20):
            for leg in (1, 2):
                for k, (cid, (lab, col)) in enumerate(comps.items()):
                    r = sub[(sub.win == win) & (sub.h == h) & (sub.leg == leg) & (sub.comp == cid)].iloc[0]
                    x = xs + (k - 1) * 0.25
                    a.plot([x, x], [r.g_lo * 100, r.g_hi * 100], color=col, lw=1.2, alpha=0.6)
                    a.plot(x, r.gain * 100, "o", ms=5, color=col, mec=SURF, mew=1,
                           label=lab if (wi == 0 and h == 5 and leg == 1) else None)
                labels.append(f"{h}d L{leg}")
                xs += 1
        xs += 0.6
    pos = [i + (0.6 if i >= 6 else 0) for i in range(12)]
    a.set_xticks(pos)
    a.set_xticklabels(labels, rotation=60, fontsize=7)
    a.axhline(0, color=INK2, lw=0.8)
    a.axvline(5.8, color=GRID, lw=1)
    a.text(2.5, a.get_ylim()[1] * 0.92, "choose 2011-16", ha="center", fontsize=8, color=INK2)
    a.text(9.1, a.get_ylim()[1] * 0.92, "check 2017-21", ha="center", fontsize=8, color=INK2)
    a.set_ylabel("Out-of-sample R² gain (pct pts)", fontsize=8, color=INK2)
    a.set_title("Q1. Disagreement adds little to realised-vol forecasts", fontsize=9, color=INK, loc="left")
    a.legend(fontsize=7, frameon=False, loc="lower left")
    b = ax[1]
    _style(b)
    for k, (var, lab) in enumerate([("unc_n_w2", "jump in 6-10 day disagreement"),
                                     ("unc_n_w3", "jump in 11-15 day disagreement"),
                                     ("unc_n_w23", "jump in mean of the two")]):
        r = pd.read_csv(D.CACHE / f"q2_event_{var}.csv")
        b.plot(r.offset, r.iv_excess, color=C[k], lw=2, label=lab)
        if var == "unc_n_w23":
            b.fill_between(r.offset, r.lo, r.hi, color=C[k], alpha=0.12, lw=0)
    b.axhline(0, color=INK2, lw=0.8)
    b.axvline(0, color=INK2, lw=0.8, ls=":")
    b.set_xlabel("Trading days from the run (0 = first settle after the jump)", fontsize=8, color=INK2)
    b.set_ylabel("Front ATM IV vs same-month normal (vol pts)", fontsize=8, color=INK2)
    b.set_title("Q2. IV does not reprice on disagreement jumps (top 10%, declustered)", fontsize=9, color=INK, loc="left")
    b.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(D.FIG / "v5_weather_uncertainty_forecast_event.png", dpi=150, facecolor=SURF)


def fig_trade():
    daily = pd.read_pickle(D.CACHE / "q3_daily.pkl")
    R = pd.read_csv(D.CACHE / "q3_random.csv")
    fig, a = plt.subplots(figsize=(10, 4.2), facecolor=SURF)
    _style(a)
    rules = [("always_long", "Always long (benchmark)"), ("storage_tight", "Long if storage tight (vol_rv)"),
             ("U23_high", "Long if disagreement high"), ("tight_x_U", "Long if tight AND disagreement high"),
             ("gap_low", "Long if IV cheap vs HAR forecast")]
    for k, (r, lab) in enumerate(rules):
        d = daily[(r, "1.5%")].sort_values("date")
        a.plot(d.date, d.net.cumsum(), color=C[k], lw=2 if k else 1.5, label=lab)
    a.axhline(0, color=INK2, lw=0.8)
    a.axvline(pd.Timestamp("2017-01-01"), color=INK2, lw=0.8, ls=":")
    a.text(pd.Timestamp("2017-02-01"), a.get_ylim()[0] * 0.92, "check window ->", fontsize=8, color=INK2)
    a.set_ylabel("Cumulative net P&L, vol pts per unit vega", fontsize=8, color=INK2)
    a.set_title("Q3. Weekly-switched long front straddles, delta-hedged daily, net of 1.5% + 2-tick costs (Oct-Apr only)",
                fontsize=9, color=INK, loc="left")
    a.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(D.FIG / "v5_weather_uncertainty_trade.png", dpi=150, facecolor=SURF)


def fig_cells():
    W = pd.read_csv(D.CACHE / "q4_contrib.csv")
    Q = pd.read_csv(D.CACHE / "q4_cells.csv")
    fig, a = plt.subplots(figsize=(8, 4), facecolor=SURF)
    _style(a)
    order = [("loose", "U_low"), ("loose", "U_high"), ("normal", "U_low"), ("normal", "U_high"),
             ("tight", "U_low"), ("tight", "U_high")]
    for i, (s, u) in enumerate(order):
        col = C[0] if u == "U_low" else C[1]
        g = W[(W.storage == s) & (W.unc == u)]
        per = g["sum"] / g.weeks
        a.scatter(np.full(len(per), i) + np.linspace(-0.12, 0.12, len(per)), per, s=18, color=col, alpha=0.5,
                  edgecolor=SURF, lw=0.8)
        q = Q[(Q.storage == s) & (Q.unc == u)].iloc[0]
        a.plot([i, i], [q.lo, q.hi], color=col, lw=2)
        a.plot(i, q["mean"], "D", color=col, ms=7, mec=SURF)
        for _, r in g.iterrows():
            if abs(r["sum"] / r.weeks) > 3.5:
                a.annotate(f"{int(r.gy) - 1}-{str(int(r.gy))[2:]}", (i + 0.15, r["sum"] / r.weeks), fontsize=7, color=INK2)
    a.set_xticks(range(6))
    a.set_xticklabels([f"{s}\n{'low' if u == 'U_low' else 'high'} disagr." for s, u in order], fontsize=8)
    a.axhline(0, color=INK2, lw=0.8)
    a.set_ylabel("Weekly long-straddle gross P&L\n(vol pts per unit vega)", fontsize=8, color=INK2)
    a.set_title("Q4. Storage x disagreement: diamonds = pooled mean [95% CI]; dots = gas years", fontsize=9,
                color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(D.FIG / "v5_weather_uncertainty_cells.png", dpi=150, facecolor=SURF)


if __name__ == "__main__":
    fig_forecast_event()
    fig_trade()
    fig_cells()
