"""Figures for fear_vs_physics. Run after run.py (uses its caches)."""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import analysis as A
from . import data as D

warnings.filterwarnings("ignore")
FIG = D.ROOT / "reports/agents/figures"
BLUE, ORANGE, AQUA, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9893", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GREY, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6})


def fig_decomp():
    t = pd.read_csv(D.CACHE / "drift_tables.csv")
    t = t[t.model == "decomp"]
    terms = [("S_fear", "Fear part of S"), ("S_phys", "Physics part of S"), ("S_time", "Time-to-expiry part"),
             ("S_res", "Unexplained part")]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for k, (y, col, lab) in enumerate((("dS_exp", BLUE, "Drift to expiry"), ("dS_4w", ORANGE, "Drift over 4 weeks"))):
        for i, (term, _) in enumerate(terms):
            r = t[(t.target == y) & (t.term == term)].iloc[0]
            yy = i + (k - 0.5) * 0.28
            ax.plot([r.lo, r.hi], [yy, yy], color=col, lw=2, solid_capstyle="round")
            ax.plot(r.coef, yy, "o", color=col, ms=7, mec="white", mew=1.5, label=lab if i == 0 else None)
            ax.text(r.hi + 0.06, yy, f"{r.coef:+.2f}", va="center", fontsize=8, color=INK)
    ax.axvline(0, color=MUTED, lw=1)
    ax.axvline(-1, color=MUTED, lw=1, ls="--")
    ax.text(-1, -0.75, "−1: decays fully", ha="center", fontsize=8, color=MUTED)
    ax.text(0, -0.75, "0: retained", ha="center", fontsize=8, color=MUTED)
    ax.set_yticks(range(len(terms)), [l for _, l in terms])
    ax.set_ylim(len(terms) - 0.5, -1.0)
    ax.set_xlabel("Coefficient of later spread change on each part of today's spread (95% winter-block CI)")
    ax.set_title("Which part of the March−April spread decays? 2010-11 to 2021, 167–171 weeks", loc="left", fontsize=10)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fear_vs_physics_decomposition.png", dpi=150)
    plt.close(fig)


def fig_episodes():
    r = pd.read_pickle(D.CACHE / "panel_signals.pkl")
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.2), gridspec_kw={"height_ratios": [2.2, 1]}, sharex="col")
    for j, w in enumerate(("2013-14", "2018-19")):
        q = r[r.winter == w]
        ax = axes[0, j]
        on = q.FX >= 0.10
        for d in q.obs[on]:
            ax.axvspan(d - pd.Timedelta(days=3), d + pd.Timedelta(days=3), color=AQUA, alpha=0.18, lw=0)
        ax.plot(q.obs, q.S, color=BLUE, lw=2, marker="o", ms=4, label="Spread S (decision settle)")
        ax.plot(q.obs, q.PP, color=ORANGE, lw=2, marker="o", ms=4, label="Physics-implied spread (walk-forward)")
        ax.plot(q.obs, q.TP if "TP" in q else q.S - q.TX, color=GREY, lw=1.2, ls="--", label="Time-only fit")
        ax.set_title(f"Winter {w}", loc="left", fontsize=10)
        ax.set_ylabel("$/MMBtu")
        if j == 0:
            ax.legend(frameon=False, fontsize=7.5, loc="upper left")
        ax2 = axes[1, j]
        ax2.plot(q.obs, q.p_tail, color=INK, lw=1.6, marker="o", ms=3)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Analog P(end-Mar\n< 5-yr avg − 500)")
        ax2.tick_params(axis="x", rotation=30)
    fig.suptitle("Shaded weeks: rule fires (S − physics-implied ≥ $0.10). Jan 2014: physics kept pace with S until the late-Feb spike",
                 fontsize=8.5, x=0.01, ha="left", color=MUTED)
    fig.tight_layout()
    fig.savefig(FIG / "fear_vs_physics_episodes.png", dpi=150)
    plt.close(fig)


def fig_rules():
    r = pd.read_pickle(D.CACHE / "panel_signals.pkl")
    chk = r[r.Y.isin(A.CHECK_Y)]
    rules = [("Short every week", lambda d: pd.Series(True, index=d.index)), ("S > $0.15", lambda d: d.S > 0.15),
             ("Fear high & physics low\n(rv4≥0.02, P_tail≤0.5)", lambda d: (d.rv4 >= 0.02) & (d.p_tail <= 0.5)),
             ("Time-only excess ≥ $0.10", lambda d: d.TX >= 0.10),
             ("Excess over physics ≥ $0.10", lambda d: d.FX >= 0.10)]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True)
    for k, (h, title) in enumerate((("4w", "Hold 4 weeks"), ("exp", "Hold to expiry"))):
        ax = axes[k]
        for i, (nm, f) in enumerate(rules):
            s = A.summ(A.trades(chk, f(chk), h))
            col = AQUA if nm.startswith("Excess") else BLUE
            ax.barh(i, s["mean"], color=col, height=0.55)
            ax.plot([s["lo"], s["hi"]], [i, i], color=INK, lw=1.2)
            ax.plot(s["worst"], i, marker="v", color=ORANGE, ms=8, ls="none", label="Worst trade" if i == 0 else None)
            ax.text(max(s["hi"], 0) + 250, i, f"${s['mean']:,.0f} (n={s['trades']}, {s['winters']} w)", va="center", fontsize=7.5)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlim(-8500, 11500)
        if k == 0:
            ax.set_yticks(range(len(rules)), [n for n, _ in rules])
            ax.legend(frameon=False, loc="lower left", fontsize=7.5)
    ax.invert_yaxis()
    fig.supxlabel("Net $ per spread per trade (bar), 95% winter-block CI (line), worst trade (triangle); check 2017-18 to Dec 2021",
                  fontsize=8.5, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fear_vs_physics_rules.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_decomp()
    fig_episodes()
    fig_rules()
    print("figures written")
