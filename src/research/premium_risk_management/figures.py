"""Figures for premium_risk_management (run after run.py).

premium_risk_management_frontier.png : premium kept vs worst-drawdown cut, every variant, with the
                                       position-scaling line and the Pareto frontier.
premium_risk_management_episodes.png : cumulative net P&L through the 2013-14 and 2018-19 drawdowns.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.research.premium_risk_management import data as D

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
FAM = {  # fixed categorical order
    "a_stop": ("Stop-loss (next-settle exit)", "#2a78d6", "o"),
    "a0_stop": ("Stop-loss (same-settle exit)", "#2a78d6", "o"),
    "b_trail": ("Trailing stop", "#eb6834", "s"),
    "c_vol": ("Volatility trigger", "#1baf7a", "^"),
    "d_phys": ("Physical (PHYS-M2) trigger", "#eda100", "D"),
    "d2_level": ("Spread-level trigger", "#e87ba4", "v"),
    "e_stagger": ("Staggered / delayed entry", "#008300", "P"),
    "f_hedge": ("Hedge (next-year H-J / outright)", "#6250d6", "X"),
    "g_cal": ("Calendar flat window", "#e34948", "h"),
}


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def frontier(R, sample, ax, title):
    x = R[(R["sample"] == sample) & (R.family != "base")]
    lim = (-0.45, 1.0, -0.6, 1.9)
    ax.plot([0, 1], [1, 0], color=INK2, lw=1.2, ls="--", zorder=1)
    ax.text(0.55, 0.30, "scaling line:\nhold fewer spreads", color=INK2, fontsize=7, va="top", ha="right")
    for fam, (lab, col, mk) in FAM.items():
        g = x[x.family == fam]
        if g.empty:
            continue
        hollow = fam == "a0_stop"
        ax.scatter(g.dd_cut.clip(lim[0], lim[1]), g.kept.clip(lim[2], lim[3]), s=34, marker=mk,
                   facecolor="none" if hollow else col, edgecolor=col, lw=1.2 if hollow else 0.6,
                   alpha=0.9, label=lab, zorder=3)
    # Pareto frontier among implementable (next-settle) variants
    y = x[x.family != "a0_stop"].sort_values(["dd_cut", "kept"], ascending=[False, False])
    pts, best = [], -np.inf
    for _, r in y.iterrows():
        if r.kept > best + 1e-9:
            pts.append((r.dd_cut, r.kept, r.variant))
            best = r.kept
    P = np.array([(a, b) for a, b, _ in pts])
    ax.step(P[:, 0], P[:, 1], where="pre", color=INK, lw=1.6, zorder=2)
    for a, b, v in pts:
        if a > 0.3:
            ax.annotate(v, (a, b), xytext=(4, 4), textcoords="offset points", fontsize=6.5, color=INK)
    ax.scatter([0], [1], s=60, color=INK, zorder=4)
    ax.annotate("base", (0, 1), xytext=(-26, 4), textcoords="offset points", fontsize=7.5, color=INK)
    ax.set_xlim(lim[0] - 0.03, lim[1] + 0.03)
    ax.set_ylim(lim[2] - 0.05, lim[3] + 0.05)
    ax.set_xlabel("Worst drawdown cut vs base (share)", color=INK, fontsize=9)
    ax.set_ylabel("Premium kept (mean P&L / base mean)", color=INK, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    style(ax)


def main():
    R = pd.read_csv(D.CACHE / "summary.csv")
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.6), facecolor=SURF)
    frontier(R, "all", axs[0], "All 12 winters 2010-11 to 2021-22 (2021-22 to 31 Dec)")
    frontier(R, "test", axs[1], "Check period 2017-18 to 2021-22")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle("Price of tail protection: premium kept vs worst drawdown cut, 123 variants "
                 "(black step = frontier of next-settle variants; values clipped to the axes)",
                 color=INK, fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(D.FIG / "premium_risk_management_frontier.png", dpi=140, facecolor=SURF)

    res = pd.read_pickle(D.CACHE / "all_res.pkl")
    W = D.winters()
    show = [("base", INK, "-"), ("stop X=3000 re=4", "#2a78d6", "-"), ("stop0 X=2000 re=4", "#2a78d6", ":"),
            ("vol k=1.5 re=1", "#1baf7a", "-"), ("phys m=2 re=1", "#eda100", "-"),
            ("stagger n=1 off=2w", "#008300", "-"), ("hedge N h=1.5xbeta", "#6250d6", "-")]
    labels = {"base": "Base (short 1 H-J)", "stop X=3000 re=4": "Stop $3k, re-enter 4w (train pick)",
              "stop0 X=2000 re=4": "Stop $2k, re-enter 4w, same-settle exit",
              "vol k=1.5 re=1": "Vol > 1.5x entry level (train pick)", "phys m=2 re=1": "PHYS-M2 ES > 2x median",
              "stagger n=1 off=2w": "Enter 2 weeks late (train pick)", "hedge N h=1.5xbeta": "Hedge 2.3x next-year H-J"}
    eps = [(2014, "2013-12-01", "2014-02-26", "Winter 2013-14 (polar vortex)"),
           (2019, "2018-11-01", "2018-12-31", "Winter 2018-19 (Nov 2018 spike)")]
    fig, axs = plt.subplots(2, 2, figsize=(13, 7.2), facecolor=SURF, gridspec_kw=dict(height_ratios=[3, 1.3]),
                            sharex="col")
    for j, (Y, a, b, t) in enumerate(eps):
        ax = axs[0, j]
        for v, col, ls in show:
            if Y not in res[v]:
                continue
            c = res[v][Y].pnl.cumsum().loc[:b]
            c = c[c.index >= a]
            ax.plot(c.index, c.values / 1000, color=col, ls=ls, lw=2.2 if v == "base" else 1.6,
                    zorder=5 if v == "base" else 3,
                    label=labels[v] if j == 1 else None)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(t, color=INK, fontsize=10, loc="left")
        ax.set_ylabel("Cumulative net P&L since entry ($k)", color=INK, fontsize=9)
        style(ax)
        s = W[Y].S.loc[a:b]
        axs[1, j].plot(s.index, s.values, color=INK2, lw=1.4)
        axs[1, j].set_ylabel("H-J spread ($/MMBtu)", color=INK, fontsize=9)
        style(axs[1, j])
        axs[1, j].tick_params(axis="x", rotation=30)
    fig.legend(*axs[0, 1].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(D.FIG / "premium_risk_management_episodes.png", dpi=140, facecolor=SURF)
    print("figures written")


if __name__ == "__main__":
    main()
