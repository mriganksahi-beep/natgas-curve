"""Figures for v4_options (aggregates and paths only; no price tables)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.research.v4_options import data as D

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # fixed categorical order


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _fig(nr, nc, w, h):
    fig, axs = plt.subplots(nr, nc, figsize=(w, h), facecolor=SURF, squeeze=False)
    for a in axs.ravel():
        _style(a)
    return fig, axs


def fig_vrp(pw):
    fig, axs = _fig(1, 2, 10, 3.8)
    for ax, inst, title, unit in ((axs[0, 0], "HJ", "March−April spread (G4X, Bachelier)", "$/MMBtu/√yr"),
                                  (axs[0, 1], "H", "March outright (ON, Black-76)", "annualised")):
        x = pw.loc[inst].dropna(subset=["rv_life"])
        xs = np.arange(len(x))
        for i, (iv, rv) in enumerate(zip(x.iv, x.rv_life)):
            ax.plot([i - 0.1, i + 0.1], [iv, rv], color=GRID, lw=2, zorder=1)
        ax.scatter(xs - 0.1, x.iv, s=40, color=C[0], label="implied (season mean)", zorder=3, edgecolor=SURF, lw=1.5)
        ax.scatter(xs + 0.1, x.rv_life, s=40, color=C[1], label="realised over remaining option life", zorder=3,
                   edgecolor=SURF, lw=1.5)
        labs = [D.wl(y) + ("\n(thin)" if inst == "HJ" and y in (2013, 2014) else "") for y in x.index]
        ax.set_xticks(xs, labs, fontsize=7)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_ylabel(unit, fontsize=8, color=INK2)
        if inst == "HJ":
            ax.set_yscale("log")
    axs[0, 0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle("Implied vs realised vol, entry (early Nov) to March option expiry, by winter", fontsize=10,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(D.FIG / "v4_options_vrp.png", dpi=150, facecolor=SURF)
    plt.close(fig)


def fig_episodes(pa, ep):
    names = list(ep.episode)
    fig, axs = _fig(2, len(names), 12, 5.2)
    for j, nm in enumerate(names):
        p = pa[pa.episode == nm]
        a0, a1 = axs[0, j], axs[1, j]
        quoted = p.nv_best.where(p.inside.fillna(False) | p.nv_hj.notna())
        a0.plot(p.k, p.nv_best, color=C[0], lw=1, ls=":", label="ATM normal vol (extrapolated smile)")
        a0.plot(p.k, quoted, color=C[0], lw=2, label="ATM normal vol (quoted strikes)", marker="o", ms=3)
        a0.axvline(0, color=INK2, lw=0.8, ls="--")
        a0.set_title(nm, fontsize=9, color=INK, loc="left")
        base = p.S[p.k == -1]
        if len(base):
            a1.plot(p.k, (p.S - base.iloc[0]) * D.MULT / 1000, color=C[1], lw=2)
        a1.axvline(0, color=INK2, lw=0.8, ls="--")
        a1.set_xlabel("trading days from blow-up start (t0)", fontsize=8, color=INK2)
    axs[0, 0].set_ylabel("H−J implied normal vol\n$/MMBtu/√yr", fontsize=8, color=INK2)
    axs[1, 0].set_ylabel("H−J change since t0−1\n$k per spread (short loses)", fontsize=8, color=INK2)
    axs[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Implied vol rises with the spread, not ahead of it", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(D.FIG / "v4_options_episodes.png", dpi=150, facecolor=SURF)
    plt.close(fig)


def fig_protection(paths, s3):
    fig, axs = _fig(1, 3, 13, 3.8)
    vs = [("naked", "naked short"), ("half", "half size"), ("call+0.25", "short + call ATM+$0.25"),
          ("call+0.50", "short + call ATM+$0.50")]
    for ax, Y in ((axs[0, 0], 2014), (axs[0, 1], 2019)):
        p = paths[Y]
        for i, (vn, lab) in enumerate(vs):
            ax.plot(np.arange(len(p)), p[vn] / 1000, color=C[i], lw=2 if i != 1 else 1.5, label=lab)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(f"Winter {D.wl(Y)}: mark-to-market P&L", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("trading days since entry", fontsize=8, color=INK2)
        ax.set_ylabel("$k per spread", fontsize=8, color=INK2)
    axs[0, 0].legend(fontsize=7, frameon=False, loc="lower left")
    ax = axs[0, 2]
    q = s3[s3.window == "all"].set_index("variant")
    ax.plot([0, 1], [1, 0], color=INK2, lw=1, ls="--")
    ax.text(0.02, 0.04, "dashed: kept + cut = 1 (what scaling the size gives)", fontsize=7, color=INK2)
    order = ["naked", "half", "call+0.25", "call+0.50", "cs+0.25/+1.00"]
    labs = {"naked": "naked", "half": "half", "call+0.25": "call +0.25", "call+0.50": "call +0.50",
            "cs+0.25/+1.00": "call spread +0.25/+1.00"}
    cols = {"naked": C[0], "half": C[1], "call+0.25": C[2], "call+0.50": C[3], "cs+0.25/+1.00": INK2}
    for vn in order:
        ax.scatter(q.at[vn, "cut"], q.at[vn, "kept"], s=50, color=cols[vn], edgecolor=SURF, lw=1.5, zorder=3)
        left = vn == "cs+0.25/+1.00"
        ax.annotate(labs[vn], (q.at[vn, "cut"], q.at[vn, "kept"]), xytext=(-6 if left else 6, 6),
                    textcoords="offset points", fontsize=7, color=INK, ha="left" if not left else "left")
    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("cut in worst drawdown (vs naked)", fontsize=8, color=INK2)
    ax.set_ylabel("share of mean P&L kept", fontsize=8, color=INK2)
    ax.set_title("All winters 2012-13..2021-22", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(D.FIG / "v4_options_protection.png", dpi=150, facecolor=SURF)
    plt.close(fig)
