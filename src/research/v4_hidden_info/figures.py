"""Figures for v4_hidden_info (reads the cached tables only)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import CACHE, FIG

COL = {"front": "#2a78d6", "second": "#eb6834", "hj": "#1baf7a"}
LAB = {"front": "Front month", "second": "Second month", "hj": "March−April"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig_predict():
    t = pd.read_csv(CACHE / "t1_daily.csv")
    t = t[(t["sample"] == "all") & t.x.str.startswith("P_")]
    ys = [("FN1", "Next 00Z run\n(d+1)"), ("FN5", "Next five runs\n(d+1..d+5)"),
          ("real_tot", "Realised weather\nvs d forecast"), ("real_tot_n1", "Realised weather\nvs d+1 forecast")]
    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=150)
    for j, ins in enumerate(COL):
        for i, (y, _) in enumerate(ys):
            r = t[(t.ins == ins) & (t.y == y)].iloc[0]
            x = i + (j - 1) * 0.22
            ax.plot([x, x], [r.ci_lo, r.ci_hi], color=COL[ins], lw=2, solid_capstyle="round")
            ax.plot(x, r.coef_sd, "o", ms=8, color=COL[ins], mec="white", mew=2, label=LAB[ins] if i == 0 else None)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(range(len(ys)), [l for _, l in ys], fontsize=9, color=INK)
    ax.set_ylabel("HDD per 1 s.d. of premium_part", fontsize=9, color=MUTED)
    ax.set_title("A day's unexplained move predicts the next GEFS runs, but not weather beyond them",
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    style(ax)
    fig.text(0.01, 0.01, "Nov–Feb 2012-13..Dec 2021, controls: GEFS levels at d (and news at d, d−1 for revisions). "
             "Bars: 95% winter-block bootstrap CI.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "v4_hidden_info_predict.png")


def fig_share():
    d = pd.read_csv(CACHE / "t3_dollar_share.csv")
    r = pd.read_csv(CACHE / "t2_reversal.csv")
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.0), dpi=150, gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]
    sets = [("all_big", "All big days"), ("up", "Up (cold scare)"), ("down", "Down")]
    for j, ins in enumerate(COL):
        for i, (s, _) in enumerate(sets):
            q = d[(d.ins == ins) & (d.set == s)].iloc[0]
            x = i + (j - 1) * 0.25
            ax.bar(x, q.share_later_gefs, width=0.22, color=COL[ins], label=LAB[ins] if i == 0 else None)
            ax.plot([x, x], [q.F_lo, q.F_hi], color=INK, lw=1)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_ylim(-0.5, 0.55)
    ax.set_xticks(range(3), [l for _, l in sets], fontsize=9, color=INK)
    ax.set_ylabel("share of premium move justified", fontsize=9, color=MUTED)
    ax.set_title("Top-decile premium days: share later justified\nby GEFS revisions d+1..d+5", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    style(ax)
    ax = axs[1]
    comps = [("P", "Raw\npremium_part"), ("pure_premium_DIAG", "Pure premium\n(diagnostic)"),
             ("anticipated", "Anticipated\nnews (diag.)")]
    for j, ins in enumerate(COL):
        for i, (c, _) in enumerate(comps):
            q = r[(r.ins == ins) & (r.h == 5) & (r.x == c)].iloc[0]
            se = abs(q.slope / q.t_hac)
            x = i + (j - 1) * 0.25
            ax.bar(x, q.slope, width=0.22, color=COL[ins])
            ax.plot([x, x], [q.slope - 1.96 * se, q.slope + 1.96 * se], color=INK, lw=1)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(range(3), [l for _, l in comps], fontsize=9, color=INK)
    ax.set_ylabel("next-5-day move per unit of component", fontsize=9, color=MUTED)
    ax.set_title("Reversal over d+1..d+5\n(−1 = full reversal)", fontsize=10, color=INK, loc="left")
    style(ax)
    fig.text(0.01, 0.01, "Bars: 95% CIs (left: winter bootstrap; right: HAC, 5 lags). "
             "Diagnostic components use future GEFS runs and realised weather: not tradable.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "v4_hidden_info_share_reversal.png")


if __name__ == "__main__":
    fig_predict()
    fig_share()
