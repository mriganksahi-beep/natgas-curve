"""Figures for v4_premium_shocks (aggregates only; no raw licensed prices)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .common import CACHE, FIG  # noqa: E402

C = {"prem": "#2a78d6", "naive": "#eb6834", "news": "#1baf7a"}
INK, MUTED = "#222222", "#777777"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(MUTED); ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color="#e6e6e6", lw=0.6)
    ax.axhline(0, color=MUTED, lw=0.8)


def fig_slopes():
    a = pd.read_csv(CACHE / "step1_reversal.csv")
    panels = [("daily", "second", "Second month, daily split"), ("daily", "hj", "H−J, daily split"),
              ("weekly", "second", "Second month, weekly split"), ("weekly", "hj", "H−J, weekly split")]
    fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharey=False)
    for ax, (fq, ins, title) in zip(axs.flat, panels):
        _style(ax)
        for j, (wn, ls, mk) in enumerate((("choose", "-", "o"), ("check", "--", "s"))):
            for key, spec, col, lab in (("prem", "split", C["prem"], "premium part"),
                                        ("move", "naive", C["naive"], "raw move (naive)"),
                                        ("prem", "split_ctrl", C["prem"], None)):
                d = a[(a.freq == fq) & (a.ins == ins) & (a.window == wn) & (a.spec == spec)].dropna(subset=[f"b_{key}"])
                if d.empty:
                    continue
                x = d.h.to_numpy() * (1 + 0.04 * (j - 0.5))
                lw = 2 if spec != "split_ctrl" else 1
                alpha = 1 if spec != "split_ctrl" else 0.45
                lbl = f"{lab}, {wn}" if lab else (f"premium | future-news ctrl, {wn}")
                ax.errorbar(x, d[f"b_{key}"], yerr=[d[f"b_{key}"] - d[f"lo_{key}"], d[f"hi_{key}"] - d[f"b_{key}"]],
                            color=col, ls=ls, marker=mk, ms=4, lw=lw, alpha=alpha, capsize=2, label=lbl)
        ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10, 20]); ax.set_xticklabels(["1", "2", "5", "10", "20"])
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("horizon h (trading days)", fontsize=8, color=MUTED)
        ax.set_ylabel("slope of future move on today's part", fontsize=8, color=MUTED)
        ax.set_ylim(-1.2, 0.8)
    axs[0, 0].legend(fontsize=6.5, frameon=False, loc="lower left", ncol=2)
    fig.suptitle("Reversal slopes (−1 = full reversal, 0 = persistence); 95% winter-bootstrap CIs", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "v4_premium_shocks_slopes.png", dpi=130)
    plt.close(fig)


def fig_pnl():
    t = pd.read_csv(CACHE / "step3_trades.csv", parse_dates=["date"])
    t["net2"] = t.gross - t.ins.map({"second": 40, "hj": 80})
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, ins, pairs in ((axs[0], "second", (("WPREM@WPREM", "fade weekly-split premium (k5, z1.5, h5)", C["prem"], "-"),
                                               ("RAW@WPREM", "naive fade, same parameters", C["naive"], "-"),
                                               ("PREM@PREM", "fade daily-split premium (k3, z1.0, h10)", C["prem"], ":"),
                                               ("RAW@PREM", "naive fade, same parameters", C["naive"], ":"))),
                           (axs[1], "hj", (("WPREM@WPREM", "fade weekly-split premium (k3, z1.5, h10)", C["prem"], "-"),
                                           ("RAW@WPREM", "naive fade, same parameters", C["naive"], "-")))):
        _style(ax)
        for rule, lab, col, ls in pairs:
            d = t[(t.ins == ins) & (t.rule == rule)].sort_values("date")
            ax.plot(np.arange(len(d)), d.net2.cumsum() / 1000, color=col, ls=ls, lw=2, label=lab)
        ax.axvline(t[(t.ins == ins) & (t.rule == pairs[0][0]) & (t.W <= 2017)].shape[0] - 0.5, color=MUTED, lw=0.8, ls="--")
        ax.text(0.02, 0.95, "choose | check →", transform=ax.transAxes, fontsize=7, color=MUTED)
        ax.set_title({"second": "Second-month outright, 1 contract", "hj": "H−J spread, 1 spread"}[ins],
                     fontsize=10, color=INK, loc="left")
        ax.set_xlabel("trade number (chronological)", fontsize=8, color=MUTED)
        ax.set_ylabel("cumulative net P&L, $k (2 ticks/side)", fontsize=8, color=MUTED)
        ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "v4_premium_shocks_pnl.png", dpi=130)
    plt.close(fig)


def fig_grid():
    g = pd.read_csv(CACHE / "step3_grid.csv")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.3))
    for ax, ins in zip(axs, ("second", "hj")):
        _style(ax)
        for wn, mk in (("choose", "o"), ("check", "s")):
            for fam, col in (("PREM", C["news"]), ("WPREM", C["prem"])):
                a = g[(g.ins == ins) & (g.fam == fam) & (g.window == wn)].set_index(["k", "thr", "h"]).mean2
                r = g[(g.ins == ins) & (g.fam == "RAW") & (g.window == wn)].set_index(["k", "thr", "h"]).mean2
                ax.scatter(r, a, s=22, marker=mk, facecolor=col if wn == "choose" else "white", edgecolor=col, lw=1.2,
                           label=f"{'daily' if fam == 'PREM' else 'weekly'}-split fade, {wn} ({(a > r).mean():.0%} above)")
        lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lim, lim, color=MUTED, lw=0.8); ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_xlabel("naive fade: mean net $/trade", fontsize=8, color=MUTED)
        ax.set_ylabel("split fade: mean net $/trade", fontsize=8, color=MUTED)
        ax.set_title({"second": "Second month: 36 parameter cells", "hj": "H−J: 36 parameter cells"}[ins],
                     fontsize=10, color=INK, loc="left")
        ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "v4_premium_shocks_grid.png", dpi=130)
    plt.close(fig)


def main():
    fig_slopes(); fig_pnl(); fig_grid()
    print("done figures")


if __name__ == "__main__":
    main()
