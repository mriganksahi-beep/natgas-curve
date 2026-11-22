import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import data as D
from . import payout as P

FIG = D.ROOT / "reports/agents/figures"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def fig_bands(r5, rm):
    ws = sorted(r5.winter.unique())
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for ax, w in zip(axes.ravel(), ws):
        a, b = r5[r5.winter == w], rm[rm.winter == w]
        ax.fill_between(a.obs, a.q10, a.q90, color=ORANGE, alpha=0.18, lw=0,
                        label="Physical 10-90% (5-yr path centre)")
        ax.plot(a.obs, a.av, color=ORANGE, lw=2, label="Actuarial value (5-yr path centre)")
        ax.plot(b.obs, b.av, color=ORANGE, lw=1.2, ls="--", label="Actuarial value (raw Model 2)")
        ax.plot(a.obs, a.S, color=BLUE, lw=2, marker="o", ms=3, label="Market H-J spread")
        if a.S_exp.notna().any():
            ax.axhline(a.S_exp.iloc[0], color=INK2, lw=1, ls=":", label="Realised expiry spread")
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(f"Winter {w}", fontsize=10, color=INK, loc="left")
        _style(ax)
        ax.tick_params(axis="x", rotation=30)
    axes[0, 0].set_ylabel("$/MMBtu", color=INK2)
    axes[1, 0].set_ylabel("$/MMBtu", color=INK2)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle("March-April spread: market price vs actuarial value of the expiry payout "
                 "(walk-forward payout curve, Model 2 scenarios)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(FIG / "widowmaker_insurance_bands.png", dpi=130)
    plt.close(fig)


def fig_payout(t, cd):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    xs = np.linspace(-1300, 1000, 200)
    c = cd[(2022, "wf")]
    band = P.predict(c["ps"], xs)
    ax.fill_between(xs, np.quantile(band, 0.1, 0), np.quantile(band, 0.9, 0), color=ORANGE, alpha=0.18, lw=0,
                    label="Parameter uncertainty 10-90% (residual bootstrap)")
    ax.plot(xs, P.predict(c["p0"], xs)[0], color=ORANGE, lw=2, label="Fit on 2011-21")
    c17 = cd[(2017, "wf")]
    ax.plot(xs, P.predict(c17["p0"], xs)[0], color=VIOLET, lw=1.2, ls="--", label="Fit on 2011-16 (used for 2016-17)")
    ax.scatter(t.x, t.y, color=BLUE, s=36, zorder=3, edgecolor="white", lw=1.5, label="Winters 2011-21")
    for Y, r in t.iterrows():
        if Y in (2014, 2012, 2017, 2019, 2021, 2011):
            ax.annotate(str(Y), (r.x, r.y), textcoords="offset points", xytext=(6, 4), fontsize=8, color=INK2)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xlabel("End-March storage minus prior-years average (Bcf)", color=INK2)
    ax.set_ylabel("H-J spread on March last trading day ($/MMBtu)", color=INK2)
    ax.set_title("Payout curve: y = a + b exp(-x / tau)", fontsize=10, color=INK, loc="left")
    _style(ax)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "widowmaker_insurance_payout.png", dpi=130)
    plt.close(fig)


def fig_theta(th5, thm, po):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    xs = np.arange(len(th5))
    ax.errorbar(xs - 0.1, th5["mean"], yerr=[th5["mean"] - th5.lo, th5.hi - th5["mean"]], color=ORANGE, marker="o",
                lw=2, capsize=3, label="5-yr path centre")
    ax.errorbar(xs + 0.1, thm["mean"], yerr=[thm["mean"] - thm.lo, thm.hi - thm["mean"]], color=VIOLET, marker="s",
                lw=1.2, ls="--", capsize=3, label="Raw Model 2")
    ax.set_xticks(xs, th5.wte_bucket)
    ax.invert_xaxis()
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xlabel("Weeks to March expiry", color=INK2)
    ax.set_ylabel("Premium = market - actuarial ($/MMBtu)", color=INK2)
    ax.set_title("Premium decay, winters 2016-17 to 2021-22 (95% winter-block CI)", fontsize=10, color=INK, loc="left")
    _style(ax)
    ax.legend(fontsize=8, frameon=False)
    yrs = po.index.values
    ax2.bar(yrs, po.net_short_nov1, color=[ORANGE if v < 0 else BLUE for v in po.net_short_nov1], width=0.6,
            label="Net P&L held to expiry")
    ax2.scatter(yrs, po.worst_mtm, marker="v", s=40, color=INK2, zorder=3, label="Worst mark-to-market while open")
    ax2.axhline(0, color=INK2, lw=0.6)
    ax2.set_xticks(yrs, [D.winter_label(y) for y in yrs], rotation=40)
    ax2.set_ylabel(r"\$ per spread (net P&L after \$160 costs)", color=INK2)
    ax2.set_title("Price only: short H-J from 1 Nov to expiry, 2010-21", fontsize=10, color=INK, loc="left")
    ax2.legend(fontsize=8, frameon=False, loc="lower left")
    _style(ax2)
    fig.tight_layout()
    fig.savefig(FIG / "widowmaker_insurance_theta.png", dpi=130)
    plt.close(fig)
