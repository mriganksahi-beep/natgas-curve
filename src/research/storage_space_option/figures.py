"""Figures for the storage-space-option report (reads cached CSVs; data < 2022 only)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.research.storage_space_option.analysis import fit_payout, payout
from src.research.storage_space_option.data import CACHE, FIG

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"axes.facecolor": SURF, "figure.facecolor": SURF, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})


def fig_payout(a):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, name, title in ((axs[0], "XV", "Nov - Oct contango at October expiry"),
                            (axs[1], "FV", "Jan - Oct contango at October expiry")):
        d = a.dropna(subset=["dev_T", f"{name}_exp"])
        ax.scatter(d.dev_T, 100 * d[f"{name}_exp"], s=36, color=C1, edgecolor=SURF, linewidth=1.5, zorder=3)
        for _, r in d.iterrows():
            ax.annotate(str(r.year), (r.dev_T, 100 * r[f"{name}_exp"]), xytext=(4, 3),
                        textcoords="offset points", fontsize=8, color=MUTED)
        xx = np.linspace(-700, 450, 200)
        b_all = fit_payout(d, f"{name}_exp", "dev_T", "exp", 150)
        b_pre = fit_payout(d[d.year < 2020], f"{name}_exp", "dev_T", "exp", 150)
        ax.plot(xx, 100 * payout(b_all, xx, "dev_T"), color=C2, lw=2, label="a + b·exp(dev/150), fit 2013-21")
        ax.plot(xx, 100 * payout(b_pre, xx, "dev_T"), color=C3, lw=2, ls="--",
                label="same, fit 2013-19 (used for 2020)")
        ax.axvspan(d.dev_T.max(), 450, color="#e6e5e0", alpha=0.5, lw=0)
        top = 100 if name == "XV" else 160
        ax.set_ylim(-10, top)
        ax.text(330, top * 0.80, "beyond\nany prior\nautumn", fontsize=8, color=MUTED, ha="center")
        ax.set_xlabel("Realised end-October storage minus 5-yr average (Bcf)")
        ax.set_ylabel("¢/MMBtu")
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "storage_space_option_payout.png", dpi=150)
    plt.close(fig)


def fig_premium(P):
    x = P[P.spread == "XV"]
    years = sorted(x.year.unique())
    fig, axs = plt.subplots(1, len(years), figsize=(13, 3.6), sharey=True)
    for ax, y in zip(axs, years):
        d = x[x.year == y].sort_values("mdate")
        ax.plot(d.mdate, 100 * d.market, color=C1, lw=2, label="Market")
        ax.plot(d.mdate, 100 * d.act_rob_main, color=C2, lw=2, label="Actuarial: convex, extrapolated")
        ax.plot(d.mdate, 100 * d.act_rob_clip, color=C3, lw=2, label="Actuarial: no extrapolation")
        ax.plot(d.mdate, 100 * d.act_pt5_main, color=C4, lw=2, ls="--", label="Payout(5-yr path)")
        ax.axhline(100 * d.expiry.iloc[0], color=INK, lw=1, ls=":", label="Realised at expiry")
        ax.set_title(str(y), fontsize=10, loc="left")
        ax.set_ylim(-10, 110)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    axs[0].set_ylabel("Nov - Oct contango, ¢/MMBtu (clipped at 110)")
    axs[-1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "storage_space_option_premium.png", dpi=150)
    plt.close(fig)


def fig_price_only(po):
    d = po[po.spread == "XV"].sort_values("year")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for _, r in d.iterrows():
        ax.plot([r.year, r.year], [100 * r.summer_avg, 100 * r.expiry], color=MUTED, lw=1.2, zorder=1)
    ax.scatter(d.year, 100 * d.summer_avg, s=40, color=C1, edgecolor=SURF, lw=1.5, zorder=3,
               label="Apr-Aug average")
    ax.scatter(d.year, 100 * d.expiry, s=40, color=C2, edgecolor=SURF, lw=1.5, zorder=3,
               label="At October's last trading day")
    ax.set_xticks(d.year)
    ax.set_ylabel("Nov - Oct contango, ¢/MMBtu")
    ax.set_title("Autumn contango usually shrinks into expiry; 2020 is the tail", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "storage_space_option_price_only.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    a = pd.read_csv(CACHE / "autumn_table.csv")
    P = pd.read_csv(CACHE / "panel.csv", parse_dates=["origin", "mdate"])
    po = pd.read_csv(CACHE / "price_only.csv")
    fig_payout(a)
    fig_premium(P)
    fig_price_only(po)
