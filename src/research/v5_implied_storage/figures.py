"""Figures for v5_implied_storage (statistics only, no raw price tables)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.research.v5_implied_storage import data as D

FIG = D.ROOT / "reports/agents/figures"
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6})


def tails():
    m = pd.read_csv(D.CACHE / "panel_weekly_final.csv", parse_dates=["obs"])
    Ys = sorted(m.Y.unique())
    fig, axs = plt.subplots(2, 4, figsize=(12, 5.4), sharey=True)
    for ax, Y in zip(axs.ravel(), Ys):
        z = m[m.Y == Y].sort_values("wte", ascending=False)
        ax.plot(z.wte, z.Q_400, color=C1, lw=2, marker="o", ms=4, label="Implied (G4X options)")
        ax.plot(z.wte, z.Psp_pl_rc_400, color=C2, lw=2, marker="s", ms=4, label="Physical: P-lite, 5-yr centred (spread space, map noise incl.)")
        if z.Psp_m2_rc_400.notna().any():
            ax.plot(z.wte, z.Psp_m2_rc_400, color=C3, lw=2, marker="^", ms=4, label="Physical: Model 2, 5-yr centred (same)")
        ax.invert_xaxis()
        xr = z.x_real.iloc[0]
        ev = "outcome n/a" if not np.isfinite(xr) else f"realised x = {xr:+.0f} Bcf"
        ax.set_title(f"{D.winter_label(Y)}  ({ev})", fontsize=9, color=INK)
        ax.set_ylim(0, 1)
    for ax in axs[1]:
        ax.set_xlabel("weeks to option expiry")
    for ax in axs[:, 0]:
        ax.set_ylabel("P(x < -400 Bcf)")
    h, l = axs[0, 2].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Market-implied vs physical probability of a 400-Bcf end-March deficit (walk-forward map)",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "v5_implied_storage_tails.png", dpi=150)
    plt.close(fig)


def quantiles():
    m = pd.read_csv(D.CACHE / "panel_weekly_final.csv", parse_dates=["obs"])
    g = m.groupby("Y").agg(q10=("xq_Q_10", "mean"), q50=("xq_Q_50", "mean"), p10=("xq_pl_rc_10", "mean"),
                           p50=("x_bench5", "mean"), m10=("xq_m2_rc_10", "mean"), xr=("x_real", "first"))
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(g))
    ax.vlines(x - 0.12, g.q10, g.q50, color=C1, lw=2)
    ax.plot(x - 0.12, g.q50, "o", color=C1, ms=8, label="Implied median (dot) to 10th pct")
    ax.vlines(x + 0.12, g.p10, g.p50, color=C2, lw=2)
    ax.plot(x + 0.12, g.p50, "s", color=C2, ms=8, label="Physical (P-lite, 5-yr centred) median to 10th pct")
    ax.plot(x, g.xr, "D", color=INK, ms=7, mfc="none", mew=1.5, label="Realised end-March x")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, [D.winter_label(Y) for Y in g.index])
    ax.set_ylabel("end-March storage minus prior-5-yr mean (Bcf)")
    ax.set_title("What storage outcome the G4X options price vs physics (winter means of weekly readings)",
                 fontsize=10, color=INK)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "v5_implied_storage_quantiles.png", dpi=150)
    plt.close(fig)


def pnl():
    f = pd.read_csv(D.CACHE / "trades_first_week.csv")
    f = f[f.Y <= 2022]
    x = np.arange(len(f))
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (col, ttl) in zip(axs, ((("pnl_fut", "half_fut", "pnl_cs", "pnl_sc"), "P&L to expiry ($ per lot)"),
                                    (("dd_fut", "half_dd", "dd_cs", "dd_sc"), "worst mark-to-market drawdown ($)"))):
        for i, (c, lab, colr) in enumerate(zip(col, ("Short H-J futures", "Half size", "Short call spread +0.25/+0.75",
                                                     "Short call +0.25"), (C1, C2, C3, C4))):
            ax.bar(x + (i - 1.5) * 0.2, f[c], width=0.18, color=colr, label=lab)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(x, [D.winter_label(Y) for Y in f.Y], rotation=45, fontsize=7)
        ax.set_title(ttl, fontsize=9, color=INK)
    axs[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Once-per-winter entries (first November decision date); 2021-22 outcome read under the brief's exception",
                 fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "v5_implied_storage_pnl.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    tails(); quantiles(); pnl()
