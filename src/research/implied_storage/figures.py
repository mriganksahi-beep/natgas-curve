"""Figures for Q2 (reads the caches written by analysis.py). Also prints a measurement-noise check.

Run: PYTHONPATH=. python -m src.research.implied_storage.figures   (~10 s)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.research.implied_storage import data as D
from src.research.implied_storage.analysis import A_LO, OUT, SE_MAX, fit, f

ROOT = Path(__file__).resolve().parents[3]
FIG = ROOT / "reports/agents/figures"
BLUE, ORANGE, INK, GREY, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#8a8984", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GREY, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})


def headline(d):
    winters = sorted(d.winter.unique())
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), sharey=True)
    for ax, w in zip(axes.ravel(), winters):
        g = d[d.winter == w].sort_values("obs_date")
        x = g.obs_date
        ax.fill_between(x, g.p10, g.p90, color=BLUE, alpha=0.18, lw=0, label="Model 2 10-90%")
        ax.plot(x, g.p50, color=BLUE, lw=2, label="Model 2 median")
        ax.plot(x, g.benchmark, color=GREY, lw=1.2, ls="--", label="5-yr path")
        if g.actual.notna().any():
            ax.axhline(g.actual.iloc[0], color=INK, lw=1.2, label="Realised end-March")
        inf = g[g.informative]
        ax.plot(inf.obs_date, inf.L_hat, "o", color=ORANGE, ms=6, mec="white", mew=1, label="Market-implied")
        flat = g[~g.informative]
        L_flat = flat.avg5 - (1000 / flat.c) * np.log(flat.sigma * 1000 / (flat.c * SE_MAX) / flat.b)
        ax.plot(flat.obs_date, L_flat, "^", color=ORANGE, mfc="white", ms=6,
                label="Market-implied: at least (curve too flat)")
        fl = inf[inf.L_hat <= 500.5]
        if len(fl):
            ax.annotate("floored at 500 Bcf", (fl.obs_date.iloc[len(fl) // 2], 500), xytext=(0, -18),
                        textcoords="offset points", fontsize=7, color=MUTED, ha="center")
        ax.set_title(f"Winter {w}", fontsize=10, color=INK, loc="left")
        ax.grid(axis="y", color="#e6e5e0", lw=0.6)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    axes[0, 0].set_ylabel("End-March storage (Bcf)")
    axes[1, 0].set_ylabel("End-March storage (Bcf)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=8)
    fig.suptitle("Market-implied end-March storage vs Model 2's scenario range (walk-forward mapping; 2021-22 to 31 Dec 2021 only)",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(FIG / "implied_storage_headline.png", dpi=150)


def mapping():
    p = D.build()
    p = p[p.x_info.notna()]
    p["y"] = 100 * p.spread / p.april
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    early, late = p[p.Y <= 2016], p[p.Y >= 2017]
    ax.plot(early.x_info, early.y, "o", color=GREY, ms=5, alpha=0.7, mec="white", label="Fit window 2011-12..2015-16")
    ax.plot(late.x_info, late.y, "o", color=ORANGE, ms=5, alpha=0.8, mec="white", label="Model 2 era 2016-17..Dec 2021")
    xs = np.linspace(p.x_info.min() - 50, p.x_info.max() + 50, 200)
    pe, se_ = fit(early.x_info.values, early.y.values / 100)
    pa, sa = fit(p[p.x_real.notna()].x_info.values, p[p.x_real.notna()].y.values / 100)
    ax.plot(xs, 100 * f(xs, *pe), color=INK, lw=1.6, label="Curve fitted on 2011-16 (used for 2016-17)")
    ax.plot(xs, 100 * f(xs, *pa), color=BLUE, lw=1.6, ls="--", label="Curve fitted on 2011-21")
    for pp, s, col in ((pe, se_, INK),):
        xf = -(1000 / pp[2]) * np.log(s * 1000 / (pp[2] * SE_MAX) / pp[1])
        ax.axvspan(xf, xs.max(), color=GREY, alpha=0.08, lw=0)
        ax.text(xf + 20, 22, "too flat to invert with\nthe 2011-16 curve\n(implied s.e. > 250 Bcf)",
                fontsize=8, color=MUTED, va="top")
    ax.axhline(0, color=GREY, lw=0.6)
    ax.set_xlabel("Ex-ante end-March surplus: storage now + avg remaining draw - prior-years avg (Bcf)")
    ax.set_ylabel("March - April spread (% of April price)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", bbox_to_anchor=(1, 1.0))
    ax.set_title("Scarcity curve: flat when storage is ample, steep when short;\nit steepened once 2018-19 entered the fit", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "implied_storage_mapping.png", dpi=150)
    print("2011-16 curve:", np.round(pe, 4), round(se_, 4))


def who_moves():
    m = pd.read_csv(OUT / "who_moves_h1.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True, sharey=True)
    for ax, col, lab, c in ((axes[0], "dL", "Change in market-implied level", ORANGE),
                            (axes[1], "dM", "Change in Model 2 median", BLUE)):
        ax.axhline(0, color=GREY, lw=0.6)
        ax.axvline(0, color=GREY, lw=0.6)
        ax.plot(m.G, m[col], "o", color=c, ms=5, mec="white", alpha=0.85)
        b = np.polyfit(m.G, m[col], 1)
        xs = np.linspace(m.G.min(), m.G.max(), 10)
        ax.plot(xs, np.polyval(b, xs), color=INK, lw=1.2)
        ax.set_title(f"{lab} next week (slope {b[0]:+.2f})", fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("Gap this week: implied - Model 2 median (Bcf)")
        ax.grid(color="#e6e5e0", lw=0.5)
    axes[0].set_ylabel("Bcf")
    fig.tight_layout()
    fig.savefig(FIG / "implied_storage_who_moves.png", dpi=150)


def noise_check(d):
    """Mechanical convergence: if the implied level carried pure i.i.d. measurement noise with
    variance se^2, regressing its next change on the gap gives about -mean(se^2)/var(G)."""
    for h in (1, 4):
        m = pd.read_csv(OUT / f"who_moves_h{h}.csv")
        print(f"h={h}: n={len(m)}  var(G)={m.G.var():.0f}  mean se^2={np.mean(m.se ** 2):.0f}  "
              f"mechanical b_market ~ {-np.mean(m.se ** 2) / m.G.var():.2f}")
    inf = d[d.informative]
    print("median se of implied level (Bcf):", round(inf.se.median()), " median M2 80% width:",
          round((inf.p90 - inf.p10).median()))


if __name__ == "__main__":
    d = pd.read_csv(OUT / "implied_weekly_main.csv", parse_dates=["obs_date", "week"])
    headline(d)
    mapping()
    who_moves()
    noise_check(d)
