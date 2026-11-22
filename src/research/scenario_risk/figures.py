"""Figures for reports/agents/scenario_risk.md. Run after evaluate.py."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.research.scenario_risk import data as D

FIG = D.ROOT / "reports/agents/figures"
COL = {"EWMA": "#2a78d6", "GARCH-t": "#eb6834", "PHYS-M2": "#1baf7a", "PHYS-WX": "#eda100",
       "CONST": "#52514e", "HS-3y": "#e87ba4", "HYB": "#6250d6"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#888", "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})


def fig_es():
    f = pd.read_csv(D.CACHE / "forecasts_summary.csv", parse_dates=["date"])
    f = f[(f.h == 1) & (f.date >= "2016-11-01")]
    winters = sorted(f[f.model == "PHYS-M2"].winter.unique())
    fig, axs = plt.subplots(1, len(winters), figsize=(13, 3.6), sharey=True)
    for ax, w in zip(axs, winters):
        g = f[f.winter == w]
        l = g[g.model == "EWMA"].set_index("date").loss
        ax.bar(l.index, l.values, width=5, color=np.where(l > 0, "#c9c8c3", "#e6e6e3"), label="realised loss")
        for m in ("EWMA", "GARCH-t", "PHYS-M2"):
            s = g[g.model == m].set_index("date").ES975
            ax.plot(s.index, s.values, color=COL[m], lw=2, label=f"{m} ES 97.5%")
        ax.set_title(w, fontsize=9)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    axs[0].set_ylabel("$ per spread, 1 week")
    axs[0].set_ylim(-6000, 15500)
    axs[0].legend(loc="upper left", fontsize=7, frameon=False)
    fig.suptitle("1-week ES of a short March-April spread vs the realised loss (Model 2 era)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "scenario_risk_es_paths.png", dpi=150)


def fig_episodes():
    e = pd.read_csv(D.CACHE / "early_warning.csv")
    models = ["HS-3y", "EWMA", "GARCH-t", "HYB", "PHYS-WX", "PHYS-M2"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), sharey=False)
    for ax, h in zip(axs, (1, 4)):
        x = e[e.h == h]
        eps = list(dict.fromkeys(x.episode))
        w = 0.13
        for i, m in enumerate(models):
            vals = [x[(x.episode == ep) & (x.model == m)].loss_over_ES.squeeze() if
                    ((x.episode == ep) & (x.model == m)).any() else np.nan for ep in eps]
            ax.bar(np.arange(len(eps)) + (i - 2.5) * w, vals, width=w * 0.9, color=COL[m], label=m)
        ax.axhline(1, color="#0b0b0b", lw=1, ls="--")
        ax.set_xticks(range(len(eps)), [f"{ep}\n(Fri {x[x.episode == ep].date.iloc[0]})" for ep in eps])
        ax.set_title(f"{h}-week horizon: realised loss / forecast ES (>1 = ES exceeded)", fontsize=9)
    axs[0].legend(fontsize=7, frameon=False, ncol=2)
    fig.text(0.01, 0.01, "PHYS-M2 needs Model 2 (Nov 2016+); no bar for Jan 2014. Nov 2021: spread fell over 4 weeks.",
             fontsize=7, color="#52514e")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "scenario_risk_episodes.png", dpi=150)


def fig_sizing():
    s = pd.read_csv(D.CACHE / "sizing_series_M2_h1.csv", index_col=0, parse_dates=True)
    t = pd.read_csv(D.CACHE / "sizing_M2_h1_k4.csv").set_index("model")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    nudge = {"GARCH-t": 450, "EWMA": -450}
    for m in ("CONST", "EWMA", "GARCH-t", "PHYS-WX", "PHYS-M2"):
        c = (s[m] / t.loc[m, "mean_N"]).cumsum().reset_index(drop=True)
        ax.plot(c.index, c.values, color=COL[m], lw=2)
        ax.text(c.index[-1] + 0.8, c.values[-1] + nudge.get(m, 0), m, color="#0b0b0b", fontsize=8, va="center")
    wint = [D.winter_label(d) for d in s.index]
    starts = [i for i in range(len(wint)) if i == 0 or wint[i] != wint[i - 1]]
    ax.set_xticks(starts, [wint[i] for i in starts])
    ax.set_xlim(0, len(s) + 9)
    ax.set_ylabel("cumulative $ per average spread held")
    ax.set_title("Short-spread book sized to a fixed 1-week ES budget, 4 ticks/side, Nov-mid-Feb weeks only", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "scenario_risk_sizing.png", dpi=150)


if __name__ == "__main__":
    fig_es(); fig_episodes(); fig_sizing()
