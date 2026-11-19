"""Figures for the README and note.  python -m src.plots"""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports/figures"


def model1_holdout(tag="v3_own"):
    p = pd.read_csv(ROOT / f"data/processed/model1_predictions_{tag}.csv", index_col=0, parse_dates=True)
    r = [c for c in p if c.startswith("Ridge")][0]
    h = p[p.index >= "2022-01-01"]
    e = h.actual - h[r]
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(h.index, h.actual, color="#222", lw=1.2, label="EIA reported")
    ax[0].plot(h.index, h[r], color="#1D9E75", lw=1.2, label="Model 1 forecast (ridge)")
    ax[0].plot(h.index, h["5-year average"], color="#B4B2A9", lw=1, ls="--", label="5-year average")
    ax[0].axhline(0, color="#888", lw=.5)
    ax[0].set_ylabel("Weekly storage change (Bcf)")
    ax[0].legend(frameon=False, ncol=3, loc="upper left", fontsize=9)
    ax[1].bar(h.index, e, width=5, color="#D85A30")
    ax[1].set_ylabel("Miss (Bcf)")
    ax[1].axhline(0, color="#888", lw=.5)
    ax[1].set_xlabel("Week the storage change refers to")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    ax[0].set_title(f"Weather-based storage forecast, out of sample 2022–2026 "
                    f"(MAE {e.abs().mean():.1f} vs {(h.actual - h['5-year average']).abs().mean():.1f} Bcf "
                    f"for the 5-year average)", loc="left", fontsize=10.5)
    plt.tight_layout()
    plt.savefig(FIG / "model1_holdout.png", dpi=150)
    plt.close()


def model2_fans(tag="v2b_from2016", targets=(("2025-03-28", "Winter 2024–25"), ("2025-10-31", "Summer 2025"))):
    b = pd.read_csv(ROOT / f"data/processed/model2_backtest_{tag}.csv", parse_dates=["origin", "target"])
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, (T, label) in zip(axs, targets):
        d = b[b.target == T].sort_values("origin")
        end = "end-March" if d.season.iloc[0] == "winter" else "end-October"
        ax.fill_between(d.origin, d.p10, d.p90, color="#9FE1CB", alpha=.6, label="Model 2: 80% range")
        ax.plot(d.origin, d["median"], color="#0F6E56", lw=1.5, label="Model 2: median forecast")
        ax.plot(d.origin, d.benchmark, color="#888780", lw=1, ls="--", label="5-year-average path")
        ax.axhline(d.actual.iloc[0], color="#D85A30", lw=1.2,
                   label=f"Actual {end} storage ({d.actual.iloc[0]:,.0f} Bcf)")
        ax.set_title(f"{label}: forecasts of {end} storage", loc="left", fontsize=10)
        ax.set_ylabel(f"Forecast of {end} storage (Bcf)")
        ax.set_xlabel("Date the forecast was made")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
        ax.legend(frameon=False, fontsize=8, loc="lower left" if end == "end-March" else "lower right")
    plt.suptitle("Model 2, out of sample: each week's forecast of one fixed end-of-season number. "
                 "The range should narrow and contain the actual.", x=0.01, ha="left", fontsize=10.5)
    plt.tight_layout()
    plt.savefig(FIG / "model2_fans.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    model1_holdout()
    model2_fans()
