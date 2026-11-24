"""Figures for reports/tool/physical_benchmark.md.  python -m src.research.tool_physical.figures"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.research.tool_physical import bench as B

OUT = B.ROOT / "reports/tool/figures"
CH = (B.CACHE / "chosen.txt").read_text().strip()
MODELS = [(CH, "Analog (new)", "#2a78d6"), ("plite", "P-lite raw", "#eb6834"),
          ("m2", "Model 2 raw", "#1baf7a"), ("clim", "Climatology", "#eda100")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
df = pd.read_csv(B.CACHE / "scores_long.csv", parse_dates=["asof"])

# 1. CRPS by winter
fig, ax = plt.subplots(figsize=(9, 3.6))
yrs = list(range(2012, 2022))
w = 0.2
for i, (m, lab, c) in enumerate(MODELS):
    s = df[df.model == m].groupby("Y").crps.mean().reindex(yrs)
    ax.bar(np.arange(len(yrs)) + (i - 1.5) * w, s.values, w * 0.9, color=c, label=lab)
ax.set_xticks(range(len(yrs)), [f"{y-1}-{str(y)[2:]}" for y in yrs])
ax.set_ylabel("mean CRPS (Bcf, lower is better)")
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.axvline(4.5, color=MUTED, lw=0.8, ls=":")
ax.text(4.5, ax.get_ylim()[1] * 0.80, "choose | check", ha="center", color=MUTED)
ax.legend(frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.0))
ax.set_title("CRPS of end-March storage deviation x, by winter (weekly as-of dates Nov-Feb)", loc="left", color=INK)
fig.tight_layout(); fig.savefig(OUT / "physical_crps_by_winter.png", dpi=150); plt.close(fig)

# 2. PIT histograms
fig, axs = plt.subplots(1, 4, figsize=(10, 2.6), sharey=True)
for ax, (m, lab, c) in zip(axs, MODELS):
    p = df[(df.model == m) & df.pit.notna()].pit
    ax.hist(p, bins=np.linspace(0, 1, 11), color=c, edgecolor="#fcfcfb", lw=2)
    ax.axhline(len(p) / 10, color=MUTED, lw=0.8, ls="--")
    ax.set_title(f"{lab} (n={len(p)}, {df[df.model == m].dropna(subset=['pit']).Y.nunique()} winters)", fontsize=8.5, loc="left")
    ax.set_xlabel("PIT of realised x")
axs[0].set_ylabel("as-of dates")
fig.suptitle("PIT histograms (flat = calibrated; dashed = uniform). Dates within a winter share one outcome.", x=0.01, ha="left", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "physical_pit.png", dpi=150); plt.close(fig)

# 3. Fan charts per winter
fig, axs = plt.subplots(2, 6, figsize=(12, 4.8), sharey=True)
for ax, Y in zip(axs.ravel(), range(2012, 2023)):
    a = df[(df.model == CH) & (df.Y == Y)].sort_values("asof")
    ax.fill_between(a["asof"], a.p10, a.p90, color="#2a78d6", alpha=0.18, lw=0, label="Analog 10-90%")
    ax.plot(a["asof"], a.med, color="#2a78d6", lw=2, label="Analog median")
    pl = df[(df.model == "plite") & (df.Y == Y)].sort_values("asof")
    ax.plot(pl["asof"], pl.med, color="#eb6834", lw=1.2, ls="--", label="P-lite median")
    m2 = df[(df.model == "m2") & (df.Y == Y)].sort_values("asof")
    if len(m2):
        ax.plot(m2["asof"], m2.med, color="#1baf7a", lw=1.2, ls="-.", label="Model 2 median")
    xr = a.x_real.iloc[0]
    if np.isfinite(xr):
        ax.axhline(xr, color=INK, lw=1.2, label="realised x")
    else:
        ax.text(0.5, 0.05, "outcome locked", transform=ax.transAxes, ha="center", color=MUTED)
    ax.axhline(0, color=GRID, lw=0.6)
    ax.set_title(f"{Y-1}-{str(Y)[2:]}", fontsize=9, loc="left")
    ax.tick_params(axis="x", labelsize=6, rotation=45)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
axs[1, 5].axis("off")
h, l = axs[0, 5].get_legend_handles_labels()
axs[1, 5].legend(h, l, frameon=False, loc="center", fontsize=8)
axs[0, 0].set_ylabel("x (Bcf vs 5-yr mean)"); axs[1, 0].set_ylabel("x (Bcf vs 5-yr mean)")
fig.suptitle("As-of distribution of end-March storage deviation x through each winter", x=0.01, ha="left")
fig.tight_layout(); fig.savefig(OUT / "physical_fans.png", dpi=150); plt.close(fig)
print("ok")
