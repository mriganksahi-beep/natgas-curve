"""README figure for the Winter Squeeze-Risk Monitor (reads the processed panels; writes reports/tool/figures/monitor.png)."""
import os, sys
os.environ.setdefault("RND_DATA_CUTOFF", "2026-12-31")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.tool import monitor as M, physical as PH, storage_map as SM

BLUE, ORANGE, INK, INK2, SURF, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
AS_OF, Y = "2026-11-17", 2027
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF})

z = np.load("data/processed/tool_live/latest.npz")
mi, _ = M.maps(Y - 1)
imp = SM.implied_storage((z["grid"], z["cdf"]), mi)
xg, dens = imp["density"]
ph = PH.physical_dist(AS_OF, baseline_bcf=M.avg5_monthly(Y) + M.wm_offset())
r = pd.read_csv("data/processed/tool_live/readings.csv").iloc[-1]

fig, ax = plt.subplots(1, 2, figsize=(12, 5.0), gridspec_kw={"width_ratios": [1.15, 1]})
a = ax[0]
step = xg[1] - xg[0]
a.plot(xg, dens / step * 1000, color=BLUE, lw=2, label="Options-implied (March–April spread options)")
h, e = np.histogram(ph["samples"], bins=np.arange(-1200, 1210, 100), density=True)
a.plot((e[:-1] + e[1:]) / 2, h * 1000, color=ORANGE, lw=2, label="Physical model (storage + weather + analogs)")
a.axvline(-400, color=INK2, lw=1, ls=(0, (3, 3)))
a.text(-410, a.get_ylim()[1] * 0.92, "400 Bcf deficit", ha="right", color=INK2, fontsize=9)
a.set_xlim(-1200, 1200); a.set_xlabel("End-March 2027 storage vs 5-year average (Bcf)"); a.set_yticks([])
a.set_title(f"Today ({AS_OF}): P(400+ Bcf deficit)  options {r.Q_def400:.0%}  ·  physical {r.P_def400_raw:.0%}",
            loc="left", color=INK, fontsize=11)
for s in ("top", "right", "left"): a.spines[s].set_visible(False)

b = ax[1]
p = pd.concat([pd.read_csv("data/processed/tool_dev/panel_dev.csv"), pd.read_csv("data/processed/tool_locked/panel_locked.csv")])
p = p[p.usable == True]
w = p.groupby("Y").agg(Q=("Q_def400", "mean"), P=("P_def400_raw", "mean"), x=("x_real", "first"))
b.axvspan(2021.5, 2026.5, color="#f0efec", zorder=0)
b.text(2024, 0.82, "locked test 2022–26\n(no deficit winters)", ha="center", color=INK2, fontsize=9)
b.plot(w.index, w.Q, "-o", color=BLUE, lw=2, ms=6, label="Options-implied (March–April spread options)")
b.plot(w.index, w.P, "-o", color=ORANGE, lw=2, ms=6, label="Physical model (storage + weather + analog years)")
ev = w[w.x < -400]
b.scatter(ev.index, [0.95] * len(ev), marker="v", s=70, color=INK, zorder=5, label="Winter ended 400+ Bcf short")
b.set_ylim(0, 1); b.set_xticks(w.index); b.set_xticklabels([f"{y%100:02d}" for y in w.index])
b.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); b.grid(axis="y", color=GRID, lw=0.8)
b.set_xlabel("Winter (March contract year, 20xx)"); b.set_title("Track record: P(400+ Bcf deficit), winter average of weekly readings", loc="left", color=INK, fontsize=11)
for s in ("top", "right"): b.spines[s].set_visible(False)
h_, l_ = b.get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK2)
fig.tight_layout(rect=(0, 0.07, 1, 1)); os.makedirs("reports/tool/figures", exist_ok=True)
fig.savefig("reports/tool/figures/monitor.png", dpi=160)
print("ok")
