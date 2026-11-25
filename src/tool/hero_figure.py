"""README hero figure: (left) short March-April futures from 1 December to expiry, P&L by winter 2001-2026;
(right) spread-option vol richness, development 2010-21 vs locked 2022-26 (study v5 test P, frozen numbers)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, INK, INK2, SURF, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF})
h = pd.read_csv("data/processed/tool_live/premium_history.csv")
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.7, 1]})
a = ax[0]
k = h.pnl / 1000
a.bar(h.Y, k, width=0.72, color=np.where(k >= 0, BLUE, ORANGE), zorder=3)
a.axhline(0, color=INK2, lw=0.8)
a.grid(axis="y", color=GRID, lw=0.8, zorder=0)
a.set_xticks(h.Y[::2]); a.set_xticklabels([f"{y%100:02d}" for y in h.Y[::2]])
a.set_ylabel("$ thousand per spread")
a.set_title(f"Short March−April futures, 1 Dec → expiry: positive in {int((h.pnl>0).sum())} of {len(h)} winters",
            loc="left", color=INK, fontsize=11)
a.annotate("2003 squeeze: −$28k", (2003, -28.4), xytext=(2005.2, -24), color=INK2, fontsize=9,
           arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
a.annotate("2014 polar vortex", (2014, -3.6), xytext=(2014.6, -12), color=INK2, fontsize=9,
           arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
a.set_xlabel("Winter (March contract year, 20xx)")
for s in ("top", "right"): a.spines[s].set_visible(False)

b = ax[1]
lab = ["2010–21\n(built here)", "2022–26\n(untouched,\ntested once)"]
m = np.array([0.33, 0.24]); lo = np.array([0.25, 0.065]); hi = np.array([0.44, 0.42])
pct = lambda v: (np.exp(v) - 1) * 100
b.bar([0, 1], pct(m), width=0.5, color=BLUE, zorder=3)
b.errorbar([0, 1], pct(m), yerr=[pct(m) - pct(lo), pct(hi) - pct(m)], fmt="none", ecolor=INK, capsize=5, lw=1.2, zorder=4)
for i, v in enumerate(pct(m)):
    b.text(i + 0.3, v, f"+{v:.0f}%", va="center", color=INK, fontsize=10)
b.axhline(0, color=INK2, lw=0.8); b.grid(axis="y", color=GRID, lw=0.8, zorder=0)
b.set_xticks([0, 1]); b.set_xticklabels(lab); b.set_xlim(-0.5, 1.7)
b.set_ylabel("Implied vs realised spread vol")
b.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"+{v:.0f}%" if v > 0 else f"{v:.0f}%"))
b.set_title("Spread options price more vol than realised", loc="left", color=INK, fontsize=11)
for s in ("top", "right"): b.spines[s].set_visible(False)
fig.text(0.01, 0.01, "Bars: mean; whiskers: 95% winter-block bootstrap. Calendar-spread options on adjacent gas months, open interest > 0.",
         color=INK2, fontsize=8)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig("reports/figures/hero.png", dpi=160)
print("ok")
