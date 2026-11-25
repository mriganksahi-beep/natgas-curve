"""Figure: the chain from weather to storage to prices, and where it breaks (README section 3)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, INK2, SURF, BLUE, ORANGE, BOX = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6", "#eb6834", "#eef3fb"
fig, ax = plt.subplots(figsize=(11, 3.6)); fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
ax.set_xlim(0, 11); ax.set_ylim(0, 3.6); ax.axis("off")
xs = {"Weather": 1.4, "Storage": 5.5, "Prices": 9.6}
for k, x in xs.items():
    ax.add_patch(FancyBboxPatch((x - 0.95, 1.05), 1.9, 0.8, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=BOX, ec=INK2, lw=1))
    ax.text(x, 1.45, k, ha="center", va="center", fontsize=13, color=INK)
def arrow(x0, x1, y, color, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                 mutation_scale=16, color=color, lw=2, ls=ls))
arrow(2.4, 4.5, 1.45, BLUE)
ax.text(3.45, 0.75, "Holds: weekly storage forecast\nerror 13.8 vs 31.3 billion cubic feet", ha="center", va="top", fontsize=9.5, color=INK2)
arrow(6.5, 8.6, 1.45, ORANGE, ls=(0, (4, 3)))
ax.text(7.55, 0.75, "Breaks: storage surprises explain\nabout 2% of spread moves", ha="center", va="top", fontsize=9.5, color=INK2)
ax.add_patch(FancyArrowPatch((1.4, 1.9), (9.6, 1.9), connectionstyle="arc3,rad=-0.28", arrowstyle="-|>",
                             mutation_scale=16, color=BLUE, lw=2))
ax.text(5.5, 3.25, "Prices respond to weather forecasts directly, before storage data confirm them",
        ha="center", va="center", fontsize=9.5, color=INK2)
fig.tight_layout(); fig.savefig("reports/figures/chain.png", dpi=160); print("ok")
