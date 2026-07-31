"""Plot seal-species validation accuracy through time."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np

SCRATCH = "/tmp/claude-4736/-blue-ewhite-b-weinstein-src-BOEM/1656d312-e1dd-44bb-9dc2-a708a1542ce5/scratchpad"
OUT = "/blue/ewhite/b.weinstein/src/BOEM/output/seal_classification_accuracy_over_time.png"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = {"Halichoerus grypus": "#2a78d6", "Phoca vitulina": "#eb6834"}
COMMON = {"Halichoerus grypus": "Gray seal", "Phoca vitulina": "Harbor seal"}

d = pd.read_csv(f"{SCRATCH}/seal_accuracy_history.csv")
d["date"] = pd.to_datetime(d["date"])
d = d.sort_values("timestamp_ms")

# point area from validation sample count (area ~ n, floored so small n stays visible)
def area(n):
    return 40 + 9.0 * np.asarray(n, dtype=float)

fig, ax = plt.subplots(figsize=(12, 6.2), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

for sp, color in SERIES.items():
    s = d[d.species == sp]
    ax.plot(s.date, s.acc_final_epoch, color=color, lw=2, zorder=2, alpha=0.9)
    ax.scatter(s.date, s.acc_final_epoch, s=area(s.n_val_species), color=color,
               edgecolor=SURFACE, linewidth=2, zorder=3)
    # direct label at the last point
    last = s.iloc[-1]
    ax.annotate(COMMON[sp], xy=(last.date, last.acc_final_epoch),
                xytext=(10, 0), textcoords="offset points",
                color=color, fontsize=11, fontweight="600", va="center")

ax.set_ylim(-0.05, 1.08)
ax.set_yticks(np.arange(0, 1.01, 0.25))
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

ax.grid(axis="y", color=GRID, lw=1, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(AXIS)
    ax.spines[side].set_linewidth(1)
ax.tick_params(colors=MUTED, labelsize=10, length=0)
ax.set_ylabel("Validation accuracy (final epoch)", color=SECONDARY, fontsize=11, labelpad=10)

ax.set_title("Seal species classification accuracy over time",
             color=INK, fontsize=16, fontweight="600", loc="left", pad=26)
ax.text(0, 1.035,
        "One point per Comet classification run  ·  point size = seal crops in that run's validation set",
        transform=ax.transAxes, color=SECONDARY, fontsize=10.5, va="bottom")

# size legend
size_handles = [
    Line2D([], [], marker="o", linestyle="none", markersize=np.sqrt(area(n)) * 0.72,
           markerfacecolor=MUTED, markeredgecolor=SURFACE, markeredgewidth=2,
           label=f"{n} crops")
    for n in (5, 20, 63)
]
color_handles = [
    Line2D([], [], color=c, lw=2, marker="o", markersize=8,
           markeredgecolor=SURFACE, markeredgewidth=2,
           label=f"{COMMON[sp]}  ({sp})")
    for sp, c in SERIES.items()
]
leg1 = ax.legend(handles=color_handles, loc="lower left", frameon=False,
                 fontsize=10, labelcolor=SECONDARY, handletextpad=0.8,
                 bbox_to_anchor=(0.0, -0.30), ncol=2)
ax.add_artist(leg1)
ax.legend(handles=size_handles, loc="lower right", frameon=False, fontsize=9.5,
          labelcolor=SECONDARY, labelspacing=1.3, handletextpad=1.2,
          bbox_to_anchor=(1.0, -0.34), ncol=3, title="Validation set size",
          title_fontsize=9.5, alignment="left")

fig.subplots_adjust(left=0.075, right=0.87, top=0.85, bottom=0.24)
fig.savefig(OUT, facecolor=SURFACE)
print("wrote", OUT)
