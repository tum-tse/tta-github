"""Fig. 1: two forecast errors with equal area but different duration (schematic)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"],
                     "mathtext.fontset": "stix", "font.size": 10.5, "axes.labelsize": 10.5, "legend.fontsize": 10.5, "xtick.labelsize": 10.5,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "xtick.major.size": 2.5, "xtick.direction": "out"})

t = np.linspace(-5, 40, 4000)
pos = np.clip(t, 0, None)
a = (1 - np.exp(-pos / 0.55)) * np.exp(-pos / 2.6)    # model A: sharp, short
b = (1 - np.exp(-pos / 1.6)) * np.exp(-pos / 9.0)     # model B: moderate, long
dt = t[1] - t[0]
a /= a.sum() * dt
b /= b.sum() * dt                                      # equal areas

blue, orange = "#1f5f99", "#d9730d"
fig, ax = plt.subplots(figsize=(7.2, 2.7))
ax.fill_between(t, b, color=orange, alpha=0.15, lw=0)
ax.fill_between(t, a, color=blue, alpha=0.15, lw=0)
ax.plot(t, b, color=orange, lw=1.5, ls=(0, (1.2, 1.2)), label="model B")
ax.plot(t, a, color=blue, lw=1.5, label="model A")
ax.axvline(0, color="0.2", lw=0.6, ls=(0, (2, 2)))
ax.text(0, 1.01, "disruption onset", fontsize=10, color="0.2", va="bottom", ha="center", transform=ax.get_xaxis_transform(), clip_on=False)


ax.set_xlim(-5, 40)
ax.set_ylim(0, a.max() * 1.12)
ax.set_xticks(range(0, 41, 10))
ax.set_yticks([])
ax.set_xlabel("time from disruption onset (min)")
ax.set_ylabel("absolute forecast error")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
h, l = ax.get_legend_handles_labels()
ax.legend(h[::-1], l[::-1], frameon=False, loc="upper right", handlelength=2.2)
fig.tight_layout(pad=0.2)
fig.savefig(OUT / "fig1_why_simple.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig1_why_simple.png", dpi=600, bbox_inches="tight")
