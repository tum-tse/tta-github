"""Fig. 2: how TTA is measured (schematic). Same onset rule on observed and forecast series."""
import numpy as np
import matplotlib.pyplot as plt
from _common import save

t = np.linspace(-12, 36, 3000)


def speed(t):
    s = np.clip(t, 0, None)
    return 112 - 35 * (1 - np.exp(-s / 2.2)) + 0.68 * np.clip(s - 6.5, 0, None) * (1 - np.exp(-np.clip(s - 6.5, 0, None) / 3))


obs, fa, fb = speed(t), speed(t - 2.6), speed(t - 9.6)
level = 86.0                                       # where each series becomes "clearly abnormal"
onset = lambda y: t[np.argmax(y <= level)]
o, oa, ob = onset(obs), onset(fa), onset(fb)
blue, orange = "#1f77b4", "#ff7f0e"

fig, (ax, bx) = plt.subplots(2, 1, figsize=(7.2, 3.4), sharex=True,
                             gridspec_kw=dict(height_ratios=[3, 1.05], hspace=0.08))
ax.plot(t, obs, color="black", lw=1.8, label="observed")
ax.plot(t, fa, color=blue, lw=1.5, ls=(0, (5, 2.5)), label="forecast A")
ax.plot(t, fb, color=orange, lw=1.5, ls=(0, (1.2, 1.4)), label="forecast B")
for x, c in ((o, "black"), (oa, blue), (ob, orange)):
    ax.plot(x, level, "o", ms=6.5, mfc="white", mec=c, mew=1.6, zorder=5)
    for a in (ax, bx):
        a.axvline(x, color=c, lw=0.7, ls=":", ymax=1 if a is bx else (level - 56) / (123 - 56))
ax.set_ylim(56, 123)
ax.set_ylabel("speed (km/h)")
ax.legend(frameon=False, ncol=3, loc="upper right")
for a in (ax, bx):
    a.spines[["top", "right"]].set_visible(False)

for y, x1, c, lab in ((0.62, oa, blue, r"$\tau_A$"), (0.25, ob, orange, r"$\tau_B$")):
    bx.annotate("", xy=(x1, y), xytext=(o, y), arrowprops=dict(arrowstyle="<->", color=c, lw=1.1))
    bx.text(x1 + 0.6, y, lab, color=c, va="center")
bx.set_ylim(0, 1)
bx.set_yticks([])
bx.spines["left"].set_visible(False)
bx.set_xlim(-12, 36)
bx.set_xlabel("time (min)")
save(fig, "fig2_operator")
