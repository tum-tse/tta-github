"""Fig. 5: each learned model relative to persistence: MAE gain against TTA delay in lead times."""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from _common import DATASETS, LEARNED, LABEL, MARK, summary, save

d = summary()
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True, sharey=True)
for ax, (key, name) in zip(axes.flat, DATASETS):
    x = d[d.dataset == key].set_index("model")
    p = x.loc["Persistence"]
    ax.axhspan(0, 3.6, color="#f4cccc", alpha=0.45, lw=0)
    ax.axhline(0, color="0.5", lw=0.6)
    mk, fc, ec = MARK["Persistence"]
    ax.plot(0, 0, mk, ms=7, mfc=fc, mec=ec)
    for m in LEARNED:
        gain = (p.MAE - x.loc[m, "MAE"]) / p.MAE * 100
        late = (x.loc[m, "TTA_mean_capped"] - p.TTA_mean_capped) / p.lead_min
        mk, fc, ec = MARK[m]
        ax.plot(gain, late, mk, ms=6, mfc=fc, mec=ec, mew=1)
    ax.set_title(name, fontsize=9)
    ax.set_xlim(-2, 34); ax.set_ylim(-0.6, 3.6)
    ax.spines[["top", "right"]].set_visible(False)
for ax in axes[:, 0]:
    ax.set_ylabel("TTA later than persistence\n(lead times)")
for ax in (*axes[1, :2], axes[0, 2]):
    ax.set_xlabel("MAE better than persistence (%)")
    ax.xaxis.set_tick_params(labelbottom=True)
leg = axes[1, 2]
leg.axis("off")
handles = [Line2D([], [], ls="", marker=MARK[m][0], mfc=MARK[m][1], mec=MARK[m][2], ms=6, label=LABEL[m])
           for m in ["Persistence", *LEARNED]]
leg.legend(handles=handles, loc="center", frameon=False)
save(fig, "fig5_persistence")
