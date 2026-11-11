"""Fig. 3: rank of each learned model by MAE against rank by mean TTA, with Kendall's tau."""
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from _common import DATASETS, LEARNED, LABEL, MARK, summary, save

d = summary()
fig, axes = plt.subplots(1, 5, figsize=(7.2, 1.95), sharey=True)
for ax, (key, name) in zip(axes, DATASETS):
    x = d[(d.dataset == key) & d.model.isin(LEARNED)].set_index("model").loc[LEARNED]
    rm, rt = x.MAE.rank(method="first"), x.TTA_mean_capped.rank(method="first")
    conc = sum((rm[i] - rm[j]) * (rt[i] - rt[j]) > 0 for i, j in combinations(LEARNED, 2))
    tau = (2 * conc - 10) / 10
    ax.plot([0.6, 5.4], [0.6, 5.4], color="0.6", lw=0.6, ls=(0, (4, 3)))
    for m in LEARNED:
        mk, fc, ec = MARK[m]
        ax.plot(rm[m], rt[m], mk, ms=6, mfc=fc, mec=ec, mew=1)
    ax.set_title(name + "\n" + f"$\\tau = {tau:+.1f}$".replace("-", "−"), fontsize=9)
    ax.set_xlim(0.5, 5.5); ax.set_ylim(0.5, 5.5)
    ax.set_xticks(range(1, 6)); ax.set_yticks(range(1, 6))
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("rank by TTA (1 = fastest)")
fig.supxlabel("rank by MAE (1 = most accurate)", fontsize=9, y=-0.06)
handles = [Line2D([], [], ls="", marker=MARK[m][0], mfc=MARK[m][1], mec=MARK[m][2], ms=6, label=LABEL[m]) for m in LEARNED]
fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.22))
save(fig, "fig3_rank")
