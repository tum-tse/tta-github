"""Fig. 4: MAE, RMSE, MAPE and mean TTA of every learned model relative to the best learned model."""
import numpy as np
import matplotlib.pyplot as plt
from _common import DATASETS, LEARNED, SHORT, summary, save

d = summary()
metrics = [("MAE", "#c6dbef"), ("RMSE", "#6baed6"), ("MAPE", "#08519c"), ("TTA_mean_capped", "#d62728")]
fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.4), sharey=True)
w = 0.2
for ax, (key, name) in zip(axes, DATASETS):
    x = d[(d.dataset == key) & d.model.isin(LEARNED)].set_index("model").loc[LEARNED]
    for k, (col, c) in enumerate(metrics):
        rel = (x[col] / x[col].min() - 1) * 100
        ax.bar(np.arange(5) + (k - 1.5) * w, rel, w, color=c, edgecolor="black", lw=0.3,
               label="TTA" if col.startswith("TTA") else col)
    ax.set_title(name, fontsize=9)
    ax.set_xticks(range(5), [SHORT[m] for m in LEARNED], rotation=45, ha="right", fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("relative to best learned model (%)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
save(fig, "fig4_models")
