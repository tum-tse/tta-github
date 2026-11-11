"""TTA at the common threshold vs. at a matched false-alarm rate, all five networks.

For each model: mean TTA when every model is read with the same threshold h_p = 5
(main table), and mean TTA when each model's threshold is set so that all of them
raise the same number of false alarms per detector-day (tab:matched). The arrow
shows how a model moves when the comparison is made at equal false-alarm cost.

The matched rate of each network is one that every model reaches at some h_p in
{2, 3, 5, 8, 12}; TTA is interpolated linearly in FAR along each model's curve, as
matched_far() in test_models_tta_all.py does.

Reads summary_all_h.csv of each run:
    models/test_results_tta/summary_all_h.csv                      (Munich)
    models/test_results_tta/benchmarks/<dataset>/d15/summary_all_h.csv
Writes fig_matched_far.pdf / .png into models/test_results_tta/.

    python3 models/plot_matched_far.py
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get("TTA_RESULTS", ROOT / "reference_results"))
OUT = ROOT / "figures" / "out"
OUT.mkdir(exist_ok=True)
H_MAIN = 5.0
# (panel, summary over all thresholds, matched rate per detector-day)
RUNS = [
    ("Munich", RES / "munich/summary_all_h.csv", 0.05),
    ("PEMS-BAY", RES / "benchmarks/pemsbay/d15/summary_all_h.csv", 0.05),
    ("METR-LA", RES / "benchmarks/metrla/d15/summary_all_h.csv", 0.12),
    ("Chengdu", RES / "benchmarks/chengdu/d15/summary_all_h.csv", 0.05),
    ("Shenzhen", RES / "benchmarks/shenzhen/d15/summary_all_h.csv", 0.05),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# one row per model, top to bottom; colours as in fig_tau
COLOR = {
    "Persistence":  "0.45",
    "RandomForest": "#1f77b4",
    "XGBoost":      "#1f77b4",
    "BiLSTM":       "#ff7f0e",
    "GCN":          "#2ca02c",
    "GCN-BiLSTM":   "#e6a700",
}
LABEL = {"RandomForest": "Random Forest"}


def tta_at(g: pd.DataFrame, far: float) -> float:
    g = g.sort_values("FAR_per_det_day")
    x = g.FAR_per_det_day.to_numpy()
    if not x.min() <= far <= x.max():
        raise ValueError(f"{g.model.iloc[0]} does not reach FAR {far} (range {x.min():.3f}-{x.max():.3f})")
    return float(np.interp(far, x, g.TTA_mean_capped))


def main() -> None:
    fig, axes = plt.subplots(1, len(RUNS), figsize=(7.16, 2.0), sharey=True)
    rows = list(COLOR)

    for ax, (name, path, far) in zip(axes, RUNS):
        df = pd.read_csv(path)
        for i, m in enumerate(rows):
            c = COLOR[m]
            g = df[df.model == m]
            a = float(g.loc[g.h_pred == H_MAIN, "TTA_mean_capped"].iloc[0])  # common threshold
            b = tta_at(g, far)                                                # matched FAR
            ax.annotate("", xy=(b, i), xytext=(a, i), zorder=2,
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=1.0, mutation_scale=7,
                                        shrinkA=2.5, shrinkB=3.5))
            ax.plot(a, i, "o", ms=4.2, mfc=c, mec="k", mew=0.5, zorder=3)
            ax.plot(b, i, "o", ms=4.2, mfc="white", mec=c, mew=1.0, zorder=3)

        ax.set_title(f"{name}\nFAR = {far:g}", fontsize=8, pad=3, linespacing=1.3)
        ax.margins(x=0.1)
        ax.grid(True, axis="x", color="0.8", lw=0.4, ls=":")
        ax.tick_params(labelsize=7.5)
        ax.tick_params(axis="y", which="both", length=0, right=False)

    axes[0].set_yticks(range(len(rows)), [LABEL.get(m, m) for m in rows])
    axes[0].set_ylim(len(rows) - 0.5, -0.5)
    fig.supxlabel("Mean TTA (min)", fontsize=8, y=0.02)

    handles = [Line2D([], [], ls="none", marker="o", ms=4.2, mfc="0.3", mec="k", mew=0.5,
                      label=r"common threshold $h_p = 5$"),
               Line2D([], [], ls="none", marker="o", ms=4.2, mfc="white", mec="0.3", mew=1.0,
                      label="matched false-alarm rate")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               fontsize=7.5, columnspacing=2.0, bbox_to_anchor=(0.55, 1.0))

    fig.subplots_adjust(left=0.105, right=0.99, top=0.74, bottom=0.18, wspace=0.12)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(OUT / f"fig_matched_far.{ext}", facecolor="white", **kw)
    print("saved", OUT / "fig_matched_far.pdf")


if __name__ == "__main__":
    main()
