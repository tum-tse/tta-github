"""Full TTA-FAR trade-off curves, all five networks (every threshold, nothing interpolated).

Reads summary_all_h.csv of each run:
    models/test_results_tta/summary_all_h.csv                      (Munich)
    models/test_results_tta/benchmarks/<dataset>/d15/summary_all_h.csv
Writes fig_tradeoff.pdf / .png into models/test_results_tta/.

    python3 models/plot_tta_far_curves.py
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get("TTA_RESULTS", ROOT / "reference_results"))
OUT = ROOT / "figures" / "out"
OUT.mkdir(exist_ok=True)
RUNS = [
    ("Munich", RES / "munich/summary_all_h.csv"),
    ("PEMS-BAY", RES / "benchmarks/pemsbay/d15/summary_all_h.csv"),
    ("METR-LA", RES / "benchmarks/metrla/d15/summary_all_h.csv"),
    ("Chengdu", RES / "benchmarks/chengdu/d15/summary_all_h.csv"),
    ("Shenzhen", RES / "benchmarks/shenzhen/d15/summary_all_h.csv"),
]
H_MAIN = 5.0          # threshold used in the main table
GUIDES = (0.05, 0.1)  # matched rates of tab:matched

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

# colour, marker, filled, dash; colours and markers as in fig_tau
STYLE = {
    "Persistence":  ("0.45", None, True, (0, (4, 2))),
    "RandomForest": ("#1f77b4", "o", False, "-"),
    "XGBoost":      ("#1f77b4", "o", True, "-"),
    "BiLSTM":       ("#ff7f0e", "s", True, "-"),
    "GCN":          ("#2ca02c", "^", True, "-"),
    "GCN-BiLSTM":   ("#e6a700", "D", True, "-"),
}
LABEL = {"RandomForest": "Random Forest"}


def main() -> None:
    fig, axes = plt.subplots(1, len(RUNS), figsize=(7.16, 2.3), sharey=True)

    for ax, (name, path) in zip(axes, RUNS):
        df = pd.read_csv(path)
        for g in GUIDES:
            ax.axvline(g, color="0.55", lw=0.5, ls=":", zorder=0)
        for m, (c, mk, filled, ls) in STYLE.items():
            s = df[df.model == m].sort_values("h_pred")
            fc = c if filled else "white"
            ax.plot(s.FAR_per_det_day, s.TTA_mean_capped, color=c, lw=0.9, ls=ls,
                    marker=mk, ms=2.6, mfc=fc, mec=c, mew=0.6, zorder=2)
            p = s[s.h_pred == H_MAIN]
            if mk:
                ax.plot(p.FAR_per_det_day, p.TTA_mean_capped, ls="none", marker=mk, ms=5,
                        mfc=fc, mec="k", mew=0.5, zorder=3)
            else:
                ax.plot(p.FAR_per_det_day, p.TTA_mean_capped, ls="none", marker="x", ms=4,
                        mec=c, mew=1.0, zorder=3)

        ax.set_xscale("symlog", linthresh=0.01, linscale=0.4)
        ax.set_xlim(0, 2)
        ax.set_xticks([0, 0.01, 0.1, 1], ["0", "0.01", "0.1", "1"])
        ax.set_title(name, fontsize=8, pad=3)
        ax.grid(True, axis="y", color="0.8", lw=0.4, ls=":")
        ax.tick_params(labelsize=7.5)
        ax.tick_params(axis="x", which="minor", length=0)

    axes[0].set_ylim(-5, 62)
    axes[0].set_ylabel("Mean TTA (min)")
    fig.supxlabel("False-alarm rate (per detector-day)", fontsize=8, y=0.02)

    handles = []
    for m, (c, mk, filled, ls) in STYLE.items():
        handles.append(Line2D([], [], color=c, lw=0.9, ls=ls, marker=mk or "x",
                              ms=4.5 if mk else 4, mfc=(c if filled else "white"),
                              mec="k" if mk else c, mew=0.5 if mk else 1.0,
                              label=LABEL.get(m, m)))
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, fontsize=7.5,
               handlelength=2.2, columnspacing=1.2, bbox_to_anchor=(0.5, 1.0))

    fig.subplots_adjust(left=0.065, right=0.99, top=0.82, bottom=0.17, wspace=0.08)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(OUT / f"fig_tradeoff.{ext}", facecolor="white", **kw)
    print("saved", OUT / "fig_tradeoff.pdf")


if __name__ == "__main__":
    main()
