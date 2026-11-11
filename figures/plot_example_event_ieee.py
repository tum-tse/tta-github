"""Single Munich event, plain IEEE-style redraw of fig_example_event_models.png.

Reads only the saved event data (no forecast cache needed):
    models/test_results_tta/fig_example_event_data.csv
    models/test_results_tta/fig_example_event_meta.json
Writes fig_event_munich.pdf (vector) and fig_event_munich.png (600 dpi) next to them.

    python3 models/plot_example_event_ieee.py
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get("TTA_RESULTS", ROOT / "reference_results"))
OUT = ROOT / "figures" / "out"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# (line colour, dash, marker); colours and markers as in fig_tau, dashes keep the
# lines separable in greyscale print
STYLE = {
    "Persistence": ("0.45", (0, (1, 1.2)), None),
    "RandomForest": ("#1f77b4", (0, (6, 1.5, 1, 1.5, 1, 1.5)), "o"),
    "XGBoost":     ("#1f77b4", (0, (5, 2)), "o"),
    "BiLSTM":      ("#ff7f0e", (0, (5, 1.5, 1, 1.5)), "s"),
    "GCN":         ("#2ca02c", "-", "^"),
    "GCN-BiLSTM":  ("#e6a700", (0, (1.5, 1.5)), "D"),
}
LABEL = {"RandomForest": "Random Forest"}


def main() -> None:
    d = pd.read_csv(RES / "munich/fig_example_event_data.csv", parse_dates=["time"])
    meta = json.loads((RES / "munich/fig_example_event_meta.json").read_text())
    onset = pd.Timestamp(meta["onset"])
    # zoom on the pick-up: from shortly before the onset to just past the last registration
    t0, t1 = onset - pd.Timedelta(minutes=14), onset + pd.Timedelta(minutes=26)
    d = d[(d["time"] >= t0) & (d["time"] <= t1)]
    t = d["time"]
    tta = {"Persistence": meta["persistence_tta_min"], **meta["tta_min"]}
    order = sorted(tta, key=lambda m: (tta[m], list(STYLE).index(m)))  # fastest on top

    fig, (axs, ax) = plt.subplots(2, 1, figsize=(7.16, 3.0), sharex=True,
                                  gridspec_kw={"height_ratios": [1, 3.2], "hspace": 0.06})

    # top strip: each model's TTA as the interval from onset to registration
    for i, m in enumerate(order):
        c = STYLE[m][0]
        tr = onset + pd.Timedelta(minutes=tta[m])
        axs.barh(i, tr - onset, left=onset, height=0.62, color=c, alpha=0.35, lw=0)
        axs.plot([tr, tr], [i - 0.31, i + 0.31], color=c, lw=1.2)
        axs.text(tr - pd.Timedelta(seconds=20), i, f"{tta[m]:.0f} min",
                 va="center", ha="right", fontsize=7)
    axs.set_yticks(range(len(order)), [LABEL.get(m, m) for m in order])
    axs.set_ylim(len(order) - 0.5, -0.5)
    axs.tick_params(axis="y", which="both", length=0, right=False)
    axs.tick_params(axis="x", which="both", labelbottom=False)

    # main panel
    ax.fill_between(t, d["normal"] - d["spread"], d["normal"] + d["spread"],
                    color="0.88", lw=0, label=r"Normal $\pm 1$ spread")
    for m, (c, ls, _) in STYLE.items():
        ax.plot(t, d[m], color=c, ls=ls, lw=0.9, label=LABEL.get(m, m))
    ax.plot(t, d["observed"], color="k", lw=1.2, label="Observed")

    # onset (solid) and each registration (dashed, model colour) through both panels;
    # equal registration times get offset dashes so both colours stay visible
    seen = {}
    for a in (axs, ax):
        a.axvline(onset, color="k", lw=0.7, zorder=1)
    for m in order:
        c = STYLE[m][0]
        tr = onset + pd.Timedelta(minutes=tta[m])
        drawn = seen.setdefault(tr, [])
        if c in drawn:  # same colour at the same minute (RF, XGBoost): one line is enough
            continue
        k = len(drawn)
        drawn.append(c)
        for a in (axs, ax):
            a.axvline(tr, color=c, lw=0.9, ls=(3 * k, (3, 3)), zorder=1)
    ax.text(onset - pd.Timedelta(seconds=30), 8, "onset", ha="right", va="bottom", fontsize=7)

    ax.set_xlim(t0, t1)
    ax.set_ylim(0, 320)
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 5)))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.yaxis.set_major_locator(plt.MultipleLocator(50))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(10))
    ax.grid(True, which="major", color="0.8", lw=0.4, ls=":")
    ax.set_xlabel("Time of day")
    ax.set_ylabel("Flow (veh/5 min)")

    leg = ax.legend(loc="lower left", ncol=2, frameon=True, fancybox=False,
                    edgecolor="k", framealpha=1, borderpad=0.3, handlelength=2.4,
                    columnspacing=0.9, labelspacing=0.15, fontsize=6,
                    borderaxespad=0.3)
    leg.get_frame().set_linewidth(0.5)

    fig.subplots_adjust(left=0.125, right=0.975, top=0.98, bottom=0.13)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(OUT / f"fig_event_munich.{ext}", facecolor="white", **kw)
    print("saved", OUT / "fig_event_munich.pdf")


if __name__ == "__main__":
    main()
