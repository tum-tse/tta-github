"""Shared paths, names and style for the paper figures."""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get("TTA_RESULTS", ROOT / "reference_results"))
OUT = ROOT / "figures" / "out"
OUT.mkdir(exist_ok=True)

DATASETS = [("munich", "Munich"), ("pemsbay", "PEMS-BAY"), ("metrla", "METR-LA"),
            ("chengdu", "Chengdu"), ("shenzhen", "Shenzhen")]
LEARNED = ["RandomForest", "XGBoost", "BiLSTM", "GCN", "GCN-BiLSTM"]
LABEL = {"RandomForest": "Random Forest", "XGBoost": "XGBoost", "BiLSTM": "BiLSTM",
         "GCN": "GCN", "GCN-BiLSTM": "GCN-BiLSTM", "Persistence": "Persistence"}
SHORT = {"RandomForest": "RF", "XGBoost": "XGB", "BiLSTM": "BiLSTM", "GCN": "GCN", "GCN-BiLSTM": "GCN-BiLSTM"}
MARK = {  # marker, face, edge
    "RandomForest": ("o", "white", "#1f77b4"),
    "XGBoost": ("o", "#1f77b4", "#1f3f66"),
    "BiLSTM": ("s", "#ff7f0e", "#b35400"),
    "GCN": ("^", "#2ca02c", "#1a5e1a"),
    "GCN-BiLSTM": ("D", "#f0a30a", "#8a5d00"),
    "Persistence": ("X", "#888888", "#333333"),
}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})


def summary() -> pd.DataFrame:
    """Main table (Table 5): one row per dataset and model."""
    return pd.read_csv(RES / "summary_all_datasets.csv")


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=600, bbox_inches="tight")
    print("saved", OUT / f"{name}.pdf")
