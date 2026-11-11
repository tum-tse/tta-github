# plot_tta_blindspot.py
"""
What the standard metrics cannot see.

Counterfactual on the held-out year: replace each model's forecast by the observed value during every
disruption (from PAD minutes before the registered onset to the end of the disruption) and re-score. How much
does each metric improve? MAE, RMSE and MAPE average over every minute of the year, and disruptions are only a
few percent of those minutes, so they barely move. TTA is computed per disruption, so it collapses.

Needs the cached predictions of test_models_tta_all.py (models/test_results_tta/cache/); uses tta_data.py.

    python3 models/plot_tta_blindspot.py        # writes models/test_results_tta/fig_tta_blindspot.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tta
import tta_data as D
import test_models_tta_all as T

SHOWN = ["XGBoost", "BiLSTM", "GCN", "GCN-BiLSTM"]
COLOR = {"XGBoost": "#2a78d6", "BiLSTM": "#eb6834", "GCN": "#1baf7a", "GCN-BiLSTM": "#eda100"}
INK, MUTED = "#0b0b0b", "#52514e"
PAD = 15            # perfect from PAD minutes before the registered onset (the tally needs time to build up)
OUT = T.OUT_DIR


def error_metrics(yt, yp):
    ok = np.isfinite(yt) & np.isfinite(yp)
    e = (yt - yp)[ok]
    nz = ok & (yt != 0)
    return dict(MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())),
                MAPE=float(100 * np.abs((yt[nz] - yp[nz]) / yt[nz]).mean()))


def compute() -> dict:
    d = D.load_obs()
    times, obs, ids = d["times"], np.asarray(d["obs"]), d["node_ids"]
    preds = D.load_predictions(SHOWN)
    w0, w1 = D.scoring_window(times)
    lo, hi = int(np.searchsorted(times, w0)), int(np.searchsorted(times, w1))

    ev = pd.read_csv(T.OUT_DIR / "events.csv", parse_dates=["onset"])
    col = {n: i for i, n in enumerate(ids)}
    mask = np.zeros(obs.shape, bool)
    for n, a, dur in zip(ev.node.astype(str), times.get_indexer(ev.onset), ev.duration_min):
        mask[max(0, a - PAD):a + int(round(dur)), col[n]] = True
    m = mask[lo:hi]
    yt = obs[lo:hi]

    cf = {name: np.where(mask, obs, g) for name, g in preds.items()}          # perfect inside disruptions
    rows = []
    for name in SHOWN:
        a, b = error_metrics(yt, preds[name][lo:hi]), error_metrics(yt, cf[name][lo:hi])
        rows.append(dict(model=name, **{f"{k}_orig": a[k] for k in a}, **{f"{k}_cf": b[k] for k in b}))
    res = pd.DataFrame(rows).set_index("model")

    abs_share = {}
    for n in SHOWN:
        e = np.abs(yt - preds[n][lo:hi])
        ok = np.isfinite(e)
        abs_share[n] = float(100 * e[ok & m].sum() / e[ok].sum())
    ok = np.isfinite(yt) & np.isfinite(preds["XGBoost"][lo:hi])
    min_share = float(100 * (ok & m).sum() / ok.sum())

    # real TTA re-evaluation of the counterfactual forecasts
    profile = D.profile_for(obs, times)
    series = {n: {"t+5min": (preds[n], 5.0)} for n in SHOWN}
    series.update({f"{n} (perfect in disruptions)": {"t+5min": (cf[n], 5.0)} for n in SHOWN})
    r = tta.evaluate(obs, times, series, profile, (w0, w1), T.TTA_CFG, scenario="cf", node_ids=ids)
    tab = tta.summary_table(r).set_index("model")
    for n in SHOWN:
        res.loc[n, "TTA_orig"] = float(tab.loc[n, "TTA_mean_capped"])
        res.loc[n, "TTA_cf"] = float(tab.loc[f"{n} (perfect in disruptions)", "TTA_mean_capped"])
    res.to_csv(OUT / "blindspot_counterfactual.csv")
    meta = {"min_share": min_share, "abs_share": abs_share, "pad": PAD}
    (OUT / "blindspot_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def plot(meta: dict) -> None:
    res = pd.read_csv(OUT / "blindspot_counterfactual.csv", index_col="model")
    min_share, abs_share = meta["min_share"], meta["abs_share"]
    print(res.round(2).to_string())
    idx = {k: 100 * (res[f"{k}_cf"] / res[f"{k}_orig"]).mean() for k in ("MAE", "RMSE", "MAPE", "TTA")}
    gap_mae = 100 * res.loc["GCN-BiLSTM", "MAE_orig"] / res.loc["XGBoost", "MAE_orig"]           # best model vs worst learned model, MAE index
    tta_before = (res.TTA_orig.min(), res.TTA_orig.max())
    tta_after = (res.TTA_cf.min(), res.TTA_cf.max())

    fig, ax = plt.subplots(figsize=(12.5, 7.4), facecolor="white")
    fig.subplots_adjust(left=0.08, right=0.75, top=0.78, bottom=0.2)
    gray = {"MAE": "#6f6e69", "RMSE": "#9c9b95", "MAPE": "#c3c2bc"}
    for k in ("MAPE", "RMSE", "MAE"):
        ax.plot([0, 1], [100, idx[k]], color=gray[k], lw=3, solid_capstyle="round", zorder=3)
        ax.scatter([0, 1], [100, idx[k]], s=70, color=gray[k], zorder=4)
    ax.plot([0, 1], [100, idx["TTA"]], color=INK, lw=4.5, solid_capstyle="round", zorder=5)
    ax.scatter([0, 1], [100, idx["TTA"]], s=110, color=INK, zorder=6)
    # scale reference: what switching from XGBoost to GCN-BiLSTM does to MAE
    ax.axhline(gap_mae, color=MUTED, lw=1.2, ls=":", zorder=1)
    ax.text(1.0, gap_mae + 1.5, f"for scale: swapping XGBoost for GCN-BiLSTM improves MAE by {100 - gap_mae:.0f}%", fontsize=10.5, color=MUTED, va="bottom", ha="right")

    ax.text(1.03, np.mean([idx["MAE"], idx["RMSE"], idx["MAPE"]]), f"MAE, RMSE, MAPE\nonly {100 - idx['MAE']:.0f}% better", fontsize=13.5, color="#4a4944",
            va="center", fontweight="bold")
    ax.text(1.03, idx["TTA"], f"TTA\n{100 - idx['TTA']:.0f}% better", fontsize=17, color=INK, va="center", fontweight="bold")
    xm = 0.66
    ax.text(xm, 100 - xm * (100 - idx["TTA"]) + 7, f"mean TTA goes from {tta_before[0]:.0f}-{tta_before[1]:.0f} min to {tta_after[0]:.0f}-{tta_after[1]:.0f} min",
            fontsize=11.5, color=INK, va="bottom", ha="left", fontweight="bold")
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(0, 108)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["the forecast as it is", f"perfect during every disruption\n(only {min_share:.0f}% of all minutes)"], fontsize=12.5, color=INK, fontweight="bold")
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], color=MUTED, fontsize=10.5)
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#b9b8b3")
    ax.tick_params(axis="x", length=0, pad=10)
    ax.grid(axis="y", color="#e6e5e1", lw=0.8)
    ax.set_axisbelow(True)

    fig.text(0.04, 0.95, "Fix every disruption and MAE barely notices. TTA drops 87%.", fontsize=19, fontweight="bold", color=INK, va="center")
    fig.text(0.04, 0.875,
             f"Munich Autobahn, held-out year 2024, forecasts 5 minutes ahead. Each line is one metric (100% = its value today; lower is better),\n"
             f"averaged over XGBoost, BiLSTM, GCN and GCN-BiLSTM. Left: the forecasts as they are. Right: the same forecasts, but\n"
             f"replaced by the real flow during every disruption (from {meta['pad']} min before it is registered until it ends), then re-scored.\n"
             f"Those minutes are {min_share:.1f}% of all minutes and {min(abs_share.values()):.1f}-{max(abs_share.values()):.1f}% of all error.",
             fontsize=11, color=MUTED, va="center", linespacing=1.55)
    fig.text(0.04, 0.115,
             "Why: MAE, RMSE and MAPE average the mistake over every minute, and disruptions are a sliver of those minutes, so even a perfect reaction to\n"
             "all of them hardly changes the score. The gap between two models in MAE comes almost entirely from ordinary traffic. TTA is computed per\n"
             "disruption, so it is the metric that responds. That is how a model can win on MAE and still be the slowest to notice that traffic has changed.",
             fontsize=10.8, color=INK, va="top", linespacing=1.6)
    out = OUT / "fig_tta_blindspot.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print("saved", out)


def main() -> None:
    meta_file = OUT / "blindspot_meta.json"
    if "--replot" in sys.argv and meta_file.exists() and (OUT / "blindspot_counterfactual.csv").exists():
        plot(json.loads(meta_file.read_text()))
    else:
        plot(compute())


if __name__ == "__main__":
    main()
