# test_tta_duration_sweep.py
"""
Minimum-event-duration sensitivity of TTA, on the cached predictions of test_models_tta_all.py.

Short disruptions mostly test ANTICIPATION, long ones ADAPTATION. For each d in DURATIONS the
disruption catalogue is rebuilt with TTAConfig(min_event_duration_min=d, depth_window_min=d)
and every model is scored on the same events. Nothing is retrained: the predictions come from
models/test_results_tta/cache/ (run test_models_tta_all.py first; this script never recomputes
a model, it stops if a cache is missing).

    python3 models/test_tta_duration_sweep.py

Outputs (models/test_results_tta/duration_sweep/):
    summary_d<d>.csv        the usual summary table for each d
    duration_sweep.csv      all four stacked (one row per model and d)
    duration_sweep.md       the four tables in Markdown
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import torch

import tta
import test_models_tta_all as T

DURATIONS = [5, 10, 15, 30]
OUT = T.OUT_DIR / "duration_sweep"
COLS = ["family", "model", "n_events", "MAE", "TTA_median", "TTA_mean_capped", "caught_pct",
        "pre_alarmed_pct", "FAR_per_det_day"]


def _md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, rule] + rows)


def _missing(*_a, **_k):
    raise RuntimeError("cache missing: run test_models_tta_all.py first")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    specs = {}
    for name, (_, path, _) in T.DL_MODELS.items():
        c = torch.load(path, map_location="cpu")["config"]
        specs[name] = dict(L=int(c["lookback"]), H=int(c["horizon"]), step=int(c["step_size"]),
                           A=int(c["accumulation_window"]))
    A = next(iter(specs.values()))["A"]

    data = T.load_period(T.EVAL_PERIOD, A, keep_X=False)
    times, obs, q, node_ids = data["times"], data["obs"], data["q"], data["node_ids"]

    predictions = {}
    if T.INCLUDE_HIST_AVERAGE:
        predictions["HistAverage"] = {f"t+{A}min": (T.hist_average_grid(q, data["lookback"], A), float(A))}
    for name in list(T.DL_MODELS) + T.TREE_MODELS:
        predictions[name] = T.cached(name, obs.shape, _missing)

    profile = tta.build_profile(obs, times, bin_min=5, spread_floor_abs=1.0)
    L_max = max([s["L"] + (s["H"] - 1) * s["step"] for s in specs.values()] + [T.TREE_LOOKBACK + A])
    H_tail = max([(s["H"] - 1) * s["step"] + s["A"] for s in specs.values()] + [A])
    window = (times[L_max + int(T.TTA_CFG.warmup_min)],
              times[len(times) - H_tail - int(T.TTA_CFG.max_wait_min) - 1])

    tables = []
    for d in DURATIONS:
        cfg = replace(T.TTA_CFG, min_event_duration_min=float(d), depth_window_min=float(d))
        res = tta.evaluate(obs, times, predictions, profile, window, cfg,
                           scenario=f"{T.EVAL_PERIOD['name']}_d{d}", node_ids=node_ids)
        t = tta.summary_table(res)
        t.insert(1, "family", t.model.map(T.FAMILY))
        t.insert(0, "min_duration_min", d)
        t.to_csv(OUT / f"summary_d{d}.csv", index=False)
        c = res.config
        print(f"[d={d}] candidates {c['n_candidate_episodes']} | low level {c['n_dropped_low_level']} | "
              f"too short {c['n_dropped_short']} | too shallow {c['n_dropped_shallow']} | "
              f"events {len(res.events)} = {c['events_per_detector_day']:.3f} per detector-day", flush=True)
        tables.append(t)

    allt = pd.concat(tables, ignore_index=True)
    allt.to_csv(OUT / "duration_sweep.csv", index=False)

    order = ["Persistence", "HistAverage", "RandomForest", "XGBoost", "GCN", "BiLSTM", "GCN-BiLSTM"]
    md = ["# TTA vs minimum disruption duration\n",
          "Munich Autobahn, held-out 2024, 5 minutes ahead. The depth is measured over the first d minutes "
          "of each disruption. Every model is scored on the same events within a table. TTA in minutes; "
          "TTA mean counts never-registered events at 60 min.\n"]
    for d, t in zip(DURATIONS, tables):
        t = t[t.model != "HistProfile"].copy()
        t["_o"] = t.model.map({m: i for i, m in enumerate(order)})
        t = t.sort_values("_o")[COLS]
        md.append(f"\n## Minimum duration {d} min  ({int(t.n_events.iloc[0]):,} events)\n")
        md.append(_md_table(t.round(2)) + "\n")
    (OUT / "duration_sweep.md").write_text("\n".join(md))
    print(f"[SAVE] {OUT}")


if __name__ == "__main__":
    main()
