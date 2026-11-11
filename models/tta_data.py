# tta_data.py
"""
Small helper for the plotting scripts: the held-out observations, the cached model predictions and the
scoring window, loaded once and cached on disk (models/test_results_tta/cache/heldout_2024__obs.*), so a
plot script starts in seconds instead of reloading the raw data (about 5 minutes).

Requires the cached predictions of test_models_tta_all.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import torch

import tta
import test_models_tta_all as T

CACHE = T.OUT_DIR / "cache"
MODELS = ["XGBoost", "RandomForest", "BiLSTM", "GCN", "GCN-BiLSTM"]


def _specs() -> dict:
    out = {}
    for name, (_, path, _) in T.DL_MODELS.items():
        c = torch.load(path, map_location="cpu")["config"]
        out[name] = dict(L=int(c["lookback"]), H=int(c["horizon"]), step=int(c["step_size"]), A=int(c["accumulation_window"]))
    return out


def load_obs() -> dict:
    """times, obs [T, N] (5-minute flow sums valid at each minute), node_ids, A."""
    f_obs, f_meta = CACHE / "heldout_2024__obs.npy", CACHE / "heldout_2024__obs_meta.json"
    if f_obs.exists() and f_meta.exists():
        meta = json.loads(f_meta.read_text())
        return {"obs": np.load(f_obs, mmap_mode="r"), "times": pd.DatetimeIndex(pd.to_datetime(meta["times"])),
                "node_ids": meta["node_ids"], "A": meta["A"]}
    A = next(iter(_specs().values()))["A"]
    d = T.load_period(T.EVAL_PERIOD, A, keep_X=False)
    np.save(f_obs, d["obs"].astype(np.float32))
    f_meta.write_text(json.dumps({"times": [str(t) for t in d["times"]], "node_ids": d["node_ids"], "A": A}))
    return {"obs": d["obs"], "times": d["times"], "node_ids": d["node_ids"], "A": A}


def load_predictions(models=MODELS) -> dict:
    """{model: grid [T, N]} for the 5-minute-ahead forecasts, from the cache (never recomputed)."""
    d = load_obs()

    def missing(*_a, **_k):
        raise RuntimeError("cache missing: run test_models_tta_all.py first")
    return {m: np.asarray(T.cached(m, d["obs"].shape, missing)[f"t+{d['A']}min"][0]) for m in models}


def scoring_window(times) -> tuple:
    specs = _specs()
    A = next(iter(specs.values()))["A"]
    L_max = max([s["L"] + (s["H"] - 1) * s["step"] for s in specs.values()] + [T.TREE_LOOKBACK + A])
    H_tail = max([(s["H"] - 1) * s["step"] + s["A"] for s in specs.values()] + [A])
    return (times[L_max + int(T.TTA_CFG.warmup_min)],
            times[len(times) - H_tail - int(T.TTA_CFG.max_wait_min) - 1])


def persistence(obs: np.ndarray, lead: int = 5) -> np.ndarray:
    p = np.full(obs.shape, np.nan, np.float32)
    p[lead:] = obs[:-lead]
    return p


def profile_for(obs, times):
    """The time-of-week 'normal' profile of the held-out year (cached next to the observations)."""
    f = CACHE / "heldout_2024__profile.npz"
    if f.exists():
        z = np.load(f, allow_pickle=False)
        return tta.Profile(z["median"], z["spread"], int(z["bin_min"]), str(z["day_classes"]))
    p = tta.build_profile(obs, times, bin_min=5, spread_floor_abs=1.0)
    np.savez(f, median=p.median, spread=p.spread, bin_min=p.bin_min, day_classes=p.day_classes)
    return p
