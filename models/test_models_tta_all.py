# test_models_tta_all.py
"""
Time to Adapt (TTA) next to MAE / RMSE / MAPE over a WHOLE held-out period,
for three model families (traffic only, no weather):

    ML         Random Forest, XGBoost        (trained here, on the DL training period)
    DL         GCN, BiLSTM                   (saved checkpoints)
    Hybrid DL  GCN-BiLSTM                    (saved checkpoint)
    Reference  Persistence, HistProfile (added by tta.py), Historical Average

All models are scored on ONE shared set of disruptions found in the observed
data, so every comparison is on identical events.

Layout (run from the repository root):
    models/test_models_tta_all.py  <- this file
    models/tta.py                  <- the metric
    python3 models/test_models_tta_all.py

Tree models
    One model shared by all detectors ("global" model), trained on a random
    sample of (minute, detector) pairs from TREE_TRAIN_PERIOD. Features, all
    known at the issue time:
        the last six 5-minute flow sums, the last three 1-minute flows,
        time of day (sin/cos), weekend flag,
        the detector's normal flow for the target time (training-period profile),
        last 5-minute sum minus its normal value.
    Target: the next 5-minute flow sum, exactly as the DL models.

Predictions are cached (OUT_DIR/cache/*.npy), so changing TTA settings and
rerunning does not rerun the models. Delete the cache after changing a model.

Outputs (models/test_results_tta/):
    summary.csv / summary_all_h.csv   MAE, RMSE, MAPE, TTA, caught %, FAR, A_idx
    families.csv                      the same averaged per family
    pairs.csv                         paired comparisons on the same events
    events.csv, per_event.csv.gz      the catalogue and every event x model
    config.json, fig_*.png
"""

from __future__ import annotations

import gc
import importlib
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = Path(__file__).resolve().parent             # .../autobahn-model/models
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import tta


# =============================================================================
# CONFIGURATION
# =============================================================================

# name: (module in models/, checkpoint, family)
DL_MODELS: dict[str, tuple[str, str, str]] = {
    # GCN: models/spatial/gcn_simple.py (plain 2-layer GCN, hidden 128, skip connection).
    # GCN-BiLSTM: models/spatio_temporal/gnn_bilstm_a.py (3-layer GCN, hidden 256, then BiLSTM with attention);
    # its GCN is NOT the same module as the standalone GCN above.
    "GCN":        ("spatial.gcn_simple",                   "checkpoints/gcn_simple/20261019_190210/best_model.pth",        "DL"),
    "BiLSTM":     ("temporal.bilstm",                      "checkpoints/bilstm/20260416_170027/best_model.pth",            "DL"),
    "GCN-BiLSTM": ("spatio_temporal.gnn_bilstm_a",         "checkpoints/gcn_bilstm/20260503_141628/best_model.pth",        "Hybrid DL"),
}
TREE_MODELS = ["RandomForest", "XGBoost"]          # family "ML"
FAMILY = {"RandomForest": "ML", "XGBoost": "ML", "HistAverage": "Reference",
          "Persistence": "Reference", "HistProfile": "Reference",
          **{k: v[2] for k, v in DL_MODELS.items()}}

# Paired comparisons (A, B) on the same events; negative difference = A faster
PAIRS = [("RandomForest", "XGBoost"), ("XGBoost", "BiLSTM"), ("XGBoost", "GCN-BiLSTM"),
         ("GCN", "GCN-BiLSTM"), ("BiLSTM", "GCN-BiLSTM"), ("XGBoost", "GCN")]

# Evaluation: held out for EVERY model (DL trained 2025-01 → 2026-02; trees trained on 2025)
EVAL_PERIOD = {"name": "heldout_2024", "start": "2024-01-01", "end": "2025-01-01"}
TREE_TRAIN_PERIOD = {"start": "2025-01-01", "end": "2026-01-01"}
TREE_TRAIN_ROWS = 2_000_000                         # random (minute, detector) samples
RF_PARAMS = dict(n_estimators=60, max_depth=16, min_samples_leaf=20, max_features=0.6, n_jobs=-1, random_state=0)
XGB_PARAMS = dict(n_estimators=600, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                  tree_method="hist", random_state=0)

# Raw 1-minute lane files (FG1_Kurz_<unit>_agg1min_<start>_bis_<end>.csv), all periods in one folder.
DATA_DIR = os.environ.get("TTA_DATA_DIR", str(REPO / "data" / "autobahn"))
DATA_SOURCES = [
    {"data_dir": DATA_DIR, "date_suffix": "2026-01-01_bis_2026-02-20"},
    {"data_dir": DATA_DIR, "date_suffix": "2025-01-01_bis_2026-01-01"},
    {"data_dir": DATA_DIR, "date_suffix": "2024-01-01_bis_2025-01-01"},
]

# What counts as a disruption. min_normal_level: only where normal flow is at
# least this many vehicles per 5 minutes (a 15 % dip on an empty road is noise).
TTA_CFG = tta.TTAConfig(direction="down", min_normal_level=50.0)

INCLUDE_HIST_AVERAGE = True
BATCH_SIZE = 512
USE_CACHE = True
N_BOOT = 2000
N_EXAMPLE_EVENTS = 4
OUT_DIR = REPO / "models" / "test_results_tta"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log(msg: str) -> None:
    print(msg, flush=True)


def find_data_source(start: str, end: str) -> dict:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    for src in DATA_SOURCES:
        a, b = (pd.Timestamp(x) for x in src["date_suffix"].split("_bis_"))
        if a <= s and e <= b:
            return src
    raise ValueError(f"Period {start} → {end} is not inside a single DATA_SOURCES entry.")


# =============================================================================
# DATA
# =============================================================================

def rolling_sum(q: np.ndarray, A: int) -> np.ndarray:
    """S[t] = q[t] + ... + q[t+A-1]: the target (A-minute flow) valid at t."""
    T = q.shape[0]
    out = np.full(q.shape, np.nan, np.float32)
    acc = np.zeros((T - A + 1,) + q.shape[1:], np.float32)
    for a in range(A):
        acc += q[a:T - A + 1 + a]
    out[:T - A + 1] = acc
    return out


def load_period(period: dict, A: int, keep_X: bool) -> dict:
    """Load + impute one period. Returns q (1-min flow), obs (A-min sums), times, nodes [, X_clean, mu, sigma]."""
    gnn = importlib.import_module("spatial.gnn")
    imputation = importlib.import_module("imputation")
    src = find_data_source(period["start"], period["end"])
    cfg = gnn.Config()
    cfg.data_dir, cfg.date_suffix = src["data_dir"], src["date_suffix"]
    gnn.Config.date_suffix = cfg.date_suffix                 # build_panel_X reads the class attribute
    nodes, _, edge_index = gnn.load_global_graph(cfg.nodes_csv, cfg.edges_csv)
    X, times = gnn.build_panel_X(nodes=nodes, site_to_units=gnn.load_selection_site_to_units(cfg.selection_csv),
                                 data_dir=cfg.data_dir, feature_cols=cfg.feature_cols,
                                 nan_fill_value=cfg.nan_fill_value, debug_mapping_out=cfg.debug_mapping_out)
    times = pd.DatetimeIndex(getattr(times, "values", times))
    i0, i1 = (int(np.searchsorted(times, pd.Timestamp(period[k]))) for k in ("start", "end"))
    X, times = X[i0:i1], times[i0:i1]
    log(f"[DATA] {period['start']} → {period['end']}: T={len(times)} N={len(nodes)}")
    X_clean, _, mu, sigma = imputation.run_imputation_pipeline(
        X=X, edge_index=edge_index, feature_cols=cfg.feature_cols, train_ratio=1.0, edges_csv=cfg.edges_csv)
    del X; gc.collect()
    as_t = lambda v: v if isinstance(v, torch.Tensor) else torch.tensor(np.asarray(v), dtype=torch.float32)
    mu, sigma = as_t(mu), as_t(sigma)
    X_clean = X_clean.nan_to_num(0.0)
    tcol = list(cfg.feature_cols).index(cfg.target_col)
    q = (X_clean[:, :, tcol] * (sigma[tcol] + 1e-8) + mu[tcol]).numpy().astype(np.float32)
    out = {"q": q, "obs": rolling_sum(q, A), "times": times, "node_ids": nodes["site"].astype(str).tolist(),
           "lookback": int(getattr(cfg, "lookback", 30))}
    if keep_X:
        out.update(X_clean=X_clean, mu=mu, sigma=sigma)
    else:
        del X_clean
    gc.collect()
    return out


# =============================================================================
# REFERENCE: HISTORICAL AVERAGE
# =============================================================================

def hist_average_grid(q: np.ndarray, lookback: int, A: int) -> np.ndarray:
    """A x mean of the last `lookback` minutes; valid at s, issued at s (inputs up to s-1)."""
    T, N = q.shape
    cs = np.vstack([np.zeros((1, N)), np.cumsum(q, axis=0, dtype=np.float64)])
    g = np.full((T, N), np.nan, np.float32)
    s = np.arange(lookback, T)
    g[s] = (A * (cs[s] - cs[s - lookback]) / lookback).astype(np.float32)
    return g


# =============================================================================
# TREE MODELS (ML)
# =============================================================================

TREE_FEATURES = ["sum_l1", "sum_l2", "sum_l3", "sum_l4", "sum_l5", "sum_l6", "q_l1", "q_l2", "q_l3",
                 "tod_sin", "tod_cos", "weekend", "normal_target", "dev_last"]
TREE_LOOKBACK = 30


def tree_features(obs, q, normal, times, s_idx, A: int) -> np.ndarray:
    """Features for target times s_idx (all detectors). Uses data up to s-1 only.
    obs[t] = sum q[t..t+A-1], so the 5-min block ending at s-1 is obs[s-A]."""
    s = np.asarray(s_idx)
    k = len(s); N = q.shape[1]
    F = np.empty((k, N, len(TREE_FEATURES)), np.float32)
    for j in range(6):
        F[:, :, j] = obs[s - A * (j + 1)]
    for j in range(3):
        F[:, :, 6 + j] = q[s - (j + 1)]
    t = times[s]
    m = np.asarray(t.hour * 60 + t.minute, dtype=np.float32)
    F[:, :, 9] = np.sin(2 * np.pi * m / 1440)[:, None]
    F[:, :, 10] = np.cos(2 * np.pi * m / 1440)[:, None]
    F[:, :, 11] = np.asarray(t.weekday >= 5, dtype=np.float32)[:, None]
    F[:, :, 12] = normal[s]
    F[:, :, 13] = obs[s - A] - normal[s - A]
    return F.reshape(k * N, -1)


def train_trees(A: int) -> dict:
    log(f"\n{'=' * 70}\nML — training tree models on {TREE_TRAIN_PERIOD}\n{'=' * 70}")
    tr = load_period(TREE_TRAIN_PERIOD, A, keep_X=False)
    prof = tta.build_profile(tr["obs"], tr["times"], bin_min=5, spread_floor_abs=1.0)
    normal_tr, _ = prof.lookup(tr["times"])
    normal_tr = np.nan_to_num(normal_tr.astype(np.float32), nan=np.nanmean(normal_tr))
    T, N = tr["q"].shape
    rng = np.random.default_rng(0)
    s = rng.integers(TREE_LOOKBACK + A, T - A, TREE_TRAIN_ROWS // N + 1)
    X = tree_features(tr["obs"], tr["q"], normal_tr, tr["times"], s, A)
    y = tr["obs"][s].reshape(-1)
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    X, y = X[ok], y[ok]
    log(f"[ML] training rows: {len(y):,} | features: {TREE_FEATURES}")
    models = {}
    try:
        from sklearn.ensemble import RandomForestRegressor
        t0 = time.time()
        models["RandomForest"] = RandomForestRegressor(**RF_PARAMS).fit(X, y)
        log(f"[ML] RandomForest trained in {(time.time() - t0) / 60:.1f} min")
    except Exception as e:
        log(f"[ML] [ERROR] RandomForest: {type(e).__name__}: {e}")
    try:
        import xgboost as xgb
        params = dict(XGB_PARAMS)
        if torch.cuda.is_available():
            params["device"] = "cuda"
        t0 = time.time()
        models["XGBoost"] = xgb.XGBRegressor(**params).fit(X, y)
        log(f"[ML] XGBoost trained in {(time.time() - t0) / 60:.1f} min")
    except Exception as e:
        log(f"[ML] [ERROR] XGBoost: {type(e).__name__}: {e}  (pip install xgboost)")
    del tr, X, y; gc.collect()
    return {"models": models, "profile": prof}


def tree_grid(model, obs, q, normal, times, A: int, chunk: int = 20_000) -> np.ndarray:
    T, N = q.shape
    g = np.full((T, N), np.nan, np.float32)
    s_all = np.arange(TREE_LOOKBACK + A, T)
    t0 = time.time()
    for c, i in enumerate(range(0, len(s_all), chunk)):
        s = s_all[i:i + chunk]
        g[s] = model.predict(tree_features(obs, q, normal, times, s, A)).reshape(len(s), N)
        if c % 5 == 0:
            done = (i + len(s)) / len(s_all); el = time.time() - t0
            log(f"  {100 * done:5.1f}%  ({el / 60:.1f} min, ~{el / done * (1 - done) / 60:.1f} min left)")
    return g


# =============================================================================
# DL MODELS
# =============================================================================

def _fast_windows(ds, X_norm: torch.Tensor, L: int):
    """get_batch(i0, i1) by slicing X_norm directly, IF identical to ds[i][0]; else None."""
    view = X_norm.unfold(0, L, 1)                     # [T-L+1, N, F, L], no copy
    probes = [0, 1, len(ds) // 2, len(ds) - 1]
    for perm in [(0, 3, 1, 2), (0, 1, 3, 2), (0, 1, 2, 3)]:
        try:
            if all(torch.allclose(torch.as_tensor(ds[i][0]).float(),
                                  view[i:i + 1].permute(*perm)[0].float(), atol=1e-6) for i in probes):
                return lambda i0, i1, perm=perm: view[i0:i1].permute(*perm).contiguous()
        except Exception:
            continue
    return None


@torch.no_grad()
def dl_grids(name: str, module: str, path: str, data: dict) -> dict:
    """Run one checkpoint over every window; return {horizon: (pred[T, N], lead_min)}."""
    gnn = importlib.import_module("spatial.gnn")
    model, ei, cd = importlib.import_module(module).load_checkpoint(path, device=DEVICE)
    model = model.to(DEVICE).eval()
    L, H, step, A = cd["lookback"], cd["horizon"], cd["step_size"], cd["accumulation_window"]
    tidx = tuple(cd["feature_cols"]).index(cd["target_col"])
    X_clean, mu, sigma, obs = data["X_clean"], data["mu"], data["sigma"], data["obs"]
    raw = torch.load(path, map_location="cpu")
    src = raw if "mu" in raw else cd
    if "mu" in src:
        mu_v, sd_v = (torch.as_tensor(src[k], dtype=torch.float32) for k in ("mu", "sigma"))
    else:
        log("  [WARN] no training mu/sigma in checkpoint — using the period's statistics")
        mu_v, sd_v = mu.clone(), sigma.clone()
    X_norm = (X_clean * (sigma + 1e-8) + mu - mu_v) / (sd_v + 1e-8)
    t_mu, t_sd = float(mu_v[tidx]), float(sd_v[tidx]) + 1e-8
    to_phys = lambda y: y * t_sd + A * t_mu                 # target = SUM of A normalised minutes

    ds = gnn.WindowedDataset(X_norm, L, H, step, A, tidx)
    n_win = len(ds)
    get_batch = _fast_windows(ds, X_norm, L)
    log(f"  windows: {n_win:,} | inputs: {'fast slicing' if get_batch else 'DataLoader'}")
    ei = ei.to(DEVICE)

    def run(bs):
        preds, t0, nxt = [], time.time(), 0.1
        if get_batch is not None:
            batches = ((i, get_batch(i, min(i + bs, n_win))) for i in range(0, n_win, bs))
        else:
            loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
            batches = ((k * bs, b[0]) for k, b in enumerate(loader))
        for i, x in batches:
            out = model(x.to(DEVICE, non_blocking=True), ei)
            preds.append(to_phys(out.float().cpu().numpy()).astype(np.float32))
            del out
            done = min(i + bs, n_win) / n_win
            if done >= nxt:
                el = time.time() - t0
                log(f"  {100 * done:5.1f}%  ({el / 60:.1f} min, ~{el / done * (1 - done) / 60:.1f} min left)")
                nxt += 0.1
        return np.concatenate(preds, 0)

    bs = BATCH_SIZE
    while True:
        try:
            preds = run(bs)
            break
        except torch.cuda.OutOfMemoryError:
            gc.collect(); torch.cuda.empty_cache()
            if bs <= 8:
                raise
            bs //= 2
            log(f"  [OOM] retrying with batch_size={bs} "
                f"(GPU free {torch.cuda.mem_get_info()[0] / 2**30:.1f} GiB — check nvidia-smi)")

    n_chk = min(n_win, 20_000)
    tgt = np.stack([to_phys(np.asarray(ds[i][-1])) for i in range(n_chk)])[:, 0]
    idx = np.arange(n_chk)
    best = min(((float(np.nanmean(np.abs(tgt - obs[idx + L + off]))), off) for off in range(-A, A + 1)
                if (idx + L + off).min() >= 0 and (idx + L + off).max() < obs.shape[0]))
    log(f"  target check: mean |target − obs| = {best[0]:.4f} (offset {best[1]})")
    if best[0] > 0.5:
        raise RuntimeError(f"{name}: dataset targets do not match observed sums")
    target_index = np.arange(preds.shape[0])[:, None] + L + best[1] + np.arange(H)[None, :] * step
    grids = tta.windows_to_valid_time(preds, target_index, T=obs.shape[0])
    del preds, model, ds, X_norm; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {f"t+{(h + 1) * step}min": (grids[h], float(h * step + A)) for h in range(H)}


# =============================================================================
# CACHE
# =============================================================================

def _cache_path(name: str, hz: str) -> Path:
    return OUT_DIR / "cache" / f"{EVAL_PERIOD['name']}__{name}__{hz}.npy"


def cached(name: str, shape: tuple, make) -> dict:
    """{horizon: (grid, lead)} from cache, or from make() and then cached."""
    meta = OUT_DIR / "cache" / f"{EVAL_PERIOD['name']}__{name}.json"
    if USE_CACHE and meta.exists():
        m = json.load(open(meta))
        try:
            out = {hz: (np.load(_cache_path(name, hz), mmap_mode="r"), float(L)) for hz, L in m.items()}
            if all(g.shape == shape for g, _ in out.values()):
                log(f"  [CACHE] loaded {name}")
                return {hz: (np.asarray(g), L) for hz, (g, L) in out.items()}
        except Exception:
            pass
    out = make()
    if USE_CACHE:
        meta.parent.mkdir(parents=True, exist_ok=True)
        for hz, (g, _) in out.items():
            np.save(_cache_path(name, hz), g)
        json.dump({hz: L for hz, (_, L) in out.items()}, open(meta, "w"))
    return out


# =============================================================================
# ANALYSIS
# =============================================================================

def _sign_test_p(n_pos: int, n_neg: int) -> float:
    n = n_pos + n_neg
    if n == 0:
        return float("nan")
    if n > 1000:
        return float(math.erfc((abs(n_pos - n_neg) - 1) / math.sqrt(n) / math.sqrt(2)))
    k = min(n_pos, n_neg)
    return float(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n))


def paired_table(pe: pd.DataFrame, h_report: float, pairs) -> pd.DataFrame:
    """B minus A on the same events (negative = B faster). CI bootstrapped over DAYS."""
    pe = pe[pe.h_pred == h_report]
    rng = np.random.default_rng(0)
    rows = []
    for a_name, b_name in pairs:
        a, b = pe[pe.model == a_name], pe[pe.model == b_name]
        if a.empty or b.empty:
            continue
        ha, hb = a.horizon.iloc[0], b.horizon.iloc[0]
        j = (a[a.horizon == ha][["node", "onset", "tta_min"]]
             .merge(b[b.horizon == hb][["node", "onset", "tta_min"]], on=["node", "onset"], suffixes=("_a", "_b"))
             .dropna())
        if j.empty:
            continue
        j["d"] = j.tta_min_b - j.tta_min_a
        days = j.groupby(pd.DatetimeIndex(j.onset).normalize()).d.agg(["sum", "count"])
        idx = rng.integers(0, len(days), (N_BOOT, len(days)))
        boot = days["sum"].to_numpy()[idx].sum(1) / days["count"].to_numpy()[idx].sum(1)
        n_b, n_a = int((j.d < 0).sum()), int((j.d > 0).sum())
        rows.append(dict(A=a_name, B=b_name, family_A=FAMILY.get(a_name), family_B=FAMILY.get(b_name),
                         n_events=len(j), n_days=len(days), mean_diff_B_minus_A=float(j.d.mean()),
                         ci_lo=float(np.percentile(boot, 2.5)), ci_hi=float(np.percentile(boot, 97.5)),
                         B_faster_pct=100 * n_b / len(j), A_faster_pct=100 * n_a / len(j),
                         tie_pct=100 * float((j.d == 0).mean()), sign_test_p=_sign_test_p(n_b, n_a)))
    return pd.DataFrame(rows)


def matched_far(table_all: pd.DataFrame, fars=(0.05, 0.10, 0.20)) -> pd.DataFrame:
    rows = []
    for m, g in table_all.groupby("model", sort=False):
        if m == "HistProfile":
            continue
        g = g.sort_values("FAR_per_det_day")
        x = g.FAR_per_det_day.to_numpy()
        r = {"model": m, "family": FAMILY.get(m), "FAR_min": x.min(), "FAR_max": x.max()}
        for f in fars:
            inside = x.min() <= f <= x.max()
            r[f"TTA_mean_capped@{f}"] = float(np.interp(f, x, g.TTA_mean_capped)) if inside else np.nan
            r[f"caught_pct@{f}"] = float(np.interp(f, x, g.caught_pct)) if inside else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    log(f"[CONFIG] eval={EVAL_PERIOD}  trees trained on {TREE_TRAIN_PERIOD}  device={DEVICE}")
    log(f"[CONFIG] DL={list(DL_MODELS)}  ML={TREE_MODELS}")
    log(f"[CONFIG] TTA {TTA_CFG}")

    specs = {}
    for name, (_, path, _) in DL_MODELS.items():
        try:
            c = torch.load(path, map_location="cpu")["config"]
            specs[name] = dict(L=int(c["lookback"]), H=int(c["horizon"]), step=int(c["step_size"]),
                               A=int(c["accumulation_window"]))
        except Exception as e:
            log(f"[WARN] cannot read config of {name}: {e}")
    A = next(iter(specs.values()))["A"] if specs else 5
    if any(s["A"] != A for s in specs.values()):
        raise ValueError(f"models disagree on accumulation_window: {specs}")

    # ── Evaluation data ──────────────────────────────────────────────────────
    log(f"\n{'=' * 70}\nEVALUATION DATA {EVAL_PERIOD}\n{'=' * 70}")
    data = load_period(EVAL_PERIOD, A, keep_X=True)
    times, obs, q, node_ids = data["times"], data["obs"], data["q"], data["node_ids"]
    shape = obs.shape
    predictions = {}
    if INCLUDE_HIST_AVERAGE:
        predictions["HistAverage"] = {f"t+{A}min": (hist_average_grid(q, data["lookback"], A), float(A))}

    # ── DL and hybrid DL ─────────────────────────────────────────────────────
    for name, (module, path, fam) in DL_MODELS.items():
        log(f"\n[{fam}] ── {name} ──")
        try:
            predictions[name] = cached(name, shape, lambda: dl_grids(name, module, path, data))
        except Exception as e:
            log(f"  [ERROR] {name} failed: {type(e).__name__}: {e}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    for k in ("X_clean", "mu", "sigma"):
        data.pop(k, None)
    gc.collect()

    # ── ML ───────────────────────────────────────────────────────────────────
    need_trees = [m for m in TREE_MODELS if not (USE_CACHE and (OUT_DIR / "cache" / f"{EVAL_PERIOD['name']}__{m}.json").exists())]
    trees = train_trees(A) if need_trees else {"models": {}, "profile": None}
    if trees["profile"] is not None:
        normal_ev, _ = trees["profile"].lookup(times)
        normal_ev = np.nan_to_num(normal_ev.astype(np.float32), nan=float(np.nanmean(normal_ev)))
    for name in TREE_MODELS:
        log(f"\n[ML] ── {name} ──")
        try:
            if name in need_trees and name not in trees["models"]:
                raise RuntimeError("model was not trained (see error above)")
            predictions[name] = cached(name, shape, lambda: {f"t+{A}min": (
                tree_grid(trees["models"][name], obs, q, normal_ev, times, A), float(A))})
        except Exception as e:
            log(f"  [ERROR] {name} failed: {type(e).__name__}: {e}")
    del trees; gc.collect()

    # ── Score ────────────────────────────────────────────────────────────────
    profile = tta.build_profile(obs, times, bin_min=5, spread_floor_abs=1.0)
    L_max = max([s["L"] + (s["H"] - 1) * s["step"] for s in specs.values()] + [TREE_LOOKBACK + A])
    H_tail = max([(s["H"] - 1) * s["step"] + s["A"] for s in specs.values()] + [A])
    window = (times[L_max + int(TTA_CFG.warmup_min)], times[len(times) - H_tail - int(TTA_CFG.max_wait_min) - 1])
    log(f"\n[TTA] scoring {window[0]} → {window[1]} | models: {list(predictions)}")
    res = tta.evaluate(obs, times, predictions, profile, window, TTA_CFG,
                       scenario=EVAL_PERIOD["name"], node_ids=node_ids)
    c = res.config
    log(f"[TTA] candidates {c['n_candidate_episodes']} | low normal level {c.get('n_dropped_low_level')} | "
        f"too short {c['n_dropped_short']} | too shallow {c['n_dropped_shallow']} | events {len(res.events)} "
        f"= {c['events_per_detector_day']:.3f} per detector-day")

    table = tta.summary_table(res)
    table_all = tta.summary_table(res, h_pred="all")
    table.insert(1, "family", table.model.map(FAMILY))
    fam = (table[table.family.isin(["ML", "DL", "Hybrid DL"])]
           .groupby("family")[["MAE", "RMSE", "MAPE", "TTA_median", "TTA_mean_capped", "caught_pct",
                                "pre_alarmed_pct", "FAR_per_det_day"]].mean()
           .reindex(["ML", "DL", "Hybrid DL"]))
    pairs = paired_table(res.per_event, c["h_report"], PAIRS)
    mf = matched_far(table_all)

    table.to_csv(OUT_DIR / "summary.csv", index=False)
    table_all.to_csv(OUT_DIR / "summary_all_h.csv", index=False)
    fam.to_csv(OUT_DIR / "families.csv")
    pairs.to_csv(OUT_DIR / "pairs.csv", index=False)
    mf.to_csv(OUT_DIR / "matched_far.csv", index=False)
    res.events.to_csv(OUT_DIR / "events.csv", index=False)
    res.per_event.to_csv(OUT_DIR / "per_event.csv.gz", index=False)
    with open(OUT_DIR / "config.json", "w") as f:
        json.dump({"eval_period": EVAL_PERIOD, "tree_train_period": TREE_TRAIN_PERIOD,
                   "dl_models": DL_MODELS, "tree_models": TREE_MODELS, "tree_features": TREE_FEATURES,
                   "rf_params": RF_PARAMS, "xgb_params": XGB_PARAMS, "tree_train_rows": TREE_TRAIN_ROWS,
                   "tta": c}, f, indent=2, default=str)

    with pd.option_context("display.width", 260, "display.max_columns", 30):
        cols = ["model", "family", "MAE", "RMSE", "MAPE", "n_events", "TTA_median", "TTA_q1", "TTA_q3",
                "TTA_mean_capped", "caught_pct", "pre_alarmed_pct", "A_idx", "FAR_per_det_day"]
        print(f"\n{'=' * 100}\nERROR AND TIME TO ADAPT — {EVAL_PERIOD['name']}\n{'=' * 100}")
        print(table[[x for x in cols if x in table]].round(2).to_string(index=False))
        print(f"\n{'=' * 100}\nPER FAMILY (mean over models)\n{'=' * 100}")
        print(fam.round(2).to_string())
        print(f"\n{'=' * 100}\nAT MATCHED FALSE-ALARM RATE\n{'=' * 100}")
        print(mf.round(2).to_string(index=False))
        print(f"\n{'=' * 100}\nPAIRED, SAME EVENTS (B minus A; negative = B adapts faster)\n{'=' * 100}")
        print(pairs.round(3).to_string(index=False))
        learned = table[table.family.isin(["ML", "DL", "Hybrid DL"])]
        if len(learned) >= 3:
            tau = learned[["MAE", "TTA_mean_capped"]].corr("kendall").iloc[0, 1]
            print(f"\nKendall tau (MAE rank vs TTA rank, learned models) = {tau:.2f}")
            print("  by MAE:", list(learned.sort_values("MAE").model))
            print("  by TTA:", list(learned.sort_values("TTA_mean_capped").model))

    # ── Figures ──────────────────────────────────────────────────────────────
    first_h = {m: g.horizon.iloc[0] for m, g in table.groupby("model", sort=False)}
    order = ["Persistence", "HistAverage"] + [m for m in TREE_MODELS + list(DL_MODELS) if m in predictions]
    for fname, fn in [("fig_adaptation_curve.png",
                       lambda ax: tta.plot_adaptation_curve(res, first_h, models=order, ax=ax)),
                      ("fig_error_vs_tta.png", lambda ax: tta.plot_error_vs_tta(table, first_h, ax=ax)),
                      ("fig_tta_vs_far.png", lambda ax: tta.plot_tta_far(table_all, first_h, ax=ax))]:
        fig, ax = plt.subplots(figsize=(7, 5))
        fn(ax); fig.tight_layout(); fig.savefig(OUT_DIR / fname, dpi=150); plt.close(fig)
    _example_events(res, first_h, "XGBoost", "GCN-BiLSTM")
    _example_events(res, first_h, "GCN", "GCN-BiLSTM")
    log(f"\n[SAVE] {OUT_DIR}  (total {(time.time() - t_start) / 60:.1f} min)")


def _example_events(res, first_h, a, b):
    """Events where a and b differ most (both directions), for the paper's real-data Fig. 1."""
    import matplotlib.pyplot as plt
    pe = res.per_event[res.per_event.h_pred == res.config["h_report"]]
    if a not in first_h or b not in first_h:
        return
    j = (pe[(pe.model == a) & (pe.horizon == first_h[a])][["node", "onset", "tta_min"]]
         .merge(pe[(pe.model == b) & (pe.horizon == first_h[b])][["node", "onset", "tta_min"]],
                on=["node", "onset"], suffixes=("_a", "_b")).dropna())
    if j.empty:
        return
    j["diff"] = j.tta_min_a - j.tta_min_b
    ev = res.events.reset_index(drop=True)
    key = {k: i for i, k in enumerate(zip(ev.node, ev.onset))}
    picks = pd.concat([j.nlargest(N_EXAMPLE_EVENTS // 2, "diff"), j.nsmallest(N_EXAMPLE_EVENTS // 2, "diff")])
    for r, row in enumerate(picks.itertuples()):
        fig, ax = plt.subplots(figsize=(8, 3.6))
        tta.plot_event(res, event=key[(row.node, row.onset)], models=[a, b, "Persistence"], ax=ax)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"fig_event_{a}_vs_{b}_{r + 1}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()