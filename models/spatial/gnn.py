"""
Spatio-temporal GNN baseline (global graph + lane aggregation) for traffic forecasting.

YOU HAVE
--------
1) Global graph definition:
   - global_nodes_roworder.csv
       * global_node_id: 0..N-1 (the authoritative node order)
       * node_key: "corridor|site" (unique identifier)
   - global_edges_roworder.csv
       * from_global_id, to_global_id (directed edges)
       * includes the connector edge (A99 -> A9)

2) A mapping table between a *site* (from node_key) and the lane-units used in filenames:
   - selection_FG1.csv  (semicolon-separated)
       * site  (e.g. MQ99-690-Stg_H)
       * unit  (e.g. DET99510_Stg_H1, DET99510_Stg_H2, ...)

3) Lane CSV files:
   - One CSV per lane unit, e.g.
       FG1_Kurz_DET99510_Stg_H1_agg1min_2026-01-01_bis_2026-02-20.csv
       FG1_Kurz_DET99510_Stg_H2_agg1min_2026-01-01_bis_2026-02-20.csv
   - 1-minute resolution, ~2 months (will extend later)
   - semicolon-separated
   - have a unified timestamp column 't_start' across files
   - may contain hourly flows q_*_h instead of per-minute q_* (we convert by /60 if needed)

WHAT THIS SCRIPT DOES
---------------------
A) Builds the static graph used by PyTorch Geometric:
   - Reads nodes + edges CSVs
   - Builds edge_index [2, E] tensor

B) Builds the spatio-temporal tensor X:
   - For each node_key in global_node_id order:
       1) Extract site from node_key
       2) Look up lane units for that site in selection_FG1.csv
       3) Find matching lane CSVs in data_dir using a filename pattern
       4) Load each lane CSV
       5) Aggregate lanes H1..H4 into one detector-level series using reasonable rules:
          * q_* summed across lanes
          * v_kfz flow-weighted by q_kfz
          * b, t_netto, s_kfz averaged across lanes
   - Align timestamps across all nodes
   - Stack into X[T, N, F]

C) Creates supervised samples via sliding windows:
   - x_hist: [L, N, F] past lookback minutes
   - y_fut:  [H, N]    future horizon minutes of target feature (default q_kfz)

D) Trains a baseline GNN forecaster:
   - Mean pool over time in the lookback window (simple temporal baseline)
   - Two GCNConv layers to mix information across graph neighbors
   - MLP head to predict horizon for each node

DEBUG OUTPUT (requested)
------------------------
Before training, the script writes a debug CSV that helps you validate the mapping:
  debug_node_to_files.csv

Each row shows:
  - global_node_id
  - node_key, corridor, site
  - lane unit name (from selection_FG1.csv)
  - resolved file path (found on disk)
  - lane_id (H1..Hn)

This lets you quickly catch mistakes like:
  - a site missing in selection_FG1.csv
  - a lane unit with no matching file in data_dir
  - unexpected lane counts

HOW TO RUN
----------
1) Put this script in the project folder, e.g. gnn_global_baseline.py
2) Ensure these files exist (paths configurable below):
     - global_nodes_roworder.csv
     - global_edges_roworder.csv
     - meta_fixed.csv
     - a folder with FG1_Kurz_*.csv lane files
3) Run:
     python gnn_global_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODELS_DIR = ROOT / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

import os

from imputation import run_imputation_pipeline

import glob
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch

import json
from pathlib import Path
from datetime import datetime
import signal
import time

# =============================================================================
# 0) CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """
    Central config object so you can change paths/hyperparameters in one place.
    """

    # --- Graph definition files (produced from your node mappings) ---
    # nodes_csv: str = "global/a8-a99-a9/a8-a99-a9-global_nodes.csv"
    # edges_csv: str = "global/a8-a99-a9/a8-a99-a9-global_edges.csv"
    # --- New csv with 30 nodes removed because of missing data ---
    nodes_csv: str = "global/a8-a99-a9_cleaned/a8-a99-a9-global_nodes_cleaned.csv"
    edges_csv: str = "global/a8-a99-a9_cleaned/a8-a99-a9-global_edges_cleaned.csv"

    # --- Site->unit mapping file (given by you) ---
    selection_csv: str = "meta_fixed.csv"

    # --- Folder containing FG1_Kurz_*.csv lane files ---
    # Example lane file:
    #   FG1_Kurz_DET99510_Stg_H1_agg1min_2026-01-01_bis_2026-02-20.csv
    data_dir: str = os.environ.get("TTA_DATA_DIR", str(ROOT / "data" / "autobahn"))

    # --- Debug output: where to write node->files mapping ---
    debug_mapping_out: str = "debug_node_to_files.csv"

    # --- Features per node after lane aggregation ---
    # You can start with fewer, e.g. ("q_kfz","v_kfz","b","t_netto","s_kfz")
    # All available features: q_kfz, q_lkw, q_pkw, v_kfz, v_lkw, v_pkw, b, t_netto, s_kfz
    feature_cols: Tuple[str, ...] = (
        "q_kfz", "q_lkw", "q_pkw",
        "v_kfz",
        "b"
    )

    # --- Target variable (must be in feature_cols) ---
    target_col: str = "q_kfz"

    # --- Supervised windowing ---
    # currently 1 since no temporal modeling implemented
    lookback: int = 30  # minutes of history
    horizon: int = 1    # number of steps ahead to predict
    step_size: int = 5   # minutes per step
    accumulation_window: int = 5  # how many 1-min steps to sum into the target

    # --- Training hyperparameters ---
    batch_size: int = 128
    lr: float = 3e-4
    epochs: int = 1000  # we'll use early stopping, so this is a max limit
    train_ratio: float = 0.8  # chronological split

    patience: int = 30  # epochs to wait for improvement before stopping
    min_delta: float = 0.01  # minimum improvement to reset patience

    # --- Model capacity ---
    hidden: int = 256

    num_nodes: int = 0  # will be set after loading graph
    # --- freeform notes about the run ---
    notes: str = ""

    # --- NaN strategy for baseline stability ---
    nan_fill_value: float = 0.0

    # --- Device ---
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # date_suffix: str = "2026-01-01_bis_2026-02-20"    # 2 month data
    date_suffix: str = "2025-01-01_bis_2026-01-01"      # 1 year data

def log(msg: str) -> None:
    """Simple print wrapper; easy to swap for logging later."""
    print(msg, flush=True)


# =============================================================================
# 1) LOAD GLOBAL GRAPH (NODES + EDGES)
# =============================================================================

def load_global_graph(nodes_csv: str, edges_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame, torch.Tensor]:
    """
    Load the global node and edge tables, and build PyG's edge_index.

    Returns
    -------
    nodes:
        DataFrame sorted by global_node_id (0..N-1).
        Must include node_key like "corridor|site".

    edges:
        DataFrame containing from_global_id and to_global_id.

    edge_index:
        torch.LongTensor shape [2, E], where:
          edge_index[0] = sources
          edge_index[1] = targets
    """
    nodes = pd.read_csv(nodes_csv).sort_values("node_id").reset_index(drop=True)
    edges = pd.read_csv(edges_csv)

    # PyTorch Geometric expects edge_index in this shape.
    edge_index = torch.tensor(
        [edges["src"].to_numpy(), edges["dst"].to_numpy()],
        dtype=torch.long
    )

    return nodes, edges, edge_index


# =============================================================================
# 2) LOAD SELECTION MAP: site -> list of unit strings (H1..H4)
# =============================================================================

def load_selection_site_to_units(selection_csv: str) -> Dict[str, List[str]]:
    """
    selection_FG1.csv is semicolon-separated and uses decimal commas in numeric columns.
    We only need 'site' and 'unit'.

    Returns
    -------
    site_to_units:
        Dict where each key is a site string and the value is a list of units (lane names).
        Example:
          site_to_units["MQ99-690-Stg_H"] = ["DET99510_Stg_H1", "DET99510_Stg_H2", ...]
    """
    sel = pd.read_csv(selection_csv, sep=";", decimal=",")

    if "site" not in sel.columns or "unit" not in sel.columns:
        raise ValueError(f"{selection_csv} must contain columns 'site' and 'unit'")

    site_to_units: Dict[str, List[str]] = {}
    for site, g in sel.groupby("site", sort=False):
        # Preserve the order in the selection file as much as possible.
        units = g["unit"].astype(str).tolist()

        # Remove duplicates while preserving order.
        seen = set()
        uniq: List[str] = []
        for u in units:
            if u not in seen:
                uniq.append(u)
                seen.add(u)

        site_to_units[str(site)] = uniq

    return site_to_units


def parse_lane_from_unit(unit: str) -> int:
    """
    Extract lane number from unit string.

    Examples:
      DET99510_Stg_H1  -> 1
      DET99-845-Stg_H2 -> 2

    Returns
    -------
    lane_id: int
    """
    # Extract the lane number from the end of a unit string.
    m = re.search(r"(\d+)\s*$", str(unit))

    if not m:
        return 1

    return int(m.group(1))


def find_lane_file(data_dir: str, unit: str, date_suffix: str) -> str:
    """
    Find the CSV file for a given unit (lane).

    Your naming convention:
      FG1_Kurz_{unit}_agg1min_YYYY-MM-DD_bis_YYYY-MM-DD.csv

    We use a glob so you can swap date ranges without changing code:
      FG1_Kurz_{unit}_agg1min_*.csv

    Returns
    -------
    path: str
        The matched file path.
    """
    pattern = os.path.join(data_dir, f"FG1_Kurz_{unit}_agg1min_{date_suffix}.csv")
    hits = sorted(glob.glob(pattern))

    if not hits:
        raise FileNotFoundError(f"No file matched pattern: {pattern}")

    # If there are multiple matches (rare), pick shortest name as a heuristic.
    if len(hits) > 1:
        hits = sorted(hits, key=lambda p: len(os.path.basename(p)))

    return hits[0]


# =============================================================================
# 3) LANE CSV LOADING + NORMALIZATION
# =============================================================================

def ensure_per_minute_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Some files contain q_*_h (per hour) instead of q_* (per minute).
    We convert by dividing by 60 if q_* is missing and q_*_h exists.
    """
    for base in ["q_kfz", "q_lkw", "q_pkw"]:
        if base not in df.columns and f"{base}_h" in df.columns:
            df[base] = df[f"{base}_h"] / 60.0
    return df


def load_lane_csv(path: str) -> pd.DataFrame:
    """
    Load one lane CSV into a DataFrame.
    Assumes semicolon separated and has 't_start'.
    """
    df = pd.read_csv(path, sep=";")
    df["t_start"] = pd.to_datetime(df["t_start"])
    df = df.sort_values("t_start").reset_index(drop=True)
    df = ensure_per_minute_cols(df)
    return df


# =============================================================================
# 4) LANE AGGREGATION (H1..H4 -> ONE NODE SERIES)
# =============================================================================

def aggregate_lanes_for_detector(
    lane_dfs: Dict[int, pd.DataFrame],
    feature_cols: Tuple[str, ...],
    nan_fill_value: float = 0.0,
) -> pd.DataFrame:
    """
    Aggregate multiple lanes for one detector/site into a single DataFrame.

    Output columns:
      t_start + feature_cols

    Baseline aggregation rules:
      - q_kfz, q_lkw, q_pkw: sum across lanes
      - v_kfz: flow-weighted mean using q_kfz weights if available
      - v_lkw: flow-weighted mean using q_lkw weights if available else mean
      - v_pkw: flow-weighted mean using q_pkw weights if available else mean
      - b, t_netto, s_kfz: mean across lanes

    NOTE:
      We inner-join timestamps across lanes so that each output row corresponds
      to a timestamp present in ALL available lanes. This avoids misalignment.
    """
    lane_ids = sorted(lane_dfs.keys())
    if not lane_ids:
        raise ValueError("No lanes to aggregate.")

    # 1) Build common timeline by inner join on t_start across lanes.
    base = lane_dfs[lane_ids[0]][["t_start"]].copy()
    for lid in lane_ids[1:]:
        base = base.merge(lane_dfs[lid][["t_start"]], on="t_start", how="inner")

    # 2) Merge lane values onto the common timeline, suffixing columns per lane.
    aligned = base
    for lid in lane_ids:
        # Ensure flows are present for weighting; include all requested features.
        cols = ["t_start"] + list(set(feature_cols) | {"q_kfz", "q_lkw", "q_pkw"})
        cols = [c for c in cols if c in lane_dfs[lid].columns]
        part = lane_dfs[lid][cols].copy()
        part = part.rename(columns={c: f"{c}_H{lid}" for c in part.columns if c != "t_start"})
        aligned = aligned.merge(part, on="t_start", how="left")

    aligned = aligned.fillna(nan_fill_value)

    out = pd.DataFrame({"t_start": aligned["t_start"]})

    def sum_across(base_name: str) -> np.ndarray:
        cols = [c for c in aligned.columns if c.startswith(base_name + "_H")]
        if not cols:
            return np.zeros(len(aligned), dtype=np.float32)
        return aligned[cols].to_numpy(np.float32).sum(axis=1)

    def mean_across(base_name: str) -> np.ndarray:
        cols = [c for c in aligned.columns if c.startswith(base_name + "_H")]
        if not cols:
            return np.zeros(len(aligned), dtype=np.float32)
        return aligned[cols].to_numpy(np.float32).mean(axis=1)

    # --- Flows: sum across lanes ---
    for flow in ["q_kfz", "q_lkw", "q_pkw"]:
        if flow in feature_cols:
            out[flow] = sum_across(flow)

    # --- Speeds: weighted means when possible ---
    # v_kfz weighted by q_kfz
    if "v_kfz" in feature_cols:
        v_cols = [c for c in aligned.columns if c.startswith("v_kfz_H")]
        q_cols = [c for c in aligned.columns if c.startswith("q_kfz_H")]
        if v_cols and q_cols:
            V = aligned[v_cols].to_numpy(np.float32)
            Q = aligned[q_cols].to_numpy(np.float32)
            out["v_kfz"] = (Q * V).sum(axis=1) / (Q.sum(axis=1) + 1e-6)
        else:
            out["v_kfz"] = mean_across("v_kfz")

    # v_lkw weighted by q_lkw if present
    if "v_lkw" in feature_cols:
        v_cols = [c for c in aligned.columns if c.startswith("v_lkw_H")]
        q_cols = [c for c in aligned.columns if c.startswith("q_lkw_H")]
        if v_cols and q_cols:
            V = aligned[v_cols].to_numpy(np.float32)
            Q = aligned[q_cols].to_numpy(np.float32)
            out["v_lkw"] = (Q * V).sum(axis=1) / (Q.sum(axis=1) + 1e-6)
        else:
            out["v_lkw"] = mean_across("v_lkw")

    # v_pkw weighted by q_pkw if present
    if "v_pkw" in feature_cols:
        v_cols = [c for c in aligned.columns if c.startswith("v_pkw_H")]
        q_cols = [c for c in aligned.columns if c.startswith("q_pkw_H")]
        if v_cols and q_cols:
            V = aligned[v_cols].to_numpy(np.float32)
            Q = aligned[q_cols].to_numpy(np.float32)
            out["v_pkw"] = (Q * V).sum(axis=1) / (Q.sum(axis=1) + 1e-6)
        else:
            out["v_pkw"] = mean_across("v_pkw")

    # --- Other features: mean across lanes ---
    for c in ["b", "t_netto", "s_kfz"]:
        if c in feature_cols:
            out[c] = mean_across(c)

    # Ensure every requested feature exists and is float32.
    for c in feature_cols:
        if c not in out.columns:
            out[c] = nan_fill_value
        out[c] = out[c].astype(np.float32)

    return out


# =============================================================================
# 5) BUILD THE SPATIO-TEMPORAL PANEL X[T, N, F]
# =============================================================================

def build_panel_X(
    nodes: pd.DataFrame,
    site_to_units: Dict[str, List[str]],
    data_dir: str,
    feature_cols: Tuple[str, ...],
    nan_fill_value: float = 0.0,
    debug_mapping_out: str = "debug_node_to_files.csv",
) -> Tuple[torch.Tensor, pd.Series]:
    """
    Build the 3D tensor X[T, N, F] in the exact global node order.

    Steps per node:
      1) Parse site from node_key ("corridor|site")
      2) Find units (H1..Hn) in selection mapping
      3) Resolve each unit to an actual file path
      4) Load lane CSVs
      5) Aggregate lanes -> one DF per node

    Additionally:
      - Write debug_node_to_files.csv so you can verify all mappings.
    """
    node_keys = nodes["site"].tolist()

    # We'll collect debug rows that show exactly which files each node uses.
    debug_rows: List[Dict[str, object]] = []

    det_dfs: List[pd.DataFrame] = []

    for i, nk in enumerate(node_keys):
        site = nk
        #print(site)

        # 1) Ensure site exists in selection file.
        if site not in site_to_units:
            raise KeyError(f"Site '{site}' not found in selection mapping (selection_FG1.csv). node_key={nk}")

        # 2) For this site, we have multiple units (one per lane).
        units = site_to_units[site]

        # 3) Load lane files for each unit.
        lane_dfs: Dict[int, pd.DataFrame] = {}
        for unit in units:
            lane_id = parse_lane_from_unit(unit)
            path = find_lane_file(data_dir, unit, Config.date_suffix)

            # record debug mapping row
            debug_rows.append({
                "global_node_id": int(nodes.loc[i, "node_id"]),
                "node_key": nk,
                #"corridor": corridor,
                "site": site,
                "unit": unit,
                "lane_id": lane_id,
                "file_path": path,
            })

            lane_dfs[lane_id] = load_lane_csv(path)

        # 4) Aggregate lanes into one node-level DF
        det_df = aggregate_lanes_for_detector(lane_dfs, feature_cols, nan_fill_value)
        det_dfs.append(det_df)

        # Progress log
        if i in (0, len(node_keys) - 1) or (i + 1) % 20 == 0:
            log(f"[DATA] node {i+1}/{len(node_keys)} {nk} -> lanes={sorted(lane_dfs.keys())}")

    # Write debug mapping file
    debug_df = pd.DataFrame(debug_rows)
    debug_df.to_csv(debug_mapping_out, index=False)
    log(f"[DEBUG] wrote mapping file: {debug_mapping_out} (rows={len(debug_df)})")

    # ------------------------------------------------------------------------
    # Build X[T, N, F] assuming timestamps are identical across files
    # (your datasource guarantees this, so we avoid expensive merges)
    # ------------------------------------------------------------------------

    times = det_dfs[0]["t_start"]
    T = len(times)
    N = len(nodes)
    F = len(feature_cols)

    X = np.zeros((T, N, F), dtype=np.float32)

    for n, df in enumerate(det_dfs):
        if len(df) != T:
            raise ValueError(
                f"Timestamp length mismatch at node {n}: got {len(df)} expected {T}"
            )

        X[:, n, :] = df[list(feature_cols)].to_numpy(np.float32)

    log(f"[PANEL] Built X with shape T={T}, N={N}, F={F} ({times.min()} -> {times.max()})")
    return torch.tensor(X, dtype=torch.float32), times


# =============================================================================
# 6) WINDOWED DATASET FOR SUPERVISED LEARNING
# =============================================================================

class WindowedDataset(Dataset):
    """
    Turn X[T, N, F] into many supervised samples using a sliding window.

    Each item:
      x_hist: [L, N, F]  (lookback history)
      y_fut:  [H, N]     (future target)
    """
    def __init__(self, X: torch.Tensor, lookback: int, horizon: int, step_size: int, accumulation_window: int, target_feat_idx: int):
        self.X = X
        self.L = lookback
        self.H = horizon
        self.S = step_size
        self.A = accumulation_window
        self.target_idx = target_feat_idx
        T = X.shape[0]
        self.max_i = T - lookback - (horizon - 1) * step_size - accumulation_window + 1
        if self.max_i <= 0:
            raise ValueError("Not enough timesteps for chosen lookback/horizon/step_size.")

    def __len__(self) -> int:
        return self.max_i

    def __getitem__(self, i: int):
        x_hist = self.X[i : i + self.L]  # [L, N, F]

        y_windows = []
        for h in range(self.H):
            start = i + self.L + h * self.S
            window = self.X[start : start + self.A, :, self.target_idx].sum(dim=0)
            y_windows.append(window)
        y_fut = torch.stack(y_windows, dim=0)  # [H, N]
        return x_hist, y_fut


# =============================================================================
# 7) MODEL: SIMPLE GNN FORECASTER
# =============================================================================

class GNNForecaster(nn.Module):
    """
    A minimal spatio-temporal baseline.

    Temporal handling:
      - mean pooling over the lookback window (very simple, debuggable)

    Spatial handling:
      - 2-layer GCNConv over the graph defined by edge_index

    Output:
      - per-node horizon predictions
    """
    def __init__(self, in_feats: int, hidden: int, horizon: int, num_nodes: int, lookback: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.lookback = lookback
        self.in_feats = in_feats
        self.drop = nn.Dropout(p=0.1)

        # GCN COMMIT
        # GCNConv adds self-loops internally by default, so you don't need self-loop edges.
        self.gcn1 = GCNConv(lookback * in_feats, hidden)
        self.gcn2 = GCNConv(hidden, hidden)
        self.gcn3 = GCNConv(hidden, hidden)

        # Re-normalizes the GCN output before the head so an out-of-distribution
        # feature spike can't propagate through the unbounded Linear+ReLU stack
        # unchecked — mirrors self.out_norm in GCNBiLSTMForecaster / GCNMamba2Forecaster
        # (models/spatio_temporal/gnn_bilstm_a.py, gnn_mamba2.py).
        self.out_norm = nn.LayerNorm(hidden)

        # Bounds the raw skip path the same way out_norm bounds the GCN path,
        # so it can't hit self.head's plain Linear unbounded — LSTM/Mamba-based
        # siblings get away without this because their gates saturate; a plain
        # Linear+ReLU head does not.
        self.skip_norm = nn.LayerNorm(in_feats)

        # Head: node embedding + this node's own most recent raw reading -> horizon
        # outputs. The raw skip connection gives the head a clean, un-smoothed
        # anchor to this node's actual current state, bypassing the 3-hop GCN
        # mixing — mirrors the torch.cat([x_hist, h_gcn], ...) skip connection in
        # GCNBiLSTMForecaster / GCNMamba2Forecaster.
        self.head = nn.Sequential(
            nn.Linear(hidden + in_feats, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden, horizon)  # final layer to produce horizon outputs
        )

    def forward(self, x_hist: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x_hist: [B, L, N, F]
        edge_index: [2, E]
        returns: [B, H, N]
        """
        B, L, N, F = x_hist.shape
        # assert N == self.num_nodes, "Mismatch between node count in data and model."

        # 1) Temporal pooling: average over lookback dimension
        # x = x_hist.mean(dim=1)  # [B, N, F]
        # x = x_hist[:, -1, :, :]
        x = x_hist.permute(0, 2, 1, 3).reshape(B, N, L * F)   # [B, N, L*F]
        x_last = x_hist[:, -1, :, :].reshape(B * N, F)         # this node's own most recent raw reading

        # 2) Spatial GNN: PyG expects [N, F] per forward call.
        # Build a PyG Batch — the correct way to handle batched graphs
        data_list = [Data(x=x[b], edge_index=edge_index) for b in range(B)]
        batch = Batch.from_data_list(data_list)

        h = self.drop(torch.relu(self.gcn1(batch.x, batch.edge_index)))
        h = self.drop(torch.relu(self.gcn2(h, batch.edge_index)))
        h = self.drop(torch.relu(self.gcn3(h, batch.edge_index)))
        h = self.out_norm(h)
        h = torch.cat([h, self.skip_norm(x_last)], dim=-1)   # skip connection, now bounded too
        y = self.head(h)                        # [B*N, H]
        y = y.view(B, N, self.horizon).permute(0, 2, 1)
        return y



# =============================================================================
# 8) TRAIN / EVAL LOOPS
# =============================================================================

def train_one_epoch(model, loader, edge_index, opt, device, target_mu=0.0, target_sigma=1.0):
    model.train()

    H = None
    total_mse_per_h = None
    total_mae_per_h = None
    total_count = 0

    for x_hist, y in loader:
        x_hist = x_hist.to(device)
        y = y.to(device)

        opt.zero_grad()
        y_hat = model(x_hist, edge_index)

        # Training loss (MSE)
        loss = ((y_hat - y) ** 2).mean()    # training loss stays normalized
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)    # gradient clipping for stability
        opt.step()

        # Metrics in real units
        y_hat_real = y_hat * target_sigma + target_mu
        y_real     = y     * target_sigma + target_mu

        # Metrics
        bs = x_hist.size(0)
        if H is None:
            H = y.shape[1]  # number of horizon steps
            total_mse_per_h = torch.zeros(H, device=device)
            total_mae_per_h = torch.zeros(H, device=device)

        # Mean over batch and nodes, but keep horizon dim separate
        # y shape: [B, H, N]
        total_mse_per_h += ((y_hat_real - y_real) ** 2).mean(dim=(0, 2)) * bs
        total_mae_per_h += torch.abs(y_hat_real - y_real).mean(dim=(0, 2)) * bs
        total_count += bs

    rmse_per_h = (total_mse_per_h / total_count).sqrt().tolist()
    mae_per_h  = (total_mae_per_h  / total_count).tolist()

    # Also keep an overall scalar for early stopping / best model tracking
    mean_rmse = sum(rmse_per_h) / len(rmse_per_h)
    mean_mae  = sum(mae_per_h)  / len(mae_per_h)

    return {
        "rmse": mean_rmse,           # scalar, used for early stopping
        "mae":  mean_mae,            # scalar
        "rmse_per_h": rmse_per_h,    # e.g. [rmse_t+5, rmse_t+10]
        "mae_per_h":  mae_per_h,
    }



@torch.no_grad()
def eval_one_epoch(model, loader, edge_index, device, target_mu=0.0, target_sigma=1.0):
    model.eval()

    H = None
    total_mse_per_h = None
    total_mae_per_h = None
    total_count = 0

    for x_hist, y in loader:
        x_hist = x_hist.to(device)
        y = y.to(device)

        y_hat = model(x_hist, edge_index)

        # Inverse transform to original units
        y_hat_real = y_hat * target_sigma + target_mu
        y_real     = y     * target_sigma + target_mu

        bs = x_hist.size(0)

        if H is None:
            H = y.shape[1]  # number of horizon steps
            total_mse_per_h = torch.zeros(H, device=device)
            total_mae_per_h = torch.zeros(H, device=device)

        # Mean over batch and nodes, but keep horizon dim separate
        # y shape: [B, H, N]
        total_mse_per_h += ((y_hat_real - y_real) ** 2).mean(dim=(0, 2)) * bs
        total_mae_per_h += torch.abs(y_hat_real - y_real).mean(dim=(0, 2)) * bs
        total_count += bs

    rmse_per_h = (total_mse_per_h / total_count).sqrt().tolist()
    mae_per_h  = (total_mae_per_h  / total_count).tolist()

    # Also keep an overall scalar for early stopping / best model tracking
    mean_rmse = sum(rmse_per_h) / len(rmse_per_h)
    mean_mae  = sum(mae_per_h)  / len(mae_per_h)

    return {
        "rmse": mean_rmse,           # scalar, used for early stopping
        "mae":  mean_mae,            # scalar
        "rmse_per_h": rmse_per_h,    # e.g. [rmse_t+5, rmse_t+10]
        "mae_per_h":  mae_per_h,
    }

# =============================================================================
# 9) SAVE THE MODEL
# =============================================================================

class ModelSaver:
    """Handles best-model tracking and final checkpoint saving."""

    def __init__(self, out_dir: str = "checkpoints/gnn", run_dir: Path | None = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_rmse = float("inf")
        self.best_epoch = -1
        # Timestamped run folder so reruns don't overwrite
        if run_dir is not None:
            # Use the pre-created directory directly — no extra timestamped folder
            self.run_dir = Path(run_dir)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = self.out_dir / ts
        
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def maybe_save_best(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        edge_index: torch.Tensor,
        epoch: int,
        val_rmse: float,
        cfg: Config,
    ) -> bool:
        """Save only when val_rmse improves. Returns True if saved."""
        if np.isnan(val_rmse) or val_rmse >= self.best_val_rmse:
            return False

        self.best_val_rmse = val_rmse
        self.best_epoch = epoch
        path = self.run_dir / "best_model.pth"

        torch.save({
            # "epoch": self.best_epoch + cfg.patience,
            # when continuing training, start from the epoch after the best one
            "epoch": self.best_epoch, 
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "edge_index": edge_index.cpu(),
            "val_rmse": val_rmse,
            "config": vars(cfg),
        }, path)
        return True

    def save_progress(
        self,
        epoch: int,
        train_metrics: dict,
        val_metrics: dict,
        cfg: Config,
        times: pd.Series,
    ) -> None:
        """Overwrite progress JSON after every epoch — survives crashes."""
        summary = {
            "run_dir": str(self.run_dir),
            "notes": cfg.notes,
            "status": "in_progress",          # ← marks incomplete runs
            "last_completed_epoch": epoch,
            "best_epoch": self.best_epoch,
            "best_val_rmse": self.best_val_rmse,
            "last_train": train_metrics,
            "last_val": val_metrics,
            "data_range": {
                "start": str(times.min()),
                "end": str(times.max()),
                "n_timesteps": len(times),
            },
            "config": vars(cfg),
        }
        with open(self.run_dir / "run_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)


    def save_final(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        edge_index: torch.Tensor,
        epoch: int,
        train_metrics: dict,
        val_metrics: dict,
        cfg: Config,
        times: pd.Series,
        elapsed: float,
        duration_str: str,
    ) -> None:
        """Save the last epoch + run summary JSON."""
        # Final checkpoint
        torch.save({
            # "epoch": self.best_epoch + cfg.patience,
            # when continuing training, start from the epoch after the best one
            "epoch": self.best_epoch, 
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "edge_index": edge_index.cpu(),
            "val_rmse": val_metrics["rmse"],
            "config": vars(cfg),
        }, self.run_dir / "final_model.pth")

        # Human-readable summary
        summary = {
            "run_dir": str(self.run_dir),
            "status": "completed",
            "notes": cfg.notes,
            "duration": duration_str,
            "duration_seconds": int(elapsed),   # raw seconds for easier sorting/analysis
            "best_epoch": self.best_epoch,
            "best_val_rmse": self.best_val_rmse,
            "final_train": train_metrics,
            "final_val": val_metrics,
            "data_range": {
                "start": str(times.min()),
                "end": str(times.max()),
                "n_timesteps": len(times),
            },
            "config": vars(cfg),
        }
        with open(self.run_dir / "run_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        log(f"[SAVE] Run artifacts in: {self.run_dir}")
        log(f"[SAVE] Best epoch={self.best_epoch}, best val RMSE={self.best_val_rmse:.6f}")


def load_checkpoint(path: str, device: str = "cpu") -> Tuple[GNNForecaster, torch.Tensor, dict]:
    """Load a saved checkpoint and reconstruct the model."""
    ckpt = torch.load(path, map_location=device)
    cfg_dict = ckpt["config"]

    model = GNNForecaster(
        in_feats=len(cfg_dict["feature_cols"]),
        hidden=cfg_dict["hidden"],
        horizon=cfg_dict["horizon"],
        num_nodes=cfg_dict["num_nodes"],
        lookback=cfg_dict["lookback"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    edge_index = ckpt["edge_index"].to(device)
    return model, edge_index, cfg_dict


# =============================================================================
# 10) MAIN
# =============================================================================

def _timeout_handler(signum, frame):
    raise TimeoutError

def main(cfg: Config) -> None:
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(120)  # 2 minute timeout to prevent hanging on data loading
    try:
        notes = input("[NOTES] Describe this run (press Enter to skip): ").strip()
    except (TimeoutError, EOFError):
        notes = ""
        print("(no notes entered)")
    finally:
        signal.alarm(0)  # disable alarm
    cfg.notes = notes

    train_start = time.time()

    # 1) Load graph
    nodes, edges, edge_index = load_global_graph(cfg.nodes_csv, cfg.edges_csv)
    cfg.num_nodes = len(nodes)
    log(f"[GRAPH] N={len(nodes)} nodes, E={edge_index.shape[1]} edges")

    print(cfg.device)

    # 2) Load mapping site -> lane units
    site_to_units = load_selection_site_to_units(cfg.selection_csv)
    log(f"[MAP] Loaded selection mapping: {len(site_to_units)} unique sites")

    # 3) Build spatio-temporal tensor
    X, times = build_panel_X(
        nodes=nodes,
        site_to_units=site_to_units,
        data_dir=cfg.data_dir,
        feature_cols=cfg.feature_cols,
        nan_fill_value=cfg.nan_fill_value,
        debug_mapping_out=cfg.debug_mapping_out,
    )

    # 4) Windowing dataset
    if cfg.target_col not in cfg.feature_cols:
        raise ValueError(f"target_col={cfg.target_col} must be in feature_cols")

    target_idx = cfg.feature_cols.index(cfg.target_col)

    # --- ADD THIS BLOCK ---
    log(f"[DATA] Mean flow: {X[:, :, target_idx].mean().item():.2f}")
    log(f"[DATA] Std flow:  {X[:, :, target_idx].std().item():.2f}")
    # ----------------------

    # NEW — impute first, then window
    X_clean, M, mu, sigma = run_imputation_pipeline(
        X=X,
        edge_index=edge_index,        # pass BEFORE .to(device)
        feature_cols=cfg.feature_cols,
        train_ratio=cfg.train_ratio,
        edges_csv=cfg.edges_csv,
    )

    # Store mu/sigma for inverse-transforming predictions later
    target_mu    = float(mu[target_idx])
    target_sigma = float(sigma[target_idx])

    ds = WindowedDataset(X_clean, cfg.lookback, cfg.horizon, cfg.step_size, cfg.accumulation_window, target_idx)
    # ds = WindowedDataset(X, cfg.lookback, cfg.horizon, target_idx)

    # Chronological split (recommended for time series)
    n = len(ds)
    n_train = int(n * cfg.train_ratio)
    train_ds = torch.utils.data.Subset(ds, range(0, n_train))
    val_ds = torch.utils.data.Subset(ds, range(n_train, n))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    # 5) Model
    device = torch.device(cfg.device)
    model = GNNForecaster(
        in_feats=len(cfg.feature_cols),
        hidden=cfg.hidden,
        horizon=cfg.horizon,
        num_nodes=len(nodes),
        lookback=cfg.lookback,
    ).to(device)

    edge_index = edge_index.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # ← Add saver
    saver = ModelSaver(out_dir="checkpoints/gnn")

    # 6) Train
    log("[TRAIN] Starting training...")
    tr, va = {}, {}
    epochs_no_improve = 0
    for epoch in range(1, cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, edge_index, opt, device, target_mu, target_sigma)
        va = eval_one_epoch(model, val_loader, edge_index, device, target_mu, target_sigma)

        improved = saver.maybe_save_best(model, opt, edge_index, epoch, va["rmse"], cfg)
        saver.save_progress(epoch, tr, va, cfg, times)

        if improved:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        h_rmse_str = " | ".join(
            f"t+{(h+1)*cfg.step_size}m {v:.4f}"
            for h, v in enumerate(va["rmse_per_h"])
        )

        log(
            f"epoch {epoch:02d} | "
            f"train RMSE {tr['rmse']:.4f} | train MAE {tr['mae']:.4f} | "
            f"val RMSE {va['rmse']:.4f} | val MAE {va['mae']:.4f}"
            + (f" | per-h: {h_rmse_str}" if len(va['rmse_per_h']) > 1 else "")
            + (" ← best" if improved else f" (no improve {epochs_no_improve}/{cfg.patience})")
        )

        if epochs_no_improve >= cfg.patience:
            log(f"[EARLY STOP] No improvement for {cfg.patience} epochs. Stopping.")
            break

    elapsed = time.time() - train_start
    hours, rem = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(rem, 60)
    duration_str = f"{hours:02d}h {minutes:02d}m {seconds:02d}s"
    log(f"[TIME] Training took {duration_str}")

    # 7) Final save
    saver.save_final(model, opt, edge_index, cfg.epochs, tr, va, cfg, times, elapsed, duration_str)

    log("[DONE]")


if __name__ == "__main__":
    main(Config())