"""
BiLSTM baseline for traffic forecasting.

Imports all shared infrastructure from gnn_global_baseline.py:
  - Config (shared hyperparameters, paths, windowing settings)
  - load_global_graph, load_selection_site_to_units
  - build_panel_X, WindowedDataset
  - train_one_epoch, eval_one_epoch
  - ModelSaver, log

Only adds:
  - BiLSTM-specific Config overrides (lookback, hidden, num_layers, dropout)
  - BiLSTMForecaster model
  - main() wired to above

HOW TO RUN
----------
    python3 bilstm.py
"""

from __future__ import annotations

import os

import signal
import time
from dataclasses import dataclass, field
from typing import Tuple

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODELS_DIR = ROOT / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

import torch
from torch import nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Import everything shared from the GNN script
# ---------------------------------------------------------------------------
from gnn import (
    Config as GNNConfig,
    log,
    load_global_graph,
    load_selection_site_to_units,
    build_panel_X,
    WindowedDataset,
    train_one_epoch,
    eval_one_epoch,
    ModelSaver,
)
from imputation import run_imputation_pipeline


# =============================================================================
# 0) BiLSTM-SPECIFIC CONFIG
#    Inherits every field from GNNConfig.
#    Only override what is meaningfully different for a BiLSTM baseline.
# =============================================================================

@dataclass
class Config(GNNConfig):
    """
    BiLSTM config.  All paths, feature_cols, target_col, step_size,
    batch_size, lr, epochs, train_ratio, patience, nan_fill_value,
    date_suffix, and device are inherited from GNNConfig unchanged.

    Changed defaults
    ----------------
    lookback  : 20  (BiLSTM can exploit a real window; GNN used 1)
    hidden    : 128 (per-direction; BiLSTM output = 2 * hidden)
    num_layers: 2   (stacked BiLSTM layers)
    dropout   : 0.2
    """
    lookback:   int   = 20     # minutes of history (overrides GNNConfig default of 1)
    accumulated_window: int = 5      # how many 1-min steps to sum into the target (overrides GNNConfig default of 1)
    hidden:     int   = 128    # per-direction hidden units
    num_layers: int   = 2      # number of stacked BiLSTM layers
    dropout:    float = 0.2
    batch_size: int   = 64    # override GNNConfig default of 256 (smaller batch size for LSTM)


# =============================================================================
# 1) MODEL: BiLSTM FORECASTER
# =============================================================================

class BiLSTMForecaster(nn.Module):
    """
    Per-node BiLSTM forecaster.

    Why BiLSTM?
    -----------
    Within the lookback window each node has a local time series [L, F].
    A standard (forward) LSTM sees only past context at each step.
    A BiLSTM additionally runs a reversed LSTM so every time step also
    sees future context *within the window* — e.g. it can detect that
    a speed dip at t-10 was followed by recovery, which is useful for
    deciding whether to predict congestion at t+5.

    Under the hood: bidirectional=True makes PyTorch run two LSTMs —
    one on the sequence left-to-right, one on the reversed sequence —
    then concatenates their hidden states. Output dim = 2 * hidden_size.
    This is equivalent to calling nn.LSTM twice manually and cat'ing,
    but faster and cleaner.

    Architecture
    ------------
    Input     : [B, L, N, F]
    Reshape   : [B*N, L, F]       treat each node independently
    BiLSTM    : [B*N, L, 2*H]     fwd + bwd concatenated
    Last step : [B*N, 2*H]        context vector
    MLP head  : [B*N, horizon]
    Reshape   : [B, horizon, N]
    """

    def __init__(
        self,
        in_feats:   int,
        hidden:     int,
        num_layers: int,
        horizon:    int,
        num_nodes:  int,
        dropout:    float = 0.2,
    ):
        super().__init__()
        self.hidden    = hidden
        self.horizon   = horizon
        self.num_nodes = num_nodes

        self.bilstm = nn.LSTM(
            input_size    = in_feats,
            hidden_size   = hidden,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,   # runs two LSTMs; output dim = 2 * hidden
            dropout       = dropout if num_layers > 1 else 0.0,
        )

        bilstm_out = 2 * hidden   # fwd hidden + bwd hidden concatenated

        self.head = nn.Sequential(
            nn.Linear(bilstm_out, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, horizon),
        )

    def forward(self, x_hist: torch.Tensor, edge_index=None) -> torch.Tensor:
        """
        x_hist     : [B, L, N, F]
        edge_index : ignored — kept for API compatibility with GNNForecaster
        returns    : [B, H, N]
        """
        B, L, N, F = x_hist.shape

        # Treat each (batch, node) pair as an independent sequence
        x = x_hist.permute(0, 2, 1, 3)   # [B, N, L, F]
        x = x.reshape(B * N, L, F)        # [B*N, L, F]

        out, _ = self.bilstm(x)            # [B*N, L, 2*hidden]
        context = out[:, -1, :]            # last time step → [B*N, 2*hidden]

        y = self.head(context)             # [B*N, H]
        y = y.view(B, N, self.horizon)     # [B, N, H]
        y = y.permute(0, 2, 1)            # [B, H, N]
        return y


# =============================================================================
# LOAD_CHECKPOINT — load a saved checkpoint and reconstruct the model
# =============================================================================

def load_checkpoint(path: str, device: str = "cpu"):
    """Load a saved BiLSTM checkpoint and reconstruct the model."""
    ckpt = torch.load(path, map_location=device)
    cfg_dict = ckpt["config"]

    model = BiLSTMForecaster(
        in_feats   = len(cfg_dict["feature_cols"]),
        hidden     = cfg_dict["hidden"],
        num_layers = cfg_dict["num_layers"],
        horizon    = cfg_dict["horizon"],
        num_nodes  = cfg_dict["num_nodes"],
        dropout    = cfg_dict.get("dropout", 0.2),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    edge_index = ckpt["edge_index"].to(device)
    return model, edge_index, cfg_dict


# =============================================================================
# 2) MAIN  (mirrors gnn_global_baseline.main exactly, model swapped out)
# =============================================================================

def _timeout_handler(signum, frame):
    raise TimeoutError


def main(cfg: Config) -> None:
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(120)
    try:
        notes = input("[NOTES] Describe this run (press Enter to skip): ").strip()
    except (TimeoutError, EOFError):
        notes = ""
        print("(no notes entered)")
    finally:
        signal.alarm(0)
    cfg.notes = notes

    train_start = time.time()

    # 1) Load graph (edges used by imputation only)
    nodes, edges, edge_index = load_global_graph(cfg.nodes_csv, cfg.edges_csv)
    cfg.num_nodes = len(nodes)
    log(f"[GRAPH] N={len(nodes)} nodes, E={edge_index.shape[1]} edges")
    log(f"[DEVICE] {cfg.device}")

    # 2) Site -> lane unit mapping
    site_to_units = load_selection_site_to_units(cfg.selection_csv)
    log(f"[MAP] {len(site_to_units)} unique sites")

    # 3) Build spatio-temporal tensor X[T, N, F]
    X, times = build_panel_X(
        nodes              = nodes,
        site_to_units      = site_to_units,
        data_dir           = cfg.data_dir,
        feature_cols       = cfg.feature_cols,
        nan_fill_value     = cfg.nan_fill_value,
        debug_mapping_out  = cfg.debug_mapping_out,
    )

    target_idx = cfg.feature_cols.index(cfg.target_col)
    log(f"[DATA] Mean flow: {X[:, :, target_idx].mean().item():.2f}")
    log(f"[DATA] Std flow:  {X[:, :, target_idx].std().item():.2f}")

    # 4) Impute + normalise
    X_clean, M, mu, sigma = run_imputation_pipeline(
        X            = X,
        edge_index   = edge_index,
        feature_cols = cfg.feature_cols,
        train_ratio  = cfg.train_ratio,
        edges_csv    = cfg.edges_csv,
    )
    target_mu    = float(mu[target_idx])
    target_sigma = float(sigma[target_idx])

    # 5) Windowed dataset + chronological split
    ds      = WindowedDataset(X_clean, cfg.lookback, cfg.horizon, cfg.step_size, cfg.accumulation_window, target_idx)
    n       = len(ds)
    n_train = int(n * cfg.train_ratio)
    train_ds = torch.utils.data.Subset(ds, range(0, n_train))
    val_ds   = torch.utils.data.Subset(ds, range(n_train, n))

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # 6) Model
    device = torch.device(cfg.device)
    model = BiLSTMForecaster(
        in_feats   = len(cfg.feature_cols),
        hidden     = cfg.hidden,
        num_layers = cfg.num_layers,
        horizon    = cfg.horizon,
        num_nodes  = cfg.num_nodes,
        dropout    = cfg.dropout,
    ).to(device)

    edge_index_dev = edge_index.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"[MODEL] BiLSTMForecaster | params={total_params:,}")

    opt   = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    saver = ModelSaver(out_dir="checkpoints/bilstm")

    # 7) Training loop
    log("[TRAIN] Starting...")
    tr, va = {}, {}
    epochs_no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        tr = train_one_epoch(
            model, train_loader, edge_index_dev, opt, device, target_mu, target_sigma
        )
        va = eval_one_epoch(
            model, val_loader, edge_index_dev, device, target_mu, target_sigma
        )

        improved = saver.maybe_save_best(model, opt, edge_index_dev, epoch, va["rmse"], cfg)
        saver.save_progress(epoch, tr, va, cfg, times)

        if improved:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        h_str = " | ".join(
            f"t+{(h+1)*cfg.step_size}m {v:.4f}"
            for h, v in enumerate(va["rmse_per_h"])
        )
        log(
            f"epoch {epoch:03d} | "
            f"train RMSE {tr['rmse']:.4f} MAE {tr['mae']:.4f} | "
            f"val RMSE {va['rmse']:.4f} MAE {va['mae']:.4f}"
            + (f" | {h_str}" if len(va["rmse_per_h"]) > 1 else "")
            + (" ← best" if improved else f" (no improve {epochs_no_improve}/{cfg.patience})")
        )

        if epochs_no_improve >= cfg.patience:
            log(f"[EARLY STOP] Stopping after {cfg.patience} epochs without improvement.")
            break

    elapsed = time.time() - train_start
    h_, rem  = divmod(int(elapsed), 3600)
    m_, s_   = divmod(rem, 60)
    duration_str = f"{h_:02d}h {m_:02d}m {s_:02d}s"
    log(f"[TIME] Training took {duration_str}")

    saver.save_final(
        model, opt, edge_index_dev, cfg.epochs,
        tr, va, cfg, times, elapsed, duration_str,
    )
    log("[DONE]")


if __name__ == "__main__":
    main(Config())