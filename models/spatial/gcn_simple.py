"""
gcn_simple.py — plain 2-layer GCN forecaster (Kipf & Welling), the standard graph baseline.

Architecture
------------
  1. Flatten each detector's lookback history:  [B, L, N, F] -> [B, N, L*F]
  2. Two graph convolutions  H' = ReLU( A_hat H W )  with A_hat = D^-1/2 (A + I) D^-1/2
     (the same normalisation as torch_geometric's GCNConv, applied as a dense N x N matmul,
     which is much faster than building a PyG Batch for every sample)
  3. Skip connection: a linear map of the detector's OWN flattened history is added to the
     graph output, so neighbour averaging cannot wash out the detector's own signal
  4. MLP head: [B*N, hidden] -> [B*N, H] -> [B, H, N]

Inherits data loading, windowing, training, evaluation and checkpointing from gnn.py,
like sat.py.  Checkpoints go to checkpoints/gcn_simple/.

Usage (from the repository root)
-----
  python models/spatial/gcn_simple.py
"""
import os

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODELS_DIR = ROOT / "models"
for _p in (MODELS_DIR, Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

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
# CONFIG
# =============================================================================

@dataclass
class GCNSimpleConfig(GNNConfig):
    """Inherits all shared fields (lookback 30, horizon 1, step 5 min, 5-min target sum)."""
    batch_size:   int   = 128
    lr:           float = 1e-3
    patience:     int   = 15
    gcn_hidden:   int   = 128
    gcn_layers:   int   = 2
    gcn_dropout:  float = 0.1


# =============================================================================
# MODEL
# =============================================================================

def normalized_adjacency(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Dense D^-1/2 (A + I) D^-1/2, rows = target nodes (same convention as PyG GCNConv)."""
    A = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
    A[edge_index[1], edge_index[0]] = 1.0                 # message j -> i stored at A[i, j]
    A.fill_diagonal_(1.0)                                 # self loops
    deg = A.sum(1)
    d = deg.pow(-0.5)
    return d[:, None] * A * d[None, :]


class GCNSimpleForecaster(nn.Module):
    def __init__(self, in_feats: int, lookback: int, hidden: int, layers: int,
                 dropout: float, horizon: int, num_nodes: int):
        super().__init__()
        self.horizon, self.num_nodes = horizon, num_nodes
        d_in = lookback * in_feats
        self.lins = nn.ModuleList([nn.Linear(d_in if i == 0 else hidden, hidden) for i in range(layers)])
        self.skip = nn.Linear(d_in, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, horizon))
        self._A = None
        self._A_key = None

    def _adj(self, edge_index: torch.Tensor) -> torch.Tensor:
        key = (edge_index.data_ptr(), edge_index.device)
        if self._A is None or self._A_key != key:
            self._A, self._A_key = normalized_adjacency(edge_index, self.num_nodes), key
        return self._A

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """The GCN itself. x [B, N, L*F] (flattened lookback per detector) -> [B, N, hidden]."""
        A = self._adj(edge_index)
        h = x
        for lin in self.lins:
            h = self.drop(torch.relu(torch.matmul(A, lin(h))))   # A_hat (H W)
        return self.norm(h + self.skip(x))

    def forward(self, x_hist: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """x_hist [B, L, N, F], edge_index [2, E] -> [B, H, N]"""
        B, L, N, F = x_hist.shape
        x = x_hist.permute(0, 2, 1, 3).reshape(B, N, L * F)
        h = self.encode(x, edge_index)
        out = self.head(h.reshape(B * N, -1))
        return out.view(B, N, self.horizon).permute(0, 2, 1)


# =============================================================================
# CHECKPOINT LOADER
# =============================================================================

def load_checkpoint(path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device)
    c = ckpt["config"]
    model = GCNSimpleForecaster(
        in_feats=len(c["feature_cols"]), lookback=c["lookback"], hidden=c["gcn_hidden"],
        layers=c["gcn_layers"], dropout=c["gcn_dropout"], horizon=c["horizon"], num_nodes=c["num_nodes"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["edge_index"].to(device), c


# =============================================================================
# MAIN
# =============================================================================

def run_training(cfg, build_model, out_dir: str, label: str) -> None:
    """Shared by the standalone GCN and the GCN-BiLSTM hybrid: same data, split, loss, early stopping."""
    train_start = time.time()

    nodes, edges, edge_index = load_global_graph(cfg.nodes_csv, cfg.edges_csv)
    cfg.num_nodes = len(nodes)
    log(f"[GRAPH] N={len(nodes)} nodes, E={edge_index.shape[1]} edges | [DEVICE] {cfg.device}")

    site_to_units = load_selection_site_to_units(cfg.selection_csv)
    X, times = build_panel_X(nodes=nodes, site_to_units=site_to_units, data_dir=cfg.data_dir,
                             feature_cols=cfg.feature_cols, nan_fill_value=cfg.nan_fill_value,
                             debug_mapping_out=cfg.debug_mapping_out)
    target_idx = cfg.feature_cols.index(cfg.target_col)

    X_clean, M, mu, sigma = run_imputation_pipeline(
        X=X, edge_index=edge_index, feature_cols=cfg.feature_cols,
        train_ratio=cfg.train_ratio, edges_csv=cfg.edges_csv)
    target_mu, target_sigma = float(mu[target_idx]), float(sigma[target_idx])

    ds = WindowedDataset(X_clean, cfg.lookback, cfg.horizon, cfg.step_size,
                         cfg.accumulation_window, target_idx)
    n_train = int(len(ds) * cfg.train_ratio)
    train_loader = DataLoader(torch.utils.data.Subset(ds, range(0, n_train)), batch_size=cfg.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(torch.utils.data.Subset(ds, range(n_train, len(ds))), batch_size=cfg.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device(cfg.device)
    model = build_model(cfg, len(nodes)).to(device)
    edge_index_dev = edge_index.to(device)
    log(f"[MODEL] {label} | params={sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    saver = ModelSaver(out_dir=out_dir)

    log(f"[TRAIN] Starting {label} training...")
    tr, va, epochs_no_improve = {}, {}, 0
    for epoch in range(1, cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, edge_index_dev, opt, device, target_mu, target_sigma)
        va = eval_one_epoch(model, val_loader, edge_index_dev, device, target_mu, target_sigma)
        improved = saver.maybe_save_best(model, opt, edge_index_dev, epoch, va["rmse"], cfg)
        saver.save_progress(epoch, tr, va, cfg, times)
        epochs_no_improve = 0 if improved else epochs_no_improve + 1
        log(f"epoch {epoch:03d} | train RMSE {tr['rmse']:.4f} MAE {tr['mae']:.4f} | "
            f"val RMSE {va['rmse']:.4f} MAE {va['mae']:.4f}"
            + (" ← best" if improved else f" (no improve {epochs_no_improve}/{cfg.patience})"))
        if epochs_no_improve >= cfg.patience:
            log(f"[EARLY STOP] No improvement for {cfg.patience} epochs. Stopping.")
            break

    elapsed = time.time() - train_start
    h_, rem = divmod(int(elapsed), 3600)
    m_, s_ = divmod(rem, 60)
    duration_str = f"{h_:02d}h {m_:02d}m {s_:02d}s"
    log(f"[TIME] Training took {duration_str}")
    saver.save_final(model, opt, edge_index_dev, cfg.epochs, tr, va, cfg, times, elapsed, duration_str)
    log("[DONE]")


def main(cfg: GCNSimpleConfig) -> None:
    cfg.notes = os.environ.get("RUN_NOTES", "plain 2-layer GCN with skip connection (dense normalised adjacency)")
    run_training(
        cfg,
        lambda c, n: GCNSimpleForecaster(len(c.feature_cols), c.lookback, c.gcn_hidden, c.gcn_layers,
                                         c.gcn_dropout, c.horizon, n),
        out_dir="checkpoints/gcn_simple", label="GCNSimpleForecaster")


if __name__ == "__main__":
    main(GCNSimpleConfig())
