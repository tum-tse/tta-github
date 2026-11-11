"""
gcn_bilstm.py — GCN + BiLSTM with temporal attention spatio-temporal forecaster.

Architecture
------------
  Spatial:   GCN encoder — 3-layer GCNConv over the global graph.
             Input per node: full lookback history flattened [L * F].
             GCN runs ONCE per sample (not once per timestep).

  Temporal:  BiLSTM-A — bidirectional LSTM with temporal attention over the
             lookback window.
             Input at each timestep: original features concatenated with the
             broadcast spatial context from GCN.

  Flow per forward pass:
    [B, L, N, F]
      → flatten lookback into node features: [B, N, L*F]
      → GCN (runs ONCE): [B, N, gcn_hidden]                 (spatial encoding)
      → broadcast: [B, L, N, gcn_hidden]
      → concat with x_hist: [B, L, N, F + gcn_hidden]
      → reshape to [B*N, L, F + gcn_hidden]
      → bridge proj: [B*N, L, bilstm_hidden]
      → BiLSTM blocks: [B*N, L, 2*bilstm_hidden]            (temporal encoding)
      → temporal attention: [B*N, 2*bilstm_hidden]
      → MLP head: [B, H, N]

Inherits all data loading, windowing, training, evaluation, and checkpointing
from gnn.py.

Usage
-----
  python gcn_bilstm.py
  python gcn_bilstm.py 2>/dev/null
"""

from __future__ import annotations

import os

import sys
import time
import signal
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODELS_DIR = ROOT / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from gnn import (
    Config as GNNConfig,
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
# LOGGING — tees every message to stdout AND a .txt file in the run directory
# =============================================================================

class TeeLogger:
    """
    Writes every log line to stdout and to <run_dir>/run_log.txt simultaneously.

    Usage
    -----
    logger = TeeLogger(run_dir)          # call once after run_dir is created
    logger.log("some message")           # replaces bare log() calls
    logger.close()                       # flush + close at end of main()
    """

    def __init__(self, run_dir: Path):
        self.run_dir  = run_dir
        self._logfile = open(run_dir / "run_log.txt", "a", buffering=1)  # line-buffered

    def log(self, msg: str) -> None:
        ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        self._logfile.write(line + "\n")

    def close(self) -> None:
        self._logfile.flush()
        self._logfile.close()


# Keep a module-level fallback so code that runs before the logger is created
# (e.g. early errors) still prints to stdout.
def log(msg: str) -> None:
    print(msg, flush=True)


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class GCNBiLSTMConfig(GNNConfig):
    """
    Combined GCN + BiLSTM-A config.
    Inherits all shared fields from GNNConfig.

    GCN spatial encoder
    -------------------
    gcn_hidden      : hidden dim for all GCNConv layers
    gcn_layers      : number of stacked GCNConv layers
    gcn_dropout     : dropout applied after each GCNConv (except last)

    BiLSTM-A temporal encoder
    -------------------------
    bilstm_hidden   : per-direction hidden units (output = 2 * bilstm_hidden)
    bilstm_layers   : number of stacked BiLSTM layers
    bilstm_dropout  : dropout between BiLSTM layers
    """
    # Data / training
    lookback:            int   = 30
    batch_size:          int   = 64
    lr:                  float = 1e-4
    accumulation_window: int   = 5
    resume_from:         str   = ""

    # GCN spatial encoder
    gcn_hidden:          int   = 256
    gcn_layers:          int   = 3
    gcn_dropout:         float = 0.1

    # BiLSTM-A temporal encoder
    bilstm_hidden:       int   = 128
    bilstm_layers:       int   = 2
    bilstm_dropout:      float = 0.1


# =============================================================================
# GCN SPATIAL ENCODER
# =============================================================================

class GCNEncoder(nn.Module):
    """
    Multi-layer GCN spatial encoder.

    Each node's input is its FULL lookback history flattened: [L * in_feats].
    GCN runs once per sample, producing a single spatial context vector per node.

    Uses PyG's Batch mechanism to handle a batch of identical graphs efficiently.

    Input:  x          — [B, N, L * in_feats]
            edge_index — [2, E]
    Output: [B, N, gcn_hidden]
    """

    def __init__(
        self,
        in_feats:    int,
        gcn_hidden:  int,
        gcn_layers:  int,
        gcn_dropout: float,
    ):
        super().__init__()
        assert gcn_layers >= 1, "gcn_layers must be >= 1"

        self.gcn_dropout = gcn_dropout
        self.drop = nn.Dropout(p=gcn_dropout)

        self.convs = nn.ModuleList()
        for i in range(gcn_layers):
            in_dim = in_feats if i == 0 else gcn_hidden
            self.convs.append(GCNConv(in_dim, gcn_hidden))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x          : [B, N, L*F]
        edge_index : [2, E]
        returns    : [B, N, gcn_hidden]
        """
        B, N, _ = x.shape

        # Build a batched PyG graph (B copies of the same topology)
        data_list = [Data(x=x[b], edge_index=edge_index) for b in range(B)]
        batch = Batch.from_data_list(data_list)

        h = batch.x
        for i, conv in enumerate(self.convs):
            h = conv(h, batch.edge_index)
            if i < len(self.convs) - 1:
                h = self.drop(torch.relu(h))
            else:
                h = torch.relu(h)   # no dropout on final layer output

        # h: [B*N, gcn_hidden]  →  [B, N, gcn_hidden]
        return h.view(B, N, -1)


# =============================================================================
# GCN-BILSTM-A FORECASTER
# =============================================================================

class GCNBiLSTMForecaster(nn.Module):
    """
    Spatio-temporal forecaster: GCN spatial encoder + BiLSTM-A temporal encoder.

    Architecture
    ------------
    1. Flatten lookback into node features and run GCN ONCE:
       [B, L, N, F] -> permute -> [B, N, L, F]
                    -> reshape  -> [B, N, L*F]
                    -> GCNEncoder (runs once) -> [B, N, gcn_hidden]
                    -> broadcast -> [B, L, N, gcn_hidden]

    2. Concatenate spatial context with original features:
       cat([B, L, N, F], [B, L, N, gcn_hidden]) -> [B, L, N, F + gcn_hidden]

    3. BiLSTM per node over time:
       [B, L, N, F+gcn_hidden] -> reshape -> [B*N, L, F+gcn_hidden]
                               -> bridge proj -> [B*N, L, bilstm_hidden]
                               -> BiLSTM blocks -> [B*N, L, 2*bilstm_hidden]
                               -> temporal attention -> [B*N, 2*bilstm_hidden]

    4. MLP head:
       [B*N, 2*bilstm_hidden] -> [B*N, H] -> [B, H, N]
    """

    def __init__(
        self,
        in_feats:       int,
        lookback:       int,
        gcn_hidden:     int,
        gcn_layers:     int,
        gcn_dropout:    float,
        bilstm_hidden:  int,
        bilstm_layers:  int,
        bilstm_dropout: float,
        horizon:        int,
        num_nodes:      int,
    ):
        super().__init__()
        self.horizon    = horizon
        self.num_nodes  = num_nodes
        self.lookback   = lookback
        self.in_feats   = in_feats
        self.gcn_hidden = gcn_hidden

        # --- 1. GCN spatial encoder ---
        self.gcn = GCNEncoder(
            in_feats    = lookback * in_feats,
            gcn_hidden  = gcn_hidden,
            gcn_layers  = gcn_layers,
            gcn_dropout = gcn_dropout,
        )

        # --- Bridge projection: (F + gcn_hidden) -> bilstm_hidden ---
        self.bridge = nn.Linear(in_feats + gcn_hidden, bilstm_hidden)

        # --- 2. BiLSTM temporal encoder ---
        self.bilstm = nn.LSTM(
            input_size    = bilstm_hidden,
            hidden_size   = bilstm_hidden,
            num_layers    = bilstm_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = bilstm_dropout if bilstm_layers > 1 else 0.0,
        )
        self.out_norm = nn.LayerNorm(2 * bilstm_hidden)

        # --- Temporal attention ---
        self.attn_proj = nn.Linear(2 * bilstm_hidden, 1)

        # --- 3. MLP prediction head ---
        self.head = nn.Sequential(
            nn.Linear(2 * bilstm_hidden, 2 * bilstm_hidden),
            nn.ReLU(),
            nn.Dropout(bilstm_dropout),
            nn.Linear(2 * bilstm_hidden, horizon),
        )

    def forward(self, x_hist: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x_hist     : [B, L, N, F]
        edge_index : [2, E]
        returns    : [B, H, N]
        """
        B, L, N, F = x_hist.shape

        # --- Step 1: GCN runs ONCE on full history per node ---
        x_spatial = x_hist.permute(0, 2, 1, 3)             # [B, N, L, F]
        x_spatial = x_spatial.reshape(B, N, L * F)         # [B, N, L*F]
        h_gcn = self.gcn(x_spatial, edge_index)             # [B, N, gcn_hidden]

        # Broadcast spatial context across all L timesteps
        h_gcn = h_gcn.unsqueeze(1).expand(-1, L, -1, -1)   # [B, L, N, gcn_hidden]

        # --- Step 2: Concatenate spatial context with original features ---
        h = torch.cat([x_hist, h_gcn], dim=-1)              # [B, L, N, F + gcn_hidden]

        # --- Step 3: BiLSTM per-node over time ---
        h = h.permute(0, 2, 1, 3)                           # [B, N, L, F+gcn_hidden]
        h = h.reshape(B * N, L, F + self.gcn_hidden)        # [B*N, L, F+gcn_hidden]
        h = self.bridge(h)                                   # [B*N, L, bilstm_hidden]

        h, _ = self.bilstm(h)                               # [B*N, L, 2*bilstm_hidden]

        # Temporal attention: score each timestep
        attn_scores  = self.attn_proj(torch.tanh(h)).squeeze(-1)    # [B*N, L]
        attn_weights = torch.softmax(attn_scores, dim=-1)           # [B*N, L]
        context      = (h * attn_weights.unsqueeze(-1)).sum(1)      # [B*N, 2*bilstm_hidden]
        context      = self.out_norm(context)

        # --- Step 4: MLP head ---
        out = self.head(context)                             # [B*N, H]
        out = out.view(B, N, self.horizon)                   # [B, N, H]
        out = out.permute(0, 2, 1)                           # [B, H, N]
        return out


# =============================================================================
# CHECKPOINT LOADER
# =============================================================================

def load_checkpoint(path: str, device: str = "cpu"):
    ckpt     = torch.load(path, map_location=device)
    cfg_dict = ckpt["config"]
    model = GCNBiLSTMForecaster(
        in_feats       = len(cfg_dict["feature_cols"]),
        lookback       = cfg_dict["lookback"],
        gcn_hidden     = cfg_dict["gcn_hidden"],
        gcn_layers     = cfg_dict["gcn_layers"],
        gcn_dropout    = cfg_dict["gcn_dropout"],
        bilstm_hidden  = cfg_dict["bilstm_hidden"],
        bilstm_layers  = cfg_dict["bilstm_layers"],
        bilstm_dropout = cfg_dict["bilstm_dropout"],
        horizon        = cfg_dict["horizon"],
        num_nodes      = cfg_dict["num_nodes"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    edge_index = ckpt["edge_index"].to(device)
    return model, edge_index, cfg_dict


# =============================================================================
# MAIN
# =============================================================================

def _timeout_handler(signum, frame):
    raise TimeoutError


def main(cfg: GCNBiLSTMConfig) -> None:
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

    # --- Create the run directory FIRST so the logger can write into it ---
    run_dir = Path("checkpoints/gcn_bilstm") / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = TeeLogger(run_dir)

    try:
        # 1) Graph
        nodes, edges, edge_index = load_global_graph(cfg.nodes_csv, cfg.edges_csv)
        cfg.num_nodes = len(nodes)
        logger.log(f"[GRAPH] N={len(nodes)} nodes, E={edge_index.shape[1]} edges")
        logger.log(f"[DEVICE] {cfg.device}")

        # 2) Site -> lane mapping
        site_to_units = load_selection_site_to_units(cfg.selection_csv)
        logger.log(f"[MAP] {len(site_to_units)} unique sites")

        # 3) Spatio-temporal tensor
        X, times = build_panel_X(
            nodes             = nodes,
            site_to_units     = site_to_units,
            data_dir          = cfg.data_dir,
            feature_cols      = cfg.feature_cols,
            nan_fill_value    = cfg.nan_fill_value,
            debug_mapping_out = cfg.debug_mapping_out,
        )

        target_idx = cfg.feature_cols.index(cfg.target_col)
        logger.log(f"[DATA] Mean flow: {X[:, :, target_idx].mean().item():.2f}")
        logger.log(f"[DATA] Std  flow: {X[:, :, target_idx].std().item():.2f}")

        # 4) Imputation + normalisation
        X_clean, M, mu, sigma = run_imputation_pipeline(
            X            = X,
            edge_index   = edge_index,
            feature_cols = cfg.feature_cols,
            train_ratio  = cfg.train_ratio,
            edges_csv    = cfg.edges_csv,
        )
        target_mu    = float(mu[target_idx])
        target_sigma = float(sigma[target_idx])

        # 5) Windowed dataset + split
        ds = WindowedDataset(
            X_clean, cfg.lookback, cfg.horizon,
            cfg.step_size, cfg.accumulation_window, target_idx,
        )
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

        # 5-6) Resume from checkpoint if specified
        device = torch.device(cfg.device)   # ← moved up from step 6
        start_epoch = 1
        epochs_no_improve = 0

        if cfg.resume_from and os.path.isfile(cfg.resume_from):
            log(f"[RESUME] Loading checkpoint: {cfg.resume_from}")
            ckpt = torch.load(cfg.resume_from, map_location=device)
            start_epoch = ckpt["epoch"] + 1
            log(f"[RESUME] Resuming from epoch {start_epoch}")
        else:
            ckpt = None

        # 6) Model
        # device = torch.device(cfg.device)
        model  = GCNBiLSTMForecaster(
            in_feats       = len(cfg.feature_cols),
            lookback       = cfg.lookback,
            gcn_hidden     = cfg.gcn_hidden,
            gcn_layers     = cfg.gcn_layers,
            gcn_dropout    = cfg.gcn_dropout,
            bilstm_hidden  = cfg.bilstm_hidden,
            bilstm_layers  = cfg.bilstm_layers,
            bilstm_dropout = cfg.bilstm_dropout,
            horizon        = cfg.horizon,
            num_nodes      = len(nodes),
        ).to(device)

        edge_index_dev = edge_index.to(device)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.log(f"[MODEL] GCNBiLSTMForecaster | params={total_params:,}")
        logger.log(
            f"[MODEL] GCN: hidden={cfg.gcn_hidden}, layers={cfg.gcn_layers}, "
            f"input=L*F={cfg.lookback}*{len(cfg.feature_cols)} | "
            f"BiLSTM: hidden={cfg.bilstm_hidden}, layers={cfg.bilstm_layers}, "
            f"output_dim={2*cfg.bilstm_hidden}, lookback={cfg.lookback}"
        )

        opt   = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        # Pass the pre-created run_dir into ModelSaver so checkpoints land
        # in the same folder as run_log.txt.
        saver = ModelSaver(out_dir=str(run_dir.parent), run_dir=run_dir)

        if ckpt is not None:
            model.load_state_dict(ckpt["model_state_dict"])
            opt.load_state_dict(ckpt["optimizer_state_dict"])
            for state in opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            model.train()
            log("[RESUME] Model and optimizer states restored.")

        # 7) Training loop
        logger.log("[TRAIN] Starting GCN-BiLSTM training...")
        tr, va            = {}, {}
        # epochs_no_improve = 0

        for epoch in range(start_epoch, cfg.epochs + 1):
            tr = train_one_epoch(model, train_loader, edge_index_dev, opt, device,
                                 target_mu, target_sigma)
            va = eval_one_epoch(model, val_loader, edge_index_dev, device,
                                target_mu, target_sigma)

            improved = saver.maybe_save_best(model, opt, edge_index_dev, epoch, va["rmse"], cfg)
            saver.save_progress(epoch, tr, va, cfg, times)

            if improved:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            h_rmse_str = " | ".join(
                f"t+{(h+1)*cfg.step_size}m {v:.4f}"
                for h, v in enumerate(va["rmse_per_h"])
            )

            logger.log(
                f"epoch {epoch:03d} | "
                f"train RMSE {tr['rmse']:.4f} MAE {tr['mae']:.4f} | "
                f"val RMSE {va['rmse']:.4f} MAE {va['mae']:.4f}"
                + (f" | {h_rmse_str}" if len(va["rmse_per_h"]) > 1 else "")
                + (" ← best" if improved else f" (no improve {epochs_no_improve}/{cfg.patience})")
            )

            if epochs_no_improve >= cfg.patience:
                logger.log(f"[EARLY STOP] No improvement for {cfg.patience} epochs. Stopping.")
                break

        elapsed = time.time() - train_start
        h_, rem  = divmod(int(elapsed), 3600)
        m_, s_   = divmod(rem, 60)
        duration_str = f"{h_:02d}h {m_:02d}m {s_:02d}s"
        logger.log(f"[TIME] Training took {duration_str}")

        saver.save_final(model, opt, edge_index_dev, cfg.epochs, tr, va, cfg, times,
                         elapsed, duration_str)
        logger.log("[DONE]")
        logger.log(f"[LOG] Full log saved to: {run_dir / 'run_log.txt'}")

    finally:
        logger.close()


if __name__ == "__main__":
    # The paper's checkpoint was trained in two runs: a first run, then resumed from its best
    # checkpoint with resume_from="checkpoints/gcn_bilstm/<first run>/best_model.pth".
    main(GCNBiLSTMConfig())