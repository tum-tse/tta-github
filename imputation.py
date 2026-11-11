from __future__ import annotations

from collections import deque
from typing import Tuple

import numpy as np
import pandas as pd
import torch


# =============================================================================
# Edge-type semantic weights
# Higher = more informative for imputation at the destination node
# =============================================================================
EDGE_TYPE_WEIGHTS: dict[str, float] = {
    "mainline_to_mainline":      1.0,   # same road, direct upstream continuation
    "A_to_mainline (A99-A9)":    0.9,   # connecting motorway, similar flow type
    "A_to_mainline (A8-A99)":    0.9,
    "A_to_A":                    0.8,   # connecting ramp between motorways
    "mainline_to_A":             0.7,   # diverge onto connecting ramp
    "E_to_mainline":             0.5,   # on-ramp merge, partial/different flow
    "self_loop":                 0.0,   # excluded — useless for imputation
}
_DEFAULT_EDGE_WEIGHT = 0.6  # fallback for unknown edge types


# =============================================================================
# Adjacency list builder
# =============================================================================

def _build_neighbors(
    edge_index_np: np.ndarray,          # [2, E]
    N: int,
    direction: str = "in",
    edge_weights_map: dict[tuple[int, int], float] | None = None,
) -> dict[int, list[int]]:
    """
    Build adjacency list respecting edge direction.

    direction:
        "in"   — neighbors[d] contains upstream nodes s (edge s→d).
                 Best default for directed traffic: upstream predicts downstream.
        "out"  — neighbors[s] contains downstream nodes d (edge s→d).
        "both" — undirected fallback.

    Self-loops (s == d) are always skipped.
    Edges with weight 0.0 (self_loop type) are always skipped.
    """
    src, dst = edge_index_np[0], edge_index_np[1]
    neighbors: dict[int, list[int]] = {n: [] for n in range(N)}

    for s, d in zip(src, dst):
        if s == d:
            continue  # skip self-loops

        # Skip if explicitly zero-weighted (e.g. self_loop edge type)
        if edge_weights_map and edge_weights_map.get((s, d), 1.0) == 0.0:
            continue

        if direction in ("in", "both"):
            neighbors[d].append(s)   # d can look upstream to s
        if direction in ("out", "both"):
            neighbors[s].append(d)   # s can look downstream to d

    return neighbors


def build_edge_weights_from_csv(edges_csv: str) -> dict[tuple[int, int], float]:
    """
    Parse the edges CSV and return a {(src, dst): weight} dict
    using semantic edge-type weights defined in EDGE_TYPE_WEIGHTS.
    """
    df = pd.read_csv(edges_csv)
    edge_weights: dict[tuple[int, int], float] = {}
    for _, row in df.iterrows():
        s, d, etype = int(row["src"]), int(row["dst"]), str(row["edge_type"]).strip()
        w = EDGE_TYPE_WEIGHTS.get(etype, _DEFAULT_EDGE_WEIGHT)
        edge_weights[(s, d)] = w
    return edge_weights


# =============================================================================
# BFS neighbor collector
# =============================================================================

def _bfs_collect(
    start: int,
    t: int,
    f: int,
    out: np.ndarray,
    neighbors: dict[int, list[int]],
    edge_weights_map: dict[tuple[int, int], float] | None,
    max_hops: int,
    max_neighbors: int,
    direction: str,
) -> tuple[list[float], list[float]]:
    """
    BFS outward from `start` up to `max_hops` hops.
    Collects up to `max_neighbors` valid (non-NaN) values.

    Combined weight = edge_type_weight / hop_depth
        - 1-hop mainline neighbor → 1.0 / 1 = 1.00
        - 2-hop mainline neighbor → 1.0 / 2 = 0.50
        - 1-hop on-ramp neighbor  → 0.5 / 1 = 0.50
        - 2-hop on-ramp neighbor  → 0.5 / 2 = 0.25

    NaN nodes are traversed through (kept in queue) to find valid data further away.
    """
    visited = {start}
    queue: deque[tuple[int, int]] = deque([(start, 0)])  # (node, hop_depth)
    vals: list[float] = []
    weights: list[float] = []

    while queue and len(vals) < max_neighbors:
        node, depth = queue.popleft()

        if depth >= max_hops:
            continue

        for nb in neighbors[node]:
            if nb in visited:
                continue
            visited.add(nb)

            val = out[t, nb, f]

            # Determine edge weight: look up in the correct direction
            if edge_weights_map is not None:
                if direction == "in":
                    # edge is nb → node (nb is upstream of node)
                    ew = edge_weights_map.get((nb, node), _DEFAULT_EDGE_WEIGHT)
                elif direction == "out":
                    # edge is node → nb
                    ew = edge_weights_map.get((node, nb), _DEFAULT_EDGE_WEIGHT)
                else:
                    # "both" — try both directions, take the higher weight
                    ew = max(
                        edge_weights_map.get((nb, node), 0.0),
                        edge_weights_map.get((node, nb), 0.0),
                    ) or _DEFAULT_EDGE_WEIGHT
            else:
                ew = 1.0  # no weight map — uniform

            if not np.isnan(val):
                vals.append(float(val))
                weights.append(ew / (depth + 1))
            else:
                # NaN node — keep traversing through it
                queue.append((nb, depth + 1))

    return vals, weights


# =============================================================================
# Core imputation pass
# =============================================================================

def _run_imputation_pass(
    out: np.ndarray,
    neighbors: dict[int, list[int]],
    edge_weights_map: dict[tuple[int, int], float] | None,
    max_hops: int,
    max_neighbors: int,
    distance_weighted: bool,
    direction: str,
) -> np.ndarray:
    T, N, F = out.shape
    for t in range(T):
        for f in range(F):
            nan_nodes = np.where(np.isnan(out[t, :, f]))[0]
            for n in nan_nodes:
                vals, weights = _bfs_collect(
                    start=n,
                    t=t,
                    f=f,
                    out=out,
                    neighbors=neighbors,
                    edge_weights_map=edge_weights_map,
                    max_hops=max_hops,
                    max_neighbors=max_neighbors,
                    direction=direction,
                )
                if vals:
                    if distance_weighted:
                        out[t, n, f] = np.average(vals, weights=weights)
                    else:
                        out[t, n, f] = np.mean(vals)
    return out


# =============================================================================
# Public imputation function
# =============================================================================

def impute_spatial_neighbors(
    X_np: np.ndarray,                                   # [T, N, F]
    edge_index_np: np.ndarray,                          # [2, E]
    max_hops: int = 2,
    max_neighbors: int = 3,
    distance_weighted: bool = True,
    direction: str = "in",
    edge_weights_map: dict[tuple[int, int], float] | None = None,
) -> np.ndarray:
    """
    Two-pass spatial imputation for directed traffic networks:

    Pass 1 — directional (default: upstream / "in" neighbors only).
              Semantically correct: upstream nodes causally predict the
              missing node's value in a directed traffic network.

    Pass 2 — fallback to both directions for any NaNs that Pass 1
              could not fill (e.g. nodes at the start of a chain with
              no upstream data, or isolated on-ramps).

    Parameters
    ----------
    X_np              : [T, N, F] float32 array, may contain NaN
    edge_index_np     : [2, E] int array (src, dst rows)
    max_hops          : BFS search depth (1 = original 1-hop behaviour)
    max_neighbors     : stop BFS once this many valid values are collected
    distance_weighted : weight closer / higher-quality edges more heavily
    direction         : "in" (recommended), "out", or "both"
    edge_weights_map  : {(src, dst): float} from build_edge_weights_from_csv()
                        Pass None to use uniform weights.
    """
    T, N, F = X_np.shape
    out = X_np.copy()

    # --- Pass 1: directional neighbors ---
    neighbors_directed = _build_neighbors(
        edge_index_np, N, direction=direction,
        edge_weights_map=edge_weights_map,
    )
    out = _run_imputation_pass(
        out, neighbors_directed, edge_weights_map,
        max_hops, max_neighbors, distance_weighted, direction,
    )

    # --- Pass 2: fallback to both directions for remaining NaNs ---
    remaining_after_pass1 = int(np.isnan(out).sum())
    if remaining_after_pass1 > 0 and direction != "both":
        neighbors_both = _build_neighbors(
            edge_index_np, N, direction="both",
            edge_weights_map=edge_weights_map,
        )
        out = _run_imputation_pass(
            out, neighbors_both, edge_weights_map,
            max_hops, max_neighbors, distance_weighted, direction="both",
        )
        remaining_after_pass2 = int(np.isnan(out).sum())
        filled_by_pass2 = remaining_after_pass1 - remaining_after_pass2
        if filled_by_pass2 > 0:
            print(f"[IMPUTE] Pass 2 (both directions): filled {filled_by_pass2} "
                  f"additional NaNs — {remaining_after_pass2} still remaining")

    return out  # remaining NaNs → safety fill in pipeline


# =============================================================================
# Normalization
# =============================================================================

def normalize_X(
    X_np: np.ndarray,
    train_end: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalization per feature, computed on the training split only.
    Returns normalized X, mean [F], std [F].
    """
    X_train = X_np[:train_end]
    mu    = np.nanmean(X_train.reshape(-1, X_train.shape[-1]), axis=0)  # [F]
    sigma = np.nanstd( X_train.reshape(-1, X_train.shape[-1]), axis=0)  # [F]
    sigma = np.where(sigma < 1e-6, 1.0, sigma)  # avoid div-by-zero

    X_norm = (X_np - mu) / sigma
    return X_norm.astype(np.float32), mu, sigma


# =============================================================================
# Pipeline entry point
# =============================================================================

def run_imputation_pipeline(
    X: torch.Tensor,                    # [T, N, F]
    edge_index: torch.Tensor,           # [2, E]
    feature_cols: Tuple[str, ...],
    train_ratio: float = 0.8,
    max_hops: int = 2,
    max_neighbors: int = 3,
    distance_weighted: bool = True,
    direction: str = "in",
    edges_csv: str | None = None,       # path to edges CSV for edge-type weights
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """
    Full imputation + normalisation pipeline.

    Steps
    -----
    1. Build observation mask M (1 = originally observed, 0 = was NaN)
    2. Load edge-type semantic weights from CSV (if provided)
    3. Pass 1: k-hop BFS fill using upstream (in) neighbors + edge-type weights
    4. Pass 2: fallback BFS using both directions for remaining NaNs
    5. Safety fill: any still-remaining NaN → 0.0
    6. Z-score normalize on training split only

    Parameters
    ----------
    edges_csv : path to your edges CSV (src, dst, edge_type columns).
                If None, uniform weights are used (still correct, less precise).

    Returns
    -------
    X_clean : torch.Tensor [T, N, F]  — imputed + normalized
    M       : torch.Tensor [T, N, F]  — binary mask (1 = originally observed)
    mu      : np.ndarray  [F]         — per-feature train mean
    sigma   : np.ndarray  [F]         — per-feature train std
    """
    X_np    = X.numpy().copy().astype(np.float32)
    edge_np = edge_index.cpu().numpy()
    T, N, F = X_np.shape

    # 1) Mask before any imputation
    M_np = (~np.isnan(X_np)).astype(np.float32)

    # 2) Load edge-type weights
    edge_weights_map = None
    if edges_csv is not None:
        edge_weights_map = build_edge_weights_from_csv(edges_csv)
        print(f"[IMPUTE] Loaded {len(edge_weights_map)} edge weights from {edges_csv}")

    # temporal forward/backward fill to reduce number of NaNs for spatial imputation to handle
    X_df = pd.DataFrame(X_np.reshape(T, -1)).ffill().bfill()
    X_np = X_df.values.reshape(T, N, F).astype(np.float32)
    
    # 3+4) Two-pass spatial fill
    X_np = impute_spatial_neighbors(
        X_np, edge_np,
        max_hops=max_hops,
        max_neighbors=max_neighbors,
        distance_weighted=distance_weighted,
        direction=direction,
        edge_weights_map=edge_weights_map,
    )

    # 5) Safety fill: nodes with no valid neighbors within max_hops in either direction
    remaining = int(np.isnan(X_np).sum())
    if remaining:
        print(f"[IMPUTE] {remaining} NaNs had no valid neighbor in either direction "
              f"within {max_hops} hops — zero-filled")
    X_np = np.nan_to_num(X_np, nan=0.0)

    # 6) Z-score normalize on training split only
    T = X_np.shape[0]
    train_end = int(T * train_ratio)
    X_norm, mu, sigma = normalize_X(X_np, train_end)

    print(f"[IMPUTE] Done. "
          f"direction={direction}, max_hops={max_hops}, "
          f"max_neighbors={max_neighbors}, "
          f"distance_weighted={distance_weighted}, "
          f"edge_weights={'yes' if edge_weights_map else 'uniform'}")
    print(f"[IMPUTE] Remaining NaNs after norm: {np.isnan(X_norm).sum()}")
    print(f"[IMPUTE] Train means: {dict(zip(feature_cols, mu.round(3)))}")

    return torch.tensor(X_norm), torch.tensor(M_np), mu, sigma