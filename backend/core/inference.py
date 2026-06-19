"""
core/inference.py

Preprocessing, sequence building, model prediction, and inverse-scaling.

The RF multi-horizon models were trained on a 35-feature engineered feature set:
  - 9 raw metrics (machine_cpu_usr, machine_cpu_kernel, machine_cpu_iowait,
                   machine_gpu, machine_load_1, machine_net_receive,
                   cap_gpu, cap_cpu, cap_mem)
  - 5 lag features × 3 targets = 15 lag features
    (target_gpu_pct_lag1/2/3/6/12, target_cpu_pct_lag1/2/3/6/12,
     target_load_lag1/2/3/6/12)
  - 3 rolling features × 3 targets = 9 rolling features
    (target_gpu_pct_rmean_1h/rstd_1h/rmax_1h, same for cpu and load)
  - 2 temporal (hour_of_day, day_of_week)

For real-time inference with a single snapshot, lag and rolling features
are estimated from the current snapshot values (since we have no history).

PatchTST (seq_len=24 steps) uses the 11 raw features with MinMaxScaler.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch

from config import FEAT_COLS, N_FEATURES, SEQ_LEN
from core.model_registry import registry

logger = logging.getLogger(__name__)

# ─── RF feature columns (35 total, order must match training in NB03) ─────────

RF_FEAT_COLS = [
    # Raw machine metrics (9)
    "machine_cpu_usr",
    "machine_cpu_kernel",
    "machine_cpu_iowait",
    "machine_gpu",
    "machine_load_1",
    "machine_net_receive",
    "cap_gpu",
    "cap_cpu",
    "cap_mem",
    # Lag features — GPU (5)
    "target_gpu_pct_lag1",
    "target_gpu_pct_lag2",
    "target_gpu_pct_lag3",
    "target_gpu_pct_lag6",
    "target_gpu_pct_lag12",
    # Lag features — CPU (5)
    "target_cpu_pct_lag1",
    "target_cpu_pct_lag2",
    "target_cpu_pct_lag3",
    "target_cpu_pct_lag6",
    "target_cpu_pct_lag12",
    # Lag features — Load (5)
    "target_load_lag1",
    "target_load_lag2",
    "target_load_lag3",
    "target_load_lag6",
    "target_load_lag12",
    # Rolling features — GPU 1h (3)
    "target_gpu_pct_rmean_1h",
    "target_gpu_pct_rstd_1h",
    "target_gpu_pct_rmax_1h",
    # Rolling features — CPU 1h (3)
    "target_cpu_pct_rmean_1h",
    "target_cpu_pct_rstd_1h",
    "target_cpu_pct_rmax_1h",
    # Rolling features — Load 1h (3)
    "target_load_rmean_1h",
    "target_load_rstd_1h",
    "target_load_rmax_1h",
    # Temporal (2)
    "hour_of_day",
    "day_of_week",
]

N_RF_FEATURES = len(RF_FEAT_COLS)   # 35


# ─── Public entry points ──────────────────────────────────────────────────────

def predict_rf(
    features: Dict[str, float],
    target: str,
    horizon: str,
) -> float:
    """
    Run a Random Forest prediction.

    Parameters
    ----------
    features : dict with keys matching RF_FEAT_COLS (or at least the raw metrics).
               Lag/rolling values may be provided; if missing, they are estimated
               from the current raw metric values.
    target   : "gpu" | "cpu" | "load"
    horizon  : "5min" | "15min" | "30min"

    Returns
    -------
    float — predicted value in original units (GPU/CPU: percentage, Load: raw)
    """
    model_key = f"rf_{target}_{horizon}"
    model     = registry.get(model_key)

    # Build the full 35-feature vector
    row = _build_rf_feature_row(features)     # (1, 35) float32
    pred = model.predict(row)                  # (1,)
    return float(pred[0])


def predict_patchtst(
    sequence: List[Dict[str, float]],
) -> Dict[str, float]:
    """
    Run PatchTST quantile prediction.

    Parameters
    ----------
    sequence : list of exactly SEQ_LEN (24) feature dicts, oldest-first.
               Keys should match FEAT_COLS (11 raw features).
               If fewer than SEQ_LEN dicts are supplied, the sequence is
               zero-padded on the left (same as training).

    Returns
    -------
    dict with keys "p10", "p50", "p90" — inverse-scaled GPU% predictions
    """
    model    = registry.get("patchtst")
    scaler_X = registry.get("scaler_X")
    scaler_y = registry.get("scaler_y")
    device   = registry.device

    # Build (SEQ_LEN, N_FEATURES) numpy array using 11-col FEAT_COLS
    arr = np.zeros((SEQ_LEN, N_FEATURES), dtype=np.float32)
    n   = min(len(sequence), SEQ_LEN)
    for i, feat_dict in enumerate(sequence[-n:]):         # take last n steps
        arr[SEQ_LEN - n + i] = _build_raw_feature_row(feat_dict)[0]

    # Scale
    arr_scaled = scaler_X.transform(arr)                  # (SEQ_LEN, 11)

    # Build batch tensor: (1, SEQ_LEN, N_FEATURES)
    tensor = torch.tensor(arr_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor).cpu().numpy()                 # (1, 3)

    # Inverse-scale each quantile
    p10 = float(scaler_y.inverse_transform(out[:, [0]])[0, 0])
    p50 = float(scaler_y.inverse_transform(out[:, [1]])[0, 0])
    p90 = float(scaler_y.inverse_transform(out[:, [2]])[0, 0])

    return {"p10": p10, "p50": p50, "p90": p90}


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _build_rf_feature_row(features: Dict[str, float]) -> np.ndarray:
    """
    Build a (1, 35) float32 RF feature row.

    For lag and rolling features that are not present in `features`, we
    estimate them from the current raw metric values:
      - All lag values = current value (i.e., no change assumed)
      - rolling mean  = current value
      - rolling std   = 0.0  (unknown)
      - rolling max   = current value
    """
    gpu  = float(features.get("machine_gpu",     0.0))
    cpu  = float(features.get("machine_cpu_usr", 0.0))
    load = float(features.get("machine_load_1",  0.0))

    estimated = {
        # Lag — GPU
        "target_gpu_pct_lag1":  gpu,
        "target_gpu_pct_lag2":  gpu,
        "target_gpu_pct_lag3":  gpu,
        "target_gpu_pct_lag6":  gpu,
        "target_gpu_pct_lag12": gpu,
        # Lag — CPU
        "target_cpu_pct_lag1":  cpu,
        "target_cpu_pct_lag2":  cpu,
        "target_cpu_pct_lag3":  cpu,
        "target_cpu_pct_lag6":  cpu,
        "target_cpu_pct_lag12": cpu,
        # Lag — Load
        "target_load_lag1":  load,
        "target_load_lag2":  load,
        "target_load_lag3":  load,
        "target_load_lag6":  load,
        "target_load_lag12": load,
        # Rolling — GPU
        "target_gpu_pct_rmean_1h": gpu,
        "target_gpu_pct_rstd_1h":  0.0,
        "target_gpu_pct_rmax_1h":  gpu,
        # Rolling — CPU
        "target_cpu_pct_rmean_1h": cpu,
        "target_cpu_pct_rstd_1h":  0.0,
        "target_cpu_pct_rmax_1h":  cpu,
        # Rolling — Load
        "target_load_rmean_1h": load,
        "target_load_rstd_1h":  0.0,
        "target_load_rmax_1h":  load,
    }

    merged = {**estimated, **features}  # caller-provided values override estimates

    row = np.array(
        [float(merged.get(col, 0.0)) for col in RF_FEAT_COLS],
        dtype=np.float32,
    ).reshape(1, -1)
    return row


def _build_raw_feature_row(features: Dict[str, float]) -> np.ndarray:
    """
    Build a (1, 11) float32 feature row for PatchTST using FEAT_COLS order.
    Missing keys default to 0.0.
    """
    row = np.array(
        [float(features.get(col, 0.0)) for col in FEAT_COLS],
        dtype=np.float32,
    ).reshape(1, -1)
    return row
