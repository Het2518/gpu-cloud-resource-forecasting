"""
config.py — Central configuration for the GPU Cloud Forecasting Backend

All paths, feature lists, model keys, scaling thresholds, and constants
are defined here so every module imports from a single source of truth.
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────

# Root of the Cloud project (one level up from backend/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR  = PROJECT_ROOT / "outputs" / "models"
DATA_DIR   = PROJECT_ROOT / "data" / "interim"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

# ─── Feature Columns (must match training order exactly) ──────────────────────

FEAT_COLS = [
    "machine_gpu",
    "machine_cpu_usr",
    "machine_cpu_kernel",
    "machine_cpu_iowait",
    "machine_load_1",
    "machine_net_receive",
    "cap_gpu",
    "cap_cpu",
    "cap_mem",
    "hour_of_day",
    "day_of_week",
]

N_FEATURES = len(FEAT_COLS)   # 11

# ─── Sequence / Forecast Config ───────────────────────────────────────────────

SEQ_LEN        = 24   # 24 × 5-min steps = 2-hour look-back window
FORECAST_STEPS = 3    # steps ahead the deep models were trained for

# ─── Scaling Thresholds ───────────────────────────────────────────────────────

SCALE_UP_THRESH   = 85.0   # % — predict GPU > this → SCALE_UP
SCALE_DOWN_THRESH = 40.0   # % — predict GPU < this → SCALE_DOWN
COOLDOWN_STEPS    = 3      # minimum steps between consecutive scale signals

# ─── Anomaly Detection ────────────────────────────────────────────────────────

ZSCORE_WINDOW = 12    # rolling window (steps) for Z-score detector
ZSCORE_THRESH = 3.0   # standard deviations to flag as spike

# ─── Model File Registry ─────────────────────────────────────────────────────
#
# Keys are the canonical model names used throughout the API.
# Values are file paths relative to MODEL_DIR.

MODEL_FILES = {
    # ── Multi-horizon Random Forest — GPU ────────────────────────────────────
    "rf_gpu_5min":  "ml_randomforest_target_gpu_pct_5min.pkl",
    "rf_gpu_15min": "ml_randomforest_target_gpu_pct_15min.pkl",
    "rf_gpu_30min": "ml_randomforest_target_gpu_pct_30min.pkl",

    # ── Multi-horizon Random Forest — CPU ────────────────────────────────────
    "rf_cpu_5min":  "ml_randomforest_target_cpu_pct_5min.pkl",
    "rf_cpu_15min": "ml_randomforest_target_cpu_pct_15min.pkl",
    "rf_cpu_30min": "ml_randomforest_target_cpu_pct_30min.pkl",

    # ── Multi-horizon Random Forest — Load ───────────────────────────────────
    "rf_load_5min":  "ml_randomforest_target_load_5min.pkl",
    "rf_load_15min": "ml_randomforest_target_load_15min.pkl",
    "rf_load_30min": "ml_randomforest_target_load_30min.pkl",

    # ── Transformer (PatchTST — quantile output: P10, P50, P90) ─────────────
    "patchtst": "tf_patchtst.pt",

    # ── Anomaly & Scaling ────────────────────────────────────────────────────
    "isolation_forest":   "isolation_forest.pkl",
    "scaling_classifier": "scaling_classifier.pkl",

    # ── Scalers ──────────────────────────────────────────────────────────────
    "scaler_X": "tf_scaler_X.pkl",
    "scaler_y": "tf_scaler_y.pkl",
}

# ─── PatchTST Architecture Params (must match training) ──────────────────────

PATCHTST_PARAMS = {
    "n_feat":    N_FEATURES,
    "seq_len":   SEQ_LEN,
    "patch_len": 4,
    "stride":    4,
    "d_model":   64,
    "n_heads":   4,
    "n_layers":  2,
    "dropout":   0.1,
}

# ─── API Metadata ─────────────────────────────────────────────────────────────

API_TITLE       = "GPU Cloud Resource Forecasting API"
API_DESCRIPTION = (
    "Intelligent GPU-Aware Cloud Resource Forecasting and Auto-Scaling "
    "Recommendation Platform using Transformer-Based Time Series Models."
)
API_VERSION = "1.0.0"

# ─── CORS Origins (add your Grafana host here) ────────────────────────────────

CORS_ORIGINS = [
    "http://localhost",
    "http://localhost:3000",   # Grafana default
    "http://127.0.0.1:3000",
    "*",                       # widen for dev; restrict in production
]
