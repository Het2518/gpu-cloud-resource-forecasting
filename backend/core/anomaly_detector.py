"""
core/anomaly_detector.py

Two-layer anomaly detection pipeline — mirrors NB06 logic exactly:

  Layer 1 — IsolationForest (isolation_forest.pkl)
            Detects structural anomalies in the full 11-feature space.
            Returns:  is_anomaly (bool), anomaly_score (float)

  Layer 2 — Z-Score rolling spike detector
            Detects sudden GPU bursts using a rolling window.
            Stateless per-call (caller supplies recent GPU history).

  Combined — OR of both layers (same as NB06).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from config import FEAT_COLS, ZSCORE_THRESH, ZSCORE_WINDOW
from core.model_registry import registry

logger = logging.getLogger(__name__)


def detect(
    features: Dict[str, float],
    recent_gpu_values: Optional[List[float]] = None,
) -> Dict:
    """
    Run both anomaly detectors and return a combined result.

    Parameters
    ----------
    features           : current 11-feature snapshot (same dict as /forecast)
    recent_gpu_values  : list of recent GPU utilisation values (oldest first).
                         Should contain at least ZSCORE_WINDOW (12) values
                         for the Z-score detector to activate.
                         The last value should be the current GPU reading.

    Returns
    -------
    {
        "is_anomaly":    bool   — combined flag (IF OR Z-score)
        "if_flag":       bool   — IsolationForest anomaly flag
        "if_score":      float  — raw IF anomaly score (lower = more anomalous)
        "zscore_flag":   bool   — Z-score spike flag
        "zscore_value":  float | None
        "label":         str    — human-readable description
    }
    """
    # ── Layer 1: IsolationForest ───────────────────────────────────────────
    if_flag, if_score = _isolation_forest(features)

    # ── Layer 2: Z-Score spike detector ───────────────────────────────────
    zscore_flag, zscore_value = _zscore(recent_gpu_values)

    # ── Combined ───────────────────────────────────────────────────────────
    combined = if_flag or zscore_flag

    label = _build_label(if_flag, zscore_flag)

    return {
        "is_anomaly":   combined,
        "if_flag":      if_flag,
        "if_score":     round(float(if_score), 6) if if_score is not None else None,
        "zscore_flag":  zscore_flag,
        "zscore_value": round(float(zscore_value), 4) if zscore_value is not None else None,
        "label":        label,
    }


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _isolation_forest(features: Dict[str, float]):
    """
    Run IsolationForest on the feature snapshot.
    Returns (is_anomaly: bool, score: float | None).
    """
    if not registry.has("isolation_forest"):
        logger.warning("IsolationForest not loaded — skipping.")
        return False, None

    row = np.array(
        [float(features.get(col, 0.0)) for col in FEAT_COLS],
        dtype=np.float32,
    ).reshape(1, -1)

    iso = registry.get("isolation_forest")
    label  = iso.predict(row)[0]          # +1 normal, -1 anomaly
    score  = float(iso.score_samples(row)[0])  # lower = more anomalous
    is_anom = label == -1
    return is_anom, score


def _zscore(recent_gpu_values: Optional[List[float]]):
    """
    Z-Score rolling spike detector.
    Returns (is_spike: bool, z_score: float | None).
    """
    if not recent_gpu_values or len(recent_gpu_values) < ZSCORE_WINDOW + 1:
        return False, None

    vals   = np.array(recent_gpu_values, dtype=np.float32)
    current = float(vals[-1])
    window  = vals[-ZSCORE_WINDOW - 1 : -1]   # last WINDOW values before current

    mu    = float(window.mean())
    sigma = float(window.std())

    if sigma <= 0:
        return False, None

    z = (current - mu) / sigma
    return abs(z) > ZSCORE_THRESH, z


def _build_label(if_flag: bool, zscore_flag: bool) -> str:
    if if_flag and zscore_flag:
        return "Structural anomaly + temporal GPU spike detected"
    if if_flag:
        return "Structural anomaly detected (IsolationForest)"
    if zscore_flag:
        return "Temporal GPU spike detected (Z-Score)"
    return "Normal — no anomaly detected"
