"""
core/scaling_engine.py

Intelligent auto-scaling recommendation engine.

Decision logic:
  • predicted GPU utilisation > SCALE_UP_THRESH   → SCALE_UP
  • predicted GPU utilisation < SCALE_DOWN_THRESH  → SCALE_DOWN
  • otherwise                                       → MAINTAIN

Optionally fuses with the trained RandomForestClassifier
(scaling_classifier.pkl) for an ML-based confidence score.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

from config import (
    FEAT_COLS,
    SCALE_DOWN_THRESH,
    SCALE_UP_THRESH,
)
from core.model_registry import registry

logger = logging.getLogger(__name__)

# Canonical action labels
SCALE_UP   = "SCALE_UP"
SCALE_DOWN = "SCALE_DOWN"
MAINTAIN   = "MAINTAIN"

# Human-readable reasons
_REASONS = {
    SCALE_UP:   (
        f"Predicted GPU utilisation is above the scale-up threshold "
        f"({SCALE_UP_THRESH}%). Recommend adding resources."
    ),
    SCALE_DOWN: (
        f"Predicted GPU utilisation is below the scale-down threshold "
        f"({SCALE_DOWN_THRESH}%). Recommend releasing idle resources."
    ),
    MAINTAIN:   (
        "Predicted GPU utilisation is within normal operating bounds. "
        "No scaling action required."
    ),
}


def recommend(
    predicted_gpu_pct: float,
    features: Optional[Dict[str, float]] = None,
    p90_forecast: float = 0.0,
    stress_score: float = 0.0,
    anomaly_flag: int = 0
) -> Dict:
    """
    Return a scaling recommendation dict.

    Parameters
    ----------
    predicted_gpu_pct : float — forecasted GPU utilisation (0–100)
    features          : optional raw feature dict
    p90_forecast      : float - the P90 upper bound forecast
    stress_score      : float - the calculated stress score
    anomaly_flag      : int - 1 if anomaly, 0 if normal

    Returns
    -------
    {
        "action":     "SCALE_UP" | "SCALE_DOWN" | "MAINTAIN",
        "reason":     str,
        "confidence": float (0–1),
        "predicted_gpu_pct": float,
        "thresholds": {"scale_up": 85.0, "scale_down": 40.0},
    }
    """
    # ── Rule-based decision ────────────────────────────────────────────────
    if predicted_gpu_pct >= SCALE_UP_THRESH:
        action = SCALE_UP
    elif predicted_gpu_pct <= SCALE_DOWN_THRESH:
        action = SCALE_DOWN
    else:
        action = MAINTAIN

    # ── ML classifier confidence (if model available + features provided) ──
    confidence = _ml_confidence(action, features, p90_forecast, stress_score, anomaly_flag)

    return {
        "action":            action,
        "reason":            _REASONS[action],
        "confidence":        round(confidence, 4),
        "predicted_gpu_pct": round(predicted_gpu_pct, 4),
        "thresholds": {
            "scale_up":   SCALE_UP_THRESH,
            "scale_down": SCALE_DOWN_THRESH,
        },
    }


# ─── Internal helpers ─────────────────────────────────────────────────────────

# Label mapping used during training (NB06)
_LABEL_MAP = {0: MAINTAIN, 1: SCALE_UP, 2: SCALE_DOWN}
# Reverse: action → class index
_ACTION_IDX = {v: k for k, v in _LABEL_MAP.items()}


def _ml_confidence(action: str, features: Optional[Dict[str, float]], p90_forecast: float, stress_score: float, anomaly_flag: int) -> float:
    """
    Return the RandomForest classifier's probability for the chosen action.
    Falls back to a simple rule-based confidence if the model is unavailable.
    """
    if features is None or not registry.has("scaling_classifier"):
        return _rule_confidence(action)

    try:
        clf = registry.get("scaling_classifier")
        
        current_gpu = float(features.get("machine_gpu", 0.0))
        current_cpu = float(features.get("machine_cpu_usr", 0.0))
        current_load = float(features.get("machine_load_1", 0.0))
        
        row = np.array(
            [current_gpu, current_cpu, current_load, p90_forecast, stress_score, float(anomaly_flag)],
            dtype=np.float32,
        ).reshape(1, -1)
        
        proba = clf.predict_proba(row)[0]          # shape: (n_classes,)
        classes = clf.classes_                     # class labels as ints

        target_idx = _ACTION_IDX.get(action, 0)
        # Find position of target_idx in classes array
        pos = np.where(np.array(classes) == target_idx)[0]
        if len(pos) == 0:
            return _rule_confidence(action)
        return float(proba[pos[0]])

    except Exception as exc:
        logger.warning("scaling_classifier prediction failed: %s", exc)
        return _rule_confidence(action)


def _rule_confidence(action: str) -> float:
    """Simple rule-based confidence placeholder (0.9 for extreme, 0.7 maintain)."""
    return 0.9 if action != MAINTAIN else 0.7
