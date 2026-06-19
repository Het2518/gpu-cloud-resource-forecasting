"""
api/schemas/anomaly.py — Request and response models for /api/v1/anomaly/detect
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ─── Request ──────────────────────────────────────────────────────────────────

class AnomalyRequest(BaseModel):
    """
    Anomaly detection request.
    Same 11-feature snapshot as ForecastRequest, plus optional GPU history
    for the Z-score temporal spike detector.
    """

    machine_gpu:         float = Field(..., ge=0.0, le=100.0)
    machine_cpu_usr:     float = Field(..., ge=0.0, le=100.0)
    machine_cpu_kernel:  float = Field(..., ge=0.0, le=100.0)
    machine_cpu_iowait:  float = Field(..., ge=0.0, le=100.0)
    machine_load_1:      float = Field(..., ge=0.0)
    machine_net_receive: float = Field(..., ge=0.0)
    cap_gpu:             float = Field(..., gt=0.0)
    cap_cpu:             float = Field(..., gt=0.0)
    cap_mem:             float = Field(..., gt=0.0)
    hour_of_day:         int   = Field(..., ge=0, le=23)
    day_of_week:         int   = Field(..., ge=0, le=6)

    # Optional: recent GPU readings for Z-score (oldest first, current last)
    recent_gpu_history: Optional[List[float]] = Field(
        None,
        description=(
            "List of recent GPU utilisation values (oldest → newest). "
            "Needs ≥13 values to activate the Z-score spike detector."
        ),
    )

    def as_feature_dict(self):
        return {
            "machine_gpu":         self.machine_gpu,
            "machine_cpu_usr":     self.machine_cpu_usr,
            "machine_cpu_kernel":  self.machine_cpu_kernel,
            "machine_cpu_iowait":  self.machine_cpu_iowait,
            "machine_load_1":      self.machine_load_1,
            "machine_net_receive": self.machine_net_receive,
            "cap_gpu":             self.cap_gpu,
            "cap_cpu":             self.cap_cpu,
            "cap_mem":             self.cap_mem,
            "hour_of_day":         float(self.hour_of_day),
            "day_of_week":         float(self.day_of_week),
        }


# ─── Response ─────────────────────────────────────────────────────────────────

class AnomalyResponse(BaseModel):
    is_anomaly:   bool
    if_flag:      bool
    if_score:     Optional[float]
    zscore_flag:  bool
    zscore_value: Optional[float]
    label:        str
    timestamp:    datetime
