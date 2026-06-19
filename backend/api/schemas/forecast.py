"""
api/schemas/forecast.py — Request and response models for /api/v1/forecast
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


# ─── Enums / Literals ─────────────────────────────────────────────────────────

Horizon = Literal["5min", "15min", "30min"]
ModelFamily = Literal["rf", "patchtst"]


# ─── Request ──────────────────────────────────────────────────────────────────

class ForecastRequest(BaseModel):
    """
    Single-snapshot forecast request.
    Provide the current machine metric snapshot; the API will predict
    GPU, CPU, and load utilisation at the requested horizon.
    """

    # ── Raw machine metrics (from pai_machine_metric.csv) ─────────────────
    machine_gpu:        float = Field(..., ge=0.0, le=100.0,  description="Current GPU utilisation (%)")
    machine_cpu_usr:    float = Field(..., ge=0.0, le=100.0,  description="CPU user-space usage (%)")
    machine_cpu_kernel: float = Field(..., ge=0.0, le=100.0,  description="CPU kernel usage (%)")
    machine_cpu_iowait: float = Field(..., ge=0.0, le=100.0,  description="CPU IO-wait (%)")
    machine_load_1:     float = Field(..., ge=0.0,             description="1-min load average")
    machine_net_receive:float = Field(..., ge=0.0,             description="Network receive (MB/s)")

    # ── Machine capacity (from pai_machine_spec.csv) ──────────────────────
    cap_gpu: float = Field(..., gt=0.0, description="GPU capacity (number of GPUs)")
    cap_cpu: float = Field(..., gt=0.0, description="CPU capacity (cores)")
    cap_mem: float = Field(..., gt=0.0, description="Memory capacity (GB)")

    # ── Temporal features ─────────────────────────────────────────────────
    hour_of_day:  int = Field(..., ge=0, le=23,  description="Hour of day (0–23)")
    day_of_week:  int = Field(..., ge=0, le=6,   description="Day of week (0=Mon, 6=Sun)")

    # ── Forecast options ──────────────────────────────────────────────────
    horizon:      Horizon     = Field("15min",  description="Forecast horizon")
    model_family: ModelFamily = Field("rf",     description="Model family to use")

    # ── PatchTST sequence (optional) ─────────────────────────────────────
    sequence: Optional[List[Dict[str, float]]] = Field(
        None,
        description=(
            "For PatchTST only — list of up to 24 historical feature snapshots "
            "(oldest first). If omitted, the current snapshot is repeated."
        ),
    )

    @model_validator(mode="after")
    def check_patchtst_sequence(self) -> "ForecastRequest":
        if self.model_family == "patchtst" and self.sequence is not None:
            if len(self.sequence) > 24:
                raise ValueError("sequence must contain at most 24 steps (= SEQ_LEN)")
        return self

    def as_feature_dict(self) -> Dict[str, float]:
        """Return a flat feature dict in FEAT_COLS order."""
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


# ─── Nested response models ────────────────────────────────────────────────────

class Predictions(BaseModel):
    gpu_pct: float = Field(description="Predicted GPU utilisation (%)")
    cpu_pct: Optional[float] = Field(None, description="Predicted CPU utilisation (%)")
    load:    Optional[float] = Field(None, description="Predicted machine load")


class Quantiles(BaseModel):
    p10: float = Field(description="10th percentile (pessimistic)")
    p50: float = Field(description="50th percentile (median forecast)")
    p90: float = Field(description="90th percentile (optimistic)")


class ScalingRecommendation(BaseModel):
    action:            str   = Field(description="SCALE_UP | SCALE_DOWN | MAINTAIN")
    reason:            str
    confidence:        float = Field(ge=0.0, le=1.0)
    predicted_gpu_pct: float
    thresholds:        Dict[str, float]


# ─── Response ─────────────────────────────────────────────────────────────────

class ForecastResponse(BaseModel):
    horizon:                 str
    model_used:              str
    model_family:            str
    predictions:             Predictions
    quantiles:               Optional[Quantiles] = None
    scaling_recommendation:  ScalingRecommendation
    timestamp:               datetime
