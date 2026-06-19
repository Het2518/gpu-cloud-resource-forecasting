"""
api/routes/anomaly.py

POST /api/v1/anomaly/detect

Runs the two-layer anomaly detection pipeline:
  1. IsolationForest  — structural anomalies
  2. Z-Score          — temporal GPU burst detection
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from api.schemas.anomaly import AnomalyRequest, AnomalyResponse
from core import anomaly_detector

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/anomaly/detect",
    response_model=AnomalyResponse,
    summary="Detect anomalies in current machine metrics",
    tags=["Anomaly"],
)
def detect_anomaly(req: AnomalyRequest) -> AnomalyResponse:
    """
    **Two-Layer Anomaly Detection**

    - **IsolationForest** — detects structural anomalies across all 11 features
    - **Z-Score** — detects sudden GPU utilisation spikes (requires ≥13 historical values)
    - **Combined** — OR of both detectors (matches NB06 logic)

    Tip: pass `recent_gpu_history` for full spike detection capability.
    """
    try:
        result = anomaly_detector.detect(
            features=req.as_feature_dict(),
            recent_gpu_values=req.recent_gpu_history,
        )
    except Exception as exc:
        logger.exception("Anomaly detection error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AnomalyResponse(
        **result,
        timestamp=datetime.now(tz=timezone.utc),
    )
