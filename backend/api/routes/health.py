"""
api/routes/health.py

GET /api/v1/health

Returns API status, loaded model list, device information, and uptime.
"""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone

import torch
from fastapi import APIRouter
from pydantic import BaseModel

from config import API_VERSION
from core.model_registry import registry

router = APIRouter()

_START_TIME = datetime.now(tz=timezone.utc)


class HealthResponse(BaseModel):
    status:         str
    version:        str
    uptime_seconds: float
    device:         str
    cuda_available: bool
    python_version: str
    models_loaded:  int
    models:         list[str]
    timestamp:      datetime


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="API health check",
    tags=["System"],
)
def health() -> HealthResponse:
    """
    **Health Check**

    Confirms the API is running, reports loaded models and hardware device.
    """
    now    = datetime.now(tz=timezone.utc)
    uptime = (now - _START_TIME).total_seconds()

    return HealthResponse(
        status="ok",
        version=API_VERSION,
        uptime_seconds=round(uptime, 1),
        device=str(registry.device),
        cuda_available=torch.cuda.is_available(),
        python_version=sys.version,
        models_loaded=len(registry.available),
        models=registry.available,
        timestamp=now,
    )
