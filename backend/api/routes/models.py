"""
api/routes/models.py

GET /api/v1/models          — list all available models with metadata
GET /api/v1/models/active   — list which models are currently loaded
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from core.model_registry import registry

router = APIRouter()


# ─── Static model catalogue ────────────────────────────────────────────────────
# Metadata enriches the raw model list with training metrics from NB07.

_MODEL_CATALOGUE: List[Dict[str, Any]] = [
    # ── Random Forest (Multi-horizon) ────────────────────────────────────────
    {"key": "rf_gpu_5min",   "family": "ML",  "algorithm": "RandomForest", "target": "GPU",  "horizon": "5min",  "rmse": None,   "r2": None,   "notes": "Best overall model (NB07)"},
    {"key": "rf_gpu_15min",  "family": "ML",  "algorithm": "RandomForest", "target": "GPU",  "horizon": "15min", "rmse": 9.4305, "r2": 0.8729, "notes": "#1 ranked model overall"},
    {"key": "rf_gpu_30min",  "family": "ML",  "algorithm": "RandomForest", "target": "GPU",  "horizon": "30min", "rmse": None,   "r2": None,   "notes": ""},
    {"key": "rf_cpu_5min",   "family": "ML",  "algorithm": "RandomForest", "target": "CPU",  "horizon": "5min",  "rmse": None,   "r2": None,   "notes": ""},
    {"key": "rf_cpu_15min",  "family": "ML",  "algorithm": "RandomForest", "target": "CPU",  "horizon": "15min", "rmse": None,   "r2": None,   "notes": ""},
    {"key": "rf_cpu_30min",  "family": "ML",  "algorithm": "RandomForest", "target": "CPU",  "horizon": "30min", "rmse": None,   "r2": None,   "notes": ""},
    {"key": "rf_load_5min",  "family": "ML",  "algorithm": "RandomForest", "target": "Load", "horizon": "5min",  "rmse": None,   "r2": None,   "notes": ""},
    {"key": "rf_load_15min", "family": "ML",  "algorithm": "RandomForest", "target": "Load", "horizon": "15min", "rmse": None,   "r2": None,   "notes": ""},
    {"key": "rf_load_30min", "family": "ML",  "algorithm": "RandomForest", "target": "Load", "horizon": "30min", "rmse": None,   "r2": None,   "notes": ""},
    # ── PatchTST (Quantile Transformer) ──────────────────────────────────────
    {"key": "patchtst",      "family": "Transformer", "algorithm": "PatchTST", "target": "GPU", "horizon": "15min", "rmse": 9.9314, "r2": 0.8576, "notes": "Quantile output P10/P50/P90"},
    # ── Anomaly / Scaling ─────────────────────────────────────────────────────
    {"key": "isolation_forest",   "family": "Anomaly",  "algorithm": "IsolationForest",      "target": "All", "horizon": "—", "rmse": None, "r2": None, "notes": "Structural anomaly detection"},
    {"key": "scaling_classifier", "family": "Scaling",  "algorithm": "RandomForestClassifier","target": "GPU", "horizon": "—", "rmse": None, "r2": None, "notes": "ML-based scaling confidence"},
]


# ─── Schemas ──────────────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    key:       str
    family:    str
    algorithm: str
    target:    str
    horizon:   str
    rmse:      Optional[float]
    r2:        Optional[float]
    notes:     str
    loaded:    bool


class ModelsResponse(BaseModel):
    total:   int
    loaded:  int
    models:  List[ModelInfo]


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.get(
    "/models",
    response_model=ModelsResponse,
    summary="List all available models with metadata",
    tags=["Models"],
)
def list_models() -> ModelsResponse:
    """
    **Model Catalogue**

    Returns every model in the system with its training metrics (RMSE, R²),
    target variable, forecast horizon, and whether it is currently loaded.
    """
    model_infos = [
        ModelInfo(**cat, loaded=registry.has(cat["key"]))
        for cat in _MODEL_CATALOGUE
    ]
    return ModelsResponse(
        total=len(model_infos),
        loaded=sum(1 for m in model_infos if m.loaded),
        models=model_infos,
    )


@router.get(
    "/models/active",
    summary="List currently loaded models",
    tags=["Models"],
)
def active_models() -> Dict:
    """Returns a simple list of all model keys currently in memory."""
    return {"loaded": registry.available, "count": len(registry.available)}
