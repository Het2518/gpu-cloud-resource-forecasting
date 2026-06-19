"""api/routes package — FastAPI route handlers."""
from fastapi import APIRouter

from .anomaly import router as anomaly_router
from .forecast import router as forecast_router
from .health import router as health_router
from .metrics import router as metrics_router
from .models import router as models_router

router = APIRouter()

router.include_router(health_router, prefix="/api/v1")
router.include_router(forecast_router, prefix="/api/v1")
router.include_router(anomaly_router, prefix="/api/v1")
router.include_router(metrics_router, prefix="/api/v1")
router.include_router(models_router, prefix="/api/v1")
