"""
api/routes/metrics.py

GET /api/v1/metrics/history

Returns historical machine metric rows from metrics_featured.parquet.
Used by Grafana to render historical overlay panels alongside forecasts.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api.schemas.metrics import MetricRow, MetricsHistoryResponse
from data.loader import load_metrics_history

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/metrics/history",
    response_model=MetricsHistoryResponse,
    summary="Retrieve historical cluster metrics",
    tags=["Metrics"],
)
def metrics_history(
    machine: Optional[str] = Query(
        None,
        description="Filter by machine ID (e.g. 'm_xxx'). If omitted, returns all machines.",
    ),
    limit: int = Query(
        100,
        ge=1,
        le=5000,
        description="Maximum number of rows to return (default 100, max 5000).",
    ),
) -> MetricsHistoryResponse:
    """
    **Historical Metrics**

    Returns the most-recent `limit` rows from the pre-processed feature dataset,
    optionally filtered to a single machine.

    Designed for Grafana time-series panels with the JSON API datasource.
    """
    try:
        rows, total = load_metrics_history(machine=machine, limit=limit)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="metrics_featured.parquet not found. Ensure data pipeline has run.",
        ) from exc
    except Exception as exc:
        logger.exception("Error loading metrics history")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return MetricsHistoryResponse(
        machine=machine,
        total_rows=total,
        rows=[MetricRow(**r) for r in rows],
    )
