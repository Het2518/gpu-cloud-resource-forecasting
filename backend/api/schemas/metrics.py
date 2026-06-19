"""
api/schemas/metrics.py — Response models for /api/v1/metrics/history
"""

from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel, Field


class MetricRow(BaseModel):
    """One row of historical machine metric data."""
    timestamp:           Optional[Union[int, str]] = None   # unix epoch (seconds)
    machine:             Optional[str]   = None
    machine_gpu:         Optional[float] = None
    machine_cpu_usr:     Optional[float] = None
    machine_cpu_kernel:  Optional[float] = None
    machine_cpu_iowait:  Optional[float] = None
    machine_load_1:      Optional[float] = None
    machine_net_receive: Optional[float] = None
    cap_gpu:             Optional[float] = None
    cap_cpu:             Optional[float] = None
    cap_mem:             Optional[float] = None
    hour_of_day:         Optional[int]   = None
    day_of_week:         Optional[int]   = None


class MetricsHistoryResponse(BaseModel):
    machine:     Optional[str]
    total_rows:  int
    rows:        List[MetricRow]
