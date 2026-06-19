"""
data/loader.py

Reads historical metrics from metrics_featured.parquet.
Used by the /metrics/history endpoint to serve Grafana time-series panels.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import polars as pl

from config import DATA_DIR

logger = logging.getLogger(__name__)

_PARQUET_PATH = DATA_DIR / "metrics_featured.parquet"

# Columns we expose via the API (subset of the full ~40-column parquet)
_EXPORT_COLS = [
    "timestamp",
    "machine",
    "machine_gpu",
    "machine_cpu_usr",
    "machine_cpu_kernel",
    "machine_cpu_iowait",
    "machine_load_1",
    "machine_net_receive",
    "cap_gpu",
    "cap_cpu",
    "cap_mem",
    "hour_of_day",
    "day_of_week",
]


def load_metrics_history(
    machine: Optional[str] = None,
    limit: int = 100,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Load the last `limit` rows from metrics_featured.parquet.

    Parameters
    ----------
    machine : optional machine ID filter (e.g. "m_0")
    limit   : maximum rows to return (1–5000)

    Returns
    -------
    (rows, total_matching_rows)
      rows  : list of dicts, each representing one metric snapshot
      total : total matching row count before limit applied
    """
    if not _PARQUET_PATH.exists():
        raise FileNotFoundError(f"Parquet file not found: {_PARQUET_PATH}")

    # Load only needed columns to minimise memory
    available_cols = [c for c in _EXPORT_COLS if True]  # keep all
    df = pl.read_parquet(str(_PARQUET_PATH), columns=_get_available_cols())

    if machine is not None:
        df = df.filter(pl.col("machine") == machine)

    total = df.shape[0]

    import time
    import datetime
    
    # Sort by timestamp descending and take the latest `limit` rows
    if "timestamp" in df.columns:
        df = df.sort("timestamp", descending=True).head(limit)
        # Shift timestamps to align with 'now'
        max_ts = df["timestamp"].max()
        if isinstance(max_ts, datetime.datetime):
            # If max_ts is naive, we can just subtract from datetime.now()
            # If it has a timezone, subtract from datetime.now(datetime.timezone.utc)
            now = datetime.datetime.now(datetime.timezone.utc) if max_ts.tzinfo else datetime.datetime.now()
            offset = now - max_ts
        else:
            offset = int(time.time()) - max_ts
        df = df.with_columns((pl.col("timestamp") + offset).alias("timestamp"))
        
        # Re-sort ascending so Grafana time-series renders correctly
        df = df.sort("timestamp", descending=False)
    else:
        df = df.tail(limit)

    rows = _to_dicts(df)
    return rows, total


def _get_available_cols():
    """Return whichever EXPORT_COLS actually exist in the parquet."""
    # Peek at schema without loading data
    try:
        schema = pl.read_parquet_schema(str(_PARQUET_PATH))
        return [c for c in _EXPORT_COLS if c in schema]
    except Exception:
        return _EXPORT_COLS


def _to_dicts(df: pl.DataFrame) -> List[Dict[str, Any]]:
    import datetime
    """Convert polars DataFrame to list of plain Python dicts."""
    rows = []
    for row in df.iter_rows(named=True):
        clean = {}
        for k, v in row.items():
            if k == "timestamp":
                if isinstance(v, (int, float)):
                    # v is in seconds
                    clean[k] = datetime.datetime.fromtimestamp(v, tz=datetime.timezone.utc).isoformat()
                elif hasattr(v, "isoformat"):
                    clean[k] = v.isoformat()
                continue
            
            # Convert polars types → Python native
            if hasattr(v, "item"):        # numpy scalar
                v = v.item()
            elif hasattr(v, "isoformat"): # datetime-like
                v = v.isoformat()
            clean[k] = v
        rows.append(clean)
    return rows
