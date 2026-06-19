"""
api/routes/forecast.py

POST /api/v1/forecast

Runs multi-target, multi-horizon GPU/CPU/Load forecasting and returns
a scaling recommendation alongside the predictions.

Supports two model families:
  • "rf"       — Random Forest (fast, single-point prediction per target)
  • "patchtst" — PatchTST Transformer (quantile: P10/P50/P90 for GPU only)

MLflow experiment tracking logs every request automatically.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from api.schemas.forecast import (
    ForecastRequest,
    ForecastResponse,
    Predictions,
    Quantiles,
    ScalingRecommendation,
)
from core import inference, scaling_engine
from core.model_registry import registry

logger = logging.getLogger(__name__)

router = APIRouter()

# ─── MLflow Setup (lazy — no network call at import time) ────────────────────
_mlflow_initialized = False

def _get_mlflow():
    """Return mlflow module with experiment set, or None if unavailable."""
    global _mlflow_initialized
    try:
        import mlflow as _mlflow
        if not _mlflow_initialized:
            uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
            _mlflow.set_tracking_uri(uri)
            _mlflow.set_experiment("gpu-resource-forecasting")
            _mlflow_initialized = True
            logger.info("MLflow tracking enabled → %s", uri)
        return _mlflow
    except Exception as exc:
        logger.debug("MLflow unavailable: %s", exc)
        return None


def _log_forecast_run(req: ForecastRequest, response: ForecastResponse, latency_ms: float):
    """Log a single forecast request/response to MLflow (non-blocking)."""
    mlflow = _get_mlflow()
    if mlflow is None:
        return
    try:
        with mlflow.start_run(run_name=f"forecast_{req.horizon}_{req.model_family}"):
            mlflow.log_params({
                "horizon":            req.horizon,
                "model_family":       req.model_family,
                "model_used":         response.model_used,
                "machine_gpu_input":  req.machine_gpu,
                "machine_cpu_input":  req.machine_cpu_usr,
                "machine_load_input": req.machine_load_1,
                "cap_gpu":            req.cap_gpu,
                "cap_cpu":            req.cap_cpu,
                "cap_mem":            req.cap_mem,
                "hour_of_day":        req.hour_of_day,
                "day_of_week":        req.day_of_week,
            })
            metrics: dict = {
                "predicted_gpu_pct":  response.predictions.gpu_pct,
                "scaling_confidence": response.scaling_recommendation.confidence,
                "latency_ms":         latency_ms,
            }
            if response.predictions.cpu_pct is not None:
                metrics["predicted_cpu_pct"] = response.predictions.cpu_pct
            if response.predictions.load is not None:
                metrics["predicted_load"] = response.predictions.load
            if response.quantiles:
                metrics["q10"] = response.quantiles.p10
                metrics["q50"] = response.quantiles.p50
                metrics["q90"] = response.quantiles.p90
            mlflow.log_metrics(metrics)
            mlflow.set_tags({
                "scaling_action": response.scaling_recommendation.action,
                "endpoint":       "/api/v1/forecast",
            })
    except Exception as exc:
        logger.debug("MLflow logging failed (non-fatal): %s", exc)


# ─── Endpoint ─────────────────────────────────────────────────────────────────

@router.post(
    "/forecast",
    response_model=ForecastResponse,
    summary="Forecast GPU / CPU / Load utilisation",
    tags=["Forecast"],
)
def forecast(req: ForecastRequest) -> ForecastResponse:
    """
    **GPU-Aware Resource Forecasting**

    Submit the current machine metric snapshot and receive:
    - Multi-target predictions (GPU%, CPU%, Load) at the requested horizon
    - Quantile intervals P10/P50/P90 (PatchTST family only)
    - Auto-scaling recommendation (SCALE_UP / SCALE_DOWN / MAINTAIN)

    Every request is automatically logged to **MLflow** for experiment tracking.

    ---
    **Horizons:** `5min` · `15min` · `30min`

    **Models:**
    - `rf` — Random Forest (default, fastest, best RMSE)
    - `patchtst` — Transformer with uncertainty quantification
    """
    features = req.as_feature_dict()
    horizon  = req.horizon
    family   = req.model_family
    t_start  = time.perf_counter()

    try:
        if family == "rf":
            gpu_pred, cpu_pred, load_pred, model_key, quantiles_out = (
                _run_rf(features, horizon)
            )
        else:  # patchtst
            gpu_pred, cpu_pred, load_pred, model_key, quantiles_out = (
                _run_patchtst(req, features, horizon)
            )
    except KeyError as exc:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {exc}") from exc
    except Exception as exc:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency_ms = (time.perf_counter() - t_start) * 1000

    # 1. Anomaly Detection
    try:
        from core import anomaly_detector
        anom_res = anomaly_detector.detect(features)
        anomaly_code = 1 if anom_res["is_anomaly"] else 0
        anomaly_label = anom_res["label"]
    except Exception:
        anomaly_code = 0
        anomaly_label = "Unknown"

    # 2. Scaling recommendation based on GPU prediction
    p90_forecast = quantiles_out.get("p90", gpu_pred * 1.10) if quantiles_out else gpu_pred * 1.10
    stress_score = 0.5 * features.get("machine_gpu", 0) + 0.3 * features.get("machine_cpu_usr", 0) + 0.2 * min(100.0, (features.get("machine_load_1", 0) / max(1.0, features.get("cap_cpu", 64))) * 100.0)
    
    scaling_raw = scaling_engine.recommend(
        predicted_gpu_pct=gpu_pred,
        features=features,
        p90_forecast=p90_forecast,
        stress_score=stress_score,
        anomaly_flag=anomaly_code
    )

    response = ForecastResponse(
        horizon=horizon,
        model_used=model_key,
        model_family=family,
        predictions=Predictions(
            gpu_pct=round(gpu_pred, 4),
            cpu_pct=round(cpu_pred, 4) if cpu_pred is not None else None,
            load=round(load_pred, 4) if load_pred is not None else None,
        ),
        quantiles=(
            Quantiles(**quantiles_out) if quantiles_out else None
        ),
        scaling_recommendation=ScalingRecommendation(**scaling_raw),
        timestamp=datetime.now(tz=timezone.utc),
    )

    _log_forecast_run(req, response, latency_ms)
    return response


@router.post(
    "/forecast/grafana_status",
    summary="Grafana Dedicated Status Endpoint",
    tags=["Forecast"],
)
def grafana_status(req: ForecastRequest):
    """
    Dedicated endpoint for Grafana's Stat panels.
    Returns an array of objects (so Infinity parses it correctly as a table)
    and uses numeric codes for deterministic mapping.
    """
    features = req.as_feature_dict()
    
    # Calculate stress score
    stress_score = 0.5 * features.get("machine_gpu", 0) + 0.3 * features.get("machine_cpu_usr", 0) + 0.2 * min(100.0, (features.get("machine_load_1", 0) / max(1.0, features.get("cap_cpu", 64))) * 100.0)

    # 1. Anomaly Detection
    try:
        from core import anomaly_detector
        anom_res = anomaly_detector.detect(features)
        anomaly_code = 1 if anom_res["is_anomaly"] else 0
        anomaly_label = anom_res["label"]
    except Exception:
        anomaly_code = 0
        anomaly_label = "Unknown"

    # 2. Forecast & Scaling
    try:
        if req.model_family == "rf":
            gpu_pred = inference.predict_rf(features, "gpu", req.horizon)
            p90_forecast = gpu_pred * 1.10
        else:
            sequence = req.sequence if req.sequence else [features] * 24
            quantiles_out = inference.predict_patchtst(sequence)
            gpu_pred = quantiles_out["p50"]
            p90_forecast = quantiles_out["p90"]
            
        scaling_raw = scaling_engine.recommend(
            predicted_gpu_pct=gpu_pred,
            features=features,
            p90_forecast=p90_forecast,
            stress_score=stress_score,
            anomaly_flag=anomaly_code
        )
        action_str = scaling_raw["action"]
        if action_str == "SCALE_UP": action_code = 0
        elif action_str == "SCALE_DOWN": action_code = 1
        else: action_code = 2
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"grafana_status failed: {e}")
        gpu_pred, action_code, action_str = 50.0, 2, "MAINTAIN"

    # Return array of 1 object for Grafana table mapping
    return [{
        "pred_gpu": gpu_pred,
        "action_code": action_code,
        "anomaly_code": anomaly_code,
        "anomaly_label": anomaly_label
    }]


# ─── Live Demo Generator ────────────────────────────────────────────────────────

import threading
import time
from data.loader import load_metrics_history

_demo_lock = threading.Lock()
_demo_generator = None
_last_demo_row = None
_last_fetch_time = 0

def _get_next_demo_row():
    global _demo_generator, _last_demo_row, _last_fetch_time
    
    # If less than 2 seconds have passed, return the cached row
    # so all Grafana panels hitting this endpoint concurrently get the same data
    if time.time() - _last_fetch_time < 2 and _last_demo_row is not None:
        return _last_demo_row

    while True:
        if _demo_generator is None:
            try:
                rows, _ = load_metrics_history(limit=5000)
                _demo_generator = iter(rows)
            except Exception:
                # Fallback static if file fails
                _last_demo_row = {"machine_gpu": 50, "machine_cpu_usr": 20, "machine_load_1": 1.5, "machine_cpu_kernel": 10, "machine_cpu_iowait": 5, "machine_net_receive": 100, "cap_gpu": 8, "cap_cpu": 64, "cap_mem": 256, "hour_of_day": 10, "day_of_week": 2}
                _last_fetch_time = time.time()
                return _last_demo_row
        try:
            _last_demo_row = next(_demo_generator)
            _last_fetch_time = time.time()
            return _last_demo_row
        except StopIteration:
            _demo_generator = None


from collections import deque

_demo_sequence = deque(maxlen=24)

@router.get(
    "/forecast/demo_live",
    summary="Live Dynamic Data Endpoint for Grafana",
    tags=["Forecast"],
)
def demo_live():
    """
    Provides a LIVE streaming endpoint for Grafana.
    Pulls a real row from the Alibaba dataset, runs predictions, and returns both
    the live features AND the predictions/anomaly codes.
    """
    with _demo_lock:
        live_row = _get_next_demo_row()
    
    # Extract features, filling defaults for safety
    features = {
        "machine_gpu": float(live_row.get("machine_gpu", 50)),
        "machine_cpu_usr": float(live_row.get("machine_cpu_usr", 20)),
        "machine_cpu_kernel": float(live_row.get("machine_cpu_kernel", 10)),
        "machine_cpu_iowait": float(live_row.get("machine_cpu_iowait", 5)),
        "machine_load_1": float(live_row.get("machine_load_1", 1.5)),
        "machine_net_receive": float(live_row.get("machine_net_receive", 100)),
        "cap_gpu": float(live_row.get("cap_gpu", 8)),
        "cap_cpu": float(live_row.get("cap_cpu", 64)),
        "cap_mem": float(live_row.get("cap_mem", 256)),
        "hour_of_day": int(live_row.get("hour_of_day", 10)),
        "day_of_week": int(live_row.get("day_of_week", 2)),
    }
    
    # Maintain live sequence
    _demo_sequence.append(features)
    
    # Calculate stress score for classifier
    stress_score = 0.5 * features["machine_gpu"] + 0.3 * features["machine_cpu_usr"] + 0.2 * min(100.0, (features["machine_load_1"] / max(1.0, features["cap_cpu"])) * 100.0)

    # 1. Anomaly Detection (must run before scaling so we have the anomaly_flag)
    try:
        from core import anomaly_detector
        anom_res = anomaly_detector.detect(features)
        anomaly_code = 1 if anom_res["is_anomaly"] else 0
        anomaly_label = anom_res["label"]
    except Exception:
        anomaly_code = 0
        anomaly_label = "Unknown"

    # 2. Forecast & Scaling
    try:
        gpu_pred = inference.predict_rf(features, "gpu", "15min")
        cpu_pred = inference.predict_rf(features, "cpu", "15min")
        load_pred = inference.predict_rf(features, "load", "15min")
        
        # P90 fallback for ML scaling classifier (assume 10% variance if using RF)
        p90_forecast = gpu_pred * 1.10
        
        scaling_raw = scaling_engine.recommend(
            predicted_gpu_pct=gpu_pred,
            features=features,
            p90_forecast=p90_forecast,
            stress_score=stress_score,
            anomaly_flag=anomaly_code
        )
        confidence = scaling_raw.get("confidence", 0.7)
        action_str = scaling_raw["action"]
        if action_str == "SCALE_UP": action_code = 0
        elif action_str == "SCALE_DOWN": action_code = 1
        else: action_code = 2
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Demo live predict failed: {e}")
        gpu_pred, cpu_pred, load_pred, confidence, action_code, action_str = 50.0, 20.0, 1.5, 0.5, 2, "MAINTAIN"

    # 3. Update Prometheus Metrics
    try:
        from core.prom_metrics import (
            PREDICTED_GPU_PCT, PREDICTED_CPU_PCT, PREDICTED_LOAD,
            LIVE_GPU_PCT, LIVE_CPU_PCT, SCALING_ACTION_CODE,
            SCALING_CONFIDENCE, ANOMALY_STATUS_CODE
        )
        PREDICTED_GPU_PCT.set(gpu_pred)
        PREDICTED_CPU_PCT.set(cpu_pred)
        PREDICTED_LOAD.set(load_pred)
        LIVE_GPU_PCT.set(features["machine_gpu"])
        LIVE_CPU_PCT.set(features["machine_cpu_usr"])
        SCALING_ACTION_CODE.set(action_code)
        SCALING_CONFIDENCE.set(confidence)
        ANOMALY_STATUS_CODE.set(anomaly_code)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Prometheus metric update failed: {e}")

    # 4. Log to MLflow
    try:
        mlflow = _get_mlflow()
        if mlflow is not None:
            with mlflow.start_run(run_name="forecast_demo_live"):
                mlflow.log_params(features)
                mlflow.log_metrics({
                    "predicted_gpu_pct": gpu_pred,
                    "predicted_cpu_pct": cpu_pred,
                    "predicted_load": load_pred,
                    "scaling_confidence": confidence
                })
                mlflow.set_tags({
                    "scaling_action": action_str if 'action_str' in locals() else "UNKNOWN",
                    "anomaly_status": anomaly_label,
                    "endpoint": "/api/v1/forecast/demo_live"
                })
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug(f"MLflow live logging failed: {e}")

    # Return array of 1 object for Grafana table mapping
    return [{
        "live_gpu": features["machine_gpu"],
        "live_cpu": features["machine_cpu_usr"],
        "live_load": features["machine_load_1"],
        "pred_gpu": gpu_pred,
        "pred_cpu": cpu_pred,
        "pred_load": load_pred,
        "confidence": confidence,
        "action_code": action_code,
        "anomaly_code": anomaly_code,
        "anomaly_label": anomaly_label
    }]

def _run_rf(features, horizon):
    """Run three RF models (GPU, CPU, Load) for the requested horizon."""
    gpu_key  = f"rf_gpu_{horizon}"
    cpu_key  = f"rf_cpu_{horizon}"
    load_key = f"rf_load_{horizon}"

    gpu_pred = inference.predict_rf(features, "gpu", horizon)

    cpu_pred  = None
    load_pred = None

    if registry.has(cpu_key):
        cpu_pred = inference.predict_rf(features, "cpu", horizon)
    if registry.has(load_key):
        load_pred = inference.predict_rf(features, "load", horizon)

    return gpu_pred, cpu_pred, load_pred, gpu_key, None


def _run_patchtst(req: ForecastRequest, features, horizon):
    """Run PatchTST quantile forecast.  GPU P50 used for scaling decision."""
    # Build sequence: use provided sequence or repeat current snapshot
    sequence = req.sequence if req.sequence else [features] * 24

    quantiles_out = inference.predict_patchtst(sequence)
    gpu_pred      = quantiles_out["p50"]  # median as point estimate

    # PatchTST only forecasts GPU (the primary target); CPU/Load via RF
    cpu_pred  = None
    load_pred = None
    if registry.has(f"rf_cpu_{horizon}"):
        cpu_pred = inference.predict_rf(features, "cpu", horizon)
    if registry.has(f"rf_load_{horizon}"):
        load_pred = inference.predict_rf(features, "load", horizon)

    return gpu_pred, cpu_pred, load_pred, "patchtst", quantiles_out
