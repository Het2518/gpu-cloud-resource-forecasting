from prometheus_client import Gauge

# Gauges for ML Predictions
PREDICTED_GPU_PCT = Gauge(
    "predicted_gpu_pct",
    "Predicted GPU Utilization Percentage 15 minutes ahead"
)

PREDICTED_CPU_PCT = Gauge(
    "predicted_cpu_pct",
    "Predicted CPU Utilization Percentage 15 minutes ahead"
)

PREDICTED_LOAD = Gauge(
    "predicted_load",
    "Predicted System Load 15 minutes ahead"
)

# Gauges for Live Features
LIVE_GPU_PCT = Gauge(
    "live_gpu_pct",
    "Live GPU Utilization Percentage"
)

LIVE_CPU_PCT = Gauge(
    "live_cpu_pct",
    "Live CPU Utilization Percentage"
)

# Gauges for Recommendations & Status
SCALING_ACTION_CODE = Gauge(
    "scaling_action_code",
    "Scaling Action Recommended (0=Scale Up, 1=Scale Down, 2=Maintain)"
)

SCALING_CONFIDENCE = Gauge(
    "scaling_confidence",
    "Confidence score of the scaling recommendation"
)

ANOMALY_STATUS_CODE = Gauge(
    "anomaly_status_code",
    "Anomaly Status (0=Normal, 1=Anomaly Detected)"
)
