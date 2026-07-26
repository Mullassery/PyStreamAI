"""PyStreamAI - Simple ML Deployment Platform"""

from .platform import Platform
from .decorators import train, serve, pipeline
from .monitoring import (
    InferenceMetric,
    MetricBackend,
    MetricCollector,
    get_metrics,
    set_metric_backend,
    log_metric,
)

__version__ = "0.1.0"
__all__ = [
    "Platform",
    "train",
    "serve",
    "pipeline",
    "InferenceMetric",
    "MetricBackend",
    "MetricCollector",
    "get_metrics",
    "set_metric_backend",
    "log_metric",
]
