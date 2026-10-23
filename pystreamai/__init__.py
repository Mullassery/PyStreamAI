"""PyStreamAI - Simple ML Deployment Platform"""

from ._core import CUDAProfiler, GPUInfo, MemoryPool
from .bentoml_manifest import build_manifest, service_template, write_manifest
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

__version__ = "2.1.2"
__all__ = [
    "CUDAProfiler",
    "GPUInfo",
    "MemoryPool",
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
    "build_manifest",
    "write_manifest",
    "service_template",
]
