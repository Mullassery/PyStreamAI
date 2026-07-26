"""PyStreamAI Observability - Pluggable metric backends"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class InferenceMetric:
    """Inference metric - plug into any observability backend"""
    request_id: str
    model_id: str
    latency_ms: float
    tokens: int
    cost_usd: float
    optimization_type: str
    batch_size: int = 1
    gpu_memory_mb: Optional[float] = None
    speedup_vs_baseline: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricBackend(ABC):
    """Abstract observability backend - implement for W&B, Datadog, etc."""

    @abstractmethod
    def log_metric(self, metric: InferenceMetric) -> None:
        """Log inference metric"""
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up resources"""
        pass


class NoOpBackend(MetricBackend):
    """Default no-op backend (metrics are discarded)"""

    def log_metric(self, metric: InferenceMetric) -> None:
        pass

    def close(self) -> None:
        pass


class MetricCollector:
    """Collects and emits metrics to pluggable backends"""

    def __init__(self, backend: Optional[MetricBackend] = None):
        self.backend = backend or NoOpBackend()
        self.metrics: List[InferenceMetric] = []

    def log(self, metric: InferenceMetric) -> None:
        """Log a metric"""
        self.metrics.append(metric)
        try:
            self.backend.log_metric(metric)
        except Exception as e:
            logger.error(f"Failed to log metric: {e}")

    def set_backend(self, backend: MetricBackend) -> None:
        """Switch observability backend at runtime"""
        if self.backend:
            self.backend.close()
        self.backend = backend
        logger.info(f"Switched to backend: {backend.__class__.__name__}")

    def get_metrics(self) -> List[InferenceMetric]:
        """Get all collected metrics"""
        return self.metrics.copy()

    def close(self) -> None:
        """Close backend"""
        if self.backend:
            self.backend.close()


# Global metric collector
_metrics = MetricCollector()


def get_metrics() -> MetricCollector:
    """Get global metric collector"""
    return _metrics


def set_metric_backend(backend: MetricBackend) -> None:
    """Set global observability backend"""
    _metrics.set_backend(backend)


def log_metric(metric: InferenceMetric) -> None:
    """Log inference metric globally"""
    _metrics.log(metric)
