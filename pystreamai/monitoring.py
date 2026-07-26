"""PyStreamAI Observability Integration - W&B, Datadog, etc."""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class InferenceMetric:
    """Single inference metric"""
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


class ObservabilityBackend(ABC):
    """Abstract base for observability integrations"""

    @abstractmethod
    def log_metric(self, metric: InferenceMetric) -> None:
        """Log a single inference metric"""
        pass

    @abstractmethod
    def log_benchmark(self, model_name: str, results: List[Dict[str, Any]]) -> None:
        """Log benchmark results"""
        pass

    @abstractmethod
    def log_deployment(self, deployment_id: str, model_id: str, config: Dict[str, Any]) -> None:
        """Log deployment event"""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close connection"""
        pass


class WeightsAndBiasesBackend(ObservabilityBackend):
    """Weights & Biases integration"""

    def __init__(self, project: str = "pystreamai", entity: Optional[str] = None):
        try:
            import wandb
            self.wandb = wandb
            self.run = wandb.init(
                project=project,
                entity=entity,
                config={
                    "framework": "pystreamai",
                    "version": "0.1.0",
                }
            )
            logger.info(f"Initialized Weights & Biases: {project}")
        except ImportError:
            logger.warning("wandb not installed. Install with: pip install wandb")
            self.wandb = None
            self.run = None

    def log_metric(self, metric: InferenceMetric) -> None:
        """Log inference metric to W&B"""
        if not self.run:
            return

        self.wandb.log({
            f"inference/{metric.model_id}/latency_ms": metric.latency_ms,
            f"inference/{metric.model_id}/tokens": metric.tokens,
            f"inference/{metric.model_id}/cost_usd": metric.cost_usd,
            f"inference/{metric.model_id}/speedup": metric.speedup_vs_baseline,
            f"inference/{metric.model_id}/optimization": metric.optimization_type,
            f"inference/{metric.model_id}/batch_size": metric.batch_size,
        })

    def log_benchmark(self, model_name: str, results: List[Dict[str, Any]]) -> None:
        """Log benchmark results as chart"""
        if not self.run:
            return

        # Create summary table
        summary_data = []
        for result in results:
            summary_data.append([
                result.get("optimization", "unknown"),
                result.get("baseline_latency_ms", 0),
                result.get("optimized_latency_ms", 0),
                result.get("speedup", 1.0),
            ])

        table = self.wandb.Table(
            columns=["Optimization", "Baseline (ms)", "Optimized (ms)", "Speedup"],
            data=summary_data
        )

        self.wandb.log({
            f"benchmarks/{model_name}": table,
            f"benchmarks/{model_name}/summary": {
                "num_optimizations": len(results),
                "max_speedup": max(r.get("speedup", 1.0) for r in results),
                "avg_speedup": sum(r.get("speedup", 1.0) for r in results) / len(results),
            }
        })

    def log_deployment(self, deployment_id: str, model_id: str, config: Dict[str, Any]) -> None:
        """Log deployment event"""
        if not self.run:
            return

        self.wandb.log({
            "deployment/id": deployment_id,
            "deployment/model": model_id,
            "deployment/replicas": config.get("replicas", 1),
            "deployment/gpu": config.get("gpu", "none"),
            "deployment/auto_optimize": config.get("auto_optimize", True),
        })

    def close(self) -> None:
        """Close W&B run"""
        if self.run:
            self.run.finish()


class DatadogBackend(ObservabilityBackend):
    """Datadog integration (stub for future)"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        logger.info("Datadog backend initialized (stub)")

    def log_metric(self, metric: InferenceMetric) -> None:
        """Log metric to Datadog"""
        # TODO: Implement Datadog API calls
        pass

    def log_benchmark(self, model_name: str, results: List[Dict[str, Any]]) -> None:
        """Log benchmark to Datadog"""
        # TODO: Implement
        pass

    def log_deployment(self, deployment_id: str, model_id: str, config: Dict[str, Any]) -> None:
        """Log deployment to Datadog"""
        # TODO: Implement
        pass

    def close(self) -> None:
        pass


class ObservabilityManager:
    """Manages multiple observability backends"""

    def __init__(self):
        self.backends: List[ObservabilityBackend] = []

    def add_backend(self, backend: ObservabilityBackend) -> None:
        """Add observability backend"""
        self.backends.append(backend)
        logger.info(f"Added observability backend: {backend.__class__.__name__}")

    def log_metric(self, metric: InferenceMetric) -> None:
        """Log metric to all backends"""
        for backend in self.backends:
            try:
                backend.log_metric(metric)
            except Exception as e:
                logger.error(f"Failed to log metric to {backend.__class__.__name__}: {e}")

    def log_benchmark(self, model_name: str, results: List[Dict[str, Any]]) -> None:
        """Log benchmark to all backends"""
        for backend in self.backends:
            try:
                backend.log_benchmark(model_name, results)
            except Exception as e:
                logger.error(f"Failed to log benchmark to {backend.__class__.__name__}: {e}")

    def log_deployment(self, deployment_id: str, model_id: str, config: Dict[str, Any]) -> None:
        """Log deployment to all backends"""
        for backend in self.backends:
            try:
                backend.log_deployment(deployment_id, model_id, config)
            except Exception as e:
                logger.error(f"Failed to log deployment to {backend.__class__.__name__}: {e}")

    def close_all(self) -> None:
        """Close all backends"""
        for backend in self.backends:
            try:
                backend.close()
            except Exception as e:
                logger.error(f"Failed to close {backend.__class__.__name__}: {e}")


# Global observability manager
_observability = ObservabilityManager()


def get_observability() -> ObservabilityManager:
    """Get global observability manager"""
    return _observability


def init_wandb(project: str = "pystreamai", entity: Optional[str] = None) -> None:
    """Initialize Weights & Biases integration"""
    backend = WeightsAndBiasesBackend(project=project, entity=entity)
    _observability.add_backend(backend)


def init_datadog(api_key: Optional[str] = None) -> None:
    """Initialize Datadog integration"""
    backend = DatadogBackend(api_key=api_key)
    _observability.add_backend(backend)
