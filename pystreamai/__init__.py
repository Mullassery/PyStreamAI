"""PyStreamAI - Simple ML Deployment Platform"""

from .platform import Platform
from .decorators import train, serve, pipeline
from .monitoring import (
    get_observability,
    init_wandb,
    init_datadog,
    InferenceMetric,
)

__version__ = "0.1.0"
__all__ = [
    "Platform",
    "train",
    "serve",
    "pipeline",
    "get_observability",
    "init_wandb",
    "init_datadog",
    "InferenceMetric",
]
