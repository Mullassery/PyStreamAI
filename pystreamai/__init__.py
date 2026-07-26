"""PyStreamAI - Simple ML Deployment Platform"""

from .platform import Platform
from .decorators import train, serve, pipeline

__version__ = "0.1.0"
__all__ = ["Platform", "train", "serve", "pipeline"]
