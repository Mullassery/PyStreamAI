#!/usr/bin/env python3
"""
Example: Start PyStreamAI HTTP Inference Server

Starts the FastAPI inference server:
  - GET  /health          - Health check
  - POST /predict         - Run inference
  - GET  /stats           - Server statistics
  - GET  /models/{id}/info - Model information

Usage:
  python examples/start_http_server.py

Then test with:
  # Health check
  curl http://localhost:8000/health

  # Run inference
  curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"data": {"text": "This is great!"}}'

  # Get stats
  curl http://localhost:8000/stats
"""

import logging
from pystreamai.api import create_api_server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

POSITIVE_WORDS = {"great", "love", "good", "cool", "excellent", "amazing"}
NEGATIVE_WORDS = {"bad", "not", "terrible", "awful", "hate"}


def toy_sentiment_model(data):
    """A real, if trivial, sentiment classifier -- swap this for anything
    with a callable .predict(data) (or that's itself callable), e.g. a
    real onnxruntime/transformers model. See
    pystreamai.platform.Endpoint's docstring for exactly what's accepted."""
    words = set(data["text"].lower().split())
    positive = len(words & POSITIVE_WORDS)
    negative = len(words & NEGATIVE_WORDS)
    label = "positive" if positive > negative else "negative" if negative > positive else "neutral"
    return {"label": label, "positive_hits": positive, "negative_hits": negative}


def main():
    # Create API server, loading a real model (required -- see
    # pystreamai.api.create_api_server's docstring)
    api_server = create_api_server(
        model_id="toy-sentiment",
        model=toy_sentiment_model,
        gpu_type="A100",
        num_gpus=1,
        port=8000,
    )

    # Start server
    api_server.run()


if __name__ == "__main__":
    main()
