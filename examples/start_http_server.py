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


def main():
    # Create API server
    api_server = create_api_server(
        model_id="bert-base-uncased",
        gpu_type="A100",
        num_gpus=1,
        port=8000,
    )

    # Start server
    api_server.run()


if __name__ == "__main__":
    main()
