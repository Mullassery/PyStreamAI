#!/usr/bin/env python3
"""
Example: Serve a model with PyStreamAI

Starts an inference server and demonstrates:
1. Loading a real model
2. Single inference request
3. Batch inference
4. Statistics/monitoring
5. Health checks

The "model" here is a deliberately tiny, dependency-free keyword-based
sentiment classifier -- not a stand-in for something fake. It's a real
callable, so `InferenceServer.predict()` calls it for real and returns
its real output. Swap it for a real transformers/onnxruntime model by
passing anything with a callable `.predict(data)` (or that's itself
callable) to `load_model()` -- see pystreamai.platform.Endpoint's
docstring for exactly what's accepted.
"""

import asyncio
import logging
from pystreamai.serving import InferenceServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

POSITIVE_WORDS = {"great", "love", "good", "cool", "excellent", "amazing"}
NEGATIVE_WORDS = {"bad", "not", "terrible", "awful", "hate"}


def toy_sentiment_model(data):
    """A real, if trivial, sentiment classifier: real keyword matching,
    not a fabricated response."""
    words = set(data["text"].lower().split())
    positive = len(words & POSITIVE_WORDS)
    negative = len(words & NEGATIVE_WORDS)
    if positive > negative:
        label = "positive"
    elif negative > positive:
        label = "negative"
    else:
        label = "neutral"
    return {"label": label, "positive_hits": positive, "negative_hits": negative}


async def main():
    # Initialize server and load a real model
    logger.info("Starting PyStreamAI Inference Server")
    server = InferenceServer(gpu_type="A100", num_gpus=1)
    await server.load_model("toy-sentiment", toy_sentiment_model)

    # Single inference
    logger.info("\n1. Single Inference Request")
    response1 = await server.predict("toy-sentiment", {"text": "This is great!"})
    print(f"   Request ID: {response1['request_id']}")
    print(f"   Output: {response1['output']}")
    print(f"   Latency: {response1['latency_ms']:.2f}ms")
    print(f"   Batch size: {response1['batch_size']}")

    # Batch of requests (will be collected and run together)
    logger.info("\n2. Batch Inference (3 concurrent requests)")
    tasks = [
        server.predict("toy-sentiment", {"text": "I love this!"}),
        server.predict("toy-sentiment", {"text": "Not good"}),
        server.predict("toy-sentiment", {"text": "Pretty cool"}),
    ]
    responses = await asyncio.gather(*tasks)
    for i, resp in enumerate(responses, 1):
        print(f"   Request {i}: {resp['output']} ({resp['latency_ms']:.2f}ms, batch_size={resp['batch_size']})")

    # Statistics
    logger.info("\n3. Server Statistics")
    stats = server.get_stats()
    print(f"   Requests processed: {stats['requests']}")
    print(f"   Avg latency: {stats['avg_latency_ms']:.2f}ms")
    print(f"   Min latency: {stats['min_latency_ms']:.2f}ms")
    print(f"   Max latency: {stats['max_latency_ms']:.2f}ms")
    print(f"   Total cost: ${stats['total_cost_usd']:.4f} (not implemented -- no real pricing table, always 0)")
    print(f"   Uptime: {stats['uptime_seconds']:.2f}s")

    # Health check
    logger.info("\n4. Health Check")
    health = server.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Requests: {health['requests_processed']}")

    logger.info("\nInference server demo complete")


if __name__ == "__main__":
    asyncio.run(main())
