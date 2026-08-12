#!/usr/bin/env python3
"""
Example: Serve a model with PyStreamAI

Starts an inference server and demonstrates:
1. Single inference request
2. Batch inference
3. Statistics/monitoring
4. Health checks
"""

import asyncio
import logging
from pystreamai.serving import InferenceServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    # Initialize server
    logger.info("Starting PyStreamAI Inference Server")
    server = InferenceServer(gpu_type="A100", num_gpus=1)

    # Single inference
    logger.info("\n1. Single Inference Request")
    response1 = await server.predict("bert-base-uncased", {"text": "This is great!"})
    print(f"   Request ID: {response1['request_id']}")
    print(f"   Latency: {response1['latency_ms']:.2f}ms")
    print(f"   Batch size: {response1['batch_size']}")
    print(f"   Cost: ${response1['cost_usd']:.6f}")

    # Batch of requests (will be collected and run together)
    logger.info("\n2. Batch Inference (3 concurrent requests)")
    tasks = [
        server.predict("bert-base-uncased", {"text": "I love this!"}),
        server.predict("bert-base-uncased", {"text": "Not good"}),
        server.predict("bert-base-uncased", {"text": "Pretty cool"}),
    ]
    responses = await asyncio.gather(*tasks)
    for i, resp in enumerate(responses, 1):
        print(f"   Request {i}: {resp['latency_ms']:.2f}ms, batch_size={resp['batch_size']}")

    # Statistics
    logger.info("\n3. Server Statistics")
    stats = server.get_stats()
    print(f"   Requests processed: {stats['requests']}")
    print(f"   Avg latency: {stats['avg_latency_ms']:.2f}ms")
    print(f"   Min latency: {stats['min_latency_ms']:.2f}ms")
    print(f"   Max latency: {stats['max_latency_ms']:.2f}ms")
    print(f"   Total cost: ${stats['total_cost_usd']:.4f}")
    print(f"   Uptime: {stats['uptime_seconds']:.2f}s")

    # Health check
    logger.info("\n4. Health Check")
    health = server.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Requests: {health['requests_processed']}")

    logger.info("\n✅ Inference server demo complete")


if __name__ == "__main__":
    asyncio.run(main())
