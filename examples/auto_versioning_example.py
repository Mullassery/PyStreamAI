"""Example: Automatic Model Versioning and Production Rollback

This example demonstrates zero-config model versioning with automatic rollback.
When a new deploy breaks in production, it automatically reverts to the last
stable version without manual intervention.
"""

import asyncio
import hashlib
import logging
from datetime import datetime

from pystreamai.auto_versioning import (
    AutoVersionManager,
    HealthThresholds,
)

logging.basicConfig(level=logging.INFO)


async def simulate_inference_traffic(
    manager: AutoVersionManager,
    model_id: str,
    requests: int,
    error_rate: float = 0.0,
    latency_ms: float = 50.0,
) -> None:
    """Simulate inference requests"""
    for i in range(requests):
        error = (i % 100) < (error_rate * 100)
        manager.record_inference(model_id, latency_ms, error)
        await asyncio.sleep(0.01)


async def main():
    print("=" * 80)
    print("PyStreamAI: Automatic Model Versioning & Production Rollback")
    print("=" * 80)

    # Initialize auto version manager
    manager = AutoVersionManager(storage_path=".pystreamai_demo")

    # Define health thresholds (more aggressive for demo)
    thresholds = HealthThresholds(
        error_rate_percent=3.0,
        p99_latency_ms=100.0,
        p95_latency_ms=75.0,
        min_samples=50,
        evaluation_window_seconds=2,  # Short windows for demo
        consecutive_bad_windows=1,  # Rollback after 1 bad window
    )

    # Register rollback handler
    def on_rollback(version_id: str, reason: str):
        print(f"\n🚨 AUTO-ROLLBACK: {version_id}")
        print(f"   Reason: {reason}\n")

    manager.on_rollback(on_rollback)

    # ========================================================================
    # SCENARIO 1: Deploy v1 (stable)
    # ========================================================================
    print("\n📦 SCENARIO 1: Deploy v1 (stable model)")
    print("-" * 80)

    model_hash_v1 = hashlib.sha256(b"model_v1_weights").hexdigest()
    v1_id = manager.deploy_model("sentiment-classifier", model_hash_v1, thresholds)
    print(f"✓ Deployed {v1_id}")

    # Simulate normal traffic (healthy metrics)
    print("  Simulating 200 requests (healthy: 0.5% error, 50ms latency)...")
    await simulate_inference_traffic(manager, "sentiment-classifier", 200, error_rate=0.005)

    manager.promote_version(v1_id)
    print(f"✓ Promoted {v1_id} to production")

    status = manager.get_version_status(v1_id)
    print(f"  Health: {status['current_health']}")

    # ========================================================================
    # SCENARIO 2: Deploy v2 (broken - triggers auto-rollback)
    # ========================================================================
    print("\n\n📦 SCENARIO 2: Deploy v2 (broken model)")
    print("-" * 80)

    model_hash_v2 = hashlib.sha256(b"model_v2_weights").hexdigest()
    v2_id = manager.deploy_model("sentiment-classifier", model_hash_v2, thresholds)
    print(f"✓ Deployed {v2_id}")

    # Simulate degraded traffic (unhealthy metrics)
    print("  Simulating 150 requests (degraded: 8% error, 120ms latency)...")
    await simulate_inference_traffic(
        manager,
        "sentiment-classifier",
        150,
        error_rate=0.08,  # 8% error - exceeds 3% threshold
        latency_ms=120.0,  # exceeds p95 threshold
    )

    # Evaluate health and trigger rollback
    print("\n  Evaluating health metrics...")
    manager.evaluate_health_windows()

    status_v2 = manager.get_version_status(v2_id)
    print(f"  v2 Health: {status_v2['current_health']}")
    print(f"  v2 Status: {status_v2['health_status']}")

    # Check that we rolled back to v1
    current_version = manager.active_versions.get("sentiment-classifier")
    print(f"\n  Active version is now: {current_version}")
    if current_version == v1_id:
        print("  ✓ Successfully rolled back to stable version!")
    else:
        print("  ✗ Failed to rollback")

    # ========================================================================
    # SCENARIO 3: Deploy v3 (improved - stable)
    # ========================================================================
    print("\n\n📦 SCENARIO 3: Deploy v3 (improved model)")
    print("-" * 80)

    model_hash_v3 = hashlib.sha256(b"model_v3_weights").hexdigest()
    v3_id = manager.deploy_model("sentiment-classifier", model_hash_v3, thresholds)
    print(f"✓ Deployed {v3_id}")

    # Simulate healthy traffic
    print("  Simulating 250 requests (healthy: 0.3% error, 45ms latency)...")
    await simulate_inference_traffic(
        manager,
        "sentiment-classifier",
        250,
        error_rate=0.003,  # 0.3% error - below threshold
        latency_ms=45.0,   # below threshold
    )

    manager.promote_version(v3_id)
    print(f"✓ Promoted {v3_id} to production")

    status_v3 = manager.get_version_status(v3_id)
    print(f"  Health: {status_v3['current_health']}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("VERSION HISTORY & SUMMARY")
    print("=" * 80)

    model_status = manager.get_model_status("sentiment-classifier")
    print(f"\nModel: {model_status['model_id']}")
    print(f"Active Version: {model_status['active_version']}\n")

    for v in model_status["versions"]:
        promoted = "✓ PROMOTED" if v["promoted_at"] else ""
        rolled_back = f"⚠ ROLLED BACK: {v['rollback_reason']}" if v["rolled_back_at"] else ""
        status_badge = promoted or rolled_back or "◊ CANARY"

        print(f"  {v['version_id']}")
        print(f"    Status: {v['health_status']} {status_badge}")
        print(f"    Requests: {v['request_count']} | Errors: {v['error_count']}")
        print(f"    Latency: avg={v['avg_latency_ms']:.1f}ms, p99={v['p99_latency_ms']:.1f}ms")
        print()

    print("\n✨ Key Benefits:")
    print("  • Zero manual configuration - sensible defaults")
    print("  • Automatic health monitoring - per-version metrics")
    print("  • Instant rollback - on detected degradation")
    print("  • Full history - audit trail and debugging")
    print("  • Integration-ready - plugs into existing deployment pipeline")
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
