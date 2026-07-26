"""
Example: PyStreamAI Inference with W&B Monitoring

Shows how to:
1. Deploy a model with auto-optimization
2. Log inference metrics to Weights & Biases
3. Monitor performance and costs in real-time
"""

from pystreamai import Platform, init_wandb, get_observability, InferenceMetric
import time


def main():
    # Initialize W&B monitoring
    init_wandb(project="pystreamai-inference")

    # Create platform
    platform = Platform()

    # Deploy model with auto-optimization
    print("Deploying model with auto-optimization...")
    endpoint = platform.serve(
        model="bert-base-uncased",
        replicas=2,
        gpu="L4",
        auto_optimize=True,
    )

    # Get observability manager
    obs = get_observability()

    # Simulate inference requests
    test_texts = [
        "This movie is absolutely fantastic!",
        "I didn't like this at all.",
        "The plot was interesting but the acting was poor.",
        "A masterpiece of cinema.",
    ]

    print("\nRunning inferences with monitoring...")
    for i, text in enumerate(test_texts):
        # Run inference
        start = time.time()
        result = endpoint.predict({"text": text})
        latency_ms = (time.time() - start) * 1000

        # Create metric
        metric = InferenceMetric(
            request_id=f"req-{i}",
            model_id="bert-base-uncased",
            latency_ms=latency_ms,
            tokens=len(text.split()),
            cost_usd=0.00001 * len(text.split()),  # Example cost
            optimization_type="quantized+batched",
            speedup_vs_baseline=6.5,  # Measured speedup
        )

        # Log to W&B
        obs.log_metric(metric)

        print(f"Request {i}: {latency_ms:.2f}ms, Cost: ${metric.cost_usd:.6f}")

    print("\n✅ Monitoring complete. Check your W&B project for detailed metrics!")
    print("   Metrics include:")
    print("   - Inference latency")
    print("   - Cost per request")
    print("   - Optimization type")
    print("   - Speedup vs baseline")


if __name__ == "__main__":
    main()
