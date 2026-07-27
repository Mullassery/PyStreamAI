# PyStreamAI: Phases 5-10 Advanced Features

Complete guide to advanced rollback strategies, cost optimization, framework integration, analytics, and multi-model orchestration.

## Table of Contents
1. [Phase 5: Advanced Rollback Strategies](#phase-5-advanced-rollback-strategies)
2. [Phase 6: Cost & Performance Optimization](#phase-6-cost--performance-optimization)
3. [Phase 8: Framework & Ecosystem Integration](#phase-8-framework--ecosystem-integration)
4. [Phase 9: Advanced Analytics](#phase-9-advanced-analytics)
5. [Phase 10: Multi-Model Orchestration](#phase-10-multi-model-orchestration)

---

## Phase 5: Advanced Rollback Strategies

Sophisticated rollback strategies beyond simple revert.

### Rollback Strategies

```python
from pystreamai import RollbackStrategy, RollbackConfig, RollbackOrchestrator

# Instant rollback - immediate 100% switch
instant_config = RollbackConfig(strategy=RollbackStrategy.INSTANT)

# Canary rollback - gradual shift with health checks
canary_config = RollbackConfig(
    strategy=RollbackStrategy.CANARY,
    traffic_steps=[10, 25, 50, 100],  # Gradual increase
    step_duration_seconds=300,
    error_rate_threshold=5.0,
)

# Blue-green rollback - instant switch, keep old running
blue_green_config = RollbackConfig(strategy=RollbackStrategy.BLUE_GREEN)

# Shadow rollback - test before switching
shadow_config = RollbackConfig(strategy=RollbackStrategy.SHADOW)
```

### Orchestrate Rollbacks

```python
orchestrator = RollbackOrchestrator(manager)

# Execute rollback
success = await orchestrator.rollback(
    version_id="model-v1",
    reason="v2 degraded",
    strategy=RollbackStrategy.CANARY,
    config=canary_config,
)

# Get rollback history
history = orchestrator.get_rollback_history("sentiment-classifier")
for rollback in history:
    print(f"{rollback['from_version']} → {rollback['to_version']}")
    print(f"  Strategy: {rollback['strategy']}")
    print(f"  Success: {rollback['success']}")

# Abort ongoing rollback
await orchestrator.abort_rollback()
```

---

## Phase 6: Cost & Performance Optimization

Track costs per model version and benchmark performance.

### Cost Tracking

```python
from pystreamai import CostModel, VersionCostTracker, VersionMetrics

# Define infrastructure costs
cost_model = CostModel(
    gpu_hourly_cost_usd=1.0,
    cpu_hourly_cost_usd=0.1,
    memory_hourly_cost_usd=0.01,
    storage_hourly_cost_usd=0.001,
)

tracker = VersionCostTracker(cost_model)

# Record inference metrics
metrics = VersionMetrics(
    version_id="model-v2",
    model_id="sentiment-classifier",
    latency_ms=50.0,
    throughput_requests_per_second=20.0,
    error_rate_percent=0.2,
    memory_mb=2048,
    gpu_memory_mb=4096,
    batch_size=32,
)

tracker.record_inference(metrics)

# Compare versions
comparison = tracker.compare_versions(["model-v1", "model-v2"])
for version, metrics in comparison.items():
    print(f"{version}:")
    print(f"  Cost/Request: ${metrics['average_cost_per_inference']:.4f}")
    print(f"  Latency: {metrics['average_latency_ms']:.1f}ms")
    print(f"  Throughput: {metrics['average_throughput_rps']:.1f} RPS")

# Get recommendation
best_version, score = tracker.recommend_version(
    "sentiment-classifier",
    weight_cost=0.4,  # 40% weight on cost
    weight_speed=0.6,  # 60% weight on speed
)
```

### Performance Benchmarking

```python
from pystreamai import PerformanceBenchmark

benchmark = PerformanceBenchmark()

# Run benchmark
test_inputs = [{"text": f"Test {i}"} for i in range(100)]

def inference_fn(version_id, model_id, data):
    # Your inference code
    return {"output": "result"}, 50.0  # (output, latency_ms)

results = benchmark.run_benchmark(
    version_ids=["model-v1", "model-v2"],
    test_inputs=test_inputs,
    metric_fn=inference_fn,
    name="monthly_benchmark",
)

# Get winner
winner = benchmark.get_winner("monthly_benchmark", metric="avg_latency_ms")
print(f"Fastest: {winner}")

# Statistical significance test
sig, p_value = benchmark.statistical_significance_test(
    version1_metrics=results["model-v1"]["latencies"],
    version2_metrics=results["model-v2"]["latencies"],
)
```

---

## Phase 8: Framework & Ecosystem Integration

Deploy models with TensorFlow Serving, TorchServe, Kubernetes.

### TensorFlow Serving

```python
from pystreamai import TensorFlowServingBackend, FrameworkDetector

# Create backend
tf_backend = TensorFlowServingBackend(server_url="http://localhost:8501")

# Deploy version
metadata = ModelMetadata(
    framework="tensorflow",
    model_path="/models/bert-saved-model",
    version="v2",
)

tf_backend.deploy_version("model-v2", "/models/bert-saved-model", metadata)

# Switch to new version
tf_backend.switch_version("model-v2")

# Health check
health = tf_backend.health_check()
print(f"Status: {health['status']}")
```

### PyTorch with TorchServe

```python
from pystreamai import TorchServeBackend

torch_backend = TorchServeBackend(server_url="http://localhost:8080")

# Deploy
torch_backend.deploy_version("model-v2", "/models/bert.pt", metadata)

# Switch
torch_backend.switch_version("model-v2")

# List deployed versions
versions = torch_backend.list_versions()
```

### Kubernetes Native

```python
from pystreamai import KubernetesDeployment

k8s_backend = KubernetesDeployment(namespace="ml-models")

# Deploy as K8s deployment
k8s_backend.deploy_version("model-v2", "/models/bert", metadata)

# Switch service to new version
k8s_backend.switch_version("model-v2")

# Check health
health = k8s_backend.health_check()
print(f"Ready replicas: {health['ready_replicas']}/{health['desired_replicas']}")
```

### Hugging Face Hub

```python
from pystreamai import HuggingFaceHub

hub = HuggingFaceHub(repo_id="bert-base-uncased")

# List available versions
versions = hub.list_versions()

# Download specific version
model_path = hub.download_version("v2.1.0")

# Get model info
info = hub.get_model_info()
print(f"Downloads: {info['downloads']}")
print(f"Likes: {info['likes']}")
```

### Auto-Detect Framework

```python
from pystreamai import FrameworkDetector

# Detect framework from model
framework = FrameworkDetector.detect_framework("/models/model.pt")
print(f"Detected: {framework}")

# Get appropriate server
server = FrameworkDetector.get_appropriate_server(
    framework,
    server_url="http://localhost:8080",
)
```

---

## Phase 9: Advanced Analytics

Detect root causes, data drift, and anomalies.

### Root Cause Analysis

```python
from pystreamai import RootCauseAnalyzer, MetricSnapshot

analyzer = RootCauseAnalyzer()

# Record metrics for versions
snapshot_v1 = MetricSnapshot(
    timestamp=datetime.now(),
    error_rate_percent=0.2,
    p95_latency_ms=100.0,
    p99_latency_ms=150.0,
    throughput_rps=100.0,
    avg_latency_ms=50.0,
    memory_usage_mb=2048,
)

snapshot_v2 = MetricSnapshot(
    timestamp=datetime.now(),
    error_rate_percent=5.0,  # Degraded
    p95_latency_ms=800.0,
    p99_latency_ms=2000.0,
    throughput_rps=50.0,  # Decreased
    avg_latency_ms=500.0,
    memory_usage_mb=4096,  # Increased
)

analyzer.record_snapshot("model-v1", snapshot_v1)
analyzer.record_snapshot("model-v2", snapshot_v2)

# Analyze degradation
analysis = analyzer.analyze_degradation("model-v2", "model-v1")
print(f"Severity: {analysis['severity']}")
print(f"Probable causes: {analysis['probable_causes']}")
print(f"Recommendations: {analysis['recommendations']}")

# Get metric correlations
correlation = analyzer.get_metric_correlation(
    "model-v2",
    "error_rate_percent",
    "memory_usage_mb",
)
print(f"Error rate ↔ Memory correlation: {correlation:.2f}")
```

### Drift Detection

```python
from pystreamai import DriftDetector

detector = DriftDetector()

# Set baseline
baseline_data = [1.0, 2.0, 3.0, 4.0, 5.0]
detector.baseline_distribution = {"mean": 3.0, "stdev": 1.4}

# Update current data
current_data = [10.0, 11.0, 12.0, 13.0, 14.0]  # Shifted distribution
detector.update_current_distribution(current_data)

# Detect data drift
has_drift, alert = detector.detect_data_drift(threshold=0.05)
if has_drift:
    print(f"🚨 {alert.drift_type}: {alert.description}")
    print(f"   Severity: {alert.severity}")
    print(f"   Action: {alert.recommended_action}")

# Detect prediction drift
has_pred_drift, alert = detector.detect_prediction_drift(
    baseline_predictions=[0.1, 0.2, 0.3],
    current_predictions=[0.5, 0.6, 0.7],
)

# Detect model drift (performance degradation)
has_model_drift, alert = detector.detect_model_drift(
    version_metrics_v1={"accuracy": 0.95, "f1": 0.92},
    version_metrics_v2={"accuracy": 0.80, "f1": 0.75},
)

# Get alerts
alerts = detector.get_drift_alerts(limit=5)
```

### Anomaly Detection

```python
from pystreamai import AnomalyDetector

detector = AnomalyDetector(contamination=0.05)

# Fit on normal data
normal_data = [10.0, 11.0, 9.0, 12.0, 11.0] * 20
detector.fit(normal_data)

# Predict anomalies
test_data = [11.0, 10.5, 100.0, 11.0, 12.0]  # 100.0 is anomaly
anomalies = detector.predict_anomalies(test_data)
print(f"Anomalies detected: {[i for i, a in enumerate(anomalies) if a]}")
```

---

## Phase 10: Multi-Model Orchestration

Deploy and manage multiple models as coordinated systems.

### Sequential Pipeline

```python
from pystreamai import ModelPipeline, PipelineStage, PipelineStageType

pipeline = ModelPipeline("nlp-processing-pipeline")

# Add stages in sequence
pipeline.add_sequential_stage("tokenizer", "tokenizer-model")
pipeline.add_sequential_stage("encoder", "bert-model")
pipeline.add_sequential_stage("classifier", "classification-model")

# Set versions
pipeline.set_version("tokenizer-model", "v1")
pipeline.set_version("bert-model", "v2")
pipeline.set_version("classification-model", "v3")

# Execute
input_data = {"text": "This is great!"}
output, metrics = await pipeline.execute(input_data, inference_fn)

print(f"Output: {output}")
print(f"Total latency: {metrics.total_latency_ms:.1f}ms")
print(f"Stage latencies: {metrics.stage_latencies}")
```

### Parallel Pipeline

```python
# Add models running in parallel
pipeline.add_parallel_stage(
    "multi_model_ensemble",
    ["model_1", "model_2", "model_3"],
)

# All three run simultaneously, outputs combined
```

### Conditional Pipeline

```python
def should_classify(outputs):
    """Only run classifier if confidence > 0.5"""
    return outputs.get("confidence", 0) > 0.5

pipeline.add_conditional_stage(
    "optional_classifier",
    "strict-classifier",
    condition=should_classify,
)
```

### Ensemble Pipeline

```python
# Create ensemble of models with weighted averaging
pipeline.add_ensemble_stage(
    "ensemble",
    ["model_v1", "model_v2", "model_v3"],
    weights={"model_v1": 0.2, "model_v2": 0.3, "model_v3": 0.5},
)

# Outputs are weighted average
```

### A/B Testing Across Models

```python
from pystreamai import CrossModelABTest

ab_test = CrossModelABTest("nlp-models-test")

# Set variants
ab_test.set_variant_a({
    "tokenizer": "v1",
    "bert": "v2",
    "classifier": "v3",
})

ab_test.set_variant_b({
    "tokenizer": "v2",
    "bert": "v3",
    "classifier": "v4",
})

ab_test.set_split(50)  # 50/50 split

# Record metrics
ab_test.record_metric("a", "accuracy", 0.92)
ab_test.record_metric("b", "accuracy", 0.94)
ab_test.record_metric("a", "latency_ms", 150)
ab_test.record_metric("b", "latency_ms", 160)

# Get stats
stats = ab_test.get_stats()
print(f"Variant A accuracy: {stats['variant_a']['accuracy']['mean']:.3f}")
print(f"Variant B accuracy: {stats['variant_b']['accuracy']['mean']:.3f}")

# Recommendation
winner, reason = ab_test.get_recommendation()
print(f"Winner: Variant {winner.upper()} - {reason}")
```

### Pipeline Registry

```python
from pystreamai import PipelineRegistry

registry = PipelineRegistry()

# Register pipeline
registry.register_pipeline(pipeline)

# Version pipeline
registry.version_pipeline("nlp-processing-pipeline", "v1.0")

# Promote to production
registry.promote_pipeline_version("nlp-processing-pipeline", "v1.0")

# Rollback if needed
registry.rollback_pipeline("nlp-processing-pipeline", "v0.9")
```

---

## Complete Example: End-to-End ML System

```python
from pystreamai import (
    AutoVersionManager,
    RollbackOrchestrator,
    RollbackStrategy,
    VersionCostTracker,
    PerformanceBenchmark,
    ModelPipeline,
    RootCauseAnalyzer,
    DriftDetector,
)

# 1. Initialize auto versioning
manager = AutoVersionManager()

# 2. Deploy models
for model_id, model_hash in [("v1", "abc"), ("v2", "def")]:
    version_id = manager.deploy_model("sentiment", model_hash)
    manager.promote_version(version_id)

# 3. Track costs
cost_tracker = VersionCostTracker()
# ... record metrics ...

# 4. Run benchmarks
benchmark = PerformanceBenchmark()
# ... run benchmarks ...

# 5. Build pipeline
pipeline = ModelPipeline("sentiment-pipeline")
pipeline.add_sequential_stage("tokenizer", "tokenizer-v1")
pipeline.add_sequential_stage("classifier", "classifier-v2")
# ... execute pipeline ...

# 6. Monitor metrics
analyzer = RootCauseAnalyzer()
detector = DriftDetector()
# ... record snapshots and detect drift ...

# 7. Rollback if needed
orchestrator = RollbackOrchestrator(manager)
await orchestrator.rollback(
    "classifier-v1",
    "v2 degraded",
    strategy=RollbackStrategy.CANARY,
)
```

---

## Performance & Scaling Considerations

### Cost Optimization
- Use VersionCostTracker to identify expensive versions
- Compare cost vs performance tradeoff
- Recommend versions based on cost+speed weighted criteria

### Performance Optimization
- Run PerformanceBenchmark regularly
- Compare latency, throughput, accuracy across versions
- Use statistical significance testing to validate improvements

### Analytics
- Track metrics per version continuously
- Use RootCauseAnalyzer to investigate degradation
- Detect drift early with DriftDetector

### Multi-Model Systems
- Sequential pipelines minimize latency
- Parallel pipelines maximize throughput
- Ensembles improve accuracy
- A/B tests validate improvements

---

## API Summary

### Phase 5
- `RollbackStrategy` — Strategy enum (INSTANT, CANARY, BLUE_GREEN, SHADOW)
- `RollbackConfig` — Configuration for rollback execution
- `RollbackOrchestrator` — Orchestrate rollbacks

### Phase 6
- `CostModel` — Define infrastructure costs
- `VersionCostTracker` — Track cost per version
- `PerformanceBenchmark` — Benchmark models

### Phase 8
- `ModelServer` — Abstract base
- `TensorFlowServingBackend` — TensorFlow Serving
- `TorchServeBackend` — TorchServe
- `KubernetesDeployment` — Kubernetes
- `HuggingFaceHub` — Hugging Face integration

### Phase 9
- `RootCauseAnalyzer` — Analyze degradation
- `DriftDetector` — Detect data/model drift
- `AnomalyDetector` — Detect anomalies

### Phase 10
- `ModelPipeline` — Orchestrate multiple models
- `PipelineRegistry` — Registry of pipelines
- `CrossModelABTest` — A/B test across models
- `PipelineStage` — Single stage in pipeline

---

See also:
- [Auto Versioning Guide](AUTO_VERSIONING.md)
- [CI/CD Integration Guide](CICD_INTEGRATION.md)
- [Deployment Guide](DEPLOYMENT.md)
