# PyStreamAI Getting Started Guide

Welcome to PyStreamAI — the simplest way to deploy AI models to production.

This guide will get you up and running in 5 minutes.

## Installation

### Prerequisites
- Python 3.10+
- pip or uv
- NVIDIA GPU (optional, for GPU acceleration)

### Basic Installation

```bash
pip install pystreamai
```

For GPU support:
```bash
pip install pystreamai[gpu]
```

For edge deployment:
```bash
pip install pystreamai[edge]
```

For full features:
```bash
pip install pystreamai[all]
```

## Your First Inference

### Step 1: Load a Model

```python
from pystreamai import Platform

# Initialize platform
platform = Platform()

# Load a model (downloads from Hugging Face automatically)
model = platform.load("distilbert-base-uncased")
```

### Step 2: Run Inference

```python
# Single inference
result = model.predict({"text": "Hello world"})
print(result)
# Output: {"output": "prediction", "latency_ms": 42.5}
```

### Step 3: Deploy as API

```python
# Start HTTP server
endpoint = platform.serve(model, port=8080)

# Now accessible at http://localhost:8080/predict
```

Test with curl:
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'
```

## Core Concepts

### Platform
Central interface for all operations.

```python
from pystreamai import Platform

platform = Platform(
    backend="local",      # or "aws", "gcp", "azure"
    gpu="A100",          # optional GPU type
    num_gpus=1           # number of GPUs
)
```

### Models
Load and manage models.

```python
# From Hugging Face
model = platform.load("bert-base-uncased")

# From local path
model = platform.load("./models/model.onnx")

# With specific version
model = platform.load("bert-base-uncased", version="v2")
```

### Endpoints
Deploy models as APIs.

```python
# Simple deployment
endpoint = platform.serve(model, replicas=2, gpu="L4")

# With configuration
endpoint = platform.serve(
    model=model,
    replicas=3,
    gpu="A100",
    batch_size=32,
    timeout_ms=5000,
    auto_optimize=True
)

# Use endpoint
result = endpoint.predict(data)
```

### Batching
Automatic request batching for efficiency.

```python
# Batching is automatic
# Collects up to 32 requests or 100ms (configurable)
endpoint = platform.serve(
    model,
    batch_size=32,
    batch_timeout_ms=100
)

# Your requests are automatically batched with others
result = endpoint.predict(data)  # May wait up to 100ms
```

### Monitoring
Built-in metrics and monitoring.

```python
# Get endpoint statistics
stats = endpoint.get_stats()
print(stats)
# {
#   "requests": 1000,
#   "avg_latency_ms": 42.5,
#   "p99_latency_ms": 125.0,
#   "throughput_req_sec": 23.5,
#   "total_cost_usd": 0.42
# }
```

## Optimization

### Automatic Optimization
By default, PyStreamAI applies all optimizations:

```python
endpoint = platform.serve(model, auto_optimize=True)
```

This enables:
- ✓ ONNX Runtime (2-3x faster)
- ✓ Quantization (INT8 or FP16)
- ✓ Batching
- ✓ Caching
- ✓ TensorRT (if GPU available)

**Result**: 40-50x speedup vs baseline

### Manual Control

```python
from pystreamai.gpu import GPUOptimizer

optimizer = GPUOptimizer("A100")
optimizer.enable_tensorrt(fp16=True, int8=False, sparsity=True)

endpoint = platform.serve(model, optimizer=optimizer)
```

## Cost Tracking

### Enable Cost Tracking

```python
from pystreamai.cost_tracking import SpendManager, CostTracker

tracker = CostTracker()
spend_manager = SpendManager(tracker, monthly_budget=1000.0)

# Check before each request
if spend_manager.should_allow_request("bert"):
    result = endpoint.predict(data)
else:
    print("Budget exceeded")

# Get cost status
status = spend_manager.get_budget_status()
print(status)
# {
#   "monthly_budget_usd": 1000.0,
#   "total_spent_usd": 234.56,
#   "remaining_usd": 765.44,
#   "percent_used": 23.5
# }
```

## Deployment to Production

### Local Development

```python
platform = Platform(backend="local")
endpoint = platform.serve(model)
# Runs on your machine
```

### Cloud Deployment

#### AWS
```python
platform = Platform(backend="aws", region="us-west-2")
endpoint = platform.serve(model, instance_type="ml.g4dn.xlarge")
```

#### GCP
```python
platform = Platform(backend="gcp", region="us-central1")
endpoint = platform.serve(model, machine_type="n1-standard-4")
```

#### Azure
```python
platform = Platform(backend="azure", region="eastus")
endpoint = platform.serve(model)
```

### Docker Deployment

```bash
# Build container
pystreamai build --model bert-base-uncased --output image.tar

# Run locally
docker run -p 8080:8080 pystreamai:latest

# Deploy to cloud
docker push myregistry.azurecr.io/pystreamai:latest
```

## Advanced Features

### Canary Deployments

```python
# Start serving old model
endpoint_v1 = platform.serve(model_v1)

# Deploy new model to 10% of traffic
endpoint_v2 = platform.serve(model_v2)
platform.start_canary_deployment(endpoint_v2, traffic_percent=10)

# Monitor metrics, then promote
platform.promote_canary()  # Promote v2 to 100%
```

### Model Versioning

```python
from pystreamai.model_registry import MLflowRegistry

registry = MLflowRegistry()

# Log model during training
registry.log_model(model, "bert-v1")

# Load specific version
model = registry.load_model("bert-v1")

# Get version history
versions = registry.get_model_versions("bert")
```

### Observability

```python
from pystreamai.observability import PrometheusBackend
from pystreamai import set_metric_backend

# Enable Prometheus metrics
set_metric_backend(PrometheusBackend(port=8001))

# Metrics now scraped at http://localhost:8001/metrics
```

## Troubleshooting

### High Latency

**Problem**: Inference is slow

**Solution**:
```python
# Check if optimization is enabled
endpoint = platform.serve(model, auto_optimize=True)

# Check stats
stats = endpoint.get_stats()
print(f"Latency: {stats['avg_latency_ms']}ms")

# If still high, check batch configuration
endpoint = platform.serve(model, batch_size=1, batch_timeout_ms=0)
# Lower batch sizes give lower latency but worse throughput
```

### Out of Memory

**Problem**: CUDA out of memory

**Solution**:
```python
# Reduce batch size
endpoint = platform.serve(model, batch_size=8)  # Default 32

# Enable quantization
endpoint = platform.serve(model, quantization="int8")

# Use smaller GPU
platform = Platform(gpu="L4")  # Instead of A100
```

### Budget Exceeded

**Problem**: Spending too much

**Solution**:
```python
# Lower budget
spend_manager = SpendManager(tracker, monthly_budget=500)

# Use cheaper GPU
platform = Platform(gpu="L4")  # vs A100

# Increase batch size (amortize cost)
endpoint = platform.serve(model, batch_size=64)
```

## Next Steps

- **[Deployment Guide](DEPLOYMENT.md)** — Deploy to production
- **[API Reference](API_REFERENCE.md)** — Complete API docs
- **[Optimization Guide](OPTIMIZATION.md)** — Tune for your use case
- **[Monitoring Guide](MONITORING.md)** — Set up observability

## Need Help?

- **GitHub Issues**: https://github.com/mullassery/pystreamai/issues
- **Discussions**: https://github.com/mullassery/pystreamai/discussions
- **Email**: mullassery@gmail.com
