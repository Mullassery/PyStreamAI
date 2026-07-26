# PyStreamAI API Reference

Complete API documentation for PyStreamAI.

## Platform

Central interface for all operations.

### `Platform(backend="local", gpu=None, num_gpus=1)`

Create a new platform instance.

**Parameters:**
- `backend` (str): Execution backend
  - `"local"` — Run on local machine (default)
  - `"aws"` — AWS EC2/SageMaker
  - `"gcp"` — Google Cloud
  - `"azure"` — Azure
- `gpu` (str, optional): GPU type
  - `"A100"`, `"H100"`, `"L4"`, `"V100"`, `"T4"`, `"RTX4090"`
- `num_gpus` (int): Number of GPUs (default: 1)

**Example:**
```python
from pystreamai import Platform

platform = Platform(backend="local", gpu="A100", num_gpus=2)
```

### `platform.load(model_id, version=None)`

Load a model.

**Parameters:**
- `model_id` (str): Model identifier
  - Hugging Face model: `"bert-base-uncased"`
  - Local path: `"./models/model.onnx"`
  - Remote URL: `"s3://bucket/model.onnx"`
- `version` (str, optional): Specific version to load

**Returns:** Model object

**Example:**
```python
model = platform.load("bert-base-uncased")
model = platform.load("./models/custom.onnx")
```

### `platform.serve(model, replicas=1, gpu=None, batch_size=32, batch_timeout_ms=100, auto_optimize=True, **kwargs)`

Deploy a model as an endpoint.

**Parameters:**
- `model`: Model object to deploy
- `replicas` (int): Number of replicas (default: 1)
- `gpu` (str, optional): GPU type for serving
- `batch_size` (int): Max batch size (default: 32)
- `batch_timeout_ms` (int): Batch timeout in ms (default: 100)
- `auto_optimize` (bool): Enable automatic optimization (default: True)

**Returns:** Endpoint object

**Example:**
```python
endpoint = platform.serve(
    model=model,
    replicas=3,
    gpu="A100",
    batch_size=64,
    batch_timeout_ms=50
)
```

## Endpoint

Deployed model endpoint.

### `endpoint.predict(data, timeout_ms=30000)`

Run inference.

**Parameters:**
- `data` (dict): Input data
- `timeout_ms` (int): Request timeout (default: 30000)

**Returns:** Prediction result with metadata

**Example:**
```python
result = endpoint.predict({"text": "Hello world"})
print(result)
# {
#   "output": "...",
#   "latency_ms": 42.5,
#   "batch_size": 8,
#   "cost_usd": 0.0001
# }
```

### `endpoint.predict_batch(data_list, max_wait_ms=100)`

Batch inference.

**Parameters:**
- `data_list` (list): List of input data
- `max_wait_ms` (int): Max wait time (default: 100)

**Returns:** List of results

**Example:**
```python
results = endpoint.predict_batch([
    {"text": "Hello"},
    {"text": "World"},
    {"text": "PyStreamAI"}
])
```

### `endpoint.get_stats()`

Get endpoint statistics.

**Returns:** Statistics dictionary

**Example:**
```python
stats = endpoint.get_stats()
print(stats)
# {
#   "requests": 1000,
#   "avg_latency_ms": 42.5,
#   "p99_latency_ms": 125.0,
#   "throughput_req_sec": 23.5,
#   "total_cost_usd": 0.42,
#   "models_active": 1,
#   "gpu_utilization_percent": 75.0
# }
```

### `endpoint.get_optimization_plan()`

Get GPU optimization plan.

**Returns:** Optimization recommendations

**Example:**
```python
plan = endpoint.get_optimization_plan()
print(plan)
```

### `endpoint.stop()`

Stop the endpoint.

**Example:**
```python
endpoint.stop()
```

## Cost Tracking

### `CostTracker(pricing=None)`

Track inference costs.

**Parameters:**
- `pricing` (Pricing, optional): Custom pricing configuration

**Example:**
```python
from pystreamai.cost_tracking import CostTracker

tracker = CostTracker()
tracker.record_inference(model_id="bert", latency_ms=42.5, batch_size=4, cost_usd=0.0001)
```

### `SpendManager(tracker, monthly_budget=1000.0)`

Manage spending with budget enforcement.

**Parameters:**
- `tracker` (CostTracker): Cost tracker instance
- `monthly_budget` (float): Monthly budget in USD (default: $1000)

**Example:**
```python
from pystreamai.cost_tracking import SpendManager

spend_manager = SpendManager(tracker, monthly_budget=500)

if spend_manager.should_allow_request("bert"):
    result = endpoint.predict(data)

status = spend_manager.get_budget_status()
print(status)
```

## Observability

### `set_metric_backend(backend)`

Set observability backend.

**Parameters:**
- `backend`: Backend instance (PrometheusBackend, DatadogBackend, etc.)

**Example:**
```python
from pystreamai.observability import PrometheusBackend
from pystreamai import set_metric_backend

set_metric_backend(PrometheusBackend(port=8001))
```

### `get_metrics()`

Get metric collector.

**Returns:** MetricCollector instance

**Example:**
```python
from pystreamai import get_metrics

metrics = get_metrics()
metrics.log(InferenceMetric(...))
```

## Deployment

### `CanaryDeployment(current_version, canary_version, canary_traffic_percent=10.0)`

Canary deployment manager.

**Parameters:**
- `current_version`: Current model version
- `canary_version`: New model version to test
- `canary_traffic_percent`: % of traffic to route to canary (default: 10)

**Example:**
```python
from pystreamai.deployment import CanaryDeployment

canary = CanaryDeployment(model_v1, model_v2, traffic_percent=10)

# Monitor metrics...

canary.promote_canary()  # Promote to 100%
```

### `ABTestDeployment(variant_a, variant_b, split_percent=50.0)`

A/B testing manager.

**Parameters:**
- `variant_a`: First model variant
- `variant_b`: Second model variant
- `split_percent`: % traffic to variant A (default: 50)

**Example:**
```python
from pystreamai.deployment import ABTestDeployment

ab_test = ABTestDeployment(model_a, model_b, split_percent=70)

# Track metrics per variant
ab_test.record_metric("variant_a", "accuracy", 0.95)
ab_test.record_metric("variant_b", "accuracy", 0.93)

stats = ab_test.get_stats()
```

## LLM Optimization

### `LLMOptimizationEngine(model)`

LLM-specific optimization engine.

**Parameters:**
- `model`: Base model

**Methods:**
- `enable_speculative_decoding(draft_model)` — Enable speculative decoding
- `enable_prompt_caching()` — Enable prompt caching
- `enable_paged_attention()` — Enable paged attention

**Example:**
```python
from pystreamai.llm_optimization import LLMOptimizationEngine

engine = LLMOptimizationEngine(model)
engine.enable_speculative_decoding(draft_model)
engine.enable_prompt_caching()

output, metrics = engine.generate("What is AI?", max_tokens=100)
```

## Edge Deployment

### `EdgeDeploymentPipeline(model_path, target_device)`

Prepare model for edge deployment.

**Parameters:**
- `model_path` (str): Path to model
- `target_device` (EdgeDevice): Target device

**Example:**
```python
from pystreamai.edge_deployment import EdgeDeploymentPipeline, EdgeDevice

pipeline = EdgeDeploymentPipeline(
    "model.onnx",
    target_device=EdgeDevice.RASPBERRY_PI
)

result = pipeline.prepare_model()
# Auto-quantizes and compiles for target device
```

**Supported Devices:**
- `EdgeDevice.RASPBERRY_PI`
- `EdgeDevice.JETSON_NANO`
- `EdgeDevice.JETSON_ORIN`
- `EdgeDevice.ESP32`
- `EdgeDevice.MOBILE_IOS`
- `EdgeDevice.MOBILE_ANDROID`
- `EdgeDevice.BROWSER_WASM`

## Caching

### `SemanticCache(embedding_model, similarity_threshold=0.95)`

Cache based on semantic similarity.

**Example:**
```python
from pystreamai.advanced_caching import SemanticCache

cache = SemanticCache(embedding_model, similarity_threshold=0.95)

# Lookup similar queries
result, similarity = cache.get("What is AI?")

# Cache results
cache.set("What is artificial intelligence?", result)
```

### `HybridCache(embedding_model)`

Combine semantic, embedding, and result caching.

**Example:**
```python
from pystreamai.advanced_caching import HybridCache

cache = HybridCache(embedding_model)

result = cache.get_result("bert", input_data)
if not result:
    result = endpoint.predict(input_data)
    cache.set_result("bert", input_data, result)
```

## Decorators

### `@platform.train`

Mark function as training job.

**Example:**
```python
@platform.train(gpu="A100", time_limit="1h")
def train_model(data):
    model = train_bert(data)
    return model

job = train_model(data="s3://bucket/data")
```

### `@platform.serve`

Mark function as serving endpoint.

**Example:**
```python
@platform.serve(replicas=3, gpu="L4")
def predict(x):
    return model.predict(x)

endpoint = predict()
```

### `@platform.pipeline`

Mark function as pipeline.

**Example:**
```python
@platform.pipeline(name="training-pipeline")
def workflow():
    data = preprocess()
    model = train(data)
    serve(model)

workflow()
```

## Configuration

### Environment Variables

```bash
# GPU selection
export PYSTREAMAI_GPU="A100"

# Batch configuration
export PYSTREAMAI_BATCH_SIZE=32
export PYSTREAMAI_BATCH_TIMEOUT_MS=100

# Cost tracking
export PYSTREAMAI_ENABLE_COST_TRACKING=true
export PYSTREAMAI_MONTHLY_BUDGET=1000

# Observability
export PYSTREAMAI_OBSERVABILITY_BACKEND="prometheus"
export PYSTREAMAI_PROMETHEUS_PORT=8001
```

### Config File

```yaml
# pystreamai.yaml
platform:
  backend: local
  gpu: A100
  num_gpus: 1

serving:
  batch_size: 32
  batch_timeout_ms: 100
  auto_optimize: true

cost_tracking:
  enabled: true
  monthly_budget: 1000

observability:
  backend: prometheus
  port: 8001
```

## Error Handling

### Common Errors

**OutOfMemoryError**: GPU memory exceeded
```python
# Solution: Reduce batch size or use smaller GPU
endpoint = platform.serve(model, batch_size=8)
```

**ModelNotFoundError**: Model not found
```python
# Solution: Verify model ID and check internet connection
model = platform.load("bert-base-uncased")  # Must exist on Hugging Face
```

**BudgetExceededError**: Monthly budget exceeded
```python
# Solution: Increase budget or stop inferences
status = spend_manager.get_budget_status()
if status["percent_used"] >= 100:
    # Handle gracefully
    print("Budget exhausted for this month")
```

## Type Hints

All PyStreamAI functions include type hints for IDE support:

```python
from pystreamai import Platform
from pystreamai.serving import Endpoint
from typing import Dict, Any

platform: Platform = Platform()
model = platform.load("bert-base-uncased")
endpoint: Endpoint = platform.serve(model)
result: Dict[str, Any] = endpoint.predict({"text": "Hello"})
```

---

For more examples, see the [Getting Started Guide](GETTING_STARTED.md) and [Examples Directory](../examples/).
