# PyStreamAI Deployment Guide

Deploy AI models to production with PyStreamAI.

## Local Development

### Quick Start

```python
from pystreamai import Platform

platform = Platform(backend="local")
model = platform.load("bert-base-uncased")
endpoint = platform.serve(model)

# Test
result = endpoint.predict({"text": "Hello world"})
print(result)
```

Run HTTP server:
```python
# Starts server at http://localhost:8080/predict
endpoint.start_server(port=8080)
```

Test with curl:
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'
```

## Docker Deployment

### Build Container

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install PyStreamAI
RUN pip install pystreamai

# Copy model and code
COPY model.onnx .
COPY app.py .

# Expose port
EXPOSE 8080

# Run
CMD ["python", "app.py"]
```

**app.py:**
```python
from pystreamai import Platform

platform = Platform(backend="local")
model = platform.load("./model.onnx")
endpoint = platform.serve(model)

endpoint.start_server(port=8080, host="0.0.0.0")
```

Build and run:
```bash
docker build -t pystreamai:latest .
docker run -p 8080:8080 pystreamai:latest
```

## Cloud Deployment

### AWS EC2

```python
from pystreamai import Platform

platform = Platform(
    backend="aws",
    region="us-west-2",
    instance_type="g4dn.xlarge"  # NVIDIA GPU
)

model = platform.load("bert-base-uncased")
endpoint = platform.serve(model, replicas=3)

# Auto-deployed with load balancer
print(endpoint.get_url())
# https://your-endpoint.us-west-2.aws.pystreamai.io
```

### AWS SageMaker

```python
platform = Platform(backend="aws", use_sagemaker=True)
model = platform.load("bert-base-uncased")

endpoint = platform.serve(
    model,
    instance_type="ml.g4dn.xlarge",
    auto_scaling_target_value=70.0  # CPU utilization
)
```

### Google Cloud

```python
from pystreamai import Platform

platform = Platform(
    backend="gcp",
    project_id="my-project",
    region="us-central1"
)

model = platform.load("bert-base-uncased")
endpoint = platform.serve(model, replicas=3)

# Deploys to Cloud Run
print(endpoint.get_url())
# https://pystreamai-xxxxx-uc.a.run.app
```

### Azure

```python
from pystreamai import Platform

platform = Platform(
    backend="azure",
    resource_group="my-rg",
    workspace_name="my-workspace",
    region="eastus"
)

model = platform.load("bert-base-uncased")
endpoint = platform.serve(model)

# Deploys to Azure Container Instances
print(endpoint.get_url())
```

## Kubernetes Deployment

### Create Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pystreamai-bert
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pystreamai
  template:
    metadata:
      labels:
        app: pystreamai
    spec:
      containers:
      - name: pystreamai
        image: pystreamai:latest
        ports:
        - containerPort: 8080
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 4Gi
          requests:
            nvidia.com/gpu: 1
            memory: 2Gi
        env:
        - name: PYSTREAMAI_GPU
          value: "A100"
```

Deploy:
```bash
kubectl apply -f deployment.yaml

# Create service
kubectl expose deployment pystreamai-bert \
  --type LoadBalancer \
  --port 8080
```

## Monitoring & Health Checks

### Health Endpoint

```bash
curl http://localhost:8080/health

# Response:
# {
#   "status": "healthy",
#   "uptime_seconds": 1234.5,
#   "requests_processed": 5000,
#   "gpu_utilization": 75.0,
#   "memory_utilization": 60.0
# }
```

### Metrics Endpoint

```bash
curl http://localhost:8080/metrics

# Prometheus-format metrics:
# pystreamai_inference_latency_ms_bucket{...}
# pystreamai_inference_latency_ms_sum{...}
# pystreamai_inference_latency_ms_count{...}
```

### Kubernetes Liveness Probe

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3
```

## Canary Deployments

### Blue-Green Deployment

```python
from pystreamai import Platform

platform = Platform(backend="aws")

# Deploy current model
model_v1 = platform.load("bert-v1")
endpoint_v1 = platform.serve(model_v1, name="bert-prod-blue")

# Deploy new model to parallel infrastructure
model_v2 = platform.load("bert-v2")
endpoint_v2 = platform.serve(model_v2, name="bert-prod-green")

# Test v2 with real traffic
# Monitor metrics...
# If healthy, switch traffic
platform.switch_traffic(
    from_endpoint="bert-prod-blue",
    to_endpoint="bert-prod-green"
)
```

### Canary with Traffic Split

```python
from pystreamai.deployment import CanaryDeployment

# 90% to v1, 10% to v2
canary = CanaryDeployment(
    current_version=model_v1,
    canary_version=model_v2,
    canary_traffic_percent=10
)

# After validation
canary.promote_canary()  # Move to 100% v2
```

## Model Versioning

### Register Models

```python
from pystreamai.model_registry import MLflowRegistry

registry = MLflowRegistry()

# Log model
registry.start_training("bert-training")
registry.log_params({"epochs": 10, "batch_size": 32})

# During training
registry.log_metrics({"loss": 0.1234, "accuracy": 0.95}, step=1)

# After training
registry.register_model(model, "bert-v2", tags={
    "framework": "pytorch",
    "task": "sentiment",
    "accuracy": "0.95"
})
```

### Load Specific Version

```python
model_v1 = registry.load_model("bert-v1")
model_v2 = registry.load_model("bert-v2")

# Rollback if needed
model_latest = registry.load_model("bert")  # Latest version
```

## Cost Management

### Monitor Spending

```python
from pystreamai.cost_tracking import SpendManager, CostTracker

tracker = CostTracker()
spend_manager = SpendManager(tracker, monthly_budget=1000)

# Check before request
if not spend_manager.should_allow_request("bert"):
    raise Exception("Budget exceeded")

# Track spending
status = spend_manager.get_budget_status()
print(f"Spent: ${status['total_spent_usd']}")
print(f"Remaining: ${status['remaining_usd']}")
```

### Cost Optimization

```python
# Strategy 1: Use cheaper GPU
platform = Platform(gpu="L4")  # vs A100

# Strategy 2: Increase batch size
endpoint = platform.serve(model, batch_size=64)  # Amortize cost

# Strategy 3: Enable aggressive quantization
endpoint = platform.serve(model, quantization="int4")  # Smaller model

# Strategy 4: Use spot instances
endpoint = platform.serve(model, use_spot_instances=True)
```

## Auto-Scaling

### Request-Based Scaling

```python
endpoint = platform.serve(
    model,
    min_replicas=1,
    max_replicas=10,
    target_requests_per_replica=100  # Scale when exceeded
)
```

### Latency-Based Scaling

```python
endpoint = platform.serve(
    model,
    min_replicas=1,
    max_replicas=10,
    target_latency_ms=50  # Scale if avg latency > 50ms
)
```

### CPU/GPU-Based Scaling

```python
endpoint = platform.serve(
    model,
    min_replicas=1,
    max_replicas=10,
    target_gpu_utilization=80  # Scale at 80% GPU usage
)
```

## Troubleshooting

### High Latency

1. Check if model is quantized:
```python
stats = endpoint.get_stats()
print(f"Model: {stats['model_optimization']}")  # Should show optimizations
```

2. Increase batch size:
```python
endpoint = platform.serve(model, batch_size=64)
```

3. Check GPU utilization:
```python
health = endpoint.get_health()
print(f"GPU: {health['gpu_utilization']}%")
```

### Out of Memory

1. Reduce batch size:
```python
endpoint = platform.serve(model, batch_size=8)
```

2. Enable quantization:
```python
endpoint = platform.serve(model, quantization="int8")
```

3. Use smaller GPU:
```python
platform = Platform(gpu="L4")  # vs A100
```

### Model Not Loading

1. Verify model exists:
```python
try:
    model = platform.load("bert-base-uncased")
except Exception as e:
    print(f"Error: {e}")
    # Check internet connection and Hugging Face API
```

2. Use local path:
```python
model = platform.load("./models/model.onnx")
```

## Production Checklist

- [ ] Model is quantized or optimized
- [ ] Monitoring/metrics are enabled
- [ ] Health checks configured
- [ ] Cost tracking enabled
- [ ] Budget alerts set
- [ ] Auto-scaling configured
- [ ] Canary deployment plan ready
- [ ] Rollback procedure documented
- [ ] Observability integrated (Prometheus/Datadog)
- [ ] Load testing completed
- [ ] Disaster recovery plan ready

---

For more information, see:
- [Getting Started Guide](GETTING_STARTED.md)
- [Optimization Guide](OPTIMIZATION.md)
- [Monitoring Guide](MONITORING.md)
