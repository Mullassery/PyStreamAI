# Automatic Model Versioning & Rollback

Deploy AI models with automatic health monitoring and one-command rollback. Zero YAML configuration.

## Features

- **Automatic Version Tracking** — Every deployment is tracked with metrics and health status
- **Health Monitoring** — Continuous monitoring of error rates, latency, and other metrics
- **Auto-Rollback** — Automatic rollback when metrics degrade below thresholds
- **Manual Rollback** — Quickly revert to any previous stable version with one command
- **Version History** — Full audit trail of all deployments and rollbacks
- **Zero Configuration** — Sensible defaults work out-of-the-box

## Quick Start

### Automatic Deployment with Health Monitoring

```python
from pystreamai.auto_versioning import AutoVersionManager, HealthThresholds
import hashlib

# Initialize the version manager
manager = AutoVersionManager()

# Define health thresholds (optional - uses defaults if not specified)
thresholds = HealthThresholds(
    error_rate_percent=5.0,      # Rollback if error rate > 5%
    p99_latency_ms=2000.0,       # Rollback if P99 latency > 2s
    p95_latency_ms=1000.0,       # Rollback if P95 latency > 1s
    min_samples=100,             # Need 100 requests before evaluating
    evaluation_window_seconds=300,  # Evaluate over 5-minute windows
    consecutive_bad_windows=2,   # Rollback after 2 consecutive bad windows
)

# Deploy a new model version
model_hash = hashlib.sha256(open("model.onnx", "rb").read()).hexdigest()
version_id = manager.deploy_model(
    model_id="sentiment-classifier",
    model_hash=model_hash,
    thresholds=thresholds,
)

print(f"Deployed version: {version_id}")

# Promote to production (mark as candidate for rollback)
manager.promote_version(version_id)
print(f"Promoted {version_id} to production")
```

### Recording Inference Metrics

Every inference should be recorded so health metrics can be computed:

```python
try:
    result = model.predict(input_data)
    latency_ms = (time.time() - start) * 1000
    manager.record_inference(
        model_id="sentiment-classifier",
        latency_ms=latency_ms,
        error=False,
    )
except Exception as e:
    manager.record_inference(
        model_id="sentiment-classifier",
        latency_ms=(time.time() - start) * 1000,
        error=True,
    )
    raise
```

### Periodic Health Evaluation

Call this periodically (e.g., every 5 minutes) to check for degradation and trigger auto-rollback:

```python
# In a background task or monitoring loop
manager.evaluate_health_windows()
```

### Register Rollback Handler

Respond to automatic rollbacks (e.g., notify ops team, restart services):

```python
def on_model_rollback(version_id: str, reason: str):
    print(f"🚨 Rollback: {version_id}")
    print(f"   Reason: {reason}")
    # TODO: Send alert to ops team
    # TODO: Update dashboards
    # TODO: Restart affected services

manager.on_rollback(on_model_rollback)
```

## Manual Rollback

### List Available Versions

```python
# All versions (newest first)
all_versions = manager.list_versions("sentiment-classifier")
for v in all_versions:
    print(f"{v['version_id']}")
    print(f"  Status: {v['health_status']}")
    print(f"  Deployed: {v['deployed_at']}")
    print(f"  Error Rate: {v.get('metrics', {}).get('error_rate_percent', 'N/A')}%")
    print()

# Only rollback candidates (healthy, promoted versions)
candidates = manager.get_rollback_candidates("sentiment-classifier")
print("\nCan rollback to:")
for v in candidates:
    print(f"  {v['version_id']} (deployed {v['deployed_at']})")
```

### Manual Rollback to Specific Version

```python
# Quick manual rollback when something breaks
success = manager.manual_rollback(
    version_id="sentiment-classifier-1697456789-a1b2c3d4",
    reason="Reverting due to customer reports",
)

if success:
    print("✓ Successfully rolled back!")
else:
    print("✗ Rollback failed")
```

## Integration with Deployment Pipeline

### With PyStreamAI Platform

```python
from pystreamai import Platform
from pystreamai.auto_versioning import AutoVersionManager
import hashlib

manager = AutoVersionManager()

# Load and deploy model
platform = Platform(backend="aws")
model = platform.load("bert-base-uncased")

# Get model hash for tracking
model_hash = hashlib.md5(str(model).encode()).hexdigest()

# Deploy and track
version_id = manager.deploy_model("bert", model_hash)

# Serve with PyStreamAI
endpoint = platform.serve(model, replicas=3)

# Promote once stable
manager.promote_version(version_id)

# Record metrics from requests
async def predict_with_tracking(input_data):
    start = time.time()
    try:
        result = await endpoint.predict(input_data)
        latency = (time.time() - start) * 1000
        manager.record_inference("bert", latency_ms=latency, error=False)
        return result
    except Exception as e:
        latency = (time.time() - start) * 1000
        manager.record_inference("bert", latency_ms=latency, error=True)
        raise
```

### With Custom Serving

```python
from pystreamai.auto_versioning import AutoVersionManager
from fastapi import FastAPI, HTTPException
import hashlib
import time

app = FastAPI()
manager = AutoVersionManager()

# Deploy on startup
@app.on_event("startup")
async def startup():
    global version_id
    model_hash = hashlib.sha256(open("model.onnx", "rb").read()).hexdigest()
    version_id = manager.deploy_model("my-model", model_hash)
    manager.promote_version(version_id)

# Endpoint with automatic tracking
@app.post("/predict")
async def predict(data: dict):
    start = time.time()
    try:
        result = model.predict(data)
        latency_ms = (time.time() - start) * 1000
        manager.record_inference("my-model", latency_ms=latency_ms, error=False)
        return {"prediction": result}
    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        manager.record_inference("my-model", latency_ms=latency_ms, error=True)
        raise HTTPException(status_code=500, detail=str(e))

# Health/status endpoint
@app.get("/model-status")
async def model_status():
    return manager.get_model_status("my-model")

# Manual rollback endpoint
@app.post("/rollback/{version_id}")
async def rollback(version_id: str, reason: str = "Manual operator request"):
    success = manager.manual_rollback(version_id, reason)
    if success:
        return {"status": "rolled_back", "version_id": version_id}
    else:
        raise HTTPException(status_code=400, detail="Rollback failed")

# Periodic health check (run in background)
@app.on_event("startup")
async def health_monitor_task():
    while True:
        manager.evaluate_health_windows()
        await asyncio.sleep(300)  # Every 5 minutes
```

### Command-Line Usage

```bash
# Deploy new version
python -c "
from pystreamai.auto_versioning import AutoVersionManager
import hashlib

manager = AutoVersionManager()
model_hash = 'abc123...'
vid = manager.deploy_model('my-model', model_hash)
print(f'Deployed: {vid}')
"

# List versions
python -c "
from pystreamai.auto_versioning import AutoVersionManager

manager = AutoVersionManager()
versions = manager.list_versions('my-model')
for v in versions:
    print(f'{v[\"version_id\"]} - {v[\"health_status\"]}')
"

# Rollback
python -c "
from pystreamai.auto_versioning import AutoVersionManager

manager = AutoVersionManager()
candidates = manager.get_rollback_candidates('my-model')
if candidates:
    success = manager.manual_rollback(candidates[0]['version_id'], 'Emergency rollback')
    print('Success!' if success else 'Failed')
"
```

## Configuration

### Default Thresholds

```python
# Error rate: rollback if > 5% of requests fail
error_rate_percent = 5.0

# P99 latency: rollback if 99th percentile latency > 2 seconds
p99_latency_ms = 2000.0

# P95 latency: rollback if 95th percentile latency > 1 second
p95_latency_ms = 1000.0

# Minimum samples: evaluate only after 100 requests
min_samples = 100

# Evaluation window: check metrics every 5 minutes
evaluation_window_seconds = 300

# Consecutive bad windows: rollback after 2 bad windows
consecutive_bad_windows = 2
```

### Custom Thresholds

```python
from pystreamai.auto_versioning import HealthThresholds

# Conservative thresholds (rollback quickly)
strict = HealthThresholds(
    error_rate_percent=1.0,
    p99_latency_ms=500.0,
    consecutive_bad_windows=1,
)

# Tolerant thresholds (absorb temporary degradation)
permissive = HealthThresholds(
    error_rate_percent=10.0,
    p99_latency_ms=5000.0,
    consecutive_bad_windows=3,
    min_samples=500,
)
```

## Monitoring & Observability

### Get Version Status

```python
status = manager.get_version_status("sentiment-classifier-1697456789-a1b2c3d4")

print(f"Model: {status['model_id']}")
print(f"Status: {status['health_status']}")
print(f"Requests: {status['request_count']}")
print(f"Errors: {status['error_count']} ({status['metrics'].get('error_rate_percent', 0):.1f}%)")
print(f"Latency: avg={status['avg_latency_ms']:.1f}ms, p99={status['p99_latency_ms']:.1f}ms")
```

### Get Model Status

```python
status = manager.get_model_status("sentiment-classifier")

print(f"Model: {status['model_id']}")
print(f"Active Version: {status['active_version']}")
print(f"\nAll Versions:")
for v in status['versions']:
    promoted = "✓" if v['promoted_at'] else " "
    print(f"  [{promoted}] {v['version_id']} - {v['health_status']}")
```

## Troubleshooting

### Version doesn't exist

```python
version = manager.registry.get_version(version_id)
if version is None:
    print(f"Version not found: {version_id}")
    # List available versions
    versions = manager.list_versions("my-model")
    for v in versions:
        print(f"  {v['version_id']}")
```

### Can't rollback to version

```python
# Only healthy, promoted versions can be rolled back to
candidates = manager.get_rollback_candidates("my-model")
if not candidates:
    print("No healthy versions available for rollback")
else:
    print("Available versions to rollback to:")
    for v in candidates:
        print(f"  {v['version_id']}")
```

### No automatic rollback occurring

Ensure you're calling `evaluate_health_windows()` periodically:

```python
import asyncio

async def health_monitor():
    while True:
        manager.evaluate_health_windows()
        await asyncio.sleep(300)  # Check every 5 minutes

asyncio.create_task(health_monitor())
```

## Best Practices

1. **Record All Inference** — Every request should call `record_inference()` so metrics are accurate
2. **Promote Before Monitoring** — Call `promote_version()` before sending production traffic
3. **Set Appropriate Thresholds** — Tune based on your service's SLOs
4. **Monitor Rollbacks** — Register handlers to be notified of automatic rollbacks
5. **Keep History** — The version registry stores 5 versions by default (configurable)
6. **Test Before Production** — Validate new versions in canary deployment first

## API Reference

### AutoVersionManager

```python
manager = AutoVersionManager(storage_path=".pystreamai_versions")

# Deploy a new version
version_id = manager.deploy_model(
    model_id: str,
    model_hash: str,
    thresholds: Optional[HealthThresholds] = None,
) -> str

# Promote version to production
manager.promote_version(version_id: str) -> None

# Record inference metric
manager.record_inference(
    model_id: str,
    latency_ms: float,
    error: bool = False,
) -> None

# Evaluate health and trigger auto-rollback
manager.evaluate_health_windows() -> None

# Manual rollback to specific version
success = manager.manual_rollback(
    version_id: str,
    reason: str = "Manual rollback",
) -> bool

# List all versions
versions = manager.list_versions(model_id: str) -> List[Dict]

# Get versions eligible for rollback
candidates = manager.get_rollback_candidates(model_id: str) -> List[Dict]

# Register rollback handler
manager.on_rollback(handler: Callable[[str, str], None]) -> None

# Get status of specific version
status = manager.get_version_status(version_id: str) -> Optional[Dict]

# Get status of model and all versions
status = manager.get_model_status(model_id: str) -> Dict
```

---

See also:
- [Getting Started Guide](GETTING_STARTED.md)
- [Deployment Guide](DEPLOYMENT.md)
- [Optimization Guide](OPTIMIZATION.md)
