# PyStreamAI Production Serving Architecture

## Overview

Simple, fast inference serving with automatic optimization.

```
┌─────────────────────────────────────────────┐
│  HTTP API / gRPC                            │
├─────────────────────────────────────────────┤
│  Request Router (load balancing)            │
├─────────────────────────────────────────────┤
│  Batch Manager (collect requests)           │
├─────────────────────────────────────────────┤
│  Inference Engine (PyTorch/ONNX/TensorRT)   │
├─────────────────────────────────────────────┤
│  GPU Scheduler (multi-GPU, multi-stream)    │
├─────────────────────────────────────────────┤
│  Metrics Collector (latency, cost, etc.)    │
└─────────────────────────────────────────────┘
```

## v0.1 Serving (Minimum Viable)

### Goals
1. ✅ Load ONNX/PyTorch models
2. ✅ Dynamic batching (collect N requests or timeout T ms)
3. ✅ Multi-GPU inference
4. ✅ Request-level metrics
5. ✅ HTTP API (FastAPI)

### Non-Goals (v0.2+)
- Model hot-reloading
- Canary deployments
- A/B testing
- Request prioritization
- gRPC (HTTP 1.1 is fine for v0.1)
- Streaming responses

## Request Flow

```
Request → Router → Batch Queue → Inference → GPU → Metrics → Response
  (1ms)    (0.1ms)  (0-100ms)     (10-100ms)  (10-50ms)  (0.1ms)   (0.1ms)
```

**Total latency: 20-300ms depending on batch timeout**

## Batching Strategy

```
Requests arrive at rate: R req/sec
Batch timeout: T ms (e.g., 100ms)
Batch size limit: B (e.g., 32)

If request_count >= B → flush
Else if time_since_first_request >= T → flush
Else → wait
```

Example with T=100ms, B=32:
- R=100 req/sec → batches of 10 (every 100ms)
- R=10 req/sec → single requests (timeout every 100ms)
- R=500 req/sec → batches of 32 (flush immediately)

## GPU Scheduling

```python
# Multi-GPU round-robin
gpus = [GPU0, GPU1, GPU2, GPU3]
batch.assign_to_gpu(gpus[current_idx % len(gpus)])

# Alternative: assign to GPU with most free memory
batch.assign_to_gpu(max(gpus, key=lambda g: g.free_memory))
```

## Metrics Collection

Per-request metrics:
- Request ID
- Model ID
- Batch size
- Latency (queue + inference + post-process)
- GPU used
- Cost (tokens × price per token)
- Speedup vs baseline

## Storage & Caching

Models stored in:
- **Hugging Face Hub** (default)
- **Local cache** (~/.cache/pystreamai)
- **S3/GCS** (for enterprise)

First request loads model (5-30s). Subsequent requests reuse cached model.

## Error Handling

```
Invalid input → 400 Bad Request
Model not found → 404 Not Found
GPU out of memory → 507 Insufficient Storage
Inference error → 500 Internal Server Error
```

## Next Steps

1. Implement `InferenceServer` class
2. Add HTTP API (FastAPI)
3. Test with BERT model
4. Benchmark against baseline
5. Deploy on sample GPU
