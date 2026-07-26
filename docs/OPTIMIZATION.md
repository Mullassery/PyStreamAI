# PyStreamAI Optimization Guide

Optimize inference performance for your use case.

## Automatic Optimization

By default, PyStreamAI applies all available optimizations:

```python
endpoint = platform.serve(model, auto_optimize=True)
```

This enables:
- ✓ ONNX Runtime (2-3x faster)
- ✓ Quantization (INT8 or FP16)
- ✓ Batching
- ✓ Caching (semantic, embedding, result)
- ✓ TensorRT (if GPU available)
- ✓ Paged Attention (for LLMs)

**Result**: 40-50x speedup vs baseline

## Performance Tiers

### Tier 1: Fast (Best for low-latency)

```python
endpoint = platform.serve(
    model,
    auto_optimize=True,
    batch_size=1,           # No batching delay
    batch_timeout_ms=0,     # Immediate response
    quantization="fp16"     # Light compression
)
```

**Latency**: ~50ms  
**Throughput**: Lower  
**Use case**: Real-time user interactions

### Tier 2: Balanced (Default)

```python
endpoint = platform.serve(
    model,
    auto_optimize=True,
    batch_size=32,          # Smart batching
    batch_timeout_ms=100,   # 100ms window
    quantization="int8"     # Moderate compression
)
```

**Latency**: ~50ms average (batched)  
**Throughput**: High  
**Use case**: API endpoints, normal production

### Tier 3: Throughput (Best for batch)

```python
endpoint = platform.serve(
    model,
    auto_optimize=True,
    batch_size=256,         # Large batches
    batch_timeout_ms=1000,  # 1s window
    quantization="int4"     # Aggressive compression
)
```

**Latency**: ~500ms average (batched)  
**Throughput**: Very high  
**Use case**: Batch processing, offline inference

## GPU Optimization

### Automatic GPU Selection

```python
from pystreamai.gpu import GPUOptimizer

optimizer = GPUOptimizer("auto")  # Auto-detect best GPU
endpoint = platform.serve(model, optimizer=optimizer)
```

### Manual GPU Configuration

```python
from pystreamai.gpu import GPUOptimizer

# A100: Best for LLMs and large models
optimizer = GPUOptimizer("A100")
optimizer.enable_tensorrt(fp16=True, int8=False, sparsity=True)
endpoint = platform.serve(model, optimizer=optimizer)

# L4: Best for cost-efficiency
optimizer = GPUOptimizer("L4")
optimizer.enable_tensorrt(fp16=True, int8=True)
endpoint = platform.serve(model, optimizer=optimizer)

# T4: Budget GPU
optimizer = GPUOptimizer("T4")
optimizer.enable_tensorrt(fp16=True)
endpoint = platform.serve(model, optimizer=optimizer)
```

### Multi-GPU Optimization

```python
from pystreamai.gpu import MultiGPUInference

# Use multiple GPUs
multi_gpu = MultiGPUInference(num_gpus=4, gpu_type="A100")
multi_gpu.enable_nccl()      # GPU-to-GPU communication
multi_gpu.enable_nvlink()    # High-bandwidth NVLink

endpoint = platform.serve(model, multi_gpu=multi_gpu)
```

## Quantization

### INT8 Quantization (Best balance)

```python
endpoint = platform.serve(
    model,
    quantization="int8"
)
# 3x faster, minimal accuracy loss
```

### INT4 Quantization (Aggressive)

```python
endpoint = platform.serve(
    model,
    quantization="int4"
)
# 5x faster, slight accuracy loss
# Good for mobile/edge deployment
```

### FP16 Quantization (Conservative)

```python
endpoint = platform.serve(
    model,
    quantization="fp16"
)
# 1.5x faster, no accuracy loss
```

## Batching Tuning

### Find Optimal Batch Size

```python
from benchmarks.bench_oss_comparison import benchmark_pytorch_baseline

# Test different batch sizes
for batch_size in [1, 4, 8, 16, 32, 64, 128]:
    endpoint = platform.serve(model, batch_size=batch_size)
    stats = endpoint.get_stats()
    
    throughput = stats["throughput_req_sec"]
    latency = stats["avg_latency_ms"]
    
    print(f"Batch {batch_size}: {throughput:.1f} req/sec, {latency:.1f}ms latency")
```

### Adaptive Batching

```python
endpoint = platform.serve(
    model,
    batch_size="adaptive"  # Auto-adjust based on load
)
```

## LLM-Specific Optimization

### Speculative Decoding (2-3x speedup)

```python
from pystreamai.llm_optimization import LLMOptimizationEngine

engine = LLMOptimizationEngine(model)
engine.enable_speculative_decoding(draft_model)

output, metrics = engine.generate("What is AI?")
print(f"Speedup: {metrics['speculative_speedup']:.2f}x")
```

### Prompt Caching (1-2x speedup for repeated prompts)

```python
engine = LLMOptimizationEngine(model)
engine.enable_prompt_caching()

# First call: slow (caches prompt)
output1 = engine.generate("Question: What is AI?")

# Second call with same prompt: faster (uses cache)
output2 = engine.generate("Question: What is AI?")  # ~2x faster
```

### Paged Attention (Memory efficient)

```python
engine = LLMOptimizationEngine(model)
engine.enable_paged_attention()

# Can now handle longer sequences without OOM
output = engine.generate("Long prompt...", max_tokens=1000)
```

## Caching Strategies

### Semantic Caching (1-2x speedup)

```python
from pystreamai.advanced_caching import SemanticCache

cache = SemanticCache(similarity_threshold=0.95)

# Query 1: "What is artificial intelligence?"
result1, _ = cache.get("What is artificial intelligence?")
if not result1:
    result1 = endpoint.predict({"text": "What is artificial intelligence?"})
    cache.set("What is artificial intelligence?", result1)

# Query 2: "Tell me about AI" (similar, hits cache)
result2, similarity = cache.get("Tell me about AI")  # Semantic match!
if result2:
    print(f"Cache hit with {similarity:.1%} similarity")
```

### Embedding Caching (90% reduction in embedding compute)

```python
from pystreamai.advanced_caching import HybridCache

cache = HybridCache(embedding_model)

# Embeddings are cached automatically
embedding1 = cache.get_embedding("Hello world")
embedding2 = cache.get_embedding("Hello world")  # Cached, instant
```

### Result Caching (Instant for deterministic models)

```python
cache = HybridCache(embedding_model)

# Results cached by input hash
result1 = cache.get_result("bert", {"text": "hello"})
if not result1:
    result1 = endpoint.predict({"text": "hello"})
    cache.set_result("bert", {"text": "hello"}, result1)

result2 = cache.get_result("bert", {"text": "hello"})  # Instant
```

## Edge Optimization

### Mobile Deployment

```python
from pystreamai.edge_deployment import EdgeDeploymentPipeline, EdgeDevice

# Prepare model for iOS
pipeline = EdgeDeploymentPipeline("model.onnx", EdgeDevice.MOBILE_IOS)
result = pipeline.prepare_model()

# Model is now:
# - Quantized to INT8 (75% smaller)
# - Compiled to Core ML
# - Optimized for A15 GPU
```

### Raspberry Pi Deployment

```python
# Prepare for Raspberry Pi (CPU only)
pipeline = EdgeDeploymentPipeline("model.onnx", EdgeDevice.RASPBERRY_PI)
result = pipeline.prepare_model()

# Model is now:
# - Quantized to INT4 (90% smaller)
# - Compiled to ONNX Runtime
# - Optimized for ARM64
```

### Browser Deployment (WASM)

```python
# Prepare for browser
pipeline = EdgeDeploymentPipeline("model.onnx", EdgeDevice.BROWSER_WASM)
result = pipeline.prepare_model()

# Model runs entirely in browser, no server needed
```

## Latency Optimization

### Profile Your Model

```python
endpoint = platform.serve(model, enable_profiling=True)

# Run inference
result = endpoint.predict(data)

# Get timing breakdown
profile = endpoint.get_profile()
print(profile)
# {
#   "preprocessing_ms": 2.5,
#   "inference_ms": 35.0,
#   "postprocessing_ms": 2.0,
#   "total_ms": 39.5
# }
```

### Target Optimization

If preprocessing is slow:
```python
# Batch preprocessing
endpoint = platform.serve(model, batch_size=64)
```

If inference is slow:
```python
# Enable TensorRT + quantization
optimizer = GPUOptimizer("A100")
optimizer.enable_tensorrt(fp16=True, int8=True)
```

If postprocessing is slow:
```python
# Move to client or async
# Configure endpoint for streaming
```

## Throughput Optimization

### Increase Replicas

```python
endpoint = platform.serve(model, replicas=4)  # 4 copies running
# Throughput ~4x
```

### Increase Batch Size

```python
endpoint = platform.serve(model, batch_size=256)
# Throughput ~8-10x (amortizes overhead)
```

### Enable Async

```python
# Process multiple requests concurrently
results = endpoint.predict_batch([data1, data2, data3, ...])
```

## Memory Optimization

### Reduce Model Size

```python
# Quantization
endpoint = platform.serve(model, quantization="int4")  # 90% smaller

# Distillation (if available)
small_model = platform.load("bert-distilled")
endpoint = platform.serve(small_model)
```

### Paged Attention (LLMs)

```python
from pystreamai.llm_optimization import LLMOptimizationEngine

engine = LLMOptimizationEngine(model)
engine.enable_paged_attention()  # Memory-efficient KV cache
```

## Cost Optimization

### Best Cost-Performance Ratio

```python
# L4 GPU is best value for most workloads
endpoint = platform.serve(model, gpu="L4")

# Batch to amortize compute cost
endpoint = platform.serve(model, batch_size=64)

# Quantize to reduce model loading time
endpoint = platform.serve(model, quantization="int8")
```

### Spot Instances (50-70% cheaper)

```python
endpoint = platform.serve(
    model,
    use_spot_instances=True  # Accept interruptions for lower cost
)
```

## Optimization Checklist

- [ ] Model is quantized (INT8 minimum)
- [ ] Batching configured for your use case
- [ ] Caching enabled (semantic, embedding, or result)
- [ ] GPU optimized (TensorRT, FP16/INT8)
- [ ] Latency profiled and optimized
- [ ] Throughput meets requirements
- [ ] Memory usage acceptable
- [ ] Cost per inference tracked
- [ ] Auto-scaling configured
- [ ] Monitoring and alerts set up

---

For more information, see:
- [Getting Started Guide](GETTING_STARTED.md)
- [API Reference](API_REFERENCE.md)
- [Deployment Guide](DEPLOYMENT.md)
