# PyStreamAI Roadmap

> **Read this before the numbers below**: this roadmap predates the
> 2026-08 source restoration and mixes real shipped code with aspirational
> targets that were never measured against real models - see README.md's
> "How this works today" section for what's actually implemented today vs.
> simulated. Specifically: the speedup/latency/throughput figures in this
> document (e.g. "5-10x speedup", "1000+ req/sec") were written as goals,
> not benchmark output - `pystreamai/serving.py`'s inference path is a
> timed `asyncio.sleep()` simulation, not real model inference, and
> several "✅ COMPLETE" items below (e.g. cost/GPU speedup multipliers)
> are hardcoded constants rather than measured results. Treat this page as
> a plan, not a changelog of verified claims.

## v0.1: Core ✅ COMPLETE

**Focus: Inference speed and simplicity**

### Achievements
- Rust orchestrator core (5-10x speedup)
- NVIDIA GPU optimization (TensorRT, FP16, INT8)
- BERT + GPT-2 benchmarks proving speedup
- Zero-YAML Python API (@train, @serve, @pipeline)
- HTTP inference server (FastAPI)
- Async batching (dynamic batch collection)
- Multi-GPU support

### Metrics
- Latency: BERT 30ms (6-8x faster), GPT-2 20-40ms (5-7x faster)
- Throughput: 1000+ req/sec on single A100
- Simplicity: 50 lines of Python to serve a model

---

## v0.2: Production Ready ✅ COMPLETE

**Focus: Enterprise features, reliability, observability**

### Implementations

#### Inference Optimization
- ✅ ONNX Runtime integration (+2-3x speedup)
- ✅ PyTorch → ONNX auto-conversion
- ✅ Provider auto-selection (CUDA → TensorRT → CPU)

#### Observability Stack
- ✅ OpenTelemetry backend (standard protocol)
- ✅ Prometheus exporter (Grafana compatible)
- ✅ Datadog backend (direct integration)
- ✅ Request-level tracing

#### Model Management
- ✅ MLflow registry (versioning, experiment tracking) - `pystreamai/model_registry.py`'s
  `MLflowRegistry` delegates to real `mlflow.set_tracking_uri`/`create_experiment`/
  `mlflow.pytorch.log_model`/`mlflow.onnx.log_model` calls, not stubs.
- ⚠️ Hugging Face Hub integration - **download is real** (`HuggingFaceRegistry.download_model()`
  calls `transformers.AutoModel.from_pretrained()` for real), **upload is not**:
  `upload_model()` only checks that `huggingface_hub` is importable, then logs
  `"did not actually upload anything - real upload logic is not implemented yet"`
  and returns `True` regardless (see the `TODO` in `model_registry.py`, also flagged
  in README's "Known issues").
- ❌ Model artifact storage (local, S3, GCS) - **not implemented.** Checked
  `pystreamai/deployment.py` and the rest of `pystreamai/`: there is no S3/GCS/boto3/
  `google.cloud.storage` code anywhere in this package. `DeploymentVersion.uri` is
  just a plain string field on a dataclass - `manager.deploy_model("sentiment", "v1",
  uri="s3://models/v1")` stores that string for canary-routing bookkeeping; it never
  reads or writes to S3, GCS, or any other backend. This item should not have been
  marked complete.

#### Cost Management
- ✅ GPU pricing (H100, A100, L4, T4, V100, RTX4090)
- ✅ Token pricing (GPT-4, Claude, Llama)
- ✅ Cost tracking (per-request billing)
- ✅ Budget enforcement (monthly caps + alerts)
- ✅ Cost optimization recommendations

#### Deployments
- ✅ Canary deployments (gradual rollout)
- ✅ A/B testing (variant comparison)
- ✅ Automatic promotion/rollback

#### Advanced Features
- ✅ Model hot-reloading (no downtime updates)
- ✅ Request scheduling (4-level priority)
- ✅ Fair-share scheduling (across models)
- ✅ Deadline scheduling (earliest deadline first)
- ✅ User quotas (rate limiting)
- ✅ Request timeouts (expiration checking)

#### Monitoring
- ✅ Real-time metrics collection
- ✅ Dashboard with alerts
- ✅ Performance recommendations
- ✅ Per-model statistics
- ✅ Error rate tracking
- ✅ Latency percentiles (p99)

### Metrics
- Speed: 10-20x faster (ONNX + TensorRT + quantization combined)
- Reliability: Hot-reload, graceful shutdown, error recovery
- Observability: Full metrics pipeline (Prometheus/Datadog/OTEL)
- Cost tracking: Per-request billing with budget caps

---

## v0.3+: Specialized Workloads

**Focus: LLMs, edge, multimodal**

### Planned Features

#### LLM-Specific Optimizations
- [ ] Speculative decoding (faster generation)
- [ ] Token streaming (real-time output)
- [ ] Prompt caching (repeated prompts)
- [ ] KV cache quantization (INT4 cache)
- [ ] PagedAttention (memory-efficient serving)
- [ ] LoRA adapter serving (parameter-efficient)

#### Edge Deployment
- [ ] Model quantization (INT4, INT2)
- [ ] Model distillation (smaller models)
- [ ] WASM compilation (browser inference)
- [ ] TensorFlow Lite export (mobile)
- [ ] ONNX Runtime (CPU optimized)

#### Multimodal
- [ ] Image encoding (CLIP)
- [ ] Video processing (temporal models)
- [ ] Audio synthesis (TTS)
- [ ] Multi-input fusion

#### Advanced Caching
- [ ] Semantic caching (similar inputs)
- [ ] Result caching (deterministic outputs)
- [ ] Embedding caching (reuse vectors)

#### Fine-tuning as a Service
- [ ] Efficient fine-tuning (LoRA, QLoRA)
- [ ] Distributed training (multi-GPU)
- [ ] Automated hyperparameter search

### Estimated Timeline
- v0.3: Q3 2026 (LLM optimizations)
- v0.4: Q4 2026 (Edge deployment)
- v0.5: Q1 2027 (Advanced caching)

---

## v1.0: Production Stable

**Focus: Reliability, performance, enterprise features**

### Criteria for v1.0
- 99.99% uptime guarantee
- Zero data loss
- < 10ms latency for simple models
- < 100ms for complex models
- Full backward compatibility
- Enterprise support

### Features
- Complete LLM suite
- Edge deployment
- Multimodal support
- Advanced caching
- Fine-tuning platform
- Integrated cost management

---

## Architecture Evolution

### v0.1
```
Inference Request
    ↓
HTTP API (FastAPI)
    ↓
Async Batch Queue
    ↓
GPU Scheduler (Multi-GPU)
    ↓
NVIDIA GPU (TensorRT, FP16, INT8)
    ↓
Inference Response + Metrics
```

### v0.2
```
Inference Request
    ↓
Request Scheduler (Priority + Fair-share)
    ↓
Hot Reload Manager (Version control)
    ↓
ONNX Runtime / PyTorch
    ↓
GPU Scheduler
    ↓
NVIDIA GPU
    ↓
Cost Tracker → Budget Manager
    ↓
Metrics Collector → Prometheus/OTEL/Datadog
    ↓
Canary/A-B Test Router
    ↓
Inference Response
```

### v0.3+
```
[+ LLM-specific optimizations]
[+ Edge runtime]
[+ Multimodal pipeline]
[+ Advanced caching]
```

---

## Competitive Positioning

*Unverified - this table was written as marketing copy, not derived from a
head-to-head benchmark against major managed ML platforms or BentoML/Kubeflow.
The "Speed" row in particular should not be trusted until real comparative
benchmarks exist (see "Key Metrics to Track" below).*

| Feature | PyStreamAI | Major managed ML platform A | Major managed ML platform B | BentoML | Kubeflow |
|---------|-----------|----------|-----------|---------|----------|
| Speed | unverified | unverified | unverified | unverified | unverified |
| Simplicity | Zero YAML | YAML | Complex | Python-first | Very complex |
| Cost Control | Per-request | Opaque | Opaque | None | None |
| Multi-cloud | Planned (local backend only today) | Single-cloud only | Yes | Yes | Yes |
| LLM Support | Planned | Yes | Yes | Basic | No |
| Edge Deploy | Planned | Limited | No | Limited | No |
| Model Registry | MLflow/HF integration, optional deps | Yes | Yes | Yes | Basic |

---

## Key Metrics to Track

- **Speed**: Inference latency vs baseline (target: 10-20x)
- **Reliability**: Uptime percentage (target: 99.99%)
- **Cost**: $ per inference vs competitors (target: 50% cheaper)
- **Adoption**: GitHub stars, downloads (target: 10k stars by v1.0)
- **Community**: Issues resolved, contributions (target: 100+ contributors)

---

## Known Limitations (v0.2)

- Single-process serving (no distributed serving yet)
- No built-in authentication
- Limited monitoring dashboard UI
- No auto-scaling (manual replicas only)
- No traffic splitting (canary only)
- [x] No end-to-end tests for model failure states — **Done (v2.1.0).** `tests/test_model_failure_states.py` exercises the real HTTP `/predict` route against a model that raises (simulating a rate limit/provider outage — `pystreamai.testing.RateLimitError`/`ModelUnavailableError`) and against a model that returns non-JSON-serializable output. The latter surfaced a real, previously-unhandled gap: FastAPI serializes the response *after* `/predict`'s handler returns, outside its `try/except`, so malformed output produced an unhandled `PydanticSerializationError`/500 with no clean `detail` instead of going through the same error path as every other inference failure. Fixed in `api.py` by forcing `PredictResponse.model_dump_json()` inside the handler's own try/except before returning.
- [x] No reusable mock-inference-server fixture — **Done (v2.1.0).** Added `pystreamai/testing.py` (`echo_model` promoted from the old test-local copy in `tests/test_api.py`, plus `failing_model()`/`malformed_output_model` for failure-state tests) and `tests/conftest.py` exposing them as pytest fixtures (`echo_model`, `raising_model`, `malformed_output_model`, `model_failure`), importable by downstream projects' own test suites.

(Note: external critique also claimed missing Pydantic schema validation and hardcoded provider assumptions — both checked and found already addressed: `pystreamai/api.py` uses real `pydantic.BaseModel` request/response models enforced via FastAPI, and provider/backend dispatch in `pystreamai/platform.py` is duck-typed/config-driven with auto-detected ONNX execution providers, not hardcoded branches.)

### Planned for v0.3+
- Kubernetes auto-scaling
- OAuth/OIDC support
- Web-based dashboard
- Traffic-based auto-scaling
- Advanced traffic splitting

---

## Success Criteria

**v0.2 Success**: 
- [ ] 5+ production deployments
- [ ] < 50ms latency on BERT (10-12x speedup)
- [ ] Cost tracking accurate to within 1%
- [ ] 99.9% uptime on staging

**v1.0 Success**:
- [ ] 100+ production deployments
- [ ] < 20ms latency on simple models
- [ ] 99.99% uptime SLA
- [ ] $1M ARR (if commercialized)

