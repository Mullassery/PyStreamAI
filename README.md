# PyStreamAI

[![PyPI](https://img.shields.io/pypi/v/pystreamai)](https://pypi.org/project/pystreamai/)
[![CI](https://github.com/Mullassery/PyStreamAI/actions/workflows/tests.yml/badge.svg)](https://github.com/Mullassery/PyStreamAI/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)

An ML deployment toolkit: canary/A-B routing, cost tracking, request scheduling, hot model reload, and ONNX Runtime inference.

## What it is

PyStreamAI is a Python package for the deployment side of machine learning: routing
traffic between model versions (canary/A-B), tracking per-inference cost and budgets,
scheduling batched requests, hot-reloading models, and serving inference through ONNX
Runtime or any callable `predict()`/callable object you supply. It also ships a small
compiled Rust extension (PyO3/maturin) that exposes `Platform`, `GPUInfo`,
`CUDAProfiler`, and `MemoryPool` as Python classes. The pure-Python deployment layer
does real training/inference against whatever model you actually pass it; the Rust
extension is a separate, self-contained prototype (see "What it is not").

## Features / Highlights

- **Canary / A-B deployment routing** — `pystreamai.deployment.DeploymentManager`
  routes requests across deployed model versions by traffic percentage.
- **Cost tracking & budgets** — `pystreamai.cost_tracking.CostTracker` records
  per-inference metrics and enforces budget rules.
- **Request scheduling** — `pystreamai.request_scheduler.RequestScheduler`, a
  priority queue for scheduled inference requests.
- **Hot model reload** — swap the model behind a running endpoint without restarting.
- **ONNX Runtime inference** — `pystreamai.onnx_runtime.ONNXModelLoader` is a real
  wrapper around `onnxruntime.InferenceSession`; it loads and runs real `.onnx` files
  and auto-selects a CUDA/TensorRT/CPU provider.
- **HTTP serving** — `pystreamai.api.create_api_server(...)` and
  `pystreamai.serving.InferenceServer` expose `/predict`, `/health`, and `/stats`.
- **Model registry integrations** — `pystreamai.model_registry` connects to MLflow and
  the Hugging Face Hub when those optional libraries are installed.
- **Real training/serving abstraction** — `pystreamai.platform.Platform.train()` runs
  your training callable/script for real; `.serve()`/`.predict()` run real inference.

## Who it's for / Use cases

- You are rolling out a new model version and want **real canary/A-B routing**, not
  config that is never read.
- You want **local ONNX Runtime inference serving** with hot model reload and cost
  tracking attached.
- You need **deterministic bookkeeping/algorithms** (priority queues, budget math,
  LRU-style caching) around your own model, independent of the model itself.
- You are integrating MLflow / Hugging Face Hub into a small deployment loop.

## What it is not

This is an honest list of known limitations. Read it before relying on the project.

- **Early-stage / source-available history.** The repository briefly shipped compiled
  wheels with the source excluded; that was reversed, and the full Python and Rust
  source is back and under active cleanup. This is not a production-hardened system.
- **Cloud backends are not implemented.** `Platform(backend=...)` only implements
  `"local"`; `"aws"`/`"gcp"`/`"azure"` raise `NotImplementedError` immediately.
- **No authentication on the HTTP servers.** `pystreamai.api` / `pystreamai.serving`
  have no auth of any kind — do not expose them to an untrusted network without a
  reverse proxy/API gateway in front.
- **No CLI.** `pystreamai --version` does not exist.
- **The Rust extension's `Platform.train()`/`serve()`/`predict()` do nothing real.**
  They `format!()` their arguments into a string and return it. The `pystreamai._core`
  extension is a separate prototype, not wired into the Python layer, and its GPU/TensorRT
  speedup numbers are hardcoded domain constants, not measurements.
- **`pystreamai.gpu` is advisory only.** `GPUOptimizer` / `InferenceOptimizationPlan` /
  `MultiGPUInference` return hardcoded industry rule-of-thumb constants; only
  `detect_available_gpus()` does real work (queries `torch.cuda`).
- **`pystreamai.llm_optimization` and `pystreamai.edge_deployment` are interface
  sketches, not working code.** `SpeculativeDecoder` never calls the models passed to
  it, `LLMOptimizationEngine`'s cache-hit path returns `prompt + "cached_response"`,
  and `ModelQuantizer`/`EdgeModelCompiler` never touch the file they are given. Both
  modules are importable directly but not exported from `pystreamai/__init__.py`.
- **No verified benchmark results.** Earlier READMEs claimed "40-50x faster" / "5ms per
  request"; those came from a script that printed a fixed string and from hardcoded
  multipliers, and have been removed. `benchmarks/` contains real scripts you can run
  yourself, but no verified results ship in this repo.
- **Known issues** (see `ROADMAP_HONEST.md` for detail):
  - `HuggingFaceRegistry.upload_model()` logs a warning and returns `True` without
    actually uploading.
  - `observability`'s OpenTelemetry backend registers an `inference_speedup` gauge with
    no callback, so it never reports a value.
  - `tests/test_onnx_runtime.py` and `tests/test_platform_onnx_predict.py` are skipped in
    CI (`onnx` not installed) and currently fail if forced, due to an `onnx` /
    `onnxruntime` IR-version mismatch.
  - `docs/GETTING_STARTED.md`, `docs/API_REFERENCE.md`, and `docs/DEPLOYMENT.md` describe
    a larger planned API surface than the code implements; this README is the source of
    truth.

## Project status

**Early / alpha.** Current version **2.1.0** (`Development Status :: 3 - Alpha`). The
Python deployment layer is real and under active cleanup; the Rust extension is a
disconnected prototype. v2.0.0 introduced breaking changes to `TrainingJob.wait()`,
`Platform.serve()`, `InferenceServer.predict()`, and `create_api_server()` — see the
changelog and `ROADMAP_HONEST.md` before upgrading.

## Installation

```bash
pip install pystreamai
```

Optional extras (not pulled in by the base install):

```bash
pip install "pystreamai[serving]"       # FastAPI + uvicorn HTTP server (pystreamai.api)
pip install "pystreamai[onnx]"          # onnxruntime + numpy (pystreamai.onnx_runtime)
pip install "pystreamai[observability]" # prometheus-client backend
pip install "pystreamai[registry]"      # mlflow + huggingface_hub integrations
```

`pystreamai.api` and `pystreamai.onnx_runtime` import their dependencies
unconditionally — install the matching extra before importing those modules. Everything
else imports optional libraries lazily and degrades to a warning/no-op if they are
missing.

Requires Python 3.10+. A Rust toolchain is only needed to build the extension from
source (not to install a prebuilt wheel).

## Quick start

```python
from pystreamai.deployment import DeploymentManager

manager = DeploymentManager()
manager.deploy_model("sentiment", "v1", uri="s3://models/v1")
manager.deploy_model("sentiment", "v2", uri="s3://models/v2")
manager.start_canary_deployment("v2", traffic_percent=10.0)
routed = manager.route_request()  # -> DeploymentVersion for v1 or v2
```

## Usage

**Track cost per inference:**

```python
from pystreamai.cost_tracking import CostTracker, CostMetric

tracker = CostTracker()
cost = tracker.record_inference(CostMetric(
    request_id="r1", model_id="sentiment", gpu_type="A100",
    latency_ms=42.0, batch_size=1, input_tokens=50, output_tokens=20,
))
```

**Priority-queue requests:**

```python
from pystreamai.request_scheduler import RequestScheduler, ScheduledRequest, RequestPriority

scheduler = RequestScheduler()
scheduler.enqueue(ScheduledRequest(
    priority=RequestPriority.HIGH.value, request_id="r1", model_id="sentiment",
))
```

**Real ONNX inference** (`pip install "pystreamai[onnx]"` first):

```python
from pystreamai.onnx_runtime import ONNXModelLoader
import numpy as np

model = ONNXModelLoader("model.onnx").load()  # auto-selects CUDA/TensorRT/CPU provider
outputs = model.infer({"input": np.zeros((1, 4), dtype=np.float32)})
```

**HTTP server** (`pip install "pystreamai[serving]"` first — no auth, see limitations):

```python
from pystreamai.api import create_api_server

def my_model(data):
    return {"prediction": data}  # replace with your real model call

server = create_api_server(model_id="demo-model", model=my_model, port=8000)
server.run()  # POST /predict, GET /health, GET /stats
```

**Real training and serving via `Platform`:**

```python
from pystreamai.platform import Platform

platform = Platform()  # only backend="local" is implemented

# code is called for real: code(dataset) -> your trained model
job = platform.train(code=lambda dataset: {"weights": "trained on " + dataset}, dataset="data.csv")
model = job.wait()  # the real object your callable returned

# model needs a callable .predict(), or to be callable, or to be a real .onnx file
endpoint = platform.serve(model=lambda data: {"echo": data})
result = endpoint.predict({"x": 1})  # real call, real measured latency_ms
```

**Development:**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,serving,onnx]"
maturin develop --release   # builds the Rust extension into the venv
pytest
ruff check .
cargo test --release
```

## Documentation

- [Getting Started](docs/GETTING_STARTED.md)
- [API Reference](docs/API_REFERENCE.md)
- [Deployment](docs/DEPLOYMENT.md)
- [Optimization](docs/OPTIMIZATION.md)
- [Serving](docs/SERVING.md)
- [Integrations](docs/INTEGRATIONS.md)
- [Roadmap](docs/ROADMAP.md) / [Honest roadmap + technical debt](ROADMAP_HONEST.md)
- [Publishing](docs/PUBLISH.md)
- [Changelog](CHANGELOG.md)

## Queue status

Shipped, blocked/failed, and rejected work for this repository is documented in
[`QUEUE_STATUS.md`](QUEUE_STATUS.md).

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) first, and report security issues per
[SECURITY.md](SECURITY.md).

## Related projects

- [PyStreamXL](https://github.com/Mullassery/PyStreamXL) — stream large Excel workbooks with constant memory
- [PyStreamPDF](https://github.com/Mullassery/PyStreamPDF) — token-efficient PDF extraction for RAG
- [PyStreamMCP](https://github.com/Mullassery/PyStreamMCP) — context and token optimization for AI agents
- [PyStreamAI](https://github.com/Mullassery/PyStreamAI) — ML deployment toolkit (canary/A-B, cost tracking, ONNX)
- [PyTokenCalc](https://github.com/Mullassery/PyTokenCalc) — LLM token counting and cost estimation
- [Browse all projects](https://github.com/Mullassery?tab=repositories)

## License

This project is licensed under the [Apache License 2.0](LICENSE).
