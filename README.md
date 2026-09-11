# PyStreamAI

An ML deployment toolkit: a Python package (canary/A-B deployment
routing, cost tracking, request scheduling, hot model reload, ONNX
Runtime inference) plus a small compiled Rust extension (PyO3), built
with maturin.

[![Tests](https://img.shields.io/github/actions/workflow/status/Mullassery/PyStreamAI/tests.yml?label=tests)](https://github.com/Mullassery/PyStreamAI/actions)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)

---

## Status: early / v2.1.0, source-available

This project's public repository briefly shipped compiled wheels with the
source code deliberately excluded ("kept locally"). That was reversed in
the 2026-08 restoration: the full Python and Rust source is back in this
repo and under active cleanup. If you're evaluating this for anything
beyond experimentation, read the **"How this works today"** and
**"Security"** sections below before you rely on it.

## Install

```bash
pip install pystreamai
```

Optional extras (installed separately - not pulled in by the base install):

```bash
pip install "pystreamai[serving]"       # FastAPI + uvicorn HTTP server (pystreamai.api)
pip install "pystreamai[onnx]"          # onnxruntime + numpy (pystreamai.onnx_runtime)
pip install "pystreamai[observability]" # prometheus-client backend
pip install "pystreamai[registry]"      # mlflow + huggingface_hub integrations
```

`pystreamai.api` and `pystreamai.onnx_runtime` import their dependencies
unconditionally - install the matching extra before importing those
specific modules, or you'll get an `ImportError`. Everything else in the
package only imports optional libraries inside `try/except ImportError`
and degrades gracefully (logs a warning, no-ops) if they're missing.

Verify:
```bash
python3 -c "import pystreamai; print(pystreamai.__version__)"
```

There is no CLI (`pystreamai --version` etc. does not exist yet).

## How this works today

PyStreamAI is two mostly-independent layers restored from an earlier,
pre-1.0 development snapshot:

**1. A pure-Python layer** (`pystreamai.platform`, `pystreamai.decorators`,
`pystreamai.serving`, `pystreamai.api`, and most of the package). As of
this pass, this layer runs **real training and real inference against
whatever model you actually give it** — there is no simulated/fabricated
response path left anywhere in it:

- `Platform.train(code, dataset)` runs `code` for real: if `code` is a
  callable, it's called as `code(dataset)` and its real return value is
  the trained model (`job.wait()` returns it, or raises the real
  exception if `code` raised). If `code` is a path to a real script, it's
  run as `python <code> --dataset <dataset> --output <path>` in a real
  subprocess, and `job.wait()` returns the real artifact path the script
  wrote — or raises if it exited non-zero or didn't write one. Anything
  else (a non-existent path, a non-callable) raises `TypeError`
  immediately, not a fake successful job.
- `Platform.serve(model)` / `Endpoint.predict()` run real inference: a
  real `.onnx` file (or `pystreamai.onnx_runtime.ONNXModelLoader`) gets
  real onnxruntime inference; an object with a callable `.predict()`
  method or a plain callable gets called for real. Anything else raises
  `TypeError` at `serve()` time — there is no "simulated" fallback
  response and no `latency_ms=42.5` constant left in the code.
- `pystreamai.serving.InferenceServer` batches and times requests for
  real *and* runs real per-request inference once you call
  `await server.load_model(model_id, model)` — `predict()` raises
  `RuntimeError` if you haven't. `pystreamai.api.create_api_server(model_id, model, ...)`
  requires the real model up front for the same reason. Fixed in this
  pass: a real concurrency bug where multiple requests batched together
  via `asyncio.gather` would deadlock (only whichever request happened to
  perform the batch flush ever got a result; the rest polled an
  already-empty queue forever) — every request in a shared batch now
  correctly gets its own real result and the real batch size it was part
  of.
- `gpu_id`/`cost_usd` on inference responses are `None`/`0.0` always —
  there's no real GPU scheduler or pricing table backing either one, so
  they no longer report a fabricated `0` or a fake round-robin GPU index.
- `Platform(backend=...)` only implements `"local"` — `"aws"`/`"gcp"`/
  `"azure"` raise `NotImplementedError` immediately instead of being
  silently accepted and behaving identically to `"local"` with zero
  actual cloud provisioning behind them.

**2. A compiled Rust extension** (`pystreamai._core`, built from `src/*.rs`
via PyO3/maturin) exposing `Platform`, `GPUInfo`, `CUDAProfiler`, and
`MemoryPool` as Python classes. This is a **separate, self-contained
prototype** - the Python `pystreamai.platform.Platform` does not call into
it, and vice versa. Its GPU/TensorRT-speedup numbers (e.g. "FP16 is 1.5x
faster") are hardcoded domain-knowledge constants, not measurements taken
on your hardware. Not touched in this pass.

**What is real and does real work:**
- `pystreamai.platform.Platform`/`Endpoint`, `pystreamai.decorators`,
  `pystreamai.serving`, `pystreamai.api` — see above; real training/
  inference against whatever model you provide.
- `pystreamai.onnx_runtime.ONNXModelLoader` - a genuine wrapper around
  `onnxruntime.InferenceSession`; it loads and runs real `.onnx` models
  (see `tests/test_onnx_runtime.py`, which builds and runs an actual ONNX
  graph, not a mock).
- `pystreamai.request_scheduler`, `pystreamai.deployment`,
  `pystreamai.cost_tracking`, `pystreamai.hot_reload`,
  `pystreamai.advanced_caching`, `pystreamai.dashboard` - real,
  deterministic bookkeeping/algorithms (priority queues, canary traffic
  routing, budget tracking, LRU-ish caches, hit-rate math). These don't
  require a real model to be useful, but they also don't call one.
- `pystreamai.model_registry` - real integration code for MLflow /
  Hugging Face Hub, active only if those optional libraries are installed
  (degrades to a documented no-op otherwise).

**Explicitly not real, and clearly labeled as such in code (not silently
faked):** `pystreamai.gpu` (`GPUOptimizer`, `InferenceOptimizationPlan`,
`MultiGPUInference`) is a generic advisory calculator — every speedup/
batch-size/VRAM number it produces is a hardcoded industry rule-of-thumb
constant, not something measured against your actual model or hardware.
The one real function in that module is `detect_available_gpus()`, which
genuinely queries `torch.cuda` if PyTorch+CUDA are available. There's no
GPU hardware in this project's development/CI environment to build real
per-model GPU benchmarking against, so this remains advisory rather than
measured — flagged here rather than presented as a benchmark.

If you want to see exactly which claims a given module can back up, read
its module docstring and the corresponding test file - every test file
under `tests/` explains what it actually exercises.

## Performance claims

Earlier versions of this README claimed "40-50x faster inference" and
"5ms per request." Digging into the restored source: those numbers came
from a benchmark script (`benchmarks/bench_oss_comparison.py`) that
printed a fixed `"~5ms (45x faster)"` string regardless of what it
measured, and from hardcoded speedup multipliers used throughout the GPU
optimization calculators (`1.5x` for FP16, `2.5x` for INT8, etc. - domain
assumptions, not benchmark output). Those claims have been removed rather
than carried forward unverified.

`benchmarks/` contains real scripts that load real BERT/GPT-2 models via
`transformers`/`torch` and time real inference
(`pip install -r requirements-benchmark.txt` first) - if you run them and
get real numbers, they'll be more trustworthy than anything stated here.
This repository does not currently ship verified benchmark results.

## Security

**`pystreamai.api` / `pystreamai.serving` have no authentication.** Any
process that can reach the HTTP server can call `/predict`, `/stats`, and
`/shutdown`. This is fine for local development or an already-isolated
network; do not expose it directly to an untrusted network without
putting your own auth (reverse proxy, API gateway, etc.) in front of it.
Built-in auth is not implemented yet.

**Model loading**: `pystreamai.onnx_runtime` loads models via
`onnxruntime.InferenceSession`, which parses ONNX's own format rather
than Python `pickle` - it does not have pickle's arbitrary-code-execution
properties. `pystreamai.model_registry.MLflowRegistry.load_model()`
delegates to `mlflow.pytorch.load_model()`, which - like plain
`torch.load()` - **can** execute arbitrary code if pointed at an
untrusted model artifact; this only runs if you have `mlflow` installed
and explicitly call it. Only load models (via any path) from sources you
trust.

## Real example (matches restored code, not aspirational API)

```python
from pystreamai.deployment import DeploymentManager
from pystreamai.cost_tracking import CostTracker, CostMetric
from pystreamai.request_scheduler import RequestScheduler, ScheduledRequest, RequestPriority

# Canary-route traffic between two deployed model versions
manager = DeploymentManager()
manager.deploy_model("sentiment", "v1", uri="s3://models/v1")
manager.deploy_model("sentiment", "v2", uri="s3://models/v2")
manager.start_canary_deployment("v2", traffic_percent=10.0)
routed = manager.route_request()  # -> DeploymentVersion for v1 or v2

# Track cost per inference
tracker = CostTracker()
cost = tracker.record_inference(CostMetric(
    request_id="r1", model_id="sentiment", gpu_type="A100",
    latency_ms=42.0, batch_size=1, input_tokens=50, output_tokens=20,
))

# Priority-queue requests
scheduler = RequestScheduler()
scheduler.enqueue(ScheduledRequest(priority=RequestPriority.HIGH.value, request_id="r1", model_id="sentiment"))
```

Real ONNX inference (`pip install "pystreamai[onnx]"` first):

```python
from pystreamai.onnx_runtime import ONNXModelLoader
import numpy as np

model = ONNXModelLoader("model.onnx").load()  # auto-selects CUDA/TensorRT/CPU provider
outputs = model.infer({"input": np.zeros((1, 4), dtype=np.float32)})
```

HTTP server (`pip install "pystreamai[serving]"` first - see Security above):

```python
from pystreamai.api import create_api_server

def my_model(data):
    return {"prediction": data}  # replace with your real model call

server = create_api_server(model_id="demo-model", model=my_model, port=8000)
server.run()  # POST /predict, GET /health, GET /stats - no auth
```

Real training and serving via `Platform` (`pystreamai.platform`):

```python
from pystreamai.platform import Platform

platform = Platform()  # only backend="local" is implemented

# code is called for real: code(dataset) -> your trained model
job = platform.train(code=lambda dataset: {"weights": "trained on " + dataset}, dataset="data.csv")
model = job.wait()  # the real object your callable returned

# model needs a callable .predict() or to be callable itself (or a real .onnx file)
endpoint = platform.serve(model=lambda data: {"echo": data})
result = endpoint.predict({"x": 1})  # real call, real measured latency_ms
```

## What's restored, what isn't

This repository's `pystreamai/` and `src/` were restored from git history
(commit `0ce6efb`) after a later commit stripped them from the public
repo. Everything under `pystreamai/*.py`, `src/*.rs`, `benchmarks/`, and
`examples/serve_model.py` / `examples/start_http_server.py` is real,
restored, and covered by `tests/`.

One thing was **not** restored, because it never existed: a prior commit
(`98f9c6d`, "Add Phases 5-10") added ~600 lines of documentation, an
example, and a test file describing `rollback_strategies`,
`cost_performance`, `framework_integration`, `advanced_analytics`,
`multi_model_orchestration`, and `auto_versioning` modules - but shipped
zero implementation for any of them (the commit touched only
README/docs/tests). Those modules do not exist anywhere in this project's
git history. `tests/test_phases_5_10.py`, `examples/auto_versioning_example.py`,
and `docs/PHASES_5_10_GUIDE.md` (which tested/documented that
non-existent code) have been removed rather than fabricated after the
fact. If you need similar functionality today, the closest real
equivalents are `pystreamai.deployment` (canary/A-B rollback routing) and
`pystreamai.cost_tracking` (cost tracking/budgets) - see the example
above.

## Rust extension

`src/*.rs` builds a PyO3 extension (`pystreamai._core`) via maturin,
targeting PyO3 0.23 (upgraded from 0.21 during restoration; the module is
built as a mixed Python/Rust layout - `python-source` = repo root,
`module-name = "pystreamai._core"`). `cargo build`/`cargo test` need the
`extension-module` PyO3 feature, which does not link against libpython -
use `maturin build`/`maturin develop`/`cargo test` (not `cargo build`
directly) to build a runnable extension. `cargo test` runs 16 unit tests
covering the non-PyO3-exposed logic in `scheduler.rs`, `storage.rs`,
`executor.rs`, and `gpu.rs`.

## Known issues

- `pystreamai.model_registry.HuggingFaceRegistry.upload_model()` checks that
  `huggingface_hub` is importable and reachable but does not perform a real
  upload yet - it logs a warning and returns `True` regardless (see the
  `TODO` in `pystreamai/model_registry.py`).
- `pystreamai.observability`'s OpenTelemetry backend creates an
  `inference_speedup` observable gauge with no callback wired up, so it
  never reports a value (see the `TODO` in `pystreamai/observability.py`).
- `docs/GETTING_STARTED.md`, `docs/API_REFERENCE.md`, and
  `docs/DEPLOYMENT.md` describe cloud backend config options that are not
  implemented - only `backend="local"` is real. Those docs have been
  reworded to say so explicitly and to drop vendor-specific naming.
  `Platform(backend=...)` now raises `NotImplementedError` immediately for
  anything other than `"local"`, instead of silently accepting e.g.
  `backend="aws"` and storing it with zero effect on behavior (the
  previous behavior — confirmed by grep, `self.backend` was never read
  anywhere after being set).
- `pystreamai.gpu` (`GPUOptimizer`/`InferenceOptimizationPlan`/
  `MultiGPUInference`) remains a generic advisory calculator, not real
  per-model/per-hardware benchmarking — see "How this works today" above.
  There's no GPU hardware in this project's environment to build and
  verify real GPU benchmarking against.
- `pystreamai._core` (the compiled Rust extension) is a separate,
  disconnected prototype — untouched by the training/serving fixes in
  this pass. Its GPU/TensorRT numbers are the same kind of hardcoded
  constant as `pystreamai.gpu`, independently.
- No open GitHub issues as of this pass (2026-08-18).
- **Breaking changes in v2.0.0** (bumped from 1.1.0 for exactly this reason):
  `TrainingJob.wait()` now returns the real trained model/artifact instead
  of always a `Path`; `Platform.serve()`/`Endpoint(...)` now raise
  `TypeError` for a model with no real way to run inference instead of
  returning it as `simulated: True`; `InferenceServer.predict()`/
  `pystreamai.api`'s `/predict` now raise/500 if you haven't called
  `load_model()` first; `create_api_server()`/`APIServer.__init__()`
  gained a required `model` parameter. Anything calling this package's
  train/serve/predict path will need updating.
- Fixed in this pass: trademarked cloud-vendor names removed from `docs/`,
  the GitHub repository's "About" description (previously claimed
  "40-50x faster... zero vendor lock-in" with no benchmark backing it),
  and every code example in this README verified against the actual
  source (`pystreamai/deployment.py`, `cost_tracking.py`,
  `request_scheduler.py`, `onnx_runtime.py`, `api.py`, `platform.py`,
  `serving.py`).

## Development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,serving,onnx]"
maturin develop --release   # builds the Rust extension into the venv
pytest                      # 146 tests (1 needs the compiled Rust extension, `maturin develop` first)
ruff check .
cargo test --release        # 16 Rust unit tests
```

## Documentation

- [Getting Started](docs/GETTING_STARTED.md)
- [API Reference](docs/API_REFERENCE.md)
- [Deployment](docs/DEPLOYMENT.md)
- [Optimization](docs/OPTIMIZATION.md)
- [Serving](docs/SERVING.md)
- [Integrations](docs/INTEGRATIONS.md)
- [Roadmap](docs/ROADMAP.md)
- [Publishing](docs/PUBLISH.md)

Several of the above predate this restoration and describe a larger
planned API surface (e.g. `Platform.load()`, `pip install pystreamai[gpu]`)
that doesn't match the code in this repository yet - this README is the
up-to-date source of truth; treat mismatches in the linked docs as
open cleanup items, not as features you can rely on.

## System Requirements

- Python 3.10+
- Rust toolchain (only if building the extension from source; not needed
  to `pip install` a prebuilt wheel)
- Linux or macOS (Windows via WSL2 - untested in this restoration pass)

## License

This project is licensed under the [Apache License 2.0](LICENSE).
