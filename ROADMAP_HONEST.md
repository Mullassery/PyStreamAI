# Honest Roadmap

This file states plainly what works, what's fake, and what hasn't been
started - no "planned"/"may be added" hedging for things that simply
don't exist or don't work. `docs/ROADMAP.md` is the older, aspirational
version of this document (kept for history, annotated with corrections).
This file is the current source of truth for status; the README's "How
this works today" and "Known issues" sections are the source of truth for
narrative detail - this file exists to make status scannable and to hold
the technical-debt list.

Verification method for every claim below: `pytest` (169 passed / 0
skipped, run 2026-09-21 with Python 3.11.16, `pip install -e
".[dev,serving,onnx]"` - up from 161 passed/2 skipped on 2026-09-19 now
that the ONNX and httpx test-dependency gaps below are fixed), `cargo
test --release` (16 passed), `ruff check .` (clean), and direct code
reading (file:line references given). Nothing below is asserted without
one of those.

## 1. Shipped and verified (tested, does what it says)

| Component | What it does | Verified by |
|---|---|---|
| `pystreamai.platform.Platform`/`Endpoint` | Real training (`code(dataset)` or subprocess script) and real inference (ONNX/`.predict()`/callable) | `tests/test_platform.py`, `tests/test_platform_onnx_predict.py` (partially - see §3), `tests/test_model_failure_states.py` |
| `pystreamai.decorators` (`@train`/`@serve`/`@pipeline`) | Real wrappers around `Platform` | `tests/test_api.py` |
| `pystreamai.serving.InferenceServer` / `pystreamai.api` | Real async batching + real per-request inference over HTTP (FastAPI); no auth (see below) | `tests/test_serving.py`, `tests/test_api.py` |
| `pystreamai.request_scheduler` | Real priority queue / fair-share / deadline scheduling | `tests/test_request_scheduler.py` |
| `pystreamai.deployment` | Real canary/A-B traffic routing (`DeploymentVersion.uri` is bookkeeping only - no S3/GCS I/O, see §3) | `tests/test_deployment.py` |
| `pystreamai.cost_tracking` | Real budget/cost bookkeeping math | `tests/test_cost_tracking.py` |
| `pystreamai.hot_reload` | Real model version swap-out logic | `tests/test_hot_reload.py` |
| `pystreamai.advanced_caching`, `pystreamai.dashboard` | Real LRU-ish caches, real metrics aggregation | `tests/test_misc_modules.py` |
| `pystreamai.model_registry.MLflowRegistry` | Real `mlflow.*` calls (not stubs) | code inspection; no dedicated test found running against a real MLflow server (see §4 tech debt) |
| `pystreamai.model_registry.HuggingFaceRegistry.download_model()` | Real `transformers.AutoModel.from_pretrained()` call | code inspection |
| Rust `scheduler.rs`/`executor.rs`/`storage.rs` | Real in-memory job/worker/model bookkeeping | `cargo test --release`, 3+3+3 passing unit tests |
| Rust `gpu.rs` (`GPUInfo`, `CUDAProfiler` logic) | Real lookup-table-driven capability checks (not measured hardware numbers - see §2) | `cargo test --release` |

## 2. Real code, but with real caveats (don't treat as unconditionally trustworthy)

- **`pystreamai.gpu`** (`GPUOptimizer`, `InferenceOptimizationPlan`,
  `MultiGPUInference`): every speedup/batch-size/VRAM number is a
  hardcoded industry rule-of-thumb constant, not a measurement. Only
  `detect_available_gpus()` genuinely queries `torch.cuda`. No GPU
  hardware exists in this project's dev/CI environment to build real
  benchmarking against.
- **`pystreamai.onnx_runtime`**: the wrapper code is real (genuine
  `onnxruntime.InferenceSession` usage). Its test suite
  (`tests/test_onnx_runtime.py`, `tests/test_platform_onnx_predict.py`)
  was **silently skipped in CI** because `pyproject.toml`'s `[onnx]`
  extra didn't install the `onnx` package the tests need to build a
  graph. **Fixed 2026-09-21**: added `onnx>=1.15.0,<1.23.0` to the
  `[onnx]` extra. The upper bound matters - onnx 1.23.0 (latest as of
  this pass) is the first release whose default IR version (14) exceeds
  what the currently-resolved `onnxruntime` (1.30.0, satisfying
  `>=1.15.0`) supports (max IR version 13, confirmed by direct test with
  a hand-built graph). onnx 1.15.0-1.22.0 all write IR versions 9-13,
  which load fine. All 8 tests now run for real and pass.
- **`pystreamai.model_registry.HuggingFaceRegistry.upload_model()`**:
  checks `huggingface_hub` is importable/reachable, then logs a warning
  and returns `True` regardless - no real upload (`model_registry.py:213`,
  `TODO` in source).
- **`pystreamai.observability.OpenTelemetryBackend`**: creates an
  `inference_speedup` observable gauge with no callback wired up - never
  reports a value (`observability.py:86`, `TODO` in source).
- **HTTP server has no authentication.** `pystreamai.api`/`pystreamai.serving`
  accept unauthenticated requests to `/predict`, `/stats`, `/shutdown`.
  Fine for local/trusted-network use; do not expose directly.
- **`tests/test_api.py` and `tests/test_model_failure_states.py` (15
  tests) were also silently skipped in CI** - found while investigating
  the ONNX gap above, same shape of bug: `pytest.importorskip("httpx",
  ...)` guards both files (needed by `fastapi.testclient.TestClient`)
  and `httpx` wasn't in any extra. **Fixed 2026-09-21**: added
  `httpx>=0.24.0` to `dev`. All 15 tests now run for real and pass.

## 3. Fake / placeholder code - needs real implementation or deletion

These are not "planned features" - they are checked-in code that returns
fabricated results regardless of input, and neither module is imported by
`pystreamai/__init__.py` or mentioned in this README's main feature list.
They were found during this audit pass and were **not previously
disclosed** anywhere except a one-line hint in a test-file docstring
(`tests/test_misc_modules.py:1-7`).

- **`pystreamai/llm_optimization.py`** (306 lines):
  - `SpeculativeDecoder.generate()` never calls `main_model`/`draft_model`.
    `_draft_tokens()` returns `"draft " * num_tokens` unconditionally
    (`llm_optimization.py:76-79`); `_verify_tokens()` just echoes it back
    (`llm_optimization.py:81-84`). The reported "acceptance rate" is
    therefore meaningless.
  - `LLMOptimizationEngine.generate()`'s cache-hit path returns
    `prompt + "cached_response"` literally (`llm_optimization.py:265`).
  - `PromptCache` and `PagedAttention` (used by the above) are real,
    deterministic bookkeeping and are genuinely tested
    (`tests/test_misc_modules.py`) - it's specifically the "LLM
    optimization" behavior wrapping them that's fake.
- **`pystreamai/edge_deployment.py`** (322 lines):
  - `ModelQuantizer.quantize_int8/int4/fp16()` never open `model_path` -
    return hardcoded dicts (`"original_size_mb": 100`, fixed compression
    ratios) regardless of the real file (`edge_deployment.py:108-142`).
  - `EdgeModelCompiler.compile_tflite/onnx_mobile/wasm/core_ml/tflite_gpu()`
    don't invoke any real compiler/converter - each just string-formats
    an output filename and returns it; no file is ever written
    (`edge_deployment.py:151-186`).
  - `docs/OPTIMIZATION.md`'s "Edge Deployment" section presents
    `EdgeDeploymentPipeline.prepare_model()`'s output as if real
    quantization/compilation happened ("Model is now: Quantized to INT8
    (75% smaller), Compiled to Core ML...") - it did not.
- **Rust `Platform.train()`/`serve()`/`predict()`** (`src/lib.rs:48-58`,
  exposed as `pystreamai._core.Platform`): each is a one-line
  `format!()` returning a string describing what it "would" do. No
  training, serving, or inference happens. The `Scheduler`/`Executor`/
  `Storage`/`InferenceOptimizer` fields constructed in `Platform::new()`
  are never read by any method - confirmed by `cargo build`'s own
  dead-code lint (`fields "scheduler", "executor", "storage", and
  "inference_optimizer" are never read`, `src/lib.rs:21`).
- **Rust dead code, never wired to anything (confirmed via `cargo build
  --release`, 40 dead-code warnings total)**:
  - `src/backend.rs` (166 lines): `Backend` trait, `LocalBackend`,
    `KubernetesBackend`, `BackendType` enum - never constructed, never
    exposed via `#[pymodule]`.
  - `src/streaming.rs` (263 lines): `InferenceProcessor`,
    `PostprocessingStage`, `StreamConnector` trait, `MQTTConnector`,
    `KafkaConnector` - never constructed, never exposed.
  - `src/inference.rs`: `TensorRTOptimizer`, `TensorBuffer` - never
    constructed (only `InferenceOptimizer`/`ModelOptimizationPlan` from
    this file are actually used).
  - `src/memory_manager.rs:105`: `GPUMemoryPool` - a second, unused
    struct in the same file as the real, PyO3-exposed `MemoryPool`
    (`memory_manager.rs:7`); don't confuse the two.

**Recommendation**: either implement these for real (speculative decoding
needs an actual draft+verify model loop; edge compilation needs real
TFLite/CoreML/wasm toolchain calls; the Rust `Backend`/`streaming`
modules need to be wired into `lib.rs`'s `#[pymodule]` or removed) or
delete them. Leaving them in place, unlabeled, contradicts this
project's own "no fabricated response path" claims elsewhere in the
README.

## 4. Not started (no code exists - not "planned", just absent)

- Cloud backends (AWS/GCP/Azure) for `Platform(backend=...)` - only
  `"local"` exists; others raise `NotImplementedError` immediately.
- Model artifact storage (S3/GCS) - `grep` confirms zero boto3/
  `google-cloud-storage` code anywhere in `pystreamai/`.
- Real GPU/TensorRT benchmarking (only the advisory calculator in §2
  exists).
- Authentication/authorization for the HTTP server.
- Kubernetes auto-scaling, OAuth/OIDC, a web-based dashboard UI,
  traffic-based auto-scaling (all listed as "Planned for v0.3+" in
  `docs/ROADMAP.md` - still true, zero code for any of them).
- A CLI (`pystreamai --version` etc. does not exist).
- Prebuilt wheels for macOS/Windows: `.github/workflows/build-wheels.yml`
  only builds on `ubuntu-latest`. A `pip install pystreamai` on macOS/
  Windows without a matching prebuilt wheel falls back to building the
  sdist from source, which needs a Rust toolchain - this isn't currently
  called out anywhere in the install instructions.

## Technical debt (concrete, file:line, for a dedicated follow-up session)

1. ~~**ONNX test suite is broken, not just optional**~~ **Fixed
   2026-09-21**: `pyproject.toml`'s `[onnx]` extra now pins
   `onnx>=1.15.0,<1.23.0` (onnx 1.23.0+ writes IR version 14, which
   onnxruntime 1.30.0 rejects - max supported IR version 13). All 8
   ONNX tests run and pass; `.github/workflows/tests.yml` already
   installs `[dev,serving,onnx]` so no workflow change was needed. Note
   for a future session: if `onnxruntime` is ever bumped past 1.30.0
   for other reasons, re-check whether the onnx upper bound can be
   relaxed (a newer onnxruntime may support IR 14+).
2. **`pystreamai/llm_optimization.py` and `pystreamai/edge_deployment.py`
   are fake** (see §3 for exact lines) - 628 lines of code with zero
   real behavior behind their public methods, and zero test coverage of
   those specific methods (only their supporting bookkeeping classes are
   tested).
3. **Rust `Platform` (`src/lib.rs`) does nothing** and **`src/backend.rs`
   (166 lines) + `src/streaming.rs` (263 lines) are entirely dead code**
   - confirmed by 40 `cargo build --release` dead-code warnings. This is
   roughly a third of `src/*.rs` by line count doing nothing.
4. **`pystreamai/observability.py:86`** - OpenTelemetry gauge callback
   not wired, never reports a value (`TODO` in source).
5. **`pystreamai/model_registry.py:213`** - `HuggingFaceRegistry.upload_model()`
   is a no-op that returns `True` (`TODO` in source).
6. **No test exercises `MLflowRegistry` against a real (or even
   in-process fake) MLflow server** - it's marked "real" based on code
   reading (genuine `mlflow.*` calls), not a passing integration test.
   Worth adding before trusting it in production.
7. **`docs/GETTING_STARTED.md`, `docs/API_REFERENCE.md`,
   `docs/DEPLOYMENT.md`, `docs/OPTIMIZATION.md`, `docs/SERVING.md`,
   `docs/INTEGRATIONS.md`** all carry a "this predates the restoration"
   banner instead of being rewritten to match current reality. That's a
   pragmatic stopgap, not a fix - a dedicated pass to either rewrite or
   delete/merge the stale portions of each into a single accurate doc
   would remove an entire class of "which claim do I believe" confusion
   for new users.
8. **Release process only produces Linux wheels** (see §4) - no macOS/
   Windows wheel build, and `build-wheels.yml`'s `maturin build --release`
   on a bare `ubuntu-latest` runner has not been verified in this pass to
   produce a `manylinux`-policy-compliant wheel (maturin auto-detects and
   labels this, but nobody has confirmed the resulting wheel installs
   cleanly on an arbitrary Linux distro, only that it builds).
9. **`Cargo.lock` was gitignored** until this pass despite this crate
   building a `cdylib` (effectively an application artifact, not a
   library meant to be depended on by other crates) - fixed here, but the
   lockfile just added has no prior commit history, so there's no
   guarantee yet that pinned versions are the ones actually used to build
   past releases (`v0.1.0`-`v2.1.0` tags predate this fix).
10. **`.github/workflows/security-audit.yml` (added in this pass) is
    unverified** - authored and `actionlint`-checked in a sandbox with no
    network access to PyPI/crates.io/the GitHub advisory database; it has
    not been confirmed to actually run green (or to correctly fail) on
    real GitHub Actions infrastructure.

## Explicitly not planned right now

- Nothing in this repository is being actively marketed with unverified
  speed claims anymore ("40-50x faster" etc. was removed in an earlier
  restoration pass - see README's "Performance claims" section). Don't
  reintroduce marketing numbers without a benchmark script backing them.
