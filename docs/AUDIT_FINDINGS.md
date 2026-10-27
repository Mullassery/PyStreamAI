# Audit findings — 2026-10-26

This file records the current portfolio-audit findings for pystreamai. It is
documentation only; no code or release artifacts were changed. The fuller
evidence report is kept in the portfolio workspace at
`.audit/reports/pystreamai/audit.md`.

## What it is

An alpha ML deployment and serving toolkit: model endpoints, request batching,
a FastAPI wrapper, ONNX Runtime integration, deployment/canary primitives, cost
accounting, scheduling, model registry, hot reload, and manifests.

## Verified strengths

- Real ONNX/callable inference dispatch in `Endpoint.predict()`.
- `APIServer` loads a real model before serving `/predict`; no fabricated model
  responses remain on the serving path.
- Batch queue, model cache, FastAPI endpoints, manifest generation, and
  monitoring are implemented as a coherent local toolkit.
- PyO3 extension provides GPU inspection and memory-pool primitives.
- Default bind is loopback; ONNX version constraint is documented.
- 18 test files and 161 static test definitions.

## Verified gaps and user-facing risks

- No authentication or authorization on the HTTP surface. `/predict`,
  `/stats`, model info, and `/shutdown` are open to anyone who can reach the
  port.
- `Endpoint.predict()` runs synchronously inside the async FastAPI handler, so
  a slow model blocks the event loop and concurrent requests.
- `request_log` grows without bound; `/stats` cost/latency data is in memory
  only and is lost on restart.
- No request size limit, inference timeout, concurrency cap, or readiness probe.
- `gpu_type`, `num_gpus`, and similar compatibility parameters do not drive a
  GPU scheduler; cost output should not be treated as a billing control.
- Only the `local` platform backend exists. Training is synchronous.
- Version/docs drift: source is `2.1.3`; README still says `2.1.0`; the
  2026-10-19 internal audit says "18 tests."
- `/shutdown` only schedules an empty hook; it is not a real shutdown control.

## Direct positioning

Lightweight local serving against BentoML, KServe, Seldon, Ray Serve, and Modal.
The niche is a small ONNX/callable deployment layer, not a distributed
inference platform or GPU scheduler.

## Ranked next actions

1. P0: add API authentication before any non-loopback deployment, and
   authenticate or remove `/shutdown`.
2. P0: move inference off the event loop and add timeouts/request limits.
3. P1: bound the request log and add readiness/metrics persistence.
4. P1: reconcile version and test-count claims.
5. P2: label GPU/cost fields as compatibility-only, or implement enforcement.
6. P2: add a committed load benchmark and production deployment example.

## Do not build

- Multi-cloud provisioning, a GPU scheduling control plane, or a hosted
  inference platform. The local serving wedge is the credible scope.

## Evidence gaps / validation needed

- No production deployment evidence.
- No reproducible benchmark for latency, throughput, or memory behavior.
- No security testing of the HTTP surface.

## Recommended disposition

Maintain narrowly as a local ONNX/callable serving toolkit. Do not position it
as a production or distributed platform until auth, non-blocking inference,
and operational bounds are in place.
