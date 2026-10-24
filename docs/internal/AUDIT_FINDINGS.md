# Audit findings — 2026-10-19

This file records the repository-specific findings from the portfolio audit
conducted on 2026-10-19. It is documentation only; no code or release artifacts
were changed in this pass unless explicitly noted under "Implementation status".

Sources: the repository's own README/source/workflows, GitHub metadata and
Actions runs, PyPI metadata, and official competitor documentation. Claims are
marked as verified, inferred, or open questions. "Absent" means not found in the
inspected repository/docs, not proof that it cannot exist.

## What it is

ML deployment toolkit: canary/A-B routing, cost tracking, request scheduling, hot model reload, and ONNX Runtime inference.

## Verified strengths

- Real Python deployment layer; canary routing; cost tracking; scheduler; hot reload; ONNX Runtime wrapper; MLflow/HF registry; HTTP API; 18 tests.
- Useful local deployment primitives.

## Verified gaps and user-facing risks

- Cloud backends absent; no auth; no CLI; Rust extension methods return formatted strings and fake speedup constants; breaking changes in 2.0; alpha.
- No production evidence.

## Direct competitors

- BentoML, KServe, Seldon, Ray Serve, Modal.

## Ranked next actions

- Freeze/archive, or narrow to 'canary routing for local ONNX' if a user asks.
- Remove fake Rust acceleration claims and dead extension paths (S).

## Do not build

- Multi-cloud backends, dashboard; no wedge.

## Evidence gaps / validation needed

- No production deployment; no benchmark.

## Recommended disposition

Archive unless a specific local-ONNX user validates it.

## Implementation status

No code changes in this pass.
