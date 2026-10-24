# Competitive intelligence in roadmap and upgrade work

**Policy date:** 2026-10-19

Competitive findings are an input to product decisions, not a one-off report.
Every pending roadmap item, new feature, and routine dependency upgrade should
check the repository's [`AUDIT_FINDINGS.md`](AUDIT_FINDINGS.md) before work
starts, and record the decision in the PR or changelog.

## Required check before building

1. **Read `AUDIT_FINDINGS.md`.** Find the relevant competitor row and the
   ranked next actions. If the finding is stale, update it with a dated source.
2. **Name the competitor and the user problem.** A feature is not justified by
   "competitor has it". State the workflow it unlocks, who needs it, and how
   often. Mark unverified demand as a hypothesis.
3. **Prefer reuse over new code.** Check whether an existing Mullassery repo
   already provides the capability (see `BUNDLES.md` in the profile repo and
   the bundle role file in this repo, when present). Typical reuse:
   - crawling/fetching/robots/SSRF → **MudFish**
   - architecture maps / PR diffs → **CodeAtlas**
   - data-quality contracts / validation → **StatGuardian**
   - token counting / cost tables → **PyTokenCalc**
   - dependency graph / advisories / SBOM → **PyDeepMesh**
   - segmentation / CLV / churn → **ClusterAudienceKit**
   - activation / lineage / PII → **PyReverseETL**
   - fleet state sync → **SyncFleet** (only after conflict resolution)
4. **Choose one disposition:** build, integrate, defer, or reject. Write it
   down. A rejection with a reason is a successful outcome.
5. **For routine upgrades**, check whether the upgrade closes a known
   competitor gap or breaks a differentiated capability. Run the tests that
   cover the affected path; do not merge a green build that silently changes a
   user-visible contract.

## PR / changelog checklist

- [ ] Competitor and source URL/date recorded (or "no direct competitor").
- [ ] User impact and frequency stated; unverified demand labelled.
- [ ] Reuse check done against sibling repos/bundles.
- [ ] Failure mode and rollback considered (especially auth, PII, cost).
- [ ] Docs/roadmap status updated honestly.
- [ ] Tests or a manual verification command included.

## Review cadence

- Refresh the competitor row when a release changes the capability.
- Re-check the full `AUDIT_FINDINGS.md` at least quarterly and before a major
  release.
- If a finding is wrong or stale, correct it in the same PR that relies on it.

## Non-goals

- Copying features for parity.
- Adding a dependency just because a competitor uses it.
- Treating star counts or download counts as proof of user need.

## OSS gap analysis — 2026-10-24

Scope: `pystreamai` as a local Python model-serving library — HTTP inference, batching, canary/A-B routing, hot reload, cost tracking, scheduling — compared against OSS model-serving platforms. Repo state: v2.1.3, Alpha, 169 tests passing (2026-09-21). "Ours" claims are checked against `README.md`, `ROADMAP_HONEST.md`, `pystreamai/serving.py`, `deployment.py`, `hot_reload.py`.

### Comparable OSS projects

| Project | GitHub | License | Last activity | Why it is the comparison |
| --- | --- | --- | --- | --- |
| BentoML | https://github.com/bentoml/BentoML | Apache-2.0 | v1.4.40, 2026-10-23 | Closest Python-first serving library: packaging, adaptive batching, runners |
| Ray Serve | https://github.com/ray-project/ray | Apache-2.0 | active 2026 | Composable Python serving with traffic splitting, autoscaling, model composition |
| KServe | https://github.com/kserve/kserve | Apache-2.0 | v0.21.0, 2026-09-25 | Kubernetes-native standard for canary, autoscaling, inference graphs |
| Seldon Core | https://github.com/SeldonIO/seldon-core | Business Source License 1.1 (not OSI; converts to Apache-2.0) | v1.19.0, 2026-01-23 | Canary, A/B, inference graphs on Kubernetes; license is a differentiator for us |

### Capability comparison

| Capability | Ours | BentoML | Ray Serve | KServe | Seldon Core | Gap |
| --- | --- | --- | --- | --- | --- | --- |
| HTTP predict/health endpoints | Yes | Yes | Yes | Yes | Yes | No |
| Weighted canary traffic split | Yes | Partial | Yes | Yes | Yes | No |
| Per-user A/B routing | Yes | Partial | Partial | Partial | Partial | No |
| Dynamic batching | Yes | Yes | Yes | Yes | Yes | No |
| Hot model reload / version activation | Yes | Yes | Yes | Yes | Yes | No |
| Autoscaling and replica management | No | Partial | Yes | Yes | Yes | Yes |
| Multi-model pipelines / inference graphs | No | Yes | Yes | Yes | Yes | Yes |
| Reproducible model packaging | No | Yes | Partial | Yes | Yes | Yes |
| Authn/authz | No | Partial | Yes | Yes | Yes | Yes |
| Kubernetes-native CRDs / operator | No | Partial | Yes | Yes | Yes | Yes |
| Per-request cost accounting | Partial | Partial | Partial | Partial | Partial | Yes |
| Priority request scheduling | Partial | Partial | Partial | Partial | Partial | Yes |
| Observability (metrics, traces) | Partial | Yes | Yes | Yes | Yes | Yes |
| Cloud deployment backends | No | Yes | Yes | Yes | Yes | Yes |
| Production maturity | No | Yes | Yes | Yes | Yes | Yes |

### Gaps that matter (ranked)

1. **No autoscaling or Kubernetes-native deployment.** We have one process and no replica/HPA/operator story. Ray Serve, KServe, and Seldon Core all make scale-out the default; BentoML at least integrates with it. This is the gap that keeps us out of real deployments.
2. **No authentication.** `README.md` states plainly that `pystreamai.api` / `pystreamai.serving` have no auth of any kind. Every comparable expects a gateway or token layer in front; ours requires the user to supply one before exposure.
3. **No multi-model pipelines.** BentoML, Ray Serve, KServe inference graphs, and Seldon graphs compose models; we serve one model per server, so multi-stage inference is out of scope today.
4. **Cost tracking and priority scheduling are unwired.** Both modules exist (`cost_tracking.py`, `request_scheduler.py`), but `serving.py:205-215` explicitly returns `gpu_id=None` and `cost_usd=0.0` because nothing backs them. The claimed differentiator is currently not on the request path.
5. **No reproducible packaging.** BentoML builds a bento + image, KServe pulls OCI/S3 artifacts, Seldon ships prepackaged servers. Our MLflow/HuggingFace registry paths are partial and `HuggingFaceRegistry.upload_model()` returns `True` without uploading.
6. **Placeholder surfaces undermine credibility.** The Rust `Platform.train()/serve()/predict()` only `format!()` strings, GPU/TensorRT speedups are hardcoded constants, and `llm_optimization`/`edge_deployment` are interface sketches. There is no CLI and no verified benchmark, while the comparables publish real numbers and release notes.

### Where we are differentiated

- Canary and per-user A/B routing (`deployment.py`) live in-process with no cluster, no CRD, no sidecar — the lightest possible version of what KServe/Seldon require a control plane for.
- Real dynamic batching against cached ONNX endpoints (`serving.py`), not a simulated queue.
- Hot reload with explicit version activation and graceful shutdown (`hot_reload.py`).
- Apache-2.0 and pip-installable, versus Seldon Core's Business Source License and the K8s/operator footprint of KServe and Seldon.
- Priority scheduling and per-request cost primitives are already written; if wired, they are a plausible niche (local, cost-aware serving) that BentoML/Ray do not center.
- Documentation is unusually honest about what is not implemented, which is itself rare in this category.

### Disposition

- **Build:** auth middleware (token/API key) and wire `cost_tracking.py` + `request_scheduler.py` into the serving path; both are small and turn existing claims into facts.
- **Build:** one multi-model pipeline primitive (sequential graph with typed hand-off), or explicitly document single-model as a constraint.
- **Integrate/reuse:** adopt BentoML's bento format or the MLflow model registry for packaging instead of inventing a store; point users at KServe/Ray Serve for autoscaling rather than writing an operator.
- **Defer/reject:** owning an autoscaler, a Kubernetes operator, or a model store — the platforms already won this and we cannot outspend them.
- **Defer/reject:** cloud backends (`aws`/`gcp`/`azure`) until a user asks; reject GPU/TensorRT speedup claims until measured, and delete or gate the placeholder Rust/LLM/edge modules.

### Sources

- https://github.com/bentoml/BentoML/releases — v1.4.40, accessed 2026-10-24.
- https://github.com/ray-project/ray/releases — active 2026, accessed 2026-10-24.
- https://github.com/kserve/kserve/releases — v0.21.0, 2026-09-25, accessed 2026-10-24.
- https://github.com/SeldonIO/seldon-core/releases and /blob/master/LICENSE — v1.19.0, 2026-01-23; BSL 1.1, accessed 2026-10-24.
- Local evidence: `README.md` (limitations), `ROADMAP_HONEST.md`, `pystreamai/serving.py`, `pystreamai/deployment.py`, `pystreamai/hot_reload.py`.
