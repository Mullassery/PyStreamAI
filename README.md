# PyStreamAI

**Simple ML Deployment - Kubeflow + KServe Replacement**

Zero YAML. No Kubernetes expertise required. Ship models to production in minutes.

## Vision

Kubeflow solved the problem of running ML workloads on Kubernetes, but accumulated complexity. KServe requires infrastructure expertise. Modern ML engineers shouldn't need to become Kubernetes experts to deploy models.

**PyStreamAI** replaces both with:

- ✨ **Dead-simple Python API** — No YAML, no manifests, no infrastructure expertise
- 🚀 **Local-to-cloud** — Test locally, deploy with one command
- 💰 **Cost visibility** — Real-time cost attribution per request
- 🌍 **Multi-cloud** — AWS, GCP, Azure, on-prem, Kubernetes (your choice)
- 🎯 **Modern AI stack** — LLMs, RAG, agents, not just classical ML
- ⚡ **Performance** — Rust core for speed + Python ergonomics

## Quick Start

### Installation

```bash
pip install pystreamai
```

### Simple Deployment

```python
from pystreamai import Platform

platform = Platform()

# Train a model
@platform.train(gpu="A100", time_limit="1h")
def train_model(data):
    model = train_bert(data)
    return model

job = train_model(data="s3://bucket/data")

# Deploy it
@platform.serve(replicas=3, gpu="L4")
def predict(x):
    return model.predict(x)

endpoint = predict()

# Use it
result = endpoint.predict({"text": "Hello world"})
```

That's it. No YAML. No Kubernetes. No infrastructure knowledge required.

## Architecture

```
┌──────────────────────────────────────┐
│   PyStreamAI Python API              │
├──────────────────────────────────────┤
│  @train  @serve  @pipeline           │
├──────────────────────────────────────┤
│   Rust Core Orchestrator             │
├──────────────────────────────────────┤
│ Scheduler │ Executor │ Storage       │
└──────────────────────────────────────┘
     ↓ Deploy to ↓
AWS / GCP / Azure / On-Prem / Kubernetes
```

## Features (v0.1)

- [x] Simple Python API
- [x] Local deployment
- [x] Training job submission
- [x] Model serving
- [ ] Cost tracking
- [ ] Multi-cloud backends
- [ ] Observability (traces, metrics, logs)
- [ ] Pipelines (DAGs + dynamic)
- [ ] Model registry
- [ ] Edge deployment

## Development

### Setup

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Python dependencies
pip install maturin pytest

# Build
maturin develop
```

### Running Tests

```bash
pytest tests/
```

## Roadmap

**v0.1** (current)
- Simple Python API
- Local & AWS backend
- Training + serving

**v0.2**
- Pipelines (workflows)
- Multi-cloud (GCP, Azure)
- Cost tracking

**v0.3**
- Model registry
- Observability layer
- Prompt management

**v0.4+**
- RAG workflows
- Agent framework
- Edge deployment

## License

Proprietary. © 2026 Georgi Mammen Mullassery.

## Status

🚧 Active development. Not ready for production yet.
