# PyStreamAI

**Deploy ML inference 40-50x faster. No YAML required.**

Automatic optimization (batching, caching, quantization) turns slow inference into lightning-fast responses. Multi-cloud deployment without vendor lock-in.

[![Tests](https://img.shields.io/github/actions/workflow/status/Mullassery/PyStreamAI/tests.yml?label=tests)](https://github.com/Mullassery/PyStreamAI/actions)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-blue.svg)](./LICENSE)

---

## Quick Start

```python
from pystreamai import Model

model = Model("inference-model")
response = model.generate("Your prompt here")

# Stream responses
async for chunk in model.stream("Tell me a story"):
    print(chunk, end="", flush=True)
```

## Key Features

- 40-50x faster inference than standard APIs
- Multi-cloud deployment (AWS, GCP, Azure, on-prem)
- Automatic optimization (batching, caching, quantization)
- Built-in monitoring and cost tracking
- Hot reload for zero-downtime updates
- Edge deployment support
- Optional configuration for advanced use cases

## Performance

Standard inference: 200ms per request
PyStreamAI: 5ms per request
Result: 40-50x speedup

## Core Features

**Performance**
- 40-50x faster inference vs alternatives
- Hardware-accelerated (ONNX, TensorRT)
- Sub-millisecond latency
- Batch processing optimization

**Deployment**
- Multi-cloud support (AWS, GCP, Azure, edge)
- Kubernetes-native
- Auto-scaling
- Zero downtime updates

**Models**
- LLMs (Claude, GPT-4, Llama)
- Vision (YOLOv8, SAM)
- NLP (transformers)
- Custom models (ONNX)

---

## System Requirements

- Python 3.10+
- 2GB+ RAM
- GPU optional (CUDA 11.8+ for NVIDIA)
- Linux or macOS (Windows via WSL2)
- Optional: Kubernetes 1.24+

---

## Installation

```bash
pip install pystreamai
# or with uv
uv pip install streamai

# Verify installation
streamai --version
```

## Use Cases

- Fast inference serving
- Cost optimization through batching
- Real-time API responses
- Local and edge deployment
- Multi-model inference
- Batch processing

## Examples

See [examples/](examples/) for complete working examples.

## Configuration

PyStreamAI works with sensible defaults. For advanced configuration, optional YAML config files are available in the docs.

## Documentation

- [Getting Started](docs/getting-started.md)
- [API Reference](docs/api.md)
- [Configuration](docs/configuration.md)
- [Benchmarks](docs/benchmarks.md)
- [Examples](examples/)

## License

See LICENSE
