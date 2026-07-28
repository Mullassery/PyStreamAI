# PyStreamAI

Run AI models 40-50x faster. Zero configuration.

[![Tests](https://img.shields.io/github/actions/workflow/status/Mullassery/PyStreamAI/tests.yml?label=tests)](https://github.com/Mullassery/PyStreamAI/actions)

Fast LLM inference with automatic optimization. No YAML, no complex setup.

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

## Installation

```bash
pip install pystreamai
```

## Use Cases

- Fast inference serving
- Cost optimization through batching
- Real-time API responses
- Local and edge deployment
- Multi-model inference

## Examples

See [examples/](examples/) for complete working examples.

## Documentation

- [Getting Started](docs/getting-started.md)
- [API Reference](docs/api.md)
- [Benchmarks](docs/benchmarks.md)
- [Examples](examples/)

## License

See LICENSE
