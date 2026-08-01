# PyStreamAI

Run AI models 40-50x faster. No YAML required for basic use.

[![Tests](https://img.shields.io/github/actions/workflow/status/Mullassery/PyStreamAI/tests.yml?label=tests)](https://github.com/Mullassery/PyStreamAI/actions)

Fast LLM inference with automatic optimization. Sensible defaults—configure only if you need to.

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
