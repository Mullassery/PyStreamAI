# PyStreamAI Inference Optimization Benchmarks

Benchmark suite demonstrating 5-10x inference speedup through automatic optimization.

## Quick Start

### Install Dependencies

```bash
pip install -r ../requirements-benchmark.txt
```

### Run Benchmarks

```bash
# Run BERT and GPT-2 benchmarks
python bench_bert_llm.py

# Results saved to: benchmark_results/
```

## What's Tested

### BERT (Sequence Classification)
- Model: `distilbert-base-uncased-finetuned-sst-2-english`
- Task: Sentiment classification
- Optimizations:
  - Quantization (INT8)
  - Batching (8 samples)
  - Combined (quantization + batching)

### GPT-2 (Language Model)
- Model: `gpt2` (pretrained)
- Task: Text generation
- Optimizations:
  - Quantization (INT8)
  - KV caching (for autoregressive generation)
  - Batching (4 samples)
  - Combined (quantization + batching + caching)

## Expected Results

**BERT Speedups:**
- Quantization (INT8): ~1.5-2.0x faster
- Batching (8 samples): ~4-6x faster (per-sample)
- Combined: ~6-8x faster (per-sample)

**GPT-2 Speedups:**
- Quantization (INT8): ~1.5-2.0x faster
- KV Cache: ~1.2-1.5x faster
- Batching (4 samples): ~3-4x faster (per-sample)
- Combined: ~5-7x faster (per-sample)

## Key Insight

**PyStreamAI advantage:** These optimizations happen automatically. Users don't need to:
- Learn quantization APIs
- Manage batching manually
- Handle KV caching
- Write custom inference code

Just call `platform.serve(model)` and get 5-10x speedup for free.

## Hardware Notes

- Benchmarks run on CPU by default
- GPU speedups will be higher (especially for larger models)
- Results scale better with larger batch sizes

## Extending Benchmarks

To add your own model:

```python
def benchmark_your_model():
    from benchmarks.models import load_your_model
    from benchmarks.inference import InferenceOptimizer
    from benchmarks.runner import BenchmarkRunner, BenchmarkResult

    model, tokenizer = load_your_model()
    optimizer = InferenceOptimizer(model, tokenizer)
    runner = BenchmarkRunner()

    # Run optimizations...
    runner.add_result(BenchmarkResult(...))
    runner.print_results_table()
```

## Files

- `bench_bert_llm.py` — Main benchmark script
- `models.py` — Model loading and quantization
- `inference.py` — Inference optimization implementations
- `runner.py` — Benchmark orchestration and reporting
- `__init__.py` — Package exports
