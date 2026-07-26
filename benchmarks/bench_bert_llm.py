#!/usr/bin/env python3
"""
Benchmark BERT and GPT-2 inference with PyStreamAI optimizations.

This script measures:
1. Baseline inference (no optimization)
2. Quantized inference (INT8)
3. Batched inference (1, 2, 4, 8 examples)
4. KV-cached inference (for LLMs)
5. Combined optimizations

Expected results: 5-10x speedup with combined optimizations
"""

import logging
import sys
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def benchmark_bert():
    """Benchmark BERT model"""
    from benchmarks.models import load_bert_model, quantize_model_int8
    from benchmarks.inference import InferenceOptimizer
    from benchmarks.runner import BenchmarkRunner, BenchmarkResult

    logger.info("=" * 80)
    logger.info("BENCHMARKING BERT (Sequence Classification)")
    logger.info("=" * 80)

    runner = BenchmarkRunner()

    try:
        model, tokenizer = load_bert_model()
    except ImportError as e:
        logger.error(f"Failed to load BERT: {e}")
        logger.error("Install: pip install transformers torch")
        return None

    optimizer = InferenceOptimizer(model, tokenizer)

    # Test samples
    test_text = "This is a great movie! I loved it so much."
    test_texts = [test_text] * 8

    # 1. Baseline (single inference)
    logger.info("\n1. Running baseline inference...")
    baseline_latencies = []
    for _ in range(10):
        latency = optimizer.infer_baseline(test_text)
        baseline_latencies.append(latency)
    baseline_avg = sum(baseline_latencies) / len(baseline_latencies)

    # 2. Quantized INT8
    logger.info("2. Running quantized inference (INT8)...")
    model_int8 = quantize_model_int8(model)
    optimizer_int8 = InferenceOptimizer(model_int8, tokenizer)
    quantized_latencies = []
    for _ in range(10):
        latency = optimizer_int8.infer_baseline(test_text)
        quantized_latencies.append(latency)
    quantized_avg = sum(quantized_latencies) / len(quantized_latencies)

    # 3. Batched (size 8)
    logger.info("3. Running batched inference (batch_size=8)...")
    batched_latencies = []
    for _ in range(10):
        latency = optimizer.infer_batched(test_texts)
        batched_latencies.append(latency)
    batched_avg = sum(batched_latencies) / len(batched_latencies)
    batched_per_sample = batched_avg / len(test_texts)

    # 4. Quantized + Batched
    logger.info("4. Running quantized + batched inference...")
    combined_latencies = []
    for _ in range(10):
        latency = optimizer_int8.infer_batched(test_texts)
        combined_latencies.append(latency)
    combined_avg = sum(combined_latencies) / len(combined_latencies)
    combined_per_sample = combined_avg / len(test_texts)

    # Add results
    runner.add_result(BenchmarkResult(
        model_name="BERT",
        optimization="Baseline",
        baseline_latency_ms=baseline_avg,
        optimized_latency_ms=baseline_avg,
        num_runs=10,
    ))

    runner.add_result(BenchmarkResult(
        model_name="BERT",
        optimization="Quantized (INT8)",
        baseline_latency_ms=baseline_avg,
        optimized_latency_ms=quantized_avg,
        num_runs=10,
    ))

    runner.add_result(BenchmarkResult(
        model_name="BERT",
        optimization="Batched (8 samples)",
        baseline_latency_ms=baseline_avg,
        optimized_latency_ms=batched_per_sample,
        batch_size=8,
        num_runs=10,
    ))

    runner.add_result(BenchmarkResult(
        model_name="BERT",
        optimization="Quantized + Batched",
        baseline_latency_ms=baseline_avg,
        optimized_latency_ms=combined_per_sample,
        batch_size=8,
        num_runs=10,
    ))

    runner.save_results("bert_results.json")
    runner.print_results_table()

    return runner


def benchmark_gpt2():
    """Benchmark GPT-2 model"""
    from benchmarks.models import load_gpt2_model, quantize_model_int8
    from benchmarks.inference import InferenceOptimizer
    from benchmarks.runner import BenchmarkRunner, BenchmarkResult

    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARKING GPT-2 (Language Model)")
    logger.info("=" * 80)

    runner = BenchmarkRunner()

    try:
        model, tokenizer = load_gpt2_model()
    except ImportError as e:
        logger.error(f"Failed to load GPT-2: {e}")
        logger.error("Install: pip install transformers torch")
        return None

    optimizer = InferenceOptimizer(model, tokenizer)

    # Test samples
    test_text = "The future of artificial intelligence is"
    test_texts = [test_text] * 4

    # 1. Baseline
    logger.info("\n1. Running baseline inference...")
    baseline_latencies = []
    for _ in range(10):
        latency = optimizer.infer_baseline(test_text)
        baseline_latencies.append(latency)
    baseline_avg = sum(baseline_latencies) / len(baseline_latencies)

    # 2. Quantized INT8
    logger.info("2. Running quantized inference (INT8)...")
    model_int8 = quantize_model_int8(model)
    optimizer_int8 = InferenceOptimizer(model_int8, tokenizer)
    quantized_latencies = []
    for _ in range(10):
        latency = optimizer_int8.infer_baseline(test_text)
        quantized_latencies.append(latency)
    quantized_avg = sum(quantized_latencies) / len(quantized_latencies)

    # 3. KV Cache
    logger.info("3. Running inference with KV cache...")
    kv_cached_latencies = []
    for _ in range(10):
        latency = optimizer.infer_kv_cached(test_text)
        kv_cached_latencies.append(latency)
    kv_cached_avg = sum(kv_cached_latencies) / len(kv_cached_latencies)

    # 4. Batched (size 4)
    logger.info("4. Running batched inference (batch_size=4)...")
    batched_latencies = []
    for _ in range(10):
        latency = optimizer.infer_batched(test_texts)
        batched_latencies.append(latency)
    batched_avg = sum(batched_latencies) / len(batched_latencies)
    batched_per_sample = batched_avg / len(test_texts)

    # 5. Combined optimizations
    logger.info("5. Running combined optimizations...")
    combined_latencies = []
    for _ in range(10):
        latency = optimizer_int8.infer_combined(test_texts)
        combined_latencies.append(latency)
    combined_avg = sum(combined_latencies) / len(combined_latencies)
    combined_per_sample = combined_avg / len(test_texts)

    # Add results
    runner.add_result(BenchmarkResult(
        model_name="GPT-2",
        optimization="Baseline",
        baseline_latency_ms=baseline_avg,
        optimized_latency_ms=baseline_avg,
        num_runs=10,
    ))

    runner.add_result(BenchmarkResult(
        model_name="GPT-2",
        optimization="Quantized (INT8)",
        baseline_latency_ms=baseline_avg,
        optimized_latency_ms=quantized_avg,
        num_runs=10,
    ))

    runner.add_result(BenchmarkResult(
        model_name="GPT-2",
        optimization="KV Cache",
        baseline_latency_ms=baseline_avg,
        optimized_latency_ms=kv_cached_avg,
        num_runs=10,
    ))

    runner.add_result(BenchmarkResult(
        model_name="GPT-2",
        optimization="Batched (4 samples)",
        baseline_latency_ms=baseline_avg,
        optimized_latency_ms=batched_per_sample,
        batch_size=4,
        num_runs=10,
    ))

    runner.add_result(BenchmarkResult(
        model_name="GPT-2",
        optimization="Quantized + Batched + KV Cache",
        baseline_latency_ms=baseline_avg,
        optimized_latency_ms=combined_per_sample,
        batch_size=4,
        num_runs=10,
    ))

    runner.save_results("gpt2_results.json")
    runner.print_results_table()

    return runner


if __name__ == "__main__":
    logger.info("PyStreamAI Inference Optimization Benchmarks")
    logger.info("Measuring speedups on BERT and GPT-2")

    bert_runner = benchmark_bert()
    gpt2_runner = benchmark_gpt2()

    if bert_runner and gpt2_runner:
        print("\n" + "=" * 80)
        print("OVERALL SUMMARY")
        print("=" * 80)
        print("\nBERT Results:")
        bert_runner.print_summary()
        print("\nGPT-2 Results:")
        gpt2_runner.print_summary()

        print("\n" + "=" * 80)
        print("✅ Benchmarking complete. Results saved to benchmark_results/")
        print("=" * 80)
    else:
        logger.error("Failed to run benchmarks")
        sys.exit(1)
