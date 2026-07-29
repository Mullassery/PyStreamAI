"""Benchmark runner for PyStreamAI inference optimization"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    model_name: str
    optimization: str
    baseline_latency_ms: float
    optimized_latency_ms: float
    batch_size: int = 1
    num_runs: int = 10
    speedup: float = field(init=False)
    latency_reduction_percent: float = field(init=False)

    def __post_init__(self):
        if self.baseline_latency_ms > 0:
            self.speedup = self.baseline_latency_ms / self.optimized_latency_ms
            self.latency_reduction_percent = (
                (self.baseline_latency_ms - self.optimized_latency_ms)
                / self.baseline_latency_ms
            ) * 100.0
        else:
            self.speedup = 1.0
            self.latency_reduction_percent = 0.0

    def __str__(self) -> str:
        return (
            f"{self.model_name} + {self.optimization:30s} | "
            f"Baseline: {self.baseline_latency_ms:7.2f}ms | "
            f"Optimized: {self.optimized_latency_ms:7.2f}ms | "
            f"Speedup: {self.speedup:5.2f}x | "
            f"Reduction: {self.latency_reduction_percent:5.1f}%"
        )


class BenchmarkRunner:
    """Orchestrates benchmarking across models and optimizations"""

    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[BenchmarkResult] = []

    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result"""
        self.results.append(result)
        logger.info(str(result))

    def save_results(self, filename: str = "results.json"):
        """Save all results to JSON"""
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(
                [asdict(r) for r in self.results],
                f,
                indent=2
            )
        logger.info(f"Results saved to {output_path}")

    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        if not self.results:
            return {}

        by_model = {}
        for result in self.results:
            if result.model_name not in by_model:
                by_model[result.model_name] = []
            by_model[result.model_name].append(result)

        summary = {}
        for model_name, results in by_model.items():
            speedups = [r.speedup for r in results]
            summary[model_name] = {
                "avg_speedup": sum(speedups) / len(speedups),
                "max_speedup": max(speedups),
                "min_speedup": min(speedups),
                "optimizations": len(results),
            }

        return summary

    def print_summary(self):
        """Print formatted summary"""
        summary = self.summary()
        if not summary:
            print("No results to summarize")
            return

        print("\n" + "=" * 100)
        print("INFERENCE OPTIMIZATION BENCHMARK SUMMARY")
        print("=" * 100)

        for model_name, stats in summary.items():
            print(f"\n{model_name}:")
            print(f"  Average Speedup: {stats['avg_speedup']:.2f}x")
            print(f"  Max Speedup: {stats['max_speedup']:.2f}x")
            print(f"  Min Speedup: {stats['min_speedup']:.2f}x")
            print(f"  Optimizations Tested: {stats['optimizations']}")

        print("\n" + "=" * 100)
        print("DETAILED RESULTS")
        print("=" * 100)
        for result in self.results:
            print(result)
        print("=" * 100 + "\n")

    def print_results_table(self):
        """Print results as formatted table"""
        if not self.results:
            print("No results to display")
            return

        print("\n" + "-" * 140)
        print(
            f"{'Model':<20} | {'Optimization':<25} | "
            f"{'Baseline (ms)':<15} | {'Optimized (ms)':<15} | "
            f"{'Speedup':<10} | {'Reduction %':<12}"
        )
        print("-" * 140)

        for result in self.results:
            print(
                f"{result.model_name:<20} | {result.optimization:<25} | "
                f"{result.baseline_latency_ms:<15.2f} | {result.optimized_latency_ms:<15.2f} | "
                f"{result.speedup:<10.2f}x | {result.latency_reduction_percent:<12.1f}%"
            )

        print("-" * 140 + "\n")
