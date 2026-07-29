"""Phase 6: Cost & Performance Optimization - Track costs, benchmark models, optimize inference"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger(__name__)


@dataclass
class InferenceCost:
    """Cost breakdown for a single inference"""
    compute_cost_usd: float
    storage_cost_usd: float
    data_transfer_cost_usd: float
    overhead_cost_usd: float = 0.0

    @property
    def total_cost_usd(self) -> float:
        return (
            self.compute_cost_usd
            + self.storage_cost_usd
            + self.data_transfer_cost_usd
            + self.overhead_cost_usd
        )


@dataclass
class VersionMetrics:
    """Performance metrics for a model version"""
    version_id: str
    model_id: str
    latency_ms: float
    throughput_requests_per_second: float
    error_rate_percent: float
    memory_mb: float
    gpu_memory_mb: float
    batch_size: int = 1


@dataclass
class CostModel:
    """Cost model for infrastructure"""
    gpu_hourly_cost_usd: float = 1.0
    cpu_hourly_cost_usd: float = 0.1
    memory_hourly_cost_usd: float = 0.01  # Per GB
    storage_hourly_cost_usd: float = 0.001  # Per GB
    data_transfer_cost_per_gb_usd: float = 0.12


class CostCalculator:
    """Calculate cost per inference"""

    def __init__(self, cost_model: CostModel):
        self.cost_model = cost_model

    def calculate_inference_cost(
        self,
        latency_ms: float,
        gpu_memory_mb: float,
        memory_mb: float,
        data_transfer_mb: float = 0.0,
    ) -> InferenceCost:
        """Calculate cost for single inference"""
        # Compute cost (based on GPU time)
        compute_hours = latency_ms / (3600 * 1000)  # Convert ms to hours
        gpu_cost = compute_hours * self.cost_model.gpu_hourly_cost_usd
        cpu_cost = compute_hours * self.cost_model.cpu_hourly_cost_usd

        # Memory cost (based on memory allocation)
        memory_hours = compute_hours
        memory_cost = (memory_mb / 1024) * memory_hours * self.cost_model.memory_hourly_cost_usd
        gpu_memory_cost = (gpu_memory_mb / 1024) * memory_hours * self.cost_model.gpu_hourly_cost_usd

        # Data transfer cost
        transfer_cost = data_transfer_mb * self.cost_model.data_transfer_cost_per_gb_usd / 1024

        return InferenceCost(
            compute_cost_usd=gpu_cost + cpu_cost,
            storage_cost_usd=gpu_memory_cost + memory_cost,
            data_transfer_cost_usd=transfer_cost,
        )

    def calculate_daily_cost(
        self,
        daily_requests: int,
        cost_per_inference: InferenceCost,
    ) -> float:
        """Calculate daily cost"""
        return daily_requests * cost_per_inference.total_cost_usd

    def calculate_monthly_cost(
        self,
        daily_requests: int,
        cost_per_inference: InferenceCost,
    ) -> float:
        """Calculate monthly cost"""
        return self.calculate_daily_cost(daily_requests, cost_per_inference) * 30


class VersionCostTracker:
    """Track cost per model version"""

    def __init__(self, cost_model: Optional[CostModel] = None):
        self.cost_model = cost_model or CostModel()
        self.calculator = CostCalculator(self.cost_model)
        self.version_costs: Dict[str, Dict[str, Any]] = {}
        self.inference_metrics: Dict[str, List[VersionMetrics]] = {}

    def record_inference(self, metrics: VersionMetrics) -> None:
        """Record inference metrics for a version"""
        version_id = metrics.version_id

        if version_id not in self.inference_metrics:
            self.inference_metrics[version_id] = []

        self.inference_metrics[version_id].append(metrics)

        # Update cost tracking
        if version_id not in self.version_costs:
            self.version_costs[version_id] = {
                "model_id": metrics.model_id,
                "total_inferences": 0,
                "total_cost_usd": 0.0,
                "average_cost_per_inference": 0.0,
                "min_cost_per_inference": float('inf'),
                "max_cost_per_inference": 0.0,
                "recorded_at": datetime.now().isoformat(),
            }

        # Calculate cost for this inference
        cost = self.calculator.calculate_inference_cost(
            latency_ms=metrics.latency_ms,
            gpu_memory_mb=metrics.gpu_memory_mb,
            memory_mb=metrics.memory_mb,
        )

        # Update tracking
        tracking = self.version_costs[version_id]
        tracking["total_inferences"] += 1
        tracking["total_cost_usd"] += cost.total_cost_usd
        tracking["average_cost_per_inference"] = tracking["total_cost_usd"] / tracking["total_inferences"]
        tracking["min_cost_per_inference"] = min(tracking["min_cost_per_inference"], cost.total_cost_usd)
        tracking["max_cost_per_inference"] = max(tracking["max_cost_per_inference"], cost.total_cost_usd)

    def get_version_cost_summary(self, version_id: str) -> Dict[str, Any]:
        """Get cost summary for a version"""
        if version_id not in self.version_costs:
            return {}

        return self.version_costs[version_id]

    def compare_versions(self, version_ids: List[str]) -> Dict[str, Any]:
        """Compare costs across versions"""
        comparison = {}

        for vid in version_ids:
            if vid in self.version_costs:
                summary = self.version_costs[vid]
                metrics = self.inference_metrics.get(vid, [])

                avg_latency = statistics.mean(m.latency_ms for m in metrics) if metrics else 0
                avg_throughput = statistics.mean(m.throughput_requests_per_second for m in metrics) if metrics else 0

                comparison[vid] = {
                    "total_cost_usd": summary["total_cost_usd"],
                    "total_inferences": summary["total_inferences"],
                    "average_cost_per_inference": summary["average_cost_per_inference"],
                    "average_latency_ms": avg_latency,
                    "average_throughput_rps": avg_throughput,
                    "cost_per_request_per_second": (
                        summary["average_cost_per_inference"] / avg_throughput
                        if avg_throughput > 0 else 0
                    ),
                }

        return comparison

    def get_cheapest_version(self, model_id: str) -> Optional[str]:
        """Get cheapest version of a model"""
        candidates = {
            vid: cost["average_cost_per_inference"]
            for vid, cost in self.version_costs.items()
            if cost["model_id"] == model_id
        }

        if not candidates:
            return None

        return min(candidates, key=candidates.get)

    def get_fastest_version(self, model_id: str) -> Optional[str]:
        """Get fastest version of a model"""
        candidates = {}

        for vid in self.inference_metrics:
            cost_info = self.version_costs.get(vid)
            if cost_info and cost_info["model_id"] == model_id:
                metrics = self.inference_metrics[vid]
                avg_latency = statistics.mean(m.latency_ms for m in metrics)
                candidates[vid] = avg_latency

        if not candidates:
            return None

        return min(candidates, key=candidates.get)

    def recommend_version(
        self,
        model_id: str,
        weight_cost: float = 0.5,
        weight_speed: float = 0.5,
    ) -> Optional[Tuple[str, float]]:
        """
        Recommend version based on cost and speed trade-off.
        Returns (version_id, score)
        """
        comparison = self.compare_versions(
            [vid for vid in self.version_costs if self.version_costs[vid]["model_id"] == model_id]
        )

        if not comparison:
            return None

        # Normalize and score
        costs = [v["average_cost_per_inference"] for v in comparison.values()]
        latencies = [v["average_latency_ms"] for v in comparison.values()]

        min_cost = min(costs) if costs else 1
        min_latency = min(latencies) if latencies else 1

        scores = {}
        for vid, metrics in comparison.items():
            cost_score = min_cost / metrics["average_cost_per_inference"] if metrics["average_cost_per_inference"] > 0 else 0
            speed_score = min_latency / metrics["average_latency_ms"] if metrics["average_latency_ms"] > 0 else 0

            combined_score = (weight_cost * cost_score) + (weight_speed * speed_score)
            scores[vid] = combined_score

        best_version = max(scores, key=scores.get)
        return best_version, scores[best_version]


class PerformanceBenchmark:
    """Benchmark model versions against each other"""

    def __init__(self):
        self.benchmarks: Dict[str, Dict[str, Any]] = {}
        self.results: Dict[str, List[Dict[str, Any]]] = {}

    def run_benchmark(
        self,
        version_ids: List[str],
        test_inputs: List[Any],
        metric_fn: callable,
        name: str = "benchmark",
    ) -> Dict[str, Dict[str, float]]:
        """
        Run benchmark comparing versions.

        Args:
            version_ids: Versions to benchmark
            test_inputs: Test inputs for inference
            metric_fn: Function that runs inference and returns metrics
            name: Benchmark name

        Returns:
            {version_id: {metric_name: value}}
        """
        results = {}

        for vid in version_ids:
            logger.info(f"Benchmarking {vid}...")
            latencies = []
            errors = []
            throughputs = []

            for test_input in test_inputs:
                try:
                    start = datetime.now()
                    output = metric_fn(vid, test_input)
                    latency = (datetime.now() - start).total_seconds() * 1000

                    latencies.append(latency)
                    if output.get("error"):
                        errors.append(1)
                    else:
                        errors.append(0)

                    throughput = 1000 / latency if latency > 0 else 0
                    throughputs.append(throughput)

                except Exception as e:
                    logger.error(f"Benchmark error for {vid}: {e}")
                    errors.append(1)

            # Calculate statistics
            results[vid] = {
                "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
                "p50_latency_ms": statistics.median(latencies) if latencies else 0,
                "p95_latency_ms": self._percentile(latencies, 0.95) if latencies else 0,
                "p99_latency_ms": self._percentile(latencies, 0.99) if latencies else 0,
                "min_latency_ms": min(latencies) if latencies else 0,
                "max_latency_ms": max(latencies) if latencies else 0,
                "throughput_rps": statistics.mean(throughputs) if throughputs else 0,
                "error_rate_percent": (sum(errors) / len(errors) * 100) if errors else 0,
            }

        self.benchmarks[name] = results
        return results

    def compare_benchmarks(self, name1: str, name2: str) -> Dict[str, Dict[str, float]]:
        """Compare two benchmarks"""
        if name1 not in self.benchmarks or name2 not in self.benchmarks:
            return {}

        bench1 = self.benchmarks[name1]
        bench2 = self.benchmarks[name2]

        comparison = {}
        for vid in bench1:
            if vid in bench2:
                comparison[vid] = {
                    "latency_change_percent": (
                        (bench2[vid]["avg_latency_ms"] - bench1[vid]["avg_latency_ms"]) /
                        bench1[vid]["avg_latency_ms"] * 100
                    ) if bench1[vid]["avg_latency_ms"] > 0 else 0,
                    "throughput_change_percent": (
                        (bench2[vid]["throughput_rps"] - bench1[vid]["throughput_rps"]) /
                        bench1[vid]["throughput_rps"] * 100
                    ) if bench1[vid]["throughput_rps"] > 0 else 0,
                }

        return comparison

    def get_winner(self, benchmark_name: str, metric: str = "avg_latency_ms") -> Optional[str]:
        """Get best version in benchmark"""
        if benchmark_name not in self.benchmarks:
            return None

        results = self.benchmarks[benchmark_name]
        metric_values = {vid: r.get(metric, float('inf')) for vid, r in results.items()}

        if metric in ["error_rate_percent"]:
            # Lower is better
            return min(metric_values, key=metric_values.get)
        else:
            # For latency, lower is better
            if "latency" in metric:
                return min(metric_values, key=metric_values.get)
            # For throughput, higher is better
            else:
                return max(metric_values, key=metric_values.get)

    def statistical_significance_test(
        self,
        version1_metrics: List[float],
        version2_metrics: List[float],
        alpha: float = 0.05,
    ) -> Tuple[bool, float]:
        """
        Test if difference between versions is statistically significant.

        Returns:
            (is_significant, p_value)
        """
        try:
            from scipy import stats
        except ImportError:
            logger.warning("scipy required for statistical tests")
            return False, 1.0

        # T-test
        t_stat, p_value = stats.ttest_ind(version1_metrics, version2_metrics)

        is_significant = p_value < alpha
        return is_significant, p_value

    @staticmethod
    def _percentile(data: List[float], percentile: float) -> float:
        """Calculate percentile"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        return sorted_data[min(index, len(sorted_data) - 1)]
