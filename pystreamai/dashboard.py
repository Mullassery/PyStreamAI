"""Real-time Monitoring Dashboard"""

import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System-wide metrics"""
    total_requests: int
    total_errors: int
    avg_latency_ms: float
    p99_latency_ms: float
    throughput_req_sec: float
    total_cost_usd: float
    models_active: int
    gpu_utilization_percent: float
    memory_utilization_percent: float


class MetricsCollector:
    """Collect and aggregate metrics"""

    def __init__(self, window_seconds: int = 300):
        self.window_seconds = window_seconds
        self.metrics_history: List[Dict[str, Any]] = []
        self.latencies: List[float] = []
        self.errors: List[str] = []
        self.model_metrics: Dict[str, Dict[str, Any]] = {}

    def record_request(
        self,
        model_id: str,
        latency_ms: float,
        batch_size: int,
        cost_usd: float,
    ) -> None:
        """Record request metrics"""
        self.latencies.append(latency_ms)

        if model_id not in self.model_metrics:
            self.model_metrics[model_id] = {
                "requests": 0,
                "total_latency": 0,
                "total_cost": 0,
                "min_latency": float("inf"),
                "max_latency": 0,
            }

        metrics = self.model_metrics[model_id]
        metrics["requests"] += 1
        metrics["total_latency"] += latency_ms
        metrics["total_cost"] += cost_usd
        metrics["min_latency"] = min(metrics["min_latency"], latency_ms)
        metrics["max_latency"] = max(metrics["max_latency"], latency_ms)

    def record_error(self, error: str) -> None:
        """Record error"""
        self.errors.append(error)

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        if not self.latencies:
            return {}

        sorted_latencies = sorted(self.latencies)
        p99_idx = int(len(sorted_latencies) * 0.99)
        p99 = sorted_latencies[p99_idx] if p99_idx < len(sorted_latencies) else 0

        total_requests = len(self.latencies)
        total_time_ms = sum(self.latencies)

        model_summaries = {}
        for model_id, metrics in self.model_metrics.items():
            if metrics["requests"] > 0:
                model_summaries[model_id] = {
                    "requests": metrics["requests"],
                    "avg_latency_ms": metrics["total_latency"] / metrics["requests"],
                    "min_latency_ms": metrics["min_latency"],
                    "max_latency_ms": metrics["max_latency"],
                    "total_cost_usd": metrics["total_cost"],
                }

        return {
            "total_requests": total_requests,
            "total_errors": len(self.errors),
            "error_rate": (len(self.errors) / total_requests * 100) if total_requests > 0 else 0,
            "avg_latency_ms": sum(self.latencies) / total_requests if total_requests > 0 else 0,
            "p99_latency_ms": p99,
            "min_latency_ms": min(self.latencies),
            "max_latency_ms": max(self.latencies),
            "models": model_summaries,
        }

    def clear_old_data(self) -> None:
        """Clear data older than window"""
        cutoff = datetime.now() - timedelta(seconds=self.window_seconds)
        # Simplified - would need timestamps on all metrics
        if len(self.latencies) > 10000:
            self.latencies = self.latencies[-5000:]


class Dashboard:
    """Real-time monitoring dashboard"""

    def __init__(self, metrics: MetricsCollector):
        self.metrics = metrics

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get all dashboard data"""
        summary = self.metrics.get_summary()

        return {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "alerts": self._generate_alerts(summary),
            "recommendations": self._generate_recommendations(summary),
        }

    def _generate_alerts(self, summary: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate alerts based on metrics"""
        alerts = []

        # High error rate
        if summary.get("error_rate", 0) > 5:
            alerts.append({
                "level": "warning",
                "message": f"High error rate: {summary['error_rate']:.1f}%",
            })

        # High latency
        if summary.get("avg_latency_ms", 0) > 500:
            alerts.append({
                "level": "warning",
                "message": f"High avg latency: {summary['avg_latency_ms']:.0f}ms",
            })

        # P99 latency too high
        if summary.get("p99_latency_ms", 0) > 1000:
            alerts.append({
                "level": "warning",
                "message": f"P99 latency high: {summary['p99_latency_ms']:.0f}ms",
            })

        return alerts

    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate recommendations"""
        recommendations = []

        # Low error rate, high throughput
        if summary.get("error_rate", 0) < 1 and summary.get("total_requests", 0) > 100:
            recommendations.append({
                "type": "scaling",
                "message": "Consider increasing batch size for better efficiency",
            })

        # Variable latency
        avg = summary.get("avg_latency_ms", 0)
        max_lat = summary.get("max_latency_ms", 0)
        if max_lat > avg * 3:
            recommendations.append({
                "type": "optimization",
                "message": "High latency variance detected - check for resource contention",
            })

        # Multiple models, uneven load
        models = summary.get("models", {})
        if len(models) > 1:
            loads = [m.get("requests", 0) for m in models.values()]
            if max(loads) > min(loads) * 2:
                recommendations.append({
                    "type": "balancing",
                    "message": "Uneven load across models - consider rebalancing",
                })

        return recommendations

    def to_json(self) -> str:
        """Export as JSON"""
        return json.dumps(self.get_dashboard_data(), indent=2)

    def print_summary(self) -> None:
        """Print dashboard summary"""
        data = self.get_dashboard_data()
        summary = data["summary"]

        print("\n" + "=" * 80)
        print("PyStreamAI MONITORING DASHBOARD")
        print("=" * 80)
        print(f"\nRequests: {summary.get('total_requests', 0)}")
        print(f"Errors: {summary.get('total_errors', 0)} ({summary.get('error_rate', 0):.1f}%)")
        print(f"\nLatency:")
        print(f"  Avg: {summary.get('avg_latency_ms', 0):.2f}ms")
        print(f"  P99: {summary.get('p99_latency_ms', 0):.2f}ms")
        print(f"  Min: {summary.get('min_latency_ms', 0):.2f}ms")
        print(f"  Max: {summary.get('max_latency_ms', 0):.2f}ms")

        if data["alerts"]:
            print(f"\nAlerts:")
            for alert in data["alerts"]:
                print(f"  [{alert['level'].upper()}] {alert['message']}")

        if data["recommendations"]:
            print(f"\nRecommendations:")
            for rec in data["recommendations"]:
                print(f"  [{rec['type']}] {rec['message']}")

        print("\n" + "=" * 80 + "\n")
