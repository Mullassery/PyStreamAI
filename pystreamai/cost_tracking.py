"""Cost Tracking and Budget Management"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


@dataclass
class CostMetric:
    """Cost per inference request"""
    request_id: str
    model_id: str
    gpu_type: str
    latency_ms: float
    batch_size: int
    input_tokens: int
    output_tokens: int
    timestamp: datetime = field(default_factory=datetime.now)

    def calculate_cost(self, pricing: "Pricing") -> float:
        """Calculate cost for this request"""
        # GPU cost (per hour → per ms)
        gpu_cost_per_ms = pricing.get_gpu_cost_per_ms(self.gpu_type)
        compute_cost = gpu_cost_per_ms * self.latency_ms

        # Token cost (LLM only)
        token_cost = pricing.get_token_cost(self.input_tokens, self.output_tokens)

        total = compute_cost + token_cost
        return total / self.batch_size  # Cost per sample


class Pricing:
    """Pricing configuration"""

    # GPU pricing (USD per hour)
    GPU_PRICING = {
        "H100": 3.18,      # $3.18/hr on AWS
        "A100": 1.69,      # $1.69/hr on AWS
        "L4": 0.35,        # $0.35/hr on AWS
        "V100": 0.90,      # $0.90/hr on AWS
        "T4": 0.35,        # $0.35/hr on AWS
        "RTX4090": 0.50,   # Approximate
    }

    # Token pricing (USD per 1M tokens)
    TOKEN_PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "claude-2": {"input": 0.008, "output": 0.024},
        "llama-2": {"input": 0.0000, "output": 0.0000},  # OSS, free
    }

    def get_gpu_cost_per_ms(self, gpu_type: str) -> float:
        """Get GPU cost per millisecond"""
        hourly_rate = self.GPU_PRICING.get(gpu_type, 1.0)
        cost_per_ms = hourly_rate / (3600 * 1000)  # Convert hour to ms
        return cost_per_ms

    def get_token_cost(self, input_tokens: int, output_tokens: int, model: str = "llama-2") -> float:
        """Get token cost"""
        if model not in self.TOKEN_PRICING:
            return 0.0

        pricing = self.TOKEN_PRICING[model]
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost


class CostTracker:
    """Track and manage costs"""

    def __init__(self, pricing: Optional[Pricing] = None):
        self.pricing = pricing or Pricing()
        self.metrics: List[CostMetric] = []
        self.daily_costs: Dict[str, float] = {}
        self.model_costs: Dict[str, float] = {}

    def record_inference(self, metric: CostMetric) -> float:
        """Record inference and calculate cost"""
        cost = metric.calculate_cost(self.pricing)

        self.metrics.append(metric)

        # Track daily cost
        date_key = metric.timestamp.strftime("%Y-%m-%d")
        self.daily_costs[date_key] = self.daily_costs.get(date_key, 0) + cost

        # Track model cost
        self.model_costs[metric.model_id] = self.model_costs.get(metric.model_id, 0) + cost

        return cost

    def get_total_cost(self) -> float:
        """Get total cost"""
        return sum(self.daily_costs.values())

    def get_daily_cost(self, date: Optional[str] = None) -> float:
        """Get cost for a specific day"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        return self.daily_costs.get(date, 0.0)

    def get_model_cost(self, model_id: str) -> float:
        """Get total cost for a model"""
        return self.model_costs.get(model_id, 0.0)

    def get_cost_breakdown(self) -> Dict[str, Any]:
        """Get detailed cost breakdown"""
        return {
            "total_cost_usd": self.get_total_cost(),
            "daily_costs": self.daily_costs,
            "model_costs": self.model_costs,
            "num_requests": len(self.metrics),
            "avg_cost_per_request": self.get_total_cost() / len(self.metrics) if self.metrics else 0,
        }


class SpendManager:
    """Manage spending with alerts and caps"""

    def __init__(self, tracker: CostTracker, monthly_budget: float = 1000.0):
        self.tracker = tracker
        self.monthly_budget = monthly_budget
        self.alert_threshold = monthly_budget * 0.8  # Alert at 80%
        self.blocked = False

    def should_allow_request(self, model_id: str) -> bool:
        """Check if request should be allowed based on spend"""
        if self.blocked:
            logger.warning("Spending blocked - monthly budget exceeded")
            return False

        total_cost = self.tracker.get_total_cost()

        if total_cost > self.monthly_budget:
            logger.error(f"Monthly budget exceeded: ${total_cost:.2f} > ${self.monthly_budget:.2f}")
            self.blocked = True
            return False

        if total_cost > self.alert_threshold:
            logger.warning(
                f"Approaching budget limit: ${total_cost:.2f} / ${self.monthly_budget:.2f} "
                f"({(total_cost/self.monthly_budget)*100:.1f}%)"
            )

        return True

    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status"""
        total_cost = self.tracker.get_total_cost()
        remaining = self.monthly_budget - total_cost
        percent_used = (total_cost / self.monthly_budget) * 100

        return {
            "monthly_budget_usd": self.monthly_budget,
            "total_spent_usd": total_cost,
            "remaining_usd": max(0, remaining),
            "percent_used": percent_used,
            "is_blocked": self.blocked,
            "requests_processed": len(self.tracker.metrics),
        }

    def reset_monthly_budget(self) -> None:
        """Reset for new month"""
        self.blocked = False
        self.tracker.daily_costs.clear()
        self.tracker.metrics.clear()
        logger.info("Monthly budget reset")


class CostOptimizer:
    """Recommend cost optimizations"""

    def __init__(self, tracker: CostTracker):
        self.tracker = tracker

    def get_recommendations(self) -> List[Dict[str, str]]:
        """Get cost optimization recommendations"""
        recommendations = []

        # Analyze GPU usage
        gpu_costs = {}
        for metric in self.tracker.metrics:
            gpu_costs[metric.gpu_type] = gpu_costs.get(metric.gpu_type, 0) + 1

        # Recommend cheaper GPU if most requests use expensive GPU
        if "H100" in gpu_costs and gpu_costs["H100"] > 10:
            recommendations.append({
                "type": "gpu_downgrade",
                "message": "Consider using A100 or L4 instead of H100 - lower cost, similar performance",
                "potential_savings": "60-80%",
            })

        # Recommend batching
        avg_batch_size = sum(m.batch_size for m in self.tracker.metrics) / len(self.tracker.metrics)
        if avg_batch_size < 4:
            recommendations.append({
                "type": "batching",
                "message": "Increase batch size to reduce per-sample compute cost",
                "potential_savings": "30-50%",
            })

        # Recommend caching
        model_requests = {}
        for metric in self.tracker.metrics:
            model_requests[metric.model_id] = model_requests.get(metric.model_id, 0) + 1

        if max(model_requests.values(), default=0) > 100:
            recommendations.append({
                "type": "caching",
                "message": "Enable model caching to reduce repeated load times",
                "potential_savings": "10-20%",
            })

        return recommendations
