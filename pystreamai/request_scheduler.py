"""Request Scheduling and Prioritization"""

import logging
import heapq
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class RequestPriority(Enum):
    """Request priority levels"""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0


@dataclass(order=True)
class ScheduledRequest:
    """Request with priority and metadata"""
    priority: int
    request_id: str = field(compare=False)
    user_id: Optional[str] = field(default=None, compare=False)
    model_id: str = field(default="", compare=False)
    data: Dict[str, Any] = field(default_factory=dict, compare=False)
    timestamp: datetime = field(default_factory=datetime.now, compare=False)
    timeout_ms: float = field(default=30000, compare=False)
    estimated_compute_ms: float = field(default=10, compare=False)

    def is_expired(self) -> bool:
        """Check if request has timed out"""
        elapsed_ms = (datetime.now() - self.timestamp).total_seconds() * 1000
        return elapsed_ms > self.timeout_ms


class RequestScheduler:
    """Schedule and prioritize requests"""

    def __init__(self, max_queue_size: int = 10000):
        self.max_queue_size = max_queue_size
        self.queue: List[ScheduledRequest] = []
        self.request_counts: Dict[str, int] = {}
        self.user_quotas: Dict[str, int] = {}

    def enqueue(self, request: ScheduledRequest) -> bool:
        """Enqueue a request"""
        if len(self.queue) >= self.max_queue_size:
            logger.warning("Queue full, rejecting request")
            return False

        # Check user quota
        if request.user_id and not self._check_user_quota(request.user_id):
            logger.warning(f"User quota exceeded: {request.user_id}")
            return False

        heapq.heappush(self.queue, request)
        self.request_counts[request.model_id] = self.request_counts.get(request.model_id, 0) + 1

        return True

    def dequeue(self) -> Optional[ScheduledRequest]:
        """Dequeue highest priority request"""
        while self.queue:
            request = heapq.heappop(self.queue)

            # Skip expired requests
            if request.is_expired():
                logger.warning(f"Request expired: {request.request_id}")
                continue

            return request

        return None

    def peek(self) -> Optional[ScheduledRequest]:
        """Peek at highest priority request without dequeuing"""
        if self.queue:
            return self.queue[0]
        return None

    def size(self) -> int:
        """Get queue size"""
        return len(self.queue)

    def _check_user_quota(self, user_id: str) -> bool:
        """Check if user has exceeded quota"""
        quota = self.user_quotas.get(user_id, 100)  # Default: 100 req/min
        current = self.request_counts.get(f"user:{user_id}", 0)
        return current < quota

    def set_user_quota(self, user_id: str, requests_per_minute: int) -> None:
        """Set quota for user"""
        self.user_quotas[user_id] = requests_per_minute
        logger.info(f"Set quota for {user_id}: {requests_per_minute} req/min")

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        model_counts = {}
        priority_counts = {p.name: 0 for p in RequestPriority}

        for request in self.queue:
            model_counts[request.model_id] = model_counts.get(request.model_id, 0) + 1
            priority_counts[RequestPriority(request.priority).name] += 1

        return {
            "queue_size": len(self.queue),
            "by_model": model_counts,
            "by_priority": priority_counts,
            "max_queue_size": self.max_queue_size,
        }


class FairShareScheduler:
    """Fair-share scheduling across models"""

    def __init__(self, models: List[str]):
        self.models = models
        self.model_queues: Dict[str, List[ScheduledRequest]] = {
            model: [] for model in models
        }
        self.model_tokens: Dict[str, float] = {model: 1.0 for model in models}
        self.token_rate = 1.0  # Tokens per second per model

    def enqueue(self, request: ScheduledRequest) -> bool:
        """Enqueue request to model queue"""
        if request.model_id not in self.model_queues:
            logger.warning(f"Unknown model: {request.model_id}")
            return False

        heapq.heappush(self.model_queues[request.model_id], request)
        return True

    def dequeue_fair(self) -> Optional[tuple[ScheduledRequest, str]]:
        """Dequeue using fair-share algorithm"""
        # Model with most tokens gets to serve
        best_model = max(
            self.models,
            key=lambda m: self.model_tokens[m] if self.model_queues[m] else -1
        )

        if not self.model_queues[best_model]:
            return None

        request = heapq.heappop(self.model_queues[best_model])

        # Deduct tokens for this request
        self.model_tokens[best_model] -= request.estimated_compute_ms / 1000.0

        return request, best_model

    def refill_tokens(self) -> None:
        """Refill tokens periodically"""
        for model in self.models:
            self.model_tokens[model] += self.token_rate

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        return {
            "model_tokens": self.model_tokens,
            "queue_sizes": {
                model: len(self.model_queues[model])
                for model in self.models
            },
        }


class DeadlineScheduler:
    """Schedule requests by deadline"""

    def __init__(self):
        self.queue: List[ScheduledRequest] = []

    def enqueue(self, request: ScheduledRequest) -> None:
        """Enqueue by deadline (earliest deadline first)"""
        heapq.heappush(self.queue, (request.timeout_ms, request))

    def dequeue(self) -> Optional[ScheduledRequest]:
        """Dequeue earliest deadline request"""
        while self.queue:
            _, request = heapq.heappop(self.queue)

            if not request.is_expired():
                return request

        return None

    def size(self) -> int:
        """Get queue size"""
        return len(self.queue)
