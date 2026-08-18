"""PyStreamAI Production Inference Server"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from .platform import Endpoint

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Single inference request"""
    request_id: str
    model_id: str
    input_data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    priority: int = 1  # 1-5, higher = more important
    timeout_ms: float = 30000.0  # 30 seconds


@dataclass
class InferenceResponse:
    """Inference response"""
    request_id: str
    model_id: str
    output: Any
    latency_ms: float
    batch_size: int
    gpu_id: Optional[int] = None
    cost_usd: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model_id": self.model_id,
            "output": self.output,
            "latency_ms": self.latency_ms,
            "batch_size": self.batch_size,
            "gpu_id": self.gpu_id,
            "cost_usd": self.cost_usd,
        }


class BatchQueue:
    """Collects requests into batches"""

    def __init__(self, max_batch_size: int = 32, max_wait_ms: float = 100.0):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue: List[InferenceRequest] = []
        self.first_request_time: Optional[float] = None
        # Created lazily on first async use (not here in __init__), so
        # constructing a BatchQueue doesn't require a running event loop --
        # asyncio.Lock() binds to "the current loop" at construction time
        # in older asyncio versions, which breaks if this object is built
        # outside any loop, or after a prior asyncio.run() call reset the
        # process's event loop policy.
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def add(self, request: InferenceRequest) -> None:
        """Add request to queue"""
        async with self._get_lock():
            if not self.queue:
                self.first_request_time = time.time()
            self.queue.append(request)

    async def should_flush(self) -> bool:
        """Check if batch should be flushed"""
        async with self._get_lock():
            if not self.queue:
                return False

            # Flush if batch is full
            if len(self.queue) >= self.max_batch_size:
                return True

            # Flush if max wait time exceeded
            if self.first_request_time:
                elapsed_ms = (time.time() - self.first_request_time) * 1000
                if elapsed_ms >= self.max_wait_ms:
                    return True

            return False

    async def flush(self) -> List[InferenceRequest]:
        """Get batch and clear queue"""
        async with self._get_lock():
            batch = self.queue.copy()
            self.queue = []
            self.first_request_time = None
            return batch

    async def size(self) -> int:
        """Get current queue size"""
        async with self._get_lock():
            return len(self.queue)


class ModelCache:
    """Cache of real, servable models, keyed by model_id.

    `set()` wraps `model` in a `pystreamai.platform.Endpoint`, which
    validates it has a real way to run inference (ONNX file, `.predict()`
    method, or is callable) and raises immediately if not -- see
    `Endpoint`'s docstring. There's no other code path to get a model
    into this cache, so anything you can `get()` back out is guaranteed
    to be real and predictable.
    """

    def __init__(self):
        self.models: Dict[str, Endpoint] = {}
        # See BatchQueue._get_lock for why this is lazy, not created here.
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def get(self, model_id: str) -> Optional[Endpoint]:
        """Get a model's Endpoint from cache, or None if not loaded."""
        async with self._get_lock():
            return self.models.get(model_id)

    async def set(self, model_id: str, model: Any) -> None:
        """Load a real model into the cache.

        Raises TypeError immediately (via Endpoint's constructor) if
        `model` has no real way to run inference.
        """
        endpoint = Endpoint(model_id=model_id, replicas=1, model=model)
        async with self._get_lock():
            self.models[model_id] = endpoint
            logger.info(f"Cached model: {model_id}")

    async def has(self, model_id: str) -> bool:
        """Check if model is cached"""
        async with self._get_lock():
            return model_id in self.models


class InferenceEngine:
    """Core inference execution engine"""

    def __init__(self, gpu_type: str = "A100", num_gpus: int = 1):
        self.gpu_type = gpu_type
        self.num_gpus = num_gpus
        self.model_cache = ModelCache()
        self.batch_queue = BatchQueue()
        self.responses: Dict[str, InferenceResponse] = {}
        # Populated by whichever concurrent infer() call actually performs
        # a given flush -- see infer()'s docstring for why this exists.
        self._results: Dict[str, tuple] = {}

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Run real inference on request.

        When multiple infer() calls run concurrently (e.g. via
        asyncio.gather), they all add themselves to the same batch queue,
        but only one of them will actually see should_flush() succeed and
        call flush() -- the rest would flush an already-empty queue and
        poll should_flush() forever if they only checked the queue.
        Instead, whichever call performs the flush computes results for
        the *whole* batch (which may include other callers' requests) and
        publishes them to self._results; every call -- including ones
        that didn't perform the flush -- waits on its own request_id
        appearing there.

        Raises RuntimeError if no model is loaded for request.model_id
        (see InferenceServer.load_model) -- there is no fabricated
        fallback response.
        """
        start_time = time.time()

        await self.batch_queue.add(request)

        while request.request_id not in self._results:
            if await self.batch_queue.should_flush():
                batch = await self.batch_queue.flush()
                if batch:
                    batch_results = await self._run_batch_inference(batch)
                    self._results.update(batch_results)
            else:
                await asyncio.sleep(0.01)  # Check every 10ms

        output, error, batch_size = self._results.pop(request.request_id)

        total_latency_ms = (time.time() - start_time) * 1000

        if error is not None:
            raise RuntimeError(f"Inference failed for model {request.model_id!r}: {error}")

        # gpu_id/cost_usd are not implemented: there's no real GPU
        # scheduler or pricing table backing them (see README), so they
        # report None/0.0 rather than a fabricated number.
        return InferenceResponse(
            request_id=request.request_id,
            model_id=request.model_id,
            output=output,
            latency_ms=total_latency_ms,
            batch_size=batch_size,
            gpu_id=None,
            cost_usd=0.0,
        )

    async def _run_batch_inference(self, batch: List[InferenceRequest]) -> Dict[str, tuple]:
        """Run real inference for every request in the batch.

        Requests are grouped by model_id and each group is predicted
        against its real cached Endpoint. Returns {request_id: (output,
        error, batch_size)} -- error is None on success, or a message
        string if no model was loaded for that request's model_id or
        predict() raised. batch_size is len(batch) for every request in
        it, real for all of them, not just whichever call performed the
        flush.
        """
        results: Dict[str, tuple] = {}
        batch_size = len(batch)

        by_model: Dict[str, List[InferenceRequest]] = {}
        for req in batch:
            by_model.setdefault(req.model_id, []).append(req)

        for model_id, requests in by_model.items():
            endpoint = await self.model_cache.get(model_id)
            if endpoint is None:
                error = (
                    f"no model loaded for model_id={model_id!r} -- "
                    "call InferenceServer.load_model(model_id, model) first"
                )
                for req in requests:
                    results[req.request_id] = (None, error, batch_size)
                continue

            for req in requests:
                try:
                    result = endpoint.predict(req.input_data)
                    results[req.request_id] = (result["output"], None, batch_size)
                except Exception as e:
                    results[req.request_id] = (None, f"{type(e).__name__}: {e}", batch_size)

        return results


class InferenceServer:
    """Inference server: real request batching/timing, and -- once you've
    called `load_model()` -- real per-request inference against the
    models you loaded. `gpu_type`/`num_gpus` are accepted but have no
    effect; there is no GPU scheduling in this package."""

    def __init__(self, gpu_type: str = "A100", num_gpus: int = 1):
        self.engine = InferenceEngine(gpu_type, num_gpus)
        self.request_log: List[Dict[str, Any]] = []
        self.start_time = time.time()

    async def load_model(self, model_id: str, model: Any) -> None:
        """Load a real model so predict(model_id, ...) can serve it.

        `model` must be a real .onnx path/ONNXModelLoader, have a
        callable `.predict()`, or be callable itself -- see
        `pystreamai.platform.Endpoint`'s docstring. Raises TypeError
        immediately otherwise.
        """
        await self.engine.model_cache.set(model_id, model)

    async def predict(self, model_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run real inference. Raises RuntimeError if load_model(model_id, ...)
        wasn't called first."""
        request = InferenceRequest(
            request_id=str(uuid.uuid4()),
            model_id=model_id,
            input_data=input_data,
        )

        # Run inference
        response = await self.engine.infer(request)

        # Log request
        self.request_log.append({
            "request_id": response.request_id,
            "model_id": response.model_id,
            "latency_ms": response.latency_ms,
            "batch_size": response.batch_size,
            "cost_usd": response.cost_usd,
            "timestamp": datetime.now().isoformat(),
        })

        return response.to_dict()

    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics.

        Always returns the same six fields (zeroed out before any
        requests) -- a previous version returned a two-key dict in the
        empty case, which didn't match pystreamai.api's StatsResponse
        schema and made GET /stats a real 500 error before any request
        had been made.
        """
        if not self.request_log:
            return {
                "requests": 0,
                "avg_latency_ms": 0.0,
                "min_latency_ms": 0.0,
                "max_latency_ms": 0.0,
                "total_cost_usd": 0.0,
                "uptime_seconds": time.time() - self.start_time,
            }

        latencies = [r["latency_ms"] for r in self.request_log]
        total_cost = sum(r["cost_usd"] for r in self.request_log)

        return {
            "requests": len(self.request_log),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "total_cost_usd": total_cost,
            "uptime_seconds": time.time() - self.start_time,
        }

    def health_check(self) -> Dict[str, Any]:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "uptime_seconds": time.time() - self.start_time,
            "requests_processed": len(self.request_log),
        }
