"""Tests for pystreamai.serving - async batching/inference engine.

The inference itself is simulated (asyncio.sleep standing in for real model
compute - see pystreamai/serving.py:_run_batch_inference and the README's
"How this works today" note), so these tests check the batching, caching
and stats bookkeeping rather than real inference correctness or latency.
"""

import asyncio

import pytest

from pystreamai.serving import BatchQueue, InferenceRequest, ModelCache, InferenceServer


def make_request(request_id="r1"):
    return InferenceRequest(request_id=request_id, model_id="m1", input_data={"x": 1})


class TestBatchQueue:
    @pytest.mark.asyncio
    async def test_should_flush_false_when_empty(self):
        queue = BatchQueue()
        assert await queue.should_flush() is False

    @pytest.mark.asyncio
    async def test_should_flush_true_when_batch_full(self):
        queue = BatchQueue(max_batch_size=2)
        await queue.add(make_request("r1"))
        await queue.add(make_request("r2"))
        assert await queue.should_flush() is True

    @pytest.mark.asyncio
    async def test_flush_clears_queue_and_returns_batch(self):
        queue = BatchQueue(max_batch_size=2)
        await queue.add(make_request("r1"))

        batch = await queue.flush()

        assert len(batch) == 1
        assert await queue.size() == 0

    @pytest.mark.asyncio
    async def test_should_flush_true_after_max_wait_elapsed(self):
        queue = BatchQueue(max_batch_size=100, max_wait_ms=1)
        await queue.add(make_request("r1"))
        await asyncio.sleep(0.01)
        assert await queue.should_flush() is True


class TestModelCache:
    @pytest.mark.asyncio
    async def test_set_then_get_returns_cached_model(self):
        cache = ModelCache()
        await cache.set("m1", object())
        assert await cache.has("m1") is True

    @pytest.mark.asyncio
    async def test_get_missing_model_returns_none(self):
        cache = ModelCache()
        assert await cache.get("missing") is None


class TestInferenceServer:
    @pytest.mark.asyncio
    async def test_predict_returns_response_dict(self):
        server = InferenceServer(gpu_type="A100", num_gpus=1)
        result = await server.predict("m1", {"x": 1})

        assert result["model_id"] == "m1"
        assert result["latency_ms"] > 0
        assert "request_id" in result

    @pytest.mark.asyncio
    async def test_get_stats_before_any_requests(self):
        server = InferenceServer()
        stats = server.get_stats()
        assert stats == {"requests": 0, "avg_latency_ms": 0}

    @pytest.mark.asyncio
    async def test_get_stats_after_requests_aggregates_latency(self):
        server = InferenceServer()
        await server.predict("m1", {"x": 1})
        await server.predict("m1", {"x": 2})

        stats = server.get_stats()
        assert stats["requests"] == 2
        assert stats["avg_latency_ms"] > 0
        assert stats["total_cost_usd"] > 0

    def test_health_check_reports_healthy(self):
        server = InferenceServer()
        health = server.health_check()
        assert health["status"] == "healthy"
        assert health["requests_processed"] == 0
