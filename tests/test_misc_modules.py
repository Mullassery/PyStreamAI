"""Smoke tests for the remaining restored modules: advanced_caching,
dashboard, edge_deployment, llm_optimization, observability, onnx_runtime.

These modules are mostly self-contained calculators/bookkeeping utilities
(several - e.g. ModelQuantizer, InferenceOptimizationPlan - return fixed
placeholder numbers rather than measuring a real model; see README). Tests
here check the real, deterministic control-flow/bookkeeping logic.
"""

import pytest

from pystreamai.advanced_caching import ResultCache, EmbeddingCache
from pystreamai.dashboard import MetricsCollector, Dashboard
from pystreamai.edge_deployment import EdgeDeviceSpec, EdgeDevice, ModelQuantizer
from pystreamai.llm_optimization import PromptCache, PagedAttention
from pystreamai.observability import OpenTelemetryBackend
from pystreamai.monitoring import InferenceMetric


class TestResultCache:
    def test_set_then_get_returns_cached_value(self):
        cache = ResultCache()
        cache.set("m1", {"x": 1}, result={"y": 2})
        assert cache.get("m1", {"x": 1}) == {"y": 2}

    def test_get_miss_returns_none(self):
        cache = ResultCache()
        assert cache.get("m1", {"x": 1}) is None

    def test_invalidate_removes_only_matching_model(self):
        # Regression test: invalidate() used to check `model_id in k` where
        # k is an MD5 hash of the cache key - that substring check never
        # matched, so invalidation was a silent no-op. Fixed to track keys
        # per model_id explicitly.
        cache = ResultCache()
        cache.set("m1", {"x": 1}, result="r1")
        cache.set("m2", {"x": 1}, result="r2")

        cache.invalidate("m1")

        assert cache.get("m1", {"x": 1}) is None
        assert cache.get("m2", {"x": 1}) == "r2"

    def test_get_stats_empty_cache(self):
        assert ResultCache().get_stats() == {"entries": 0, "hit_rate": 0, "total_size_mb": 0}


class TestEmbeddingCache:
    def test_set_then_get_returns_embedding(self):
        cache = EmbeddingCache(embedding_model=None)
        cache.set("hello", [0.1, 0.2, 0.3])
        assert cache.get("hello") == [0.1, 0.2, 0.3]

    def test_stats_report_cached_count(self):
        cache = EmbeddingCache(embedding_model=None)
        cache.set("a", [0.1])
        cache.set("b", [0.2])
        assert cache.get_stats()["cached_embeddings"] == 2


class TestDashboard:
    def test_metrics_collector_summary_after_requests(self):
        collector = MetricsCollector()
        collector.record_request("m1", latency_ms=10.0, batch_size=1, cost_usd=0.01)
        collector.record_request("m1", latency_ms=20.0, batch_size=1, cost_usd=0.02)

        summary = collector.get_summary()
        assert summary["total_requests"] == 2
        assert summary["avg_latency_ms"] == pytest.approx(15.0)
        assert summary["models"]["m1"]["requests"] == 2

    def test_dashboard_generates_alert_on_high_error_rate(self):
        collector = MetricsCollector()
        for _ in range(10):
            collector.record_request("m1", latency_ms=10.0, batch_size=1, cost_usd=0.0)
        for _ in range(2):
            collector.record_error("boom")

        dashboard = Dashboard(collector)
        data = dashboard.get_dashboard_data()

        alert_messages = [a["message"] for a in data["alerts"]]
        assert any("error rate" in m for m in alert_messages)

    def test_dashboard_no_alerts_when_healthy(self):
        collector = MetricsCollector()
        collector.record_request("m1", latency_ms=5.0, batch_size=1, cost_usd=0.0)

        dashboard = Dashboard(collector)
        data = dashboard.get_dashboard_data()
        assert data["alerts"] == []


class TestEdgeDeviceSpec:
    def test_jetson_orin_has_gpu(self):
        spec = EdgeDeviceSpec.specs(EdgeDevice.JETSON_ORIN)
        assert spec.gpu is True
        assert spec.ram_mb == 16384

    def test_esp32_has_tiny_model_budget(self):
        spec = EdgeDeviceSpec.specs(EdgeDevice.ESP32)
        assert spec.max_model_size_mb == 1


class TestModelQuantizer:
    def test_int4_is_smaller_than_int8(self):
        int8 = ModelQuantizer.quantize_int8("m.onnx", "out")
        int4 = ModelQuantizer.quantize_int4("m.onnx", "out")
        assert int4["quantized_size_mb"] < int8["quantized_size_mb"]
        assert int4["compression_ratio"] > int8["compression_ratio"]


class TestPromptCache:
    def test_lookup_miss_then_hit(self):
        cache = PromptCache()
        assert cache.lookup("hello world") is None

        cache.store("hello world", embeddings={}, tokens=2)
        assert cache.lookup("hello world") is not None

    def test_hit_rate_tracks_hits_and_misses(self):
        cache = PromptCache()
        cache.lookup("miss")  # miss
        cache.store("hit", embeddings={}, tokens=1)
        cache.lookup("hit")  # hit

        assert cache.get_hit_rate() == pytest.approx(50.0)


class TestPagedAttention:
    def test_allocate_and_deallocate_pages(self):
        paged = PagedAttention(page_size_tokens=16, num_pages=10)
        pages = paged.allocate_pages("req1", num_tokens=32)

        assert len(pages) == 2
        assert paged.get_memory_usage()["used_pages"] == 2

        paged.deallocate_pages("req1")
        assert paged.get_memory_usage()["used_pages"] == 0

    def test_allocate_more_than_available_returns_partial(self):
        paged = PagedAttention(page_size_tokens=16, num_pages=1)
        pages = paged.allocate_pages("req1", num_tokens=1000)
        assert len(pages) == 1  # capped by num_pages


class TestOpenTelemetryBackend:
    def test_initializes_without_opentelemetry_installed(self):
        # opentelemetry is not a declared dependency; must degrade gracefully.
        backend = OpenTelemetryBackend(service_name="test")
        assert backend.meter is None

    def test_log_metric_without_meter_does_not_raise(self):
        backend = OpenTelemetryBackend(service_name="test")
        metric = InferenceMetric(
            request_id="r1",
            model_id="m1",
            latency_ms=1.0,
            tokens=1,
            cost_usd=0.0,
            optimization_type="none",
        )
        backend.log_metric(metric)  # should no-op, not raise
