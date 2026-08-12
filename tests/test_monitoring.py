"""Tests for pystreamai.monitoring - pluggable metric backends."""

from pystreamai.monitoring import (
    InferenceMetric,
    MetricBackend,
    MetricCollector,
    NoOpBackend,
    get_metrics,
    set_metric_backend,
    log_metric,
)


def make_metric(request_id="r1"):
    return InferenceMetric(
        request_id=request_id,
        model_id="m1",
        latency_ms=10.0,
        tokens=5,
        cost_usd=0.001,
        optimization_type="none",
    )


class RecordingBackend(MetricBackend):
    def __init__(self):
        self.logged = []
        self.closed = False

    def log_metric(self, metric):
        self.logged.append(metric)

    def close(self):
        self.closed = True


class TestInferenceMetric:
    def test_to_dict_round_trips_fields(self):
        metric = make_metric()
        d = metric.to_dict()
        assert d["request_id"] == "r1"
        assert d["model_id"] == "m1"
        assert d["batch_size"] == 1  # default


class TestNoOpBackend:
    def test_log_and_close_are_no_ops(self):
        backend = NoOpBackend()
        backend.log_metric(make_metric())  # should not raise
        backend.close()  # should not raise


class TestMetricCollector:
    def test_defaults_to_noop_backend(self):
        collector = MetricCollector()
        assert isinstance(collector.backend, NoOpBackend)

    def test_log_appends_and_forwards_to_backend(self):
        backend = RecordingBackend()
        collector = MetricCollector(backend=backend)
        metric = make_metric()

        collector.log(metric)

        assert collector.get_metrics() == [metric]
        assert backend.logged == [metric]

    def test_log_survives_backend_exception(self):
        class BrokenBackend(MetricBackend):
            def log_metric(self, metric):
                raise RuntimeError("boom")

            def close(self):
                pass

        collector = MetricCollector(backend=BrokenBackend())
        collector.log(make_metric())  # must not raise
        assert len(collector.get_metrics()) == 1

    def test_set_backend_closes_previous_backend(self):
        old_backend = RecordingBackend()
        new_backend = RecordingBackend()
        collector = MetricCollector(backend=old_backend)

        collector.set_backend(new_backend)

        assert old_backend.closed is True
        assert collector.backend is new_backend

    def test_get_metrics_returns_a_copy(self):
        collector = MetricCollector()
        collector.log(make_metric())
        metrics = collector.get_metrics()
        metrics.append(make_metric("intruder"))

        assert len(collector.get_metrics()) == 1


class TestModuleLevelHelpers:
    def test_log_metric_and_get_metrics_use_shared_global_collector(self):
        backend = RecordingBackend()
        set_metric_backend(backend)
        try:
            metric = make_metric("global-1")
            log_metric(metric)
            assert metric in get_metrics().get_metrics()
            assert metric in backend.logged
        finally:
            set_metric_backend(NoOpBackend())
