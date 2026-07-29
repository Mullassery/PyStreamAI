"""OpenTelemetry Integration - Standard observability"""

import logging
from typing import Dict, Any, Optional
from .monitoring import MetricBackend, InferenceMetric

logger = logging.getLogger(__name__)


class OpenTelemetryBackend(MetricBackend):
    """OpenTelemetry backend for metrics, traces, and logs"""

    def __init__(self, service_name: str = "pystreamai"):
        self.service_name = service_name
        self.tracer = None
        self.meter = None
        self.logger_provider = None

        self._initialize()

    def _initialize(self):
        """Initialize OpenTelemetry"""
        try:
            from opentelemetry import trace, metrics
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
            from opentelemetry.exporter.jaeger.thrift import JaegerExporter
            from opentelemetry.exporter.prometheus import PrometheusMetricReader

            # Jaeger exporter for traces
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=6831,
            )

            trace_provider = TracerProvider()
            trace_provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
            trace.set_tracer_provider(trace_provider)

            # Prometheus exporter for metrics
            prometheus_reader = PrometheusMetricReader(
                prefix="pystreamai_"
            )
            meter_provider = MeterProvider(metric_readers=[prometheus_reader])
            metrics.set_meter_provider(meter_provider)

            self.tracer = trace.get_tracer(__name__)
            self.meter = metrics.get_meter(__name__)

            logger.info("OpenTelemetry initialized (Jaeger + Prometheus)")
        except ImportError:
            logger.warning(
                "OpenTelemetry not installed. "
                "Install: pip install opentelemetry-api opentelemetry-sdk "
                "opentelemetry-exporter-jaeger opentelemetry-exporter-prometheus"
            )

    def log_metric(self, metric: InferenceMetric) -> None:
        """Log metric via OpenTelemetry"""
        if not self.meter:
            return

        try:
            # Record latency histogram
            latency_counter = self.meter.create_histogram(
                name="inference_latency_ms",
                description="Inference latency in milliseconds",
                unit="ms",
            )
            latency_counter.record(metric.latency_ms, {
                "model": metric.model_id,
                "optimization": metric.optimization_type,
            })

            # Record cost counter
            cost_counter = self.meter.create_counter(
                name="inference_cost_usd",
                description="Total inference cost in USD",
                unit="usd",
            )
            cost_counter.add(metric.cost_usd, {
                "model": metric.model_id,
            })

            # Record speedup gauge
            speedup_gauge = self.meter.create_observable_gauge(
                name="inference_speedup",
                description="Speedup vs baseline",
            )
            # Note: Observable gauges need callbacks, this is simplified
        except Exception as e:
            logger.error(f"Failed to record metric: {e}")

    def close(self) -> None:
        """Clean up resources"""
        pass


class PrometheusBackend(MetricBackend):
    """Prometheus metrics backend"""

    def __init__(self, port: int = 8001):
        self.port = port
        self._initialize()

    def _initialize(self):
        """Initialize Prometheus"""
        try:
            from prometheus_client import Counter, Histogram, Gauge, start_http_server

            self.latency_histogram = Histogram(
                "pystreamai_inference_latency_ms",
                "Inference latency in milliseconds",
                ["model_id", "optimization_type"],
                buckets=(10, 50, 100, 200, 500, 1000),
            )

            self.cost_counter = Counter(
                "pystreamai_cost_usd_total",
                "Total inference cost in USD",
                ["model_id"],
            )

            self.batch_size_histogram = Histogram(
                "pystreamai_batch_size",
                "Inference batch size",
                ["model_id"],
                buckets=(1, 2, 4, 8, 16, 32, 64),
            )

            self.speedup_gauge = Gauge(
                "pystreamai_speedup_vs_baseline",
                "Speedup compared to baseline",
                ["model_id"],
            )

            # Start Prometheus HTTP server
            start_http_server(self.port)
            logger.info(f"Prometheus metrics server started on port {self.port}")
        except ImportError:
            logger.warning(
                "prometheus-client not installed. "
                "Install: pip install prometheus-client"
            )

    def log_metric(self, metric: InferenceMetric) -> None:
        """Log metric to Prometheus"""
        try:
            self.latency_histogram.labels(
                model_id=metric.model_id,
                optimization_type=metric.optimization_type,
            ).observe(metric.latency_ms)

            self.cost_counter.labels(
                model_id=metric.model_id,
            ).inc(metric.cost_usd)

            self.batch_size_histogram.labels(
                model_id=metric.model_id,
            ).observe(metric.batch_size)

            self.speedup_gauge.labels(
                model_id=metric.model_id,
            ).set(metric.speedup_vs_baseline)
        except Exception as e:
            logger.error(f"Failed to log to Prometheus: {e}")

    def close(self) -> None:
        pass


class DatadogBackend(MetricBackend):
    """Datadog metrics backend"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._initialize()

    def _initialize(self):
        """Initialize Datadog"""
        try:
            from datadog import initialize, api

            options = {
                "api_key": self.api_key,
                "app_key": self.api_key,
            }
            initialize(**options)
            logger.info("Datadog backend initialized")
        except ImportError:
            logger.warning(
                "datadog not installed. "
                "Install: pip install datadog"
            )

    def log_metric(self, metric: InferenceMetric) -> None:
        """Log metric to Datadog"""
        try:
            from datadog import api
            import time

            api.Metric.send(
                metric=f"pystreamai.inference.latency",
                points=[(int(time.time()), metric.latency_ms)],
                tags=[
                    f"model:{metric.model_id}",
                    f"optimization:{metric.optimization_type}",
                    f"batch_size:{metric.batch_size}",
                ],
            )
        except Exception as e:
            logger.error(f"Failed to log to Datadog: {e}")

    def close(self) -> None:
        pass
