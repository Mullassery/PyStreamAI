"""Tests for pystreamai.platform and pystreamai.decorators.

Note: Platform.train()/serve() and Endpoint.predict() in this pure-Python
layer are intentionally lightweight simulations (no real training/inference
happens) - see README "How this works today" section. These tests verify
the simulated control-flow and bookkeeping behave as documented, not that
real ML work occurs.
"""

from pathlib import Path

import pytest

from pystreamai.platform import Platform, Endpoint, TrainingJob, get_platform, set_platform
from pystreamai import decorators


class TestPlatform:
    def test_train_creates_job(self):
        platform = Platform(backend="local")
        job = platform.train(code="train.py", dataset="data.csv")

        assert isinstance(job, TrainingJob)
        assert job.status == "running"
        assert job.job_id in platform.jobs

    def test_train_job_wait_returns_path(self):
        platform = Platform()
        job = platform.train(code="train.py", dataset="data.csv", model_id="my-model")
        result = job.wait()

        assert isinstance(result, Path)
        assert "my-model" in str(result)

    def test_train_job_cancel(self):
        platform = Platform()
        job = platform.train(code="train.py", dataset="data.csv")
        job.cancel()

        assert job.status == "cancelled"

    def test_serve_creates_endpoint(self):
        platform = Platform()
        endpoint = platform.serve(model="my-model.pkl", replicas=2, model_id="svc")

        assert isinstance(endpoint, Endpoint)
        assert endpoint.replicas == 2
        assert "svc" in platform.deployments

    def test_endpoint_predict_without_gpu(self):
        endpoint = Endpoint(model_id="m1", replicas=1)
        result = endpoint.predict({"x": 1})

        assert result["model"] == "m1"
        assert result["latency_ms"] == pytest.approx(42.5)
        assert endpoint.gpu_optimizer is None

    def test_endpoint_predict_with_gpu_is_faster_than_without(self):
        endpoint = Endpoint(model_id="m1", replicas=1, gpu="A100")
        result = endpoint.predict({"x": 1})

        assert endpoint.gpu_optimizer is not None
        assert result["latency_ms"] < 42.5

    def test_endpoint_stop(self):
        endpoint = Endpoint(model_id="m1", replicas=1)
        assert endpoint.status == "running"
        endpoint.stop()
        assert endpoint.status == "stopped"

    def test_list_deployments_and_jobs(self):
        platform = Platform()
        platform.train(code="t.py", dataset="d.csv", model_id="j1")
        platform.serve(model="m", model_id="d1")

        assert "j1" not in platform.list_deployments()  # jobs != deployments
        assert "d1" in platform.list_deployments()
        assert len(platform.list_jobs()) == 1


class TestGlobalPlatform:
    def test_get_platform_returns_singleton(self):
        p1 = get_platform()
        p2 = get_platform()
        assert p1 is p2

    def test_set_platform_replaces_singleton(self):
        original = get_platform()
        new_platform = set_platform(backend="aws")
        try:
            assert new_platform is get_platform()
            assert new_platform is not original
            assert new_platform.backend == "aws"
        finally:
            set_platform(backend="local")  # restore for other tests


class TestDecorators:
    def test_train_decorator_returns_job(self):
        set_platform(backend="local")

        @decorators.train(gpu="A100")
        def make_model(x):
            return {"weights": x}

        job = make_model(42)
        assert isinstance(job, TrainingJob)

    def test_serve_decorator_wraps_predict(self):
        set_platform(backend="local")

        @decorators.serve(replicas=1)
        def predict(x):
            return x * 2

        endpoint = predict(21)
        assert isinstance(endpoint, Endpoint)
        assert endpoint.predict(21) == 42

    def test_pipeline_decorator_runs_and_returns_value(self, capsys):
        @decorators.pipeline(name="my-pipeline")
        def workflow():
            return "done"

        result = workflow()
        assert result == "done"
        captured = capsys.readouterr()
        assert "my-pipeline" in captured.out
