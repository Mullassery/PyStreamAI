"""Tests for pystreamai.platform and pystreamai.decorators.

Platform.train() and Endpoint.predict() run real code: train() calls a
real callable (or runs a real training script as a subprocess) and
serve()/predict() call a real model's .predict()/__call__, or run real
ONNX inference (see tests/test_platform_onnx_predict.py for the ONNX
case). There is no simulated fallback anywhere in this module -- passing
something with no real way to run is a TypeError, not a fake response.
"""

from pathlib import Path

import pytest

from pystreamai.platform import Platform, Endpoint, TrainingJob, get_platform, set_platform
from pystreamai import decorators


REAL_TRAIN_SCRIPT = """\
import argparse, pickle

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

with open(args.dataset) as f:
    row_count = sum(1 for _ in f)

with open(args.output, "wb") as f:
    pickle.dump({"trained_on": args.dataset, "row_count": row_count}, f)
"""

FAILING_TRAIN_SCRIPT = """\
import sys
print("intentional failure for test", file=sys.stderr)
sys.exit(1)
"""


@pytest.fixture
def real_train_script(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(REAL_TRAIN_SCRIPT)
    return script


@pytest.fixture
def failing_train_script(tmp_path):
    script = tmp_path / "fail.py"
    script.write_text(FAILING_TRAIN_SCRIPT)
    return script


@pytest.fixture
def real_dataset(tmp_path):
    dataset = tmp_path / "data.csv"
    dataset.write_text("a,b\n1,2\n3,4\n")
    return dataset


class EchoModel:
    """A minimal real model: predict() just echoes its input, but it's a
    real method call, not a fabricated response."""

    def predict(self, data):
        return {"echoed": data}


class TestPlatform:
    def test_train_with_callable_runs_it_for_real(self):
        platform = Platform(backend="local")
        calls = []

        def train_fn(dataset):
            calls.append(dataset)
            return {"weights": [1, 2, 3]}

        job = platform.train(code=train_fn, dataset="some-dataset")

        assert calls == ["some-dataset"]
        assert isinstance(job, TrainingJob)
        assert job.status == "completed"
        assert job.job_id in platform.jobs

    def test_train_job_wait_returns_the_real_model_for_callable_code(self):
        platform = Platform()

        def train_fn(dataset):
            return {"weights": [1, 2, 3], "dataset": dataset}

        job = platform.train(code=train_fn, dataset="data.csv", model_id="my-model")
        result = job.wait()

        assert result == {"weights": [1, 2, 3], "dataset": "data.csv"}

    def test_train_with_real_script_runs_a_subprocess_and_writes_a_real_artifact(
        self, real_train_script, real_dataset
    ):
        platform = Platform()
        job = platform.train(code=str(real_train_script), dataset=str(real_dataset), model_id="my-model")

        artifact_path = job.wait()

        assert isinstance(artifact_path, Path)
        assert artifact_path.exists()
        assert "my-model" in str(artifact_path)

        import pickle
        with open(artifact_path, "rb") as f:
            artifact = pickle.load(f)
        assert artifact["row_count"] == 3  # header + 2 data rows

    def test_train_with_failing_callable_reports_real_failure(self):
        platform = Platform()

        def train_fn(dataset):
            raise ValueError("dataset is empty")

        job = platform.train(code=train_fn, dataset="data.csv")

        assert job.status == "failed"
        assert "dataset is empty" in job.error
        with pytest.raises(RuntimeError, match="dataset is empty"):
            job.wait()

    def test_train_with_failing_script_reports_real_failure(self, failing_train_script, real_dataset):
        platform = Platform()
        job = platform.train(code=str(failing_train_script), dataset=str(real_dataset))

        assert job.status == "failed"
        assert "intentional failure" in job.error

    def test_train_with_nonexistent_script_and_non_callable_raises_immediately(self):
        platform = Platform()
        with pytest.raises(TypeError):
            platform.train(code="does-not-exist.py", dataset="data.csv")

    def test_train_job_cancel(self):
        platform = Platform()
        job = platform.train(code=lambda dataset: "model", dataset="data.csv")
        job.cancel()

        assert job.status == "cancelled"

    def test_serve_with_real_model_creates_endpoint(self):
        platform = Platform()
        endpoint = platform.serve(model=EchoModel(), replicas=2, model_id="svc")

        assert isinstance(endpoint, Endpoint)
        assert endpoint.replicas == 2
        assert "svc" in platform.deployments

    def test_serve_with_no_real_way_to_run_inference_raises(self):
        platform = Platform()
        with pytest.raises(TypeError):
            platform.serve(model="just-a-string-path.pkl", replicas=1)
        with pytest.raises(TypeError):
            platform.serve(model=None, replicas=1)

    def test_endpoint_predict_calls_real_predict_method(self):
        endpoint = Endpoint(model_id="m1", replicas=1, model=EchoModel())
        result = endpoint.predict({"x": 1})

        assert result["model"] == "m1"
        assert result["output"] == {"echoed": {"x": 1}}
        assert result["latency_ms"] >= 0

    def test_endpoint_predict_calls_real_callable_model(self):
        endpoint = Endpoint(model_id="m1", replicas=1, model=lambda data: {"doubled": data["x"] * 2})
        result = endpoint.predict({"x": 21})

        assert result["output"] == {"doubled": 42}

    def test_endpoint_stop(self):
        endpoint = Endpoint(model_id="m1", replicas=1, model=EchoModel())
        assert endpoint.status == "running"
        endpoint.stop()
        assert endpoint.status == "stopped"

    def test_list_deployments_and_jobs(self):
        platform = Platform()
        platform.train(code=lambda dataset: "model", dataset="d.csv", model_id="j1")
        platform.serve(model=EchoModel(), model_id="d1")

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
        new_platform = set_platform(backend="local")
        try:
            assert new_platform is get_platform()
            assert new_platform is not original
            assert new_platform.backend == "local"
        finally:
            set_platform(backend="local")  # restore for other tests

    def test_set_platform_with_unimplemented_backend_raises(self):
        """backend="aws"/"gcp"/"azure" used to be silently accepted and
        stored with zero effect on behavior. There's no cloud-provisioning
        code in this package, so this must fail loudly instead."""
        with pytest.raises(NotImplementedError):
            set_platform(backend="aws")


class TestDecorators:
    def test_train_decorator_runs_the_function_for_real_exactly_once(self):
        set_platform(backend="local")
        calls = []

        @decorators.train(gpu="A100")
        def make_model(x):
            calls.append(x)
            return {"weights": x}

        job = make_model(42)
        assert isinstance(job, TrainingJob)
        assert calls == [42]  # called exactly once, not twice
        assert job.wait() == {"weights": 42}

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
