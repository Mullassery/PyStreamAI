"""Tests for pystreamai.model_registry - MLflow/HuggingFace integrations.

mlflow/transformers/huggingface_hub are optional (see pyproject.toml
[project.optional-dependencies].registry) and are not installed in the
base test environment, so these tests exercise the documented
graceful-degradation path when those libraries are absent.
"""

from pystreamai.model_registry import MLflowRegistry, ModelTracker, HuggingFaceRegistry


class TestMLflowRegistryWithoutMLflowInstalled:
    def test_initialize_without_mlflow_leaves_mlflow_none(self):
        registry = MLflowRegistry()
        assert registry.mlflow is None

    def test_start_experiment_falls_back_to_local(self):
        registry = MLflowRegistry()
        assert registry.start_experiment("exp1") == "local"

    def test_log_model_falls_back_to_local(self):
        registry = MLflowRegistry()
        assert registry.log_model(model=object(), model_name="m1") == "local"

    def test_log_params_and_log_metrics_do_not_raise(self):
        registry = MLflowRegistry()
        registry.log_params({"lr": 0.001})  # should no-op silently
        registry.log_metrics({"accuracy": 0.9})

    def test_load_model_returns_none(self):
        registry = MLflowRegistry()
        assert registry.load_model("runs:/x/model") is None

    def test_get_model_versions_returns_empty_list(self):
        registry = MLflowRegistry()
        assert registry.get_model_versions("m1") == []


class TestModelTracker:
    def test_start_training_without_mlflow_does_not_raise(self):
        tracker = ModelTracker(experiment_name="test-exp")
        tracker.start_training("m1", {"lr": 0.01})  # no-op without mlflow
        assert tracker.experiment_id == "local"

    def test_end_training_without_mlflow_does_not_raise(self):
        tracker = ModelTracker(experiment_name="test-exp")
        tracker.end_training(model=object(), model_name="m1")


class TestHuggingFaceRegistry:
    def test_download_model_without_transformers_returns_none(self):
        # transformers is not installed in this environment; the function
        # must catch ImportError rather than propagate it.
        assert HuggingFaceRegistry.download_model("bert-base-uncased") is None

    def test_upload_model_without_huggingface_hub_returns_false(self):
        assert HuggingFaceRegistry.upload_model(object(), "m1", "repo") is False
