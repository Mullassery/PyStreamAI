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


class TestHuggingFaceUpload:
    def _fake_hub(self, monkeypatch):
        import sys
        import types

        calls = []

        class FakeApi:
            def __init__(self, token=None):
                calls.append(("init", token))

            def create_repo(self, repo_id, exist_ok=False, private=False):
                calls.append(("create_repo", repo_id, exist_ok, private))

            def upload_folder(self, folder_path, repo_id, token=None):
                calls.append(("upload_folder", folder_path, repo_id, token))

            def upload_file(self, path_or_fileobj, path_in_repo, repo_id, token=None):
                calls.append(
                    ("upload_file", path_or_fileobj, path_in_repo, repo_id, token)
                )

        module = types.ModuleType("huggingface_hub")
        module.HfApi = FakeApi
        monkeypatch.setitem(sys.modules, "huggingface_hub", module)
        return calls

    def test_upload_local_directory_calls_create_repo_and_upload_folder(
        self, tmp_path, monkeypatch
    ):
        calls = self._fake_hub(monkeypatch)
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")

        assert (
            HuggingFaceRegistry.upload_model(
                str(model_dir), "local-model", "org/repo", token="t", private=True
            )
            is True
        )
        assert ("init", "t") in calls
        assert ("create_repo", "org/repo", True, True) in calls
        assert any(call[0] == "upload_folder" and call[2] == "org/repo" for call in calls)

    def test_upload_model_object_uses_save_pretrained(self, tmp_path, monkeypatch):
        calls = self._fake_hub(monkeypatch)

        class FakeModel:
            def save_pretrained(self, path):
                with open(f"{path}/weights.txt", "w") as handle:
                    handle.write("weights")

        assert HuggingFaceRegistry.upload_model(FakeModel(), "m1", "org/repo") is True
        assert any(call[0] == "upload_folder" for call in calls)

    def test_upload_missing_path_returns_false(self, monkeypatch):
        self._fake_hub(monkeypatch)
        assert (
            HuggingFaceRegistry.upload_model("/no/such/path", "m1", "org/repo")
            is False
        )
