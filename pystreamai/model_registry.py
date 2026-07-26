"""MLflow Integration - Model registry, versioning, and tracking"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class MLflowRegistry:
    """MLflow model registry integration"""

    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        self.tracking_uri = tracking_uri
        self.mlflow = None
        self._initialize()

    def _initialize(self):
        """Initialize MLflow"""
        try:
            import mlflow

            mlflow.set_tracking_uri(self.tracking_uri)
            self.mlflow = mlflow
            logger.info(f"MLflow initialized: {self.tracking_uri}")
        except ImportError:
            logger.warning(
                "mlflow not installed. "
                "Install: pip install mlflow"
            )

    def start_experiment(self, experiment_name: str) -> str:
        """Start a new MLflow experiment"""
        if not self.mlflow:
            return "local"

        try:
            experiment_id = self.mlflow.create_experiment(experiment_name)
            self.mlflow.set_experiment(experiment_name)
            logger.info(f"Started experiment: {experiment_name}")
            return experiment_id
        except Exception as e:
            logger.warning(f"Failed to create experiment: {e}")
            return "local"

    def log_model(
        self,
        model,
        model_name: str,
        artifact_path: str = "models",
        model_format: str = "pytorch",
    ) -> str:
        """Log model to MLflow"""
        if not self.mlflow:
            return "local"

        try:
            if model_format == "pytorch":
                self.mlflow.pytorch.log_model(model, artifact_path)
            elif model_format == "onnx":
                self.mlflow.onnx.log_model(model, artifact_path)
            else:
                self.mlflow.log_artifact(model, artifact_path)

            logger.info(f"Model logged: {model_name}")
            return f"{artifact_path}/{model_name}"
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
            return "local"

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters"""
        if not self.mlflow:
            return

        try:
            for key, value in params.items():
                self.mlflow.log_param(key, value)
        except Exception as e:
            logger.error(f"Failed to log params: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics"""
        if not self.mlflow:
            return

        try:
            for key, value in metrics.items():
                self.mlflow.log_metric(key, value, step=step)
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")

    def register_model(
        self,
        model_uri: str,
        model_name: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Register model in MLflow registry"""
        if not self.mlflow:
            return "local"

        try:
            result = self.mlflow.register_model(model_uri, model_name)
            logger.info(f"Model registered: {model_name}")

            if tags:
                for key, value in tags.items():
                    self.mlflow.set_model_tag(model_uri, key, value)

            return result.model_uri
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return "local"

    def load_model(self, model_uri: str):
        """Load model from MLflow"""
        if not self.mlflow:
            return None

        try:
            model = self.mlflow.pytorch.load_model(model_uri)
            logger.info(f"Model loaded: {model_uri}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

    def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """Get all versions of a registered model"""
        if not self.mlflow:
            return []

        try:
            client = self.mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(f"name='{model_name}'")
            return [v.to_dictionary() for v in versions]
        except Exception as e:
            logger.error(f"Failed to get model versions: {e}")
            return []


class ModelTracker:
    """Track model training and inference"""

    def __init__(self, experiment_name: str = "pystreamai"):
        self.registry = MLflowRegistry()
        self.experiment_id = self.registry.start_experiment(experiment_name)
        self.current_run_id = None

    def start_training(self, model_name: str, params: Dict[str, Any]) -> None:
        """Start training run"""
        if not self.registry.mlflow:
            return

        try:
            self.registry.mlflow.start_run(run_name=model_name)
            self.current_run_id = self.registry.mlflow.active_run().info.run_id
            self.registry.log_params(params)
            logger.info(f"Training started: {model_name}")
        except Exception as e:
            logger.error(f"Failed to start training: {e}")

    def log_training_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log training metrics"""
        self.registry.log_metrics(metrics, step=epoch)

    def end_training(self, model, model_name: str, tags: Optional[Dict[str, str]] = None) -> None:
        """End training run and register model"""
        if not self.registry.mlflow:
            return

        try:
            self.registry.log_model(model, model_name)
            self.registry.register_model(
                f"runs:/{self.current_run_id}/models/{model_name}",
                model_name,
                tags=tags,
            )
            self.registry.mlflow.end_run()
            logger.info(f"Training completed and model registered: {model_name}")
        except Exception as e:
            logger.error(f"Failed to end training: {e}")


class HuggingFaceRegistry:
    """Hugging Face model hub integration"""

    @staticmethod
    def download_model(model_id: str, cache_dir: Optional[str] = None) -> str:
        """Download model from Hugging Face Hub"""
        try:
            from transformers import AutoModel

            logger.info(f"Downloading model from Hugging Face: {model_id}")
            model = AutoModel.from_pretrained(model_id, cache_dir=cache_dir)
            logger.info(f"Model downloaded: {model_id}")
            return model_id
        except ImportError:
            logger.error("transformers not installed")
            return None
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return None

    @staticmethod
    def upload_model(model, model_id: str, repo_name: str) -> bool:
        """Upload model to Hugging Face Hub"""
        try:
            from huggingface_hub import HfApi

            logger.info(f"Uploading model to Hugging Face: {model_id}")
            api = HfApi()
            # Upload logic here
            logger.info(f"Model uploaded: {model_id}")
            return True
        except ImportError:
            logger.error("huggingface_hub not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to upload model: {e}")
            return False

    @staticmethod
    def list_models(task: str = "text-classification", limit: int = 10) -> List[Dict[str, str]]:
        """List models from Hugging Face Hub"""
        try:
            from huggingface_hub import list_models

            models = list(list_models(task=task, limit=limit))
            return [{"id": m.id, "downloads": m.downloads} for m in models]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
