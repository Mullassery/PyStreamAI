"""Phase 8: Framework & Ecosystem Integration - TensorFlow Serving, TorchServe, Kubernetes, HuggingFace"""

import logging
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Model metadata for framework detection"""
    framework: str  # tensorflow, pytorch, onnx, transformers
    model_path: str
    version: str
    input_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None
    requires: Optional[List[str]] = None  # Dependencies


class ModelServer(ABC):
    """Abstract base for model serving platforms"""

    @abstractmethod
    def deploy_version(self, version_id: str, model_path: str, metadata: ModelMetadata) -> bool:
        """Deploy model version to server"""
        pass

    @abstractmethod
    def get_active_version(self) -> Optional[str]:
        """Get currently active model version"""
        pass

    @abstractmethod
    def list_versions(self) -> List[str]:
        """List all deployed versions"""
        pass

    @abstractmethod
    def switch_version(self, version_id: str) -> bool:
        """Switch active version"""
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Check server health"""
        pass


class TensorFlowServingBackend(ModelServer):
    """TensorFlow Serving integration"""

    def __init__(self, server_url: str = "http://localhost:8501"):
        self.server_url = server_url
        self.deployed_versions: Dict[str, str] = {}
        self.active_version: Optional[str] = None

        try:
            import tensorflow_serving
            self.tf_serving = tensorflow_serving
        except ImportError:
            logger.warning("tensorflow-serving-api not installed")
            self.tf_serving = None

    def deploy_version(self, version_id: str, model_path: str, metadata: ModelMetadata) -> bool:
        """Deploy SavedModel to TensorFlow Serving"""
        try:
            import tensorflow as tf

            logger.info(f"Loading TensorFlow SavedModel: {model_path}")

            # Verify it's a SavedModel
            model = tf.saved_model.load(model_path)

            # Create version-specific path
            version_path = f"{model_path}/{version_id}"
            tf.saved_model.save(model, version_path)

            self.deployed_versions[version_id] = version_path
            logger.info(f"Deployed {version_id} to TensorFlow Serving")

            return True

        except Exception as e:
            logger.error(f"Failed to deploy to TensorFlow Serving: {e}")
            return False

    def get_active_version(self) -> Optional[str]:
        return self.active_version

    def list_versions(self) -> List[str]:
        return list(self.deployed_versions.keys())

    def switch_version(self, version_id: str) -> bool:
        """Switch to new version in TensorFlow Serving"""
        if version_id not in self.deployed_versions:
            logger.error(f"Version {version_id} not deployed")
            return False

        try:
            self.active_version = version_id
            logger.info(f"Switched to TensorFlow Serving version {version_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to switch version: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Check TensorFlow Serving health"""
        try:
            import requests
            response = requests.get(f"{self.server_url}/v1/models/default")

            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "framework": "tensorflow",
                    "server_url": self.server_url,
                }

            return {"status": "unhealthy", "error": response.text}

        except Exception as e:
            return {"status": "error", "error": str(e)}


class TorchServeBackend(ModelServer):
    """TorchServe integration"""

    def __init__(self, server_url: str = "http://localhost:8080"):
        self.server_url = server_url
        self.deployed_versions: Dict[str, str] = {}
        self.active_version: Optional[str] = None

        try:
            import requests
            self.requests = requests
        except ImportError:
            logger.warning("requests library not installed")

    def deploy_version(self, version_id: str, model_path: str, metadata: ModelMetadata) -> bool:
        """Deploy to TorchServe"""
        try:
            logger.info(f"Deploying PyTorch model to TorchServe: {model_path}")

            # Register model with TorchServe
            response = self.requests.post(
                f"{self.server_url}/models?url={model_path}&model_name={version_id}"
            )

            if response.status_code == 200:
                self.deployed_versions[version_id] = model_path
                logger.info(f"Deployed {version_id} to TorchServe")
                return True

            logger.error(f"TorchServe deployment failed: {response.text}")
            return False

        except Exception as e:
            logger.error(f"Failed to deploy to TorchServe: {e}")
            return False

    def get_active_version(self) -> Optional[str]:
        return self.active_version

    def list_versions(self) -> List[str]:
        try:
            response = self.requests.get(f"{self.server_url}/models")
            if response.status_code == 200:
                models = response.json()
                return [m["modelName"] for m in models]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")

        return list(self.deployed_versions.keys())

    def switch_version(self, version_id: str) -> bool:
        """Set default version in TorchServe"""
        if version_id not in self.deployed_versions:
            logger.error(f"Version {version_id} not deployed")
            return False

        try:
            # Update model version preference
            response = self.requests.put(
                f"{self.server_url}/models/{version_id}/set-default",
            )

            if response.status_code == 200:
                self.active_version = version_id
                logger.info(f"Switched to TorchServe version {version_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to switch version: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Check TorchServe health"""
        try:
            response = self.requests.get(f"{self.server_url}/ping")

            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "framework": "pytorch",
                    "server_url": self.server_url,
                }

            return {"status": "unhealthy"}

        except Exception as e:
            return {"status": "error", "error": str(e)}


class KubernetesDeployment(ModelServer):
    """Kubernetes native deployment"""

    def __init__(self, namespace: str = "default"):
        self.namespace = namespace
        self.deployed_versions: Dict[str, str] = {}
        self.active_version: Optional[str] = None

        try:
            from kubernetes import client, config
            config.load_incluster_config()  # Load from cluster
            self.k8s_client = client.AppsV1Api()
            self.core_api = client.CoreV1Api()
        except Exception:
            logger.warning("Kubernetes client not configured")
            self.k8s_client = None

    def deploy_version(self, version_id: str, model_path: str, metadata: ModelMetadata) -> bool:
        """Deploy model as Kubernetes deployment"""
        if not self.k8s_client:
            logger.error("Kubernetes client not initialized")
            return False

        try:
            from kubernetes import client

            # Create deployment spec
            deployment = client.V1Deployment(
                api_version="apps/v1",
                kind="Deployment",
                metadata=client.V1ObjectMeta(name=f"model-{version_id}"),
                spec=client.V1DeploymentSpec(
                    replicas=3,
                    selector=client.V1LabelSelector(
                        match_labels={"app": "model", "version": version_id}
                    ),
                    template=client.V1PodTemplateSpec(
                        metadata=client.V1ObjectMeta(
                            labels={"app": "model", "version": version_id}
                        ),
                        spec=client.V1PodSpec(
                            containers=[
                                client.V1Container(
                                    name="model-server",
                                    image=f"model-server:{version_id}",
                                    ports=[client.V1ContainerPort(container_port=8080)],
                                    env=[
                                        client.V1EnvVar(name="MODEL_PATH", value=model_path),
                                        client.V1EnvVar(name="MODEL_VERSION", value=version_id),
                                    ],
                                )
                            ]
                        ),
                    ),
                ),
            )

            # Create deployment
            self.k8s_client.create_namespaced_deployment(
                namespace=self.namespace,
                body=deployment,
            )

            self.deployed_versions[version_id] = model_path
            logger.info(f"Deployed {version_id} to Kubernetes")
            return True

        except Exception as e:
            logger.error(f"Failed to deploy to Kubernetes: {e}")
            return False

    def get_active_version(self) -> Optional[str]:
        return self.active_version

    def list_versions(self) -> List[str]:
        return list(self.deployed_versions.keys())

    def switch_version(self, version_id: str) -> bool:
        """Update service to route to version"""
        if version_id not in self.deployed_versions:
            logger.error(f"Version {version_id} not deployed")
            return False

        try:
            from kubernetes import client

            # Update service selector to route to new version
            service = self.core_api.read_namespaced_service(
                name="model-service",
                namespace=self.namespace,
            )

            service.spec.selector = {"app": "model", "version": version_id}

            self.core_api.patch_namespaced_service(
                name="model-service",
                namespace=self.namespace,
                body=service,
            )

            self.active_version = version_id
            logger.info(f"Switched Kubernetes service to {version_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to switch version: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Check Kubernetes deployment health"""
        if not self.k8s_client:
            return {"status": "error", "error": "Kubernetes client not initialized"}

        try:
            if not self.active_version:
                return {"status": "unhealthy", "error": "No active version"}

            deployment = self.k8s_client.read_namespaced_deployment(
                name=f"model-{self.active_version}",
                namespace=self.namespace,
            )

            ready_replicas = deployment.status.ready_replicas or 0
            desired_replicas = deployment.spec.replicas

            if ready_replicas >= desired_replicas:
                return {
                    "status": "healthy",
                    "platform": "kubernetes",
                    "version": self.active_version,
                    "ready_replicas": ready_replicas,
                    "desired_replicas": desired_replicas,
                }

            return {
                "status": "degraded",
                "ready_replicas": ready_replicas,
                "desired_replicas": desired_replicas,
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}


class HuggingFaceHub:
    """Hugging Face Model Hub integration"""

    def __init__(self, repo_id: str, cache_dir: Optional[str] = None):
        self.repo_id = repo_id
        self.cache_dir = cache_dir
        self.versions: Dict[str, Dict[str, Any]] = {}

        try:
            from huggingface_hub import model_info
            self.model_info = model_info
        except ImportError:
            logger.warning("huggingface-hub not installed")

    def list_versions(self) -> List[str]:
        """List all revisions of model on Hub"""
        try:
            info = self.model_info(self.repo_id)
            return [sibling.revision for sibling in info.siblings]
        except Exception as e:
            logger.error(f"Failed to list HF versions: {e}")
            return []

    def download_version(self, version: str = "main") -> Optional[str]:
        """Download specific version from Hub"""
        try:
            from huggingface_hub import hf_hub_download

            logger.info(f"Downloading {self.repo_id}@{version}")

            # Download model files
            model_path = hf_hub_download(
                repo_id=self.repo_id,
                revision=version,
                cache_dir=self.cache_dir,
            )

            self.versions[version] = {
                "path": model_path,
                "downloaded_at": datetime.now().isoformat(),
            }

            logger.info(f"Downloaded {version} to {model_path}")
            return model_path

        except Exception as e:
            logger.error(f"Failed to download from HF Hub: {e}")
            return None

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information from Hub"""
        try:
            info = self.model_info(self.repo_id)
            return {
                "repo_id": self.repo_id,
                "library_name": info.library_name,
                "tags": info.tags,
                "downloads": info.downloads,
                "likes": info.likes,
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {}


class FrameworkDetector:
    """Auto-detect model framework"""

    @staticmethod
    def detect_framework(model_path: str) -> str:
        """Detect framework from model file"""
        import os

        try:
            # Check file extensions and directories
            if os.path.isfile(model_path):
                if model_path.endswith(".pt"):
                    return "pytorch"
                elif model_path.endswith(".pb"):
                    return "tensorflow"
                elif model_path.endswith(".onnx"):
                    return "onnx"

            # Check for SavedModel structure
            if os.path.isdir(model_path):
                if os.path.exists(os.path.join(model_path, "saved_model.pb")):
                    return "tensorflow"
                elif os.path.exists(os.path.join(model_path, "model.pt")):
                    return "pytorch"

            # Try loading with different frameworks
            try:
                import tensorflow
                tensorflow.saved_model.load(model_path)
                return "tensorflow"
            except:
                pass

            try:
                import torch
                torch.load(model_path)
                return "pytorch"
            except:
                pass

            return "unknown"

        except Exception as e:
            logger.error(f"Failed to detect framework: {e}")
            return "unknown"

    @staticmethod
    def get_appropriate_server(framework: str, **kwargs) -> Optional[ModelServer]:
        """Get appropriate model server for framework"""
        if framework == "tensorflow":
            return TensorFlowServingBackend(**kwargs)
        elif framework == "pytorch":
            return TorchServeBackend(**kwargs)
        elif framework == "kubernetes":
            return KubernetesDeployment(**kwargs)
        else:
            logger.warning(f"No server for framework: {framework}")
            return None


from datetime import datetime
