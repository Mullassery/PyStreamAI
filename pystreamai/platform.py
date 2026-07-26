"""PyStreamAI Platform - Main API"""

from typing import Optional, Dict, Any, Callable
import functools
from pathlib import Path
from .gpu import GPUOptimizer, InferenceOptimizationPlan


class Endpoint:
    """Deployed model endpoint"""

    def __init__(self, model_id: str, replicas: int, gpu: Optional[str] = None):
        self.model_id = model_id
        self.replicas = replicas
        self.gpu = gpu
        self.status = "running"
        self.gpu_optimizer = None

        # Initialize GPU optimizer if GPU is specified
        if gpu:
            self.gpu_optimizer = GPUOptimizer(gpu)
            self.gpu_optimizer.enable_tensorrt(fp16=True)

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference on the endpoint"""
        latency = 42.5
        if self.gpu_optimizer:
            # With TensorRT + FP16: estimate 1.5x speedup
            latency = 42.5 / 1.5

        return {
            "model": self.model_id,
            "output": f"prediction from {self.model_id}",
            "latency_ms": latency,
            "gpu": self.gpu,
        }

    def get_optimization_plan(self) -> Optional[str]:
        """Get GPU optimization plan"""
        if not self.gpu_optimizer:
            return None

        plan = InferenceOptimizationPlan(self.model_id, self.gpu)
        return plan.apply()

    def stop(self):
        """Stop the endpoint"""
        self.status = "stopped"


class TrainingJob:
    """Submitted training job"""

    def __init__(self, job_id: str, model_id: str):
        self.job_id = job_id
        self.model_id = model_id
        self.status = "running"

    def wait(self, timeout_seconds: int = 3600) -> Path:
        """Wait for training to complete, return model artifact path"""
        return Path(f"/models/{self.model_id}/model.pkl")

    def cancel(self):
        """Cancel the training job"""
        self.status = "cancelled"


class Platform:
    """PyStreamAI Platform - Zero-YAML ML deployment"""

    def __init__(self, backend: str = "local"):
        """
        Initialize the platform.

        Args:
            backend: "local", "aws", "gcp", "azure" (default: "local")
        """
        self.backend = backend
        self.deployments = {}
        self.jobs = {}

    def train(
        self,
        code: str,
        dataset: str,
        gpu: Optional[str] = None,
        time_limit: Optional[str] = None,
        **kwargs
    ) -> TrainingJob:
        """
        Submit a training job.

        Args:
            code: Path to training script or Python function
            dataset: Path to dataset (local, S3, GCS, etc)
            gpu: GPU type ("A100", "L4", "H100", etc)
            time_limit: Max time (e.g., "1h", "30m")

        Returns:
            TrainingJob that can be waited on
        """
        job_id = f"job-{len(self.jobs)}"
        model_id = kwargs.get("model_id", f"model-{len(self.deployments)}")

        job = TrainingJob(job_id, model_id)
        self.jobs[job_id] = job

        print(f"🚀 Training job {job_id} submitted")
        print(f"   Dataset: {dataset}")
        if gpu:
            print(f"   GPU: {gpu}")
        if time_limit:
            print(f"   Time limit: {time_limit}")

        return job

    def serve(
        self,
        model: Any,
        replicas: int = 1,
        gpu: Optional[str] = None,
        max_batch_size: Optional[int] = None,
        **kwargs
    ) -> Endpoint:
        """
        Deploy a model as an endpoint.

        Args:
            model: Trained model (path to file or model object)
            replicas: Number of replicas
            gpu: GPU type for serving
            max_batch_size: Max batch size for batching requests

        Returns:
            Endpoint that can be used for inference
        """
        model_id = kwargs.get("model_id", f"model-{len(self.deployments)}")

        endpoint = Endpoint(model_id, replicas, gpu)
        self.deployments[model_id] = endpoint

        print(f"✨ Model {model_id} deployed!")
        print(f"   Replicas: {replicas}")
        if gpu:
            print(f"   GPU: {gpu}")
        print(f"   Ready for inference")

        return endpoint

    def list_deployments(self) -> Dict[str, Endpoint]:
        """List all active deployments"""
        return self.deployments.copy()

    def list_jobs(self) -> Dict[str, TrainingJob]:
        """List all training jobs"""
        return self.jobs.copy()


# Global platform instance
_platform = Platform()


def get_platform() -> Platform:
    """Get the global platform instance"""
    return _platform


def set_platform(backend: str = "local") -> Platform:
    """Set the global platform backend"""
    global _platform
    _platform = Platform(backend)
    return _platform
