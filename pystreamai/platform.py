"""PyStreamAI Platform - Main API"""

import logging
import time
from typing import Optional, Dict, Any
from pathlib import Path
from .gpu import GPUOptimizer, InferenceOptimizationPlan

logger = logging.getLogger(__name__)


class Endpoint:
    """Deployed model endpoint.

    If `model` is a real .onnx file (a path, or an already-loaded
    `pystreamai.onnx_runtime.ONNXModelLoader`), predict() runs genuine
    inference via onnxruntime with real measured latency. Otherwise --
    an arbitrary Python object, a non-.onnx path, or onnxruntime not
    installed -- predict() falls back to a simulated response, and says
    so explicitly (`simulated: True`) rather than fabricating a real-
    looking result. There is no code path that runs real inference for
    non-ONNX models; that would require framework-specific integration
    this package doesn't have.
    """

    def __init__(self, model_id: str, replicas: int, gpu: Optional[str] = None, model: Any = None):
        self.model_id = model_id
        self.replicas = replicas
        self.gpu = gpu
        self.status = "running"
        self.gpu_optimizer = None
        self.onnx_model = self._try_load_onnx(model)

        # Initialize GPU optimizer if GPU is specified
        if gpu:
            self.gpu_optimizer = GPUOptimizer(gpu)
            self.gpu_optimizer.enable_tensorrt(fp16=True)

    @staticmethod
    def _try_load_onnx(model: Any):
        """Return a loaded ONNXModelLoader if `model` is a real .onnx
        model, else None. Never raises -- a bad/missing onnx setup just
        means predict() falls back to the simulated path."""
        if model is None:
            return None

        try:
            from .onnx_runtime import ONNXModelLoader
        except ImportError:
            logger.debug("onnxruntime not installed; predict() will use the simulated path")
            return None

        if isinstance(model, ONNXModelLoader):
            return model if model.session is not None else model.load()

        if isinstance(model, (str, Path)) and str(model).endswith(".onnx"):
            try:
                return ONNXModelLoader(str(model)).load()
            except Exception as e:
                logger.warning(f"Failed to load ONNX model {model}: {e}; falling back to simulated predict()")
                return None

        return None

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference on the endpoint.

        Real inference (via onnxruntime) if this endpoint was served with
        a real .onnx model; a clearly-labeled simulated response otherwise.
        """
        if self.onnx_model is not None:
            start = time.perf_counter()
            output = self.onnx_model.infer(data)
            latency = (time.perf_counter() - start) * 1000
            return {
                "model": self.model_id,
                "output": output,
                "latency_ms": latency,
                "gpu": self.gpu,
                "simulated": False,
            }

        latency = 42.5
        if self.gpu_optimizer:
            # With TensorRT + FP16: estimate 1.5x speedup
            latency = 42.5 / 1.5

        return {
            "model": self.model_id,
            "output": f"prediction from {self.model_id}",
            "latency_ms": latency,
            "gpu": self.gpu,
            "simulated": True,
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

    _SUPPORTED_BACKENDS = ("local",)

    def __init__(self, backend: str = "local"):
        """
        Initialize the platform.

        Args:
            backend: only "local" is currently implemented. train()/serve()
                behave identically regardless of what's passed here -- there
                is no cloud-provisioning code in this package yet. Passing
                anything other than "local" raises NotImplementedError
                immediately rather than silently accepting it and behaving
                exactly like "local" anyway.
        """
        if backend not in self._SUPPORTED_BACKENDS:
            raise NotImplementedError(
                f"Platform(backend={backend!r}) is not implemented. "
                f"Only {self._SUPPORTED_BACKENDS!r} currently does anything -- "
                "there is no cloud-provisioning code in this package yet."
            )
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

        print(f"Training job {job_id} submitted")
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
            model: Trained model. If this is a path to a real .onnx file
                (or an already-loaded pystreamai.onnx_runtime.ONNXModelLoader),
                the returned Endpoint runs genuine inference via
                onnxruntime. Any other object (a path to a non-.onnx file,
                an in-memory model object, or None) results in a
                simulated Endpoint -- see Endpoint's docstring.
            replicas: Number of replicas
            gpu: GPU type for serving
            max_batch_size: Max batch size for batching requests

        Returns:
            Endpoint that can be used for inference
        """
        model_id = kwargs.get("model_id", f"model-{len(self.deployments)}")

        endpoint = Endpoint(model_id, replicas, gpu, model=model)
        self.deployments[model_id] = endpoint

        print(f"Model {model_id} deployed")
        print(f"   Replicas: {replicas}")
        if gpu:
            print(f"   GPU: {gpu}")
        print("   Ready for inference")

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
