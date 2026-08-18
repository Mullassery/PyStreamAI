"""PyStreamAI Platform - Main API"""

import logging
import re
import subprocess
import sys
import time
from typing import Optional, Dict, Any
from pathlib import Path
from .gpu import GPUOptimizer, InferenceOptimizationPlan

logger = logging.getLogger(__name__)

_TIME_LIMIT_RE = re.compile(r"^(\d+)([smh])$")


def _parse_time_limit_seconds(time_limit: Optional[str]) -> Optional[float]:
    """Parse "30m"/"1h"/"90s" into seconds for subprocess timeout. Returns
    None (no timeout) for None or an unrecognized format -- an
    unenforceable time_limit shouldn't silently kill a real training run."""
    if not time_limit:
        return None
    match = _TIME_LIMIT_RE.match(time_limit.strip())
    if not match:
        return None
    value, unit = match.groups()
    return int(value) * {"s": 1, "m": 60, "h": 3600}[unit]


class Endpoint:
    """Deployed model endpoint. predict() runs real inference, trying in
    order:

    1. Real ONNX inference via onnxruntime, if `model` is a real .onnx
       file (a path, or an already-loaded
       `pystreamai.onnx_runtime.ONNXModelLoader`).
    2. Calling `model.predict(data)` for real, if `model` has a callable
       `.predict` attribute (the scikit-learn/most-frameworks convention).
    3. Calling `model(data)` for real, if `model` is itself callable (a
       plain function, or a callable object).

    There is no simulated/fake fallback: if `model` was `None`, or is a
    plain object with no `.onnx` file, no `.predict()`, and isn't
    callable, `serve()` raises `TypeError` immediately (see `serve()`'s
    docstring) rather than creating an Endpoint that would return a fake
    response later.
    """

    def __init__(self, model_id: str, replicas: int, gpu: Optional[str] = None, model: Any = None):
        self.model_id = model_id
        self.replicas = replicas
        self.gpu = gpu
        self.status = "running"
        self.gpu_optimizer = None
        self.model = model
        self.onnx_model = self._try_load_onnx(model)

        if self.onnx_model is None and not hasattr(model, "predict") and not callable(model):
            raise TypeError(
                f"serve(model={model!r}) has no real way to run inference: it's not a "
                "real .onnx file/ONNXModelLoader, has no callable .predict(), and isn't "
                "itself callable. Pass one of those, or wrap your model in a small "
                "callable/class with a .predict(data) method."
            )

        # Initialize GPU optimizer if GPU is specified
        if gpu:
            self.gpu_optimizer = GPUOptimizer(gpu)
            self.gpu_optimizer.enable_tensorrt(fp16=True)

    @staticmethod
    def _try_load_onnx(model: Any):
        """Return a loaded ONNXModelLoader if `model` is a real .onnx
        model, else None."""
        if model is None:
            return None

        try:
            from .onnx_runtime import ONNXModelLoader
        except ImportError:
            return None

        if isinstance(model, ONNXModelLoader):
            return model if model.session is not None else model.load()

        if isinstance(model, (str, Path)) and str(model).endswith(".onnx"):
            return ONNXModelLoader(str(model)).load()

        return None

    def predict(self, data: Any) -> Dict[str, Any]:
        """Run real inference on the endpoint: real ONNX inference, real
        `model.predict(data)`, or real `model(data)` -- whichever
        `serve()` determined this endpoint can actually do. There is no
        simulated path; construction already failed if none applied."""
        start = time.perf_counter()

        if self.onnx_model is not None:
            output = self.onnx_model.infer(data)
        elif hasattr(self.model, "predict"):
            output = self.model.predict(data)
        else:
            output = self.model(data)

        latency = (time.perf_counter() - start) * 1000

        return {
            "model": self.model_id,
            "output": output,
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
    """A training job. Platform.train() runs training synchronously (there
    is no background worker/queue), so by the time this object exists,
    training has already either produced a real result or failed.

    Exactly one of `model` / `artifact_path` is set on success:
    - `model` is set when `code` was a callable: the real object it
      returned.
    - `artifact_path` is set when `code` was a script path: the real
      filesystem path the script wrote its model artifact to.
    """

    def __init__(
        self,
        job_id: str,
        model_id: str,
        model: Any = None,
        artifact_path: Optional[Path] = None,
        error: Optional[str] = None,
    ):
        self.job_id = job_id
        self.model_id = model_id
        self.model = model
        self.artifact_path = artifact_path
        self.error = error
        if error:
            self.status = "failed"
        elif model is not None or artifact_path is not None:
            self.status = "completed"
        else:
            self.status = "running"

    def wait(self, timeout_seconds: int = 3600) -> Any:
        """Wait for training to complete.

        Training already ran synchronously by the time this job exists, so
        this returns immediately: the real trained model object if `code`
        was a callable, the real artifact `Path` if `code` was a script,
        or raises if training failed. `timeout_seconds` is accepted for
        API compatibility with an eventual async execution path but has no
        effect today.
        """
        if self.error:
            raise RuntimeError(f"Training job {self.job_id} failed: {self.error}")
        if self.artifact_path is not None:
            return self.artifact_path
        if self.model is not None:
            return self.model
        raise RuntimeError(
            f"Training job {self.job_id} has no result. This should not "
            "happen for a job that isn't in 'failed' status -- please report it."
        )

    def cancel(self):
        """Cancel the training job.

        Training already ran synchronously by the time this job exists
        (there is no background process to cancel) -- this just marks the
        job's status.
        """
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
        code,
        dataset: str,
        gpu: Optional[str] = None,
        time_limit: Optional[str] = None,
        **kwargs
    ) -> TrainingJob:
        """
        Run a training job synchronously (there is no queue/background
        worker -- by the time this returns, training has already run).

        `gpu` and `time_limit` are accepted but have no effect: there is
        no GPU scheduling or timeout enforcement in this package yet.

        Args:
            code: Either
                (a) a callable, called as `code(dataset)` -- its return
                    value is the real trained model, available via
                    `job.wait()` / `job.model`, or
                (b) a path to an existing, real training script -- run as
                    `python <code> --dataset <dataset> --output <artifact_path>`
                    in a subprocess. The script is responsible for actually
                    training a model and writing it to the `--output` path
                    it's given; if it doesn't, this reports a failed job
                    rather than fabricating a path.
                Anything else (a non-existent path, a non-callable object)
                raises TypeError immediately.
            dataset: Passed to `code` -- a path, or anything your callable
                expects as its first argument.
            gpu: GPU type, e.g. "A100" (accepted, currently has no effect).
            time_limit: e.g. "1h" (accepted, currently has no effect).

        Returns:
            TrainingJob wrapping the real result (or real failure).
        """
        job_id = f"job-{len(self.jobs)}"
        model_id = kwargs.get("model_id", f"model-{len(self.deployments)}")

        print(f"Training job {job_id} submitted")
        print(f"   Dataset: {dataset}")
        if gpu:
            print(f"   GPU: {gpu} (accepted, not currently used to schedule anything)")
        if time_limit:
            print(f"   Time limit: {time_limit} (accepted, not currently enforced)")

        model = None
        artifact_path = None
        error = None

        if callable(code):
            try:
                model = code(dataset)
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
        elif isinstance(code, (str, Path)) and Path(code).is_file():
            artifact_dir = Path("pystreamai_artifacts") / model_id
            artifact_dir.mkdir(parents=True, exist_ok=True)
            candidate_path = artifact_dir / "model.pkl"
            try:
                result = subprocess.run(
                    [sys.executable, str(code), "--dataset", str(dataset), "--output", str(candidate_path)],
                    capture_output=True,
                    text=True,
                    timeout=_parse_time_limit_seconds(time_limit),
                )
            except subprocess.TimeoutExpired:
                error = f"training script {code} exceeded time_limit={time_limit!r}"
            else:
                if result.returncode != 0:
                    error = (result.stderr or result.stdout or "").strip() or (
                        f"training script {code} exited with code {result.returncode}"
                    )
                elif not candidate_path.exists():
                    error = (
                        f"training script {code} exited 0 but did not write "
                        f"an artifact to the --output path it was given ({candidate_path})"
                    )
                else:
                    artifact_path = candidate_path
        else:
            raise TypeError(
                f"code={code!r} is neither callable nor an existing file path. "
                "Platform.train() needs a real callable (called as code(dataset)) "
                "or a real training script path -- see train()'s docstring."
            )

        job = TrainingJob(job_id, model_id, model=model, artifact_path=artifact_path, error=error)
        self.jobs[job_id] = job

        if error:
            print(f"   Training job {job_id} failed: {error}")
        else:
            print(f"   Training job {job_id} completed")

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
        Deploy a model as an endpoint that runs real inference.

        Args:
            model: One of:
                - a path to a real .onnx file, or an already-loaded
                  pystreamai.onnx_runtime.ONNXModelLoader -- real inference
                  via onnxruntime.
                - an object with a callable `.predict(data)` method (the
                  scikit-learn/most-frameworks convention) -- real calls
                  to that method.
                - a plain callable (function, or callable object) -- real
                  calls to it directly.
                Anything else -- None, a plain data object with no
                .predict() and not callable, or a .onnx path that fails to
                load -- raises TypeError/the real load error immediately.
                There is no simulated fallback.
            replicas: Number of replicas
            gpu: GPU type for serving
            max_batch_size: Max batch size for batching requests

        Returns:
            Endpoint that can be used for inference

        Raises:
            TypeError: if `model` has no real way to run inference.
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
