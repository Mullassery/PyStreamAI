"""ONNX Runtime Integration - 2-3x speedup on inference"""

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import onnxruntime as ort
import numpy as np

logger = logging.getLogger(__name__)


class ONNXModelLoader:
    """Load and optimize ONNX models"""

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.session = None
        self.input_names = []
        self.output_names = []

    def load(self, providers: Optional[List[str]] = None) -> "ONNXModelLoader":
        """Load ONNX model with optimal providers"""
        if providers is None:
            # Auto-select providers: CUDA > TensorRT > CPU
            providers = []
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.append("CUDAExecutionProvider")
            if "TensorrtExecutionProvider" in ort.get_available_providers():
                providers.append("TensorrtExecutionProvider")
            providers.append("CPUExecutionProvider")

        logger.info(f"Loading ONNX model: {self.model_path}")
        logger.info(f"Providers: {providers}")

        self.session = ort.InferenceSession(str(self.model_path), providers=providers)

        # Get I/O info
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        logger.info(f"Model loaded. Inputs: {self.input_names}, Outputs: {self.output_names}")
        return self

    def infer(self, input_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run inference"""
        if not self.session:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self.session.run(self.output_names, input_data)

    def infer_batch(self, batch_inputs: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
        """Batch inference"""
        results = []
        for inputs in batch_inputs:
            result = self.infer(inputs)
            results.append(dict(zip(self.output_names, result)))
        return results

    def get_input_info(self) -> Dict[str, Any]:
        """Get model input specifications"""
        info = {}
        for inp in self.session.get_inputs():
            info[inp.name] = {
                "shape": inp.shape,
                "type": str(inp.type),
            }
        return info

    def get_output_info(self) -> Dict[str, Any]:
        """Get model output specifications"""
        info = {}
        for out in self.session.get_outputs():
            info[out.name] = {
                "shape": out.shape,
                "type": str(out.type),
            }
        return info


class PyTorchToONNXConverter:
    """Convert PyTorch models to ONNX format"""

    @staticmethod
    def convert(
        model,
        input_sample,
        output_path: str,
        opset_version: int = 14,
        optimize: bool = True,
    ) -> str:
        """Convert PyTorch model to ONNX"""
        import torch
        from pathlib import Path

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Converting PyTorch model to ONNX: {output_path}")

        # Export to ONNX
        torch.onnx.export(
            model,
            input_sample,
            str(output_path),
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            verbose=False,
        )

        logger.info(f"Model exported to ONNX: {output_path}")

        # Optimize ONNX model
        if optimize:
            PyTorchToONNXConverter._optimize_onnx(output_path)

        return str(output_path)

    @staticmethod
    def _optimize_onnx(model_path: Path) -> None:
        """Optimize ONNX model using onnxruntime tools"""
        try:
            from onnxruntime.transformers import optimizer

            opt_model_path = model_path.parent / f"{model_path.stem}_optimized.onnx"
            opt_model = optimizer.optimize_model(str(model_path))
            opt_model.save_model_to_file(str(opt_model_path))

            logger.info(f"Model optimized: {opt_model_path}")
        except Exception as e:
            logger.warning(f"ONNX optimization failed: {e}")


class ONNXBenchmark:
    """Benchmark ONNX inference performance"""

    def __init__(self, onnx_model: ONNXModelLoader):
        self.model = onnx_model

    def benchmark_throughput(
        self,
        batch_size: int = 1,
        num_iterations: int = 100,
    ) -> Dict[str, float]:
        """Benchmark throughput (requests/sec)"""
        import time

        # Create dummy input
        input_info = self.model.get_input_info()
        input_name = list(input_info.keys())[0]
        shape = input_info[input_name]["shape"]

        # Prepare batch input
        dummy_input = np.random.randn(batch_size, *shape[1:]).astype(np.float32)
        inputs = {input_name: dummy_input}

        # Warmup
        for _ in range(10):
            self.model.infer(inputs)

        # Benchmark
        start = time.time()
        for _ in range(num_iterations):
            self.model.infer(inputs)
        elapsed = time.time() - start

        throughput = (num_iterations * batch_size) / elapsed
        latency_ms = (elapsed / num_iterations) * 1000

        return {
            "throughput_req_sec": throughput,
            "latency_ms": latency_ms,
            "batch_size": batch_size,
        }

    def compare_with_pytorch(self, pytorch_model, input_sample) -> Dict[str, Any]:
        """Compare ONNX vs PyTorch inference speed"""
        import time
        import torch

        # ONNX benchmark
        onnx_latencies = []
        for _ in range(50):
            start = time.perf_counter()
            self.model.infer({"input": input_sample.numpy()})
            onnx_latencies.append((time.perf_counter() - start) * 1000)

        # PyTorch benchmark
        pytorch_latencies = []
        pytorch_model.eval()
        with torch.no_grad():
            for _ in range(50):
                start = time.perf_counter()
                pytorch_model(input_sample)
                pytorch_latencies.append((time.perf_counter() - start) * 1000)

        onnx_avg = sum(onnx_latencies) / len(onnx_latencies)
        pytorch_avg = sum(pytorch_latencies) / len(pytorch_latencies)
        speedup = pytorch_avg / onnx_avg

        return {
            "onnx_latency_ms": onnx_avg,
            "pytorch_latency_ms": pytorch_avg,
            "speedup": speedup,
        }
