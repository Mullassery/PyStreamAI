"""NVIDIA GPU Optimization Integration"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TensorRTConfig:
    """TensorRT optimization configuration"""
    fp16: bool = True
    int8: bool = False
    sparsity: bool = False
    max_batch_size: int = 32
    workspace_mb: int = 4096

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fp16": self.fp16,
            "int8": self.int8,
            "sparsity": self.sparsity,
            "max_batch_size": self.max_batch_size,
            "workspace_mb": self.workspace_mb,
        }


class GPUOptimizer:
    """NVIDIA GPU inference optimization"""

    def __init__(self, gpu_type: str = "A100"):
        self.gpu_type = gpu_type
        self.tensorrt_config = TensorRTConfig()
        self._validate_gpu()

    def _validate_gpu(self):
        """Validate GPU type"""
        valid_gpus = ["A100", "H100", "L4", "V100", "T4", "RTX4090"]
        if self.gpu_type not in valid_gpus:
            logger.warning(f"Unknown GPU type: {self.gpu_type}")

    def enable_tensorrt(
        self,
        fp16: bool = True,
        int8: bool = False,
        sparsity: bool = False,
    ) -> "GPUOptimizer":
        """Enable TensorRT optimizations"""
        self.tensorrt_config.fp16 = fp16
        self.tensorrt_config.int8 = int8
        self.tensorrt_config.sparsity = sparsity

        optimizations = []
        if fp16:
            optimizations.append("FP16")
        if int8:
            optimizations.append("INT8")
        if sparsity:
            optimizations.append("Sparsity")

        logger.info(f"TensorRT enabled on {self.gpu_type}: {', '.join(optimizations)}")
        return self

    def get_expected_speedup(self) -> float:
        """Get expected speedup from TensorRT optimizations"""
        speedup = 1.0

        if self.tensorrt_config.fp16:
            speedup *= 1.5  # FP16 is 1.5x faster

        if self.tensorrt_config.int8:
            speedup *= 2.5  # INT8 is 2.5x faster

        if self.tensorrt_config.sparsity:
            speedup *= 1.2  # Sparsity adds 20%

        return speedup

    def get_batch_size_recommendation(self) -> int:
        """Get recommended batch size for this GPU"""
        recommendations = {
            "H100": 256,
            "A100": 128,
            "L4": 64,
            "V100": 32,
            "T4": 16,
            "RTX4090": 128,
        }
        return recommendations.get(self.gpu_type, 32)

    def get_memory_recommendation_mb(self, model_params_millions: int) -> int:
        """Estimate VRAM needed (rough estimate)"""
        # Rough: ~4 bytes per parameter + activations
        model_vram = model_params_millions * 4
        # Add 50% for activations and batching
        total = int(model_vram * 1.5)
        return total


class MultiGPUInference:
    """Multi-GPU inference orchestration"""

    def __init__(self, num_gpus: int, gpu_type: str = "A100"):
        self.num_gpus = num_gpus
        self.gpu_type = gpu_type
        self.optimizers = [GPUOptimizer(gpu_type) for _ in range(num_gpus)]
        self.current_gpu = 0

    def next_gpu(self) -> GPUOptimizer:
        """Round-robin GPU selection"""
        optimizer = self.optimizers[self.current_gpu]
        self.current_gpu = (self.current_gpu + 1) % self.num_gpus
        return optimizer

    def total_vram_gb(self) -> float:
        """Get total VRAM across all GPUs"""
        vram_per_gpu = {
            "H100": 80,
            "A100": 80,
            "L4": 24,
            "V100": 32,
            "T4": 16,
            "RTX4090": 24,
        }
        per_gpu = vram_per_gpu.get(self.gpu_type, 0)
        return per_gpu * self.num_gpus

    def enable_nccl(self) -> "MultiGPUInference":
        """Enable NCCL for multi-GPU communication"""
        logger.info(f"NCCL enabled for {self.num_gpus} GPUs")
        return self

    def enable_nvlink(self) -> "MultiGPUInference":
        """Enable NVLink (if available on H100/A100)"""
        if self.gpu_type in ["H100", "A100"]:
            logger.info(f"NVLink enabled: ~1.7TB/s GPU-to-GPU bandwidth")
        else:
            logger.warning(f"NVLink not available on {self.gpu_type}")
        return self


class InferenceOptimizationPlan:
    """Complete GPU optimization plan for a model"""

    def __init__(self, model_name: str, gpu_type: str):
        self.model_name = model_name
        self.gpu_type = gpu_type
        self.gpu_optimizer = GPUOptimizer(gpu_type)
        self.multi_gpu = None
        self.plan_steps = []

    def recommend(self) -> Dict[str, Any]:
        """Get optimization recommendations"""
        plan = {
            "model": self.model_name,
            "gpu": self.gpu_type,
            "steps": [],
        }

        # Step 1: TensorRT with FP16
        self.gpu_optimizer.enable_tensorrt(fp16=True, int8=False, sparsity=False)
        plan["steps"].append({
            "step": "TensorRT + FP16",
            "speedup": 1.5,
            "description": "Convert to TensorRT, enable FP16"
        })

        # Step 2: Add INT8 (if no accuracy concerns)
        plan["steps"].append({
            "step": "INT8 Quantization",
            "speedup": 2.5,
            "description": "Post-training quantization to INT8 (requires calibration)"
        })

        # Step 3: Add Sparsity (if on A100/H100)
        if self.gpu_type in ["H100", "A100"]:
            plan["steps"].append({
                "step": "Structured Sparsity",
                "speedup": 1.2,
                "description": "Leverage tensor sparsity (2:4 sparsity pattern)"
            })

        # Total speedup
        total_speedup = 1.0
        for step in plan["steps"]:
            total_speedup *= step["speedup"]

        plan["total_speedup"] = total_speedup
        plan["batch_size_recommended"] = self.gpu_optimizer.get_batch_size_recommendation()

        return plan

    def apply(self) -> str:
        """Apply optimization plan"""
        plan = self.recommend()
        result = f"Optimization Plan for {self.model_name} on {self.gpu_type}\n"
        result += "=" * 60 + "\n"

        for i, step in enumerate(plan["steps"], 1):
            result += f"{i}. {step['step']}\n"
            result += f"   Speedup: {step['speedup']}x\n"
            result += f"   {step['description']}\n\n"

        result += f"Total Expected Speedup: {plan['total_speedup']:.1f}x\n"
        result += f"Recommended Batch Size: {plan['batch_size_recommended']}\n"

        return result


# GPU detection utilities
def detect_available_gpus() -> Dict[str, Any]:
    """Detect available NVIDIA GPUs"""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"count": 0, "gpus": []}

        count = torch.cuda.device_count()
        gpus = []
        for i in range(count):
            gpus.append({
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "capability": torch.cuda.get_device_capability(i),
                "memory_mb": torch.cuda.get_device_properties(i).total_memory / 1024 / 1024,
            })

        return {"count": count, "gpus": gpus}
    except ImportError:
        logger.warning("PyTorch not installed, GPU detection skipped")
        return {"count": 0, "gpus": []}
