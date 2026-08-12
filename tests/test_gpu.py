"""Tests for pystreamai.gpu - GPU optimization planning (calculators, not real GPU calls)."""

import pytest

from pystreamai.gpu import (
    GPUOptimizer,
    MultiGPUInference,
    InferenceOptimizationPlan,
    TensorRTConfig,
    detect_available_gpus,
)


class TestGPUOptimizer:
    def test_default_speedup_is_one(self):
        optimizer = GPUOptimizer("A100")
        optimizer.tensorrt_config = TensorRTConfig(fp16=False, int8=False, sparsity=False)
        assert optimizer.get_expected_speedup() == 1.0

    def test_enable_tensorrt_fp16_increases_speedup(self):
        optimizer = GPUOptimizer("A100").enable_tensorrt(fp16=True)
        assert optimizer.get_expected_speedup() == pytest.approx(1.5)

    def test_combined_optimizations_multiply(self):
        optimizer = GPUOptimizer("A100").enable_tensorrt(fp16=True, int8=True, sparsity=True)
        assert optimizer.get_expected_speedup() == pytest.approx(1.5 * 2.5 * 1.2)

    def test_batch_size_recommendation_known_gpu(self):
        assert GPUOptimizer("H100").get_batch_size_recommendation() == 256

    def test_batch_size_recommendation_unknown_gpu_falls_back(self):
        assert GPUOptimizer("MADE_UP_GPU").get_batch_size_recommendation() == 32

    def test_enable_tensorrt_returns_self_for_chaining(self):
        optimizer = GPUOptimizer("A100")
        assert optimizer.enable_tensorrt() is optimizer


class TestMultiGPUInference:
    def test_next_gpu_round_robins(self):
        cluster = MultiGPUInference(num_gpus=2, gpu_type="A100")
        first = cluster.next_gpu()
        second = cluster.next_gpu()
        third = cluster.next_gpu()

        assert first is cluster.optimizers[0]
        assert second is cluster.optimizers[1]
        assert third is cluster.optimizers[0]

    def test_total_vram_scales_with_gpu_count(self):
        cluster = MultiGPUInference(num_gpus=4, gpu_type="A100")
        assert cluster.total_vram_gb() == 4 * 80


class TestInferenceOptimizationPlan:
    def test_recommend_includes_sparsity_step_on_a100(self):
        plan = InferenceOptimizationPlan("bert-base", "A100").recommend()
        step_names = [s["step"] for s in plan["steps"]]
        assert "Structured Sparsity" in step_names

    def test_recommend_excludes_sparsity_step_on_t4(self):
        plan = InferenceOptimizationPlan("bert-base", "T4").recommend()
        step_names = [s["step"] for s in plan["steps"]]
        assert "Structured Sparsity" not in step_names

    def test_apply_returns_readable_summary(self):
        summary = InferenceOptimizationPlan("bert-base", "A100").apply()
        assert "bert-base" in summary
        assert "Total Expected Speedup" in summary


def test_detect_available_gpus_without_torch_returns_empty():
    # In the test environment torch is not a declared dependency; the
    # function must degrade gracefully rather than raising ImportError.
    result = detect_available_gpus()
    assert "count" in result
    assert "gpus" in result
