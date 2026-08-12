"""Tests for the compiled Rust extension (pystreamai._core).

This exercises the PyO3 bindings built from src/*.rs to make sure the
extension actually loads and its exposed classes behave as implemented.
Note this Rust layer is currently a separate, self-contained prototype -
the pure-Python pystreamai.platform.Platform does not call into it (see
README "How this works today"). These tests hold it to its own contract,
not to the Python layer's.
"""

import pytest

pystreamai_core = pytest.importorskip(
    "pystreamai._core", reason="compiled extension not built (run `maturin develop`)"
)


class TestPlatform:
    def test_train_returns_confirmation_string(self):
        platform = pystreamai_core.Platform()
        result = platform.train("model-1", "dataset.csv")
        assert "model-1" in result
        assert "dataset.csv" in result

    def test_serve_returns_confirmation_string(self):
        platform = pystreamai_core.Platform()
        result = platform.serve("model-1", 3)
        assert "model-1" in result
        assert "3" in result

    def test_predict_returns_confirmation_string(self):
        platform = pystreamai_core.Platform()
        result = platform.predict("model-1", "input-data")
        assert "model-1" in result

    def test_optimize_inference_reports_a_speedup(self):
        platform = pystreamai_core.Platform()
        result = platform.optimize_inference("model-1")
        assert "speedup" in result.lower()


class TestGPUInfo:
    def test_known_gpu_reports_expected_tensor_cores(self):
        gpu = pystreamai_core.GPUInfo("A100", 0)
        assert gpu.tensor_cores() == 6912

    def test_unknown_gpu_reports_zero_capabilities(self):
        gpu = pystreamai_core.GPUInfo("NOT_A_REAL_GPU", 0)
        assert gpu.tensor_cores() == 0
        assert gpu.supports_tensorrt() is False

    def test_memory_utilization_starts_at_zero(self):
        gpu = pystreamai_core.GPUInfo("A100", 0)
        assert gpu.memory_utilization() == 0.0

    def test_case_insensitive_gpu_name(self):
        gpu = pystreamai_core.GPUInfo("a100", 0)
        assert gpu.tensor_cores() == 6912


class TestCUDAProfiler:
    def test_record_and_kernel_launch_overhead(self):
        profiler = pystreamai_core.CUDAProfiler()
        profiler.record("kernel_launch")
        assert profiler.kernel_launch_overhead_us() == pytest.approx(5.0)

    def test_memory_transfer_bandwidth_known_gpu(self):
        profiler = pystreamai_core.CUDAProfiler()
        assert profiler.memory_transfer_bandwidth_gbps("H100") > 0

    def test_memory_transfer_bandwidth_unknown_gpu_is_zero(self):
        profiler = pystreamai_core.CUDAProfiler()
        assert profiler.memory_transfer_bandwidth_gbps("UNKNOWN") == 0.0


class TestMemoryPool:
    def test_allocate_within_budget_succeeds(self):
        pool = pystreamai_core.MemoryPool(1024, 512)  # 1GB budget, 512MB GPU
        assert pool.allocate("buf1", 10 * 1024 * 1024) is True

    def test_get_memory_stats_reports_utilization(self):
        pool = pystreamai_core.MemoryPool(1024, 512)
        pool.allocate("buf1", 100 * 1024 * 1024)
        stats = pool.get_memory_stats()
        assert stats["used_mb"] == 100
        assert stats["gpu_memory_mb"] == 512

    def test_deallocate_frees_memory(self):
        pool = pystreamai_core.MemoryPool(1024, 512)
        pool.allocate("buf1", 100 * 1024 * 1024)
        pool.deallocate("buf1")
        assert pool.get_memory_stats()["used_mb"] == 0

    def test_clear_resets_all_buffers(self):
        pool = pystreamai_core.MemoryPool(1024, 512)
        pool.allocate("buf1", 10 * 1024 * 1024)
        pool.allocate("buf2", 10 * 1024 * 1024)
        pool.clear()
        assert pool.get_memory_stats()["used_mb"] == 0
        assert pool.get_buffer("buf1") is None
