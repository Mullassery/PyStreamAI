import pystreamai


def test_native_gpu_types_are_exported():
    gpu = pystreamai.GPUInfo("a100", 0)
    assert gpu.tensor_cores() > 0
    assert pystreamai.CUDAProfiler is not None
    assert pystreamai.MemoryPool(max_buffer_size_mb=64, gpu_memory_mb=0).get_memory_stats()["used_mb"] == 0
