use pyo3::prelude::*;

#[derive(Debug, Clone, Copy)]
pub enum GPUType {
    A100,
    H100,
    L4,
    V100,
    T4,
    RTX4090,
    Unknown,
}

impl GPUType {
    pub fn from_string(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "A100" => GPUType::A100,
            "H100" => GPUType::H100,
            "L4" => GPUType::L4,
            "V100" => GPUType::V100,
            "T4" => GPUType::T4,
            "RTX4090" => GPUType::RTX4090,
            _ => GPUType::Unknown,
        }
    }

    pub fn tensor_cores(&self) -> u32 {
        match self {
            GPUType::H100 => 18176,
            GPUType::A100 => 6912,
            GPUType::L4 => 5888,
            GPUType::V100 => 5120,
            GPUType::T4 => 2560,
            GPUType::RTX4090 => 16384,
            GPUType::Unknown => 0,
        }
    }

    pub fn memory_gb(&self) -> u32 {
        match self {
            GPUType::H100 => 80,
            GPUType::A100 => 80,
            GPUType::L4 => 24,
            GPUType::V100 => 32,
            GPUType::T4 => 16,
            GPUType::RTX4090 => 24,
            GPUType::Unknown => 0,
        }
    }

    pub fn fp16_tflops(&self) -> f32 {
        match self {
            GPUType::H100 => 1456.0,
            GPUType::A100 => 312.0,
            GPUType::L4 => 183.0,
            GPUType::V100 => 125.0,
            GPUType::T4 => 32.0,
            GPUType::RTX4090 => 331.0,
            GPUType::Unknown => 0.0,
        }
    }

    pub fn int8_tflops(&self) -> f32 {
        match self {
            GPUType::H100 => 1456.0,
            GPUType::A100 => 624.0,
            GPUType::L4 => 366.0,
            GPUType::V100 => 0.0, // V100 doesn't support native INT8
            GPUType::T4 => 64.0,
            GPUType::RTX4090 => 331.0,
            GPUType::Unknown => 0.0,
        }
    }

    pub fn supports_tensorrt(&self) -> bool {
        !matches!(self, GPUType::Unknown)
    }

    pub fn supports_mixed_precision(&self) -> bool {
        matches!(
            self,
            GPUType::A100 | GPUType::H100 | GPUType::L4 | GPUType::RTX4090
        )
    }

    pub fn supports_sparsity(&self) -> bool {
        matches!(self, GPUType::A100 | GPUType::H100)
    }
}

#[pyclass]
pub struct GPUInfo {
    pub gpu_type: GPUType,
    pub device_id: u32,
    pub memory_mb: u32,
    pub memory_free_mb: u32,
}

#[pymethods]
impl GPUInfo {
    #[new]
    fn new(gpu_type: String, device_id: u32) -> Self {
        let gpu_type = GPUType::from_string(&gpu_type);
        let memory_mb = gpu_type.memory_gb() * 1024;

        GPUInfo {
            gpu_type,
            device_id,
            memory_mb,
            memory_free_mb: memory_mb,
        }
    }

    fn tensor_cores(&self) -> u32 {
        self.gpu_type.tensor_cores()
    }

    fn fp16_tflops(&self) -> f32 {
        self.gpu_type.fp16_tflops()
    }

    fn int8_tflops(&self) -> f32 {
        self.gpu_type.int8_tflops()
    }

    fn supports_tensorrt(&self) -> bool {
        self.gpu_type.supports_tensorrt()
    }

    fn supports_mixed_precision(&self) -> bool {
        self.gpu_type.supports_mixed_precision()
    }

    fn supports_sparsity(&self) -> bool {
        self.gpu_type.supports_sparsity()
    }

    fn memory_utilization(&self) -> f32 {
        let used = self.memory_mb - self.memory_free_mb;
        (used as f32 / self.memory_mb as f32) * 100.0
    }
}

pub struct TensorRTOptimizer {
    gpu_type: GPUType,
    fp16_enabled: bool,
    int8_enabled: bool,
    sparsity_enabled: bool,
    max_workspace_mb: u32,
}

impl TensorRTOptimizer {
    pub fn new(gpu_type: GPUType) -> Self {
        TensorRTOptimizer {
            gpu_type,
            fp16_enabled: gpu_type.supports_mixed_precision(),
            int8_enabled: false, // Requires calibration
            sparsity_enabled: gpu_type.supports_sparsity(),
            max_workspace_mb: 4096, // 4GB workspace
        }
    }

    pub fn enable_fp16(&mut self) -> &mut Self {
        if self.gpu_type.supports_mixed_precision() {
            self.fp16_enabled = true;
        }
        self
    }

    pub fn enable_int8(&mut self) -> &mut Self {
        self.int8_enabled = true;
        self
    }

    pub fn enable_sparsity(&mut self) -> &mut Self {
        if self.gpu_type.supports_sparsity() {
            self.sparsity_enabled = true;
        }
        self
    }

    pub fn expected_speedup(&self) -> f32 {
        let mut speedup = 1.0;

        if self.fp16_enabled {
            speedup *= 1.5; // FP16 is 1.5x faster than FP32
        }

        if self.int8_enabled {
            speedup *= 2.5; // INT8 is 2.5x faster than FP32
        }

        if self.sparsity_enabled {
            speedup *= 1.2; // Sparsity adds ~20% speedup
        }

        speedup
    }

    pub fn tensorrt_config(&self) -> String {
        let mut config = String::from("TensorRT Config:\n");
        config.push_str(&format!("  GPU: {:?}\n", self.gpu_type));
        config.push_str(&format!("  FP16: {}\n", self.fp16_enabled));
        config.push_str(&format!("  INT8: {}\n", self.int8_enabled));
        config.push_str(&format!("  Sparsity: {}\n", self.sparsity_enabled));
        config.push_str(&format!("  Expected Speedup: {:.2}x\n", self.expected_speedup()));
        config
    }
}

pub struct MultiGPUScheduler {
    gpus: Vec<GPUInfo>,
    current_gpu: usize,
}

impl MultiGPUScheduler {
    pub fn new(gpu_count: usize, gpu_type: String) -> Self {
        let mut gpus = Vec::new();
        for i in 0..gpu_count {
            gpus.push(GPUInfo::new(gpu_type.clone(), i as u32));
        }

        MultiGPUScheduler {
            gpus,
            current_gpu: 0,
        }
    }

    pub fn next_gpu(&mut self) -> &GPUInfo {
        let gpu = &self.gpus[self.current_gpu];
        self.current_gpu = (self.current_gpu + 1) % self.gpus.len();
        gpu
    }

    pub fn gpu_with_most_memory(&self) -> &GPUInfo {
        self.gpus
            .iter()
            .max_by_key(|g| g.memory_free_mb)
            .unwrap_or(&self.gpus[0])
    }

    pub fn total_vram_gb(&self) -> u32 {
        self.gpus.iter().map(|g| g.memory_mb).sum::<u32>() / 1024
    }
}

#[pyclass]
pub struct CUDAProfiler {
    pub events: Vec<String>,
}

#[pymethods]
impl CUDAProfiler {
    #[new]
    fn new() -> Self {
        CUDAProfiler {
            events: Vec::new(),
        }
    }

    fn record(&mut self, event_name: String) {
        self.events.push(event_name);
    }

    fn kernel_launch_overhead_us(&self) -> f32 {
        5.0 // Typical kernel launch overhead: 5 microseconds
    }

    fn memory_transfer_bandwidth_gbps(&self, gpu_type: String) -> f32 {
        match gpu_type.to_uppercase().as_str() {
            "H100" => 3.35 * 1024.0, // 3.35 TB/s (NVLink)
            "A100" => 2.0 * 1024.0,  // 2 TB/s (NVLink)
            "L4" => 0.432 * 1024.0,  // 432 GB/s (PCIe 4.0)
            "H100_PCIe" => 0.576 * 1024.0, // 576 GB/s (PCIe 5.0)
            _ => 0.0,
        }
    }
}
