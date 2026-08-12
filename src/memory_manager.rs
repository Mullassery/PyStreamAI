use std::collections::HashMap;
use std::sync::Arc;
use pyo3::prelude::*;

/// Memory pool for zero-copy buffer management
#[pyclass]
pub struct MemoryPool {
    buffers: HashMap<String, Vec<u8>>,
    buffer_size: usize,
    gpu_memory_mb: u32,
    cpu_memory_mb: u32,
    used_memory_mb: u32,
}

#[pymethods]
impl MemoryPool {
    #[new]
    fn new(max_buffer_size_mb: u32, gpu_memory_mb: u32) -> Self {
        MemoryPool {
            buffers: HashMap::new(),
            buffer_size: (max_buffer_size_mb as usize) * 1024 * 1024,
            gpu_memory_mb,
            cpu_memory_mb: max_buffer_size_mb,
            used_memory_mb: 0,
        }
    }

    fn allocate(&mut self, buffer_id: &str, size_bytes: usize) -> PyResult<bool> {
        if self.used_memory_mb * 1024 * 1024 + size_bytes as u32 > self.buffer_size as u32 {
            return Ok(false);  // Out of memory
        }

        let buffer = vec![0u8; size_bytes];
        self.buffers.insert(buffer_id.to_string(), buffer);
        self.used_memory_mb += (size_bytes / (1024 * 1024)) as u32;

        Ok(true)
    }

    fn deallocate(&mut self, buffer_id: &str) -> PyResult<()> {
        if let Some(buffer) = self.buffers.remove(buffer_id) {
            self.used_memory_mb -= (buffer.len() / (1024 * 1024)) as u32;
        }
        Ok(())
    }

    fn get_buffer(&self, buffer_id: &str) -> PyResult<Option<Vec<u8>>> {
        Ok(self.buffers.get(buffer_id).cloned())
    }

    fn get_memory_stats(&self) -> PyResult<std::collections::HashMap<String, u32>> {
        let mut stats = std::collections::HashMap::new();
        stats.insert("used_mb".to_string(), self.used_memory_mb);
        stats.insert("gpu_memory_mb".to_string(), self.gpu_memory_mb);
        stats.insert("cpu_memory_mb".to_string(), self.cpu_memory_mb);
        stats.insert("utilization_percent".to_string(),
            (self.used_memory_mb * 100 / self.cpu_memory_mb).min(100));
        Ok(stats)
    }

    fn clear(&mut self) -> PyResult<()> {
        self.buffers.clear();
        self.used_memory_mb = 0;
        Ok(())
    }
}

/// Tensor lifecycle management
pub struct TensorBuffer {
    pub id: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub device: String,  // "cpu" or "cuda:0"
    pub size_bytes: usize,
    pub reference_count: u32,
}

impl TensorBuffer {
    pub fn new(id: String, shape: Vec<usize>, dtype: String, device: String) -> Self {
        let size_bytes = shape.iter().product::<usize>() * 8;  // Assume 8 bytes per element

        TensorBuffer {
            id,
            shape,
            dtype,
            device,
            size_bytes,
            reference_count: 1,
        }
    }

    pub fn increment_ref(&mut self) {
        self.reference_count += 1;
    }

    pub fn decrement_ref(&mut self) -> bool {
        if self.reference_count > 0 {
            self.reference_count -= 1;
        }
        self.reference_count == 0  // Should be deallocated
    }
}

/// GPU memory pooling for inference
pub struct GPUMemoryPool {
    total_memory_mb: u32,
    used_memory_mb: u32,
    max_allocations: usize,
    current_allocations: usize,
}

impl GPUMemoryPool {
    pub fn new(total_memory_mb: u32, max_allocations: usize) -> Self {
        GPUMemoryPool {
            total_memory_mb,
            used_memory_mb: 0,
            max_allocations,
            current_allocations: 0,
        }
    }

    pub fn allocate(&mut self, size_mb: u32) -> Result<String, String> {
        if self.used_memory_mb + size_mb > self.total_memory_mb {
            return Err("GPU out of memory".to_string());
        }

        if self.current_allocations >= self.max_allocations {
            return Err("Too many allocations".to_string());
        }

        let allocation_id = format!("gpu_alloc_{}", self.current_allocations);
        self.used_memory_mb += size_mb;
        self.current_allocations += 1;

        Ok(allocation_id)
    }

    pub fn deallocate(&mut self, size_mb: u32) {
        self.used_memory_mb = self.used_memory_mb.saturating_sub(size_mb);
        self.current_allocations = self.current_allocations.saturating_sub(1);
    }

    pub fn memory_utilization(&self) -> f32 {
        (self.used_memory_mb as f32 / self.total_memory_mb as f32) * 100.0
    }
}
