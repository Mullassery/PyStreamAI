use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationType {
    None,
    INT8,
    FP16,
    INT4,
}

#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub id: String,
    pub model_id: String,
    pub input_tokens: usize,
    pub priority: RequestPriority,
    pub submitted_at: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RequestPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

pub struct InferenceOptimizer {
    quantization: QuantizationType,
    batch_size: usize,
    batch_timeout_ms: u64,
    request_queue: VecDeque<InferenceRequest>,
    cache_enabled: bool,
    cache_size_mb: usize,
}

impl InferenceOptimizer {
    pub fn new() -> Self {
        InferenceOptimizer {
            quantization: QuantizationType::FP16,
            batch_size: 32,
            batch_timeout_ms: 100,
            request_queue: VecDeque::new(),
            cache_enabled: true,
            cache_size_mb: 512,
        }
    }

    pub fn enable_quantization(&mut self, quant_type: QuantizationType) {
        self.quantization = quant_type;
    }

    pub fn get_expected_speedup(&self) -> f32 {
        match self.quantization {
            QuantizationType::None => 1.0,
            QuantizationType::FP16 => 1.5,
            QuantizationType::INT8 => 3.0,
            QuantizationType::INT4 => 5.0,
        }
    }

    pub fn queue_request(&mut self, request: InferenceRequest) {
        self.request_queue.push_back(request);
    }

    pub fn should_flush_batch(&self, elapsed_ms: u64) -> bool {
        self.request_queue.len() >= self.batch_size || elapsed_ms >= self.batch_timeout_ms
    }

    pub fn next_batch(&mut self) -> Vec<InferenceRequest> {
        let batch_size = std::cmp::min(self.batch_size, self.request_queue.len());
        let mut batch = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            if let Some(req) = self.request_queue.pop_front() {
                batch.push(req);
            }
        }

        // Sort by priority (high priority first)
        batch.sort_by(|a, b| b.priority.cmp(&a.priority));

        batch
    }

    pub fn estimated_latency_ms(&self, input_tokens: usize) -> u64 {
        let base_latency = (input_tokens / 100).max(10) as u64;
        let speedup = self.get_expected_speedup();
        (base_latency as f32 / speedup) as u64
    }
}

pub struct ModelOptimizationPlan {
    pub model_id: String,
    pub quantization: QuantizationType,
    pub batching_enabled: bool,
    pub caching_enabled: bool,
    pub speculative_decoding: bool,
    pub flash_attention: bool,
}

impl ModelOptimizationPlan {
    pub fn for_model(model_id: String) -> Self {
        ModelOptimizationPlan {
            model_id,
            quantization: QuantizationType::FP16,
            batching_enabled: true,
            caching_enabled: true,
            speculative_decoding: true,
            flash_attention: true,
        }
    }

    pub fn expected_speedup(&self) -> f32 {
        let mut speedup = 1.0;

        if self.batching_enabled {
            speedup *= 1.3; // 30% faster with batching
        }
        if self.caching_enabled {
            speedup *= 1.2; // 20% faster with cache
        }
        if self.speculative_decoding {
            speedup *= 1.5; // 50% faster for LLM generation
        }
        if self.flash_attention {
            speedup *= 1.4; // 40% faster attention
        }

        match self.quantization {
            QuantizationType::FP16 => speedup *= 1.5,
            QuantizationType::INT8 => speedup *= 3.0,
            QuantizationType::INT4 => speedup *= 5.0,
            QuantizationType::None => {}
        }

        speedup
    }

    pub fn estimated_latency_reduction(&self) -> String {
        let speedup = self.expected_speedup();
        let reduction = ((speedup - 1.0) / speedup) * 100.0;
        format!("{:.0}% faster", reduction)
    }
}
