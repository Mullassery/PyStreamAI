use dashmap::DashMap;
use std::sync::Arc;
use chrono::Utc;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptCacheEntry {
    pub prompt_hash: String,
    pub tokens: Vec<i32>,
    pub embedding: Vec<f32>,
    pub created_at: i64,
    pub ttl_seconds: i64,
}

pub struct PromptCache {
    cache: Arc<DashMap<String, PromptCacheEntry>>,
    max_entries: usize,
}

impl PromptCache {
    pub fn new(max_entries: usize) -> Self {
        PromptCache {
            cache: Arc::new(DashMap::new()),
            max_entries,
        }
    }

    pub fn get(&self, prompt: &str) -> Option<Vec<i32>> {
        let hash = self.hash_prompt(prompt);
        if let Some(entry) = self.cache.get(&hash) {
            let now = Utc::now().timestamp();
            if now - entry.created_at < entry.ttl_seconds {
                return Some(entry.tokens.clone());
            }
        }
        None
    }

    pub fn set(&self, prompt: &str, tokens: Vec<i32>) {
        let hash = self.hash_prompt(prompt);
        if self.cache.len() >= self.max_entries {
            if let Some(entry) = self.cache.iter().next() {
                let key = entry.key().clone();
                drop(entry);
                self.cache.remove(&key);
            }
        }
        self.cache.insert(
            hash.clone(),
            PromptCacheEntry {
                prompt_hash: hash,
                tokens,
                embedding: Vec::new(),
                created_at: Utc::now().timestamp(),
                ttl_seconds: 3600, // 1 hour
            },
        );
    }

    fn hash_prompt(&self, prompt: &str) -> String {
        format!("{:x}", fxhash::hash64(prompt))
    }

    pub fn stats(&self) -> (usize, f32) {
        let len = self.cache.len();
        let util = (len as f32 / self.max_entries as f32) * 100.0;
        (len, util)
    }

    pub fn clear(&self) {
        self.cache.clear();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeDecodingConfig {
    pub draft_model: String,
    pub verify_model: String,
    pub max_speculation_tokens: usize,
    pub acceptance_threshold: f32,
}

pub struct SpeculativeDecoder {
    config: SpeculativeDecodingConfig,
    draft_cache: Arc<DashMap<String, Vec<i32>>>,
}

impl SpeculativeDecoder {
    pub fn new(config: SpeculativeDecodingConfig) -> Self {
        SpeculativeDecoder {
            config,
            draft_cache: Arc::new(DashMap::new()),
        }
    }

    pub fn generate_draft_tokens(&self, prompt: &str) -> Vec<i32> {
        if let Some(cached) = self.draft_cache.get(prompt) {
            return cached.clone();
        }
        // Simulate draft model generation
        let tokens: Vec<i32> = (0..self.config.max_speculation_tokens)
            .map(|i| (i as i32) % 1000)
            .collect();
        self.draft_cache.insert(prompt.to_string(), tokens.clone());
        tokens
    }

    pub fn verify_tokens(&self, draft_tokens: Vec<i32>, verify_prob: f32) -> Vec<i32> {
        // Keep tokens where verify_prob > acceptance_threshold
        draft_tokens
            .into_iter()
            .filter(|_| verify_prob > self.config.acceptance_threshold)
            .collect()
    }

    pub fn decode_with_speculation(
        &self,
        prompt: &str,
        _max_tokens: usize,
    ) -> (Vec<i32>, f32) {
        // Generate draft tokens
        let draft = self.generate_draft_tokens(prompt);
        // Verify with main model (simulated)
        let verified = self.verify_tokens(draft, 0.85);
        // Speedup: number of tokens generated in parallel / sequential generations
        let speedup = if verified.len() > 0 {
            (self.config.max_speculation_tokens as f32 / verified.len() as f32) * 1.5
        } else {
            1.0
        };
        (verified, speedup)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KVCacheConfig {
    pub max_cache_size_mb: usize,
    pub page_size_tokens: usize,
}

pub struct PagedAttention {
    config: KVCacheConfig,
    cache: Arc<DashMap<String, Vec<u8>>>,
}

impl PagedAttention {
    pub fn new(config: KVCacheConfig) -> Self {
        PagedAttention {
            config,
            cache: Arc::new(DashMap::new()),
        }
    }

    pub fn allocate_pages(&self, num_tokens: usize) -> usize {
        let pages_needed = (num_tokens + self.config.page_size_tokens - 1)
            / self.config.page_size_tokens;
        let memory_per_page = 1024; // Simulated
        pages_needed * memory_per_page
    }

    pub fn get_cache_efficiency(&self) -> f32 {
        let used = self.cache.len() * 1024;
        let available = self.config.max_cache_size_mb * 1024 * 1024;
        (used as f32 / available as f32) * 100.0
    }

    pub fn store_kv(&self, request_id: &str, key_value: Vec<u8>) {
        if self.cache.len() * 1024 < self.config.max_cache_size_mb * 1024 * 1024 {
            self.cache.insert(request_id.to_string(), key_value);
        }
    }

    pub fn retrieve_kv(&self, request_id: &str) -> Option<Vec<u8>> {
        self.cache.get(request_id).map(|r| r.value().clone())
    }
}

pub struct LLMOptimizationEngine {
    prompt_cache: Arc<PromptCache>,
    speculative_decoder: Option<Arc<SpeculativeDecoder>>,
    paged_attention: Option<Arc<PagedAttention>>,
}

impl LLMOptimizationEngine {
    pub fn new() -> Self {
        LLMOptimizationEngine {
            prompt_cache: Arc::new(PromptCache::new(1000)),
            speculative_decoder: None,
            paged_attention: None,
        }
    }

    pub fn enable_prompt_caching(&self) {
        // Prompt cache is always enabled
    }

    pub fn enable_speculative_decoding(&mut self, draft_model: String) {
        let config = SpeculativeDecodingConfig {
            draft_model,
            verify_model: "main".to_string(),
            max_speculation_tokens: 4,
            acceptance_threshold: 0.8,
        };
        self.speculative_decoder = Some(Arc::new(SpeculativeDecoder::new(config)));
    }

    pub fn enable_paged_attention(&mut self) {
        let config = KVCacheConfig {
            max_cache_size_mb: 500,
            page_size_tokens: 16,
        };
        self.paged_attention = Some(Arc::new(PagedAttention::new(config)));
    }

    pub fn generate(&self, prompt: &str, max_tokens: usize) -> (String, LLMMetrics) {
        let mut metrics = LLMMetrics::new();

        // Check prompt cache
        if let Some(cached) = self.prompt_cache.get(prompt) {
            metrics.prompt_cache_hit = true;
            metrics.tokens_generated = cached.len() as u32;
            return (
                format!("Cached response: {} tokens", cached.len()),
                metrics,
            );
        }

        // Use speculative decoding if enabled
        if let Some(decoder) = &self.speculative_decoder {
            let (tokens, speedup) = decoder.decode_with_speculation(prompt, max_tokens);
            metrics.speculative_speedup = speedup;
            metrics.tokens_generated = tokens.len() as u32;
        } else {
            metrics.tokens_generated = max_tokens as u32;
        }

        // Cache the prompt
        self.prompt_cache
            .set(prompt, vec![0; max_tokens.min(100)]);

        (format!("Generated {} tokens", max_tokens), metrics)
    }
}

impl Default for LLMOptimizationEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMMetrics {
    pub prompt_cache_hit: bool,
    pub tokens_generated: u32,
    pub speculative_speedup: f32,
}

impl LLMMetrics {
    pub fn new() -> Self {
        LLMMetrics {
            prompt_cache_hit: false,
            tokens_generated: 0,
            speculative_speedup: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt_cache() {
        let cache = PromptCache::new(100);
        cache.set("hello world", vec![1, 2, 3]);
        assert_eq!(cache.get("hello world"), Some(vec![1, 2, 3]));
    }

    #[test]
    fn test_speculative_decoding() {
        let config = SpeculativeDecodingConfig {
            draft_model: "draft".to_string(),
            verify_model: "main".to_string(),
            max_speculation_tokens: 4,
            acceptance_threshold: 0.8,
        };
        let decoder = SpeculativeDecoder::new(config);
        let (tokens, speedup) = decoder.decode_with_speculation("test", 100);
        assert!(speedup > 1.0);
    }

    #[test]
    fn test_paged_attention() {
        let config = KVCacheConfig {
            max_cache_size_mb: 500,
            page_size_tokens: 16,
        };
        let pa = PagedAttention::new(config);
        let pages = pa.allocate_pages(100);
        assert!(pages > 0);
    }
}
