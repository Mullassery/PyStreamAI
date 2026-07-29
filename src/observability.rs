use prometheus::{
    Gauge, HistogramOpts, HistogramVec, IntCounter, IntGauge,
    Registry,
};
use lazy_static::lazy_static;
use std::sync::Arc;
use parking_lot::RwLock;
use std::collections::HashMap;

lazy_static! {
    static ref REGISTRY: Registry = Registry::new();

    static ref INFERENCE_LATENCY: HistogramVec = HistogramVec::new(
        HistogramOpts::new("inference_latency_ms", "Inference latency in milliseconds"),
        &["model", "optimization"]
    ).expect("Failed to create histogram");

    static ref INFERENCE_COUNT: IntCounter = IntCounter::new(
        "inference_total",
        "Total number of inferences"
    ).expect("Failed to create counter");

    static ref ERROR_COUNT: IntCounter = IntCounter::new(
        "inference_errors_total",
        "Total number of inference errors"
    ).expect("Failed to create counter");

    static ref THROUGHPUT: Gauge = Gauge::new(
        "throughput_req_per_sec",
        "Current throughput in requests per second"
    ).expect("Failed to create gauge");

    static ref COST_TOTAL: Gauge = Gauge::new(
        "cost_usd_total",
        "Total cost in USD"
    ).expect("Failed to create gauge");

    static ref ACTIVE_REQUESTS: IntGauge = IntGauge::new(
        "active_requests",
        "Number of active requests"
    ).expect("Failed to create gauge");

    static ref MODEL_LOAD_TIME: HistogramVec = HistogramVec::new(
        HistogramOpts::new("model_load_time_ms", "Model loading time in milliseconds"),
        &["model"]
    ).expect("Failed to create histogram");
}

pub struct MetricsCollector {
    latencies: Arc<RwLock<Vec<f64>>>,
    errors: Arc<RwLock<usize>>,
    total_requests: Arc<RwLock<usize>>,
    costs: Arc<RwLock<f64>>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        // Register all metrics
        let _ = REGISTRY.register(Box::new(INFERENCE_LATENCY.clone()));
        let _ = REGISTRY.register(Box::new(INFERENCE_COUNT.clone()));
        let _ = REGISTRY.register(Box::new(ERROR_COUNT.clone()));
        let _ = REGISTRY.register(Box::new(THROUGHPUT.clone()));
        let _ = REGISTRY.register(Box::new(COST_TOTAL.clone()));
        let _ = REGISTRY.register(Box::new(ACTIVE_REQUESTS.clone()));
        let _ = REGISTRY.register(Box::new(MODEL_LOAD_TIME.clone()));

        MetricsCollector {
            latencies: Arc::new(RwLock::new(Vec::new())),
            errors: Arc::new(RwLock::new(0)),
            total_requests: Arc::new(RwLock::new(0)),
            costs: Arc::new(RwLock::new(0.0)),
        }
    }

    pub fn record_inference(&self, model: &str, latency_ms: f64, optimization: &str) {
        INFERENCE_LATENCY
            .with_label_values(&[model, optimization])
            .observe(latency_ms);
        INFERENCE_COUNT.inc();

        let mut lats = self.latencies.write();
        lats.push(latency_ms);
        if lats.len() > 10000 {
            lats.drain(0..1000);
        }

        let mut total = self.total_requests.write();
        *total += 1;
    }

    pub fn record_error(&self, _model: &str) {
        ERROR_COUNT.inc();
        let mut errors = self.errors.write();
        *errors += 1;
    }

    pub fn record_cost(&self, _model: &str, cost_usd: f64) {
        COST_TOTAL.set(self.costs.read().clone() + cost_usd);
        let mut costs = self.costs.write();
        *costs += cost_usd;
    }

    pub fn set_active_requests(&self, count: i64) {
        ACTIVE_REQUESTS.set(count);
    }

    pub fn record_model_load(&self, model: &str, load_time_ms: f64) {
        MODEL_LOAD_TIME
            .with_label_values(&[model])
            .observe(load_time_ms);
    }

    pub fn get_metrics(&self) -> HashMap<String, f64> {
        let lats = self.latencies.read();
        let mut metrics = HashMap::new();

        if !lats.is_empty() {
            let sorted: Vec<f64> = {
                let mut v = lats.clone();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                v
            };

            metrics.insert("latency_p50".to_string(), sorted[sorted.len() / 2]);
            metrics.insert(
                "latency_p95".to_string(),
                sorted[(sorted.len() as f64 * 0.95) as usize],
            );
            metrics.insert(
                "latency_p99".to_string(),
                sorted[(sorted.len() as f64 * 0.99) as usize],
            );
            metrics.insert("latency_min".to_string(), sorted[0]);
            metrics.insert("latency_max".to_string(), sorted[sorted.len() - 1]);
            metrics.insert(
                "latency_mean".to_string(),
                lats.iter().sum::<f64>() / lats.len() as f64,
            );
        }

        let total = self.total_requests.read();
        metrics.insert("total_requests".to_string(), *total as f64);
        metrics.insert("error_count".to_string(), *self.errors.read() as f64);
        metrics.insert("total_cost_usd".to_string(), *self.costs.read());

        metrics
    }

    pub fn export_prometheus(&self) -> String {
        let mut output = String::new();
        output.push_str("# HELP inference_latency_ms Inference latency in milliseconds\n");
        output.push_str("# TYPE inference_latency_ms histogram\n");
        output.push_str("inference_total_count 0\n");
        output.push_str("inference_total_sum 0\n");
        output.push_str("inference_total_bucket{le=\"+Inf\"} 0\n");
        output.push_str("\n");
        output.push_str("# HELP throughput_req_per_sec Current throughput in requests per second\n");
        output.push_str("# TYPE throughput_req_per_sec gauge\n");
        output.push_str("throughput_req_per_sec 0\n");
        output
    }

    pub fn reset(&self) {
        self.latencies.write().clear();
        *self.errors.write() = 0;
        *self.total_requests.write() = 0;
        *self.costs.write() = 0.0;
        ACTIVE_REQUESTS.set(0);
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_collection() {
        let collector = MetricsCollector::new();
        collector.record_inference("bert", 50.0, "full_stack");
        collector.record_inference("bert", 60.0, "full_stack");
        collector.record_cost("bert", 0.0001);

        let metrics = collector.get_metrics();
        assert_eq!(metrics.get("total_requests"), Some(&2.0));
        assert!(metrics.get("latency_p50").is_some());
    }
}
