use pyo3::prelude::*;

mod scheduler;
mod executor;
mod storage;
mod inference;
mod backend;
mod gpu;
mod observability;
mod edge;
mod llm;
mod errors;

use scheduler::Scheduler;
use executor::Executor;
use storage::Storage;
use inference::{InferenceOptimizer, ModelOptimizationPlan};
use gpu::{GPUInfo, CUDAProfiler};
use observability::MetricsCollector;
use edge::{EdgeDeploymentPipeline, EdgeDevice};

#[pyclass]
pub struct Platform {
    scheduler: Scheduler,
    executor: Executor,
    storage: Storage,
    inference_optimizer: InferenceOptimizer,
    metrics: std::sync::Arc<std::sync::Mutex<MetricsCollector>>,
}

#[pymethods]
impl Platform {
    #[new]
    fn new() -> Self {
        Platform {
            scheduler: Scheduler::new(),
            executor: Executor::new(),
            storage: Storage::new(),
            inference_optimizer: InferenceOptimizer::new(),
            metrics: std::sync::Arc::new(std::sync::Mutex::new(MetricsCollector::new())),
        }
    }

    fn optimize_inference(&self, model_id: String) -> PyResult<String> {
        let plan = ModelOptimizationPlan::for_model(model_id);
        Ok(format!(
            "Optimization: {} | Expected speedup: {:.1}x",
            plan.estimated_latency_reduction(),
            plan.expected_speedup()
        ))
    }

    fn train(&self, model_id: String, dataset: String) -> PyResult<String> {
        Ok(format!("Training model {} on dataset {}", model_id, dataset))
    }

    fn serve(&self, model_id: String, replicas: usize) -> PyResult<String> {
        Ok(format!("Serving model {} with {} replicas", model_id, replicas))
    }

    fn predict(&self, model_id: String, data: String) -> PyResult<String> {
        Ok(format!("Prediction from model {}: {}", model_id, data))
    }

    fn record_inference(&self, model: String, latency_ms: f64, optimization: String) -> PyResult<()> {
        let metrics = self.metrics.lock().unwrap();
        metrics.record_inference(&model, latency_ms, &optimization);
        Ok(())
    }

    fn record_error(&self, model: String) -> PyResult<()> {
        let metrics = self.metrics.lock().unwrap();
        metrics.record_error(&model);
        Ok(())
    }

    fn record_cost(&self, model: String, cost_usd: f64) -> PyResult<()> {
        let metrics = self.metrics.lock().unwrap();
        metrics.record_cost(&model, cost_usd);
        Ok(())
    }

    fn get_metrics(&self) -> PyResult<String> {
        let metrics = self.metrics.lock().unwrap();
        let data = metrics.get_metrics();
        Ok(serde_json::to_string(&data).unwrap_or_default())
    }

    fn export_prometheus(&self) -> PyResult<String> {
        let metrics = self.metrics.lock().unwrap();
        Ok(metrics.export_prometheus())
    }

    fn deploy_edge(&self, model_path: String, device: String) -> PyResult<String> {
        let edge_device = match device.as_str() {
            "ios" => EdgeDevice::MobileIOS,
            "android" => EdgeDevice::MobileAndroid,
            "raspberry_pi" => EdgeDevice::RaspberryPi,
            "jetson" => EdgeDevice::Jetson,
            "wasm" => EdgeDevice::WASM,
            "browser" => EdgeDevice::Browser,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Unknown device")),
        };

        let pipeline = EdgeDeploymentPipeline::new(&model_path, edge_device);
        match pipeline.prepare_model() {
            Ok(result) => Ok(result),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)),
        }
    }

    fn estimate_edge_latency(&self, device: String) -> PyResult<u32> {
        let edge_device = match device.as_str() {
            "ios" => EdgeDevice::MobileIOS,
            "android" => EdgeDevice::MobileAndroid,
            "raspberry_pi" => EdgeDevice::RaspberryPi,
            "jetson" => EdgeDevice::Jetson,
            "wasm" => EdgeDevice::WASM,
            "browser" => EdgeDevice::Browser,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Unknown device")),
        };

        let pipeline = EdgeDeploymentPipeline::new("dummy.onnx", edge_device);
        Ok(pipeline.estimate_latency())
    }
}

#[pymodule]
fn pystreamai(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Platform>()?;
    m.add_class::<GPUInfo>()?;
    m.add_class::<CUDAProfiler>()?;
    Ok(())
}
