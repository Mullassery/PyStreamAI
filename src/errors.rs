use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error, Clone, Serialize, Deserialize)]
pub enum PyStreamAIError {
    #[error("Model not found: {model}")]
    ModelNotFound { model: String },

    #[error("Invalid model format: {format}. Supported: ONNX, SavedModel, PyTorch")]
    InvalidModelFormat { format: String },

    #[error("Inference failed: {reason}")]
    InferenceFailed { reason: String },

    #[error("Insufficient GPU memory: required {required_mb} MB, available {available_mb} MB")]
    InsufficientGPUMemory {
        required_mb: u32,
        available_mb: u32,
    },

    #[error("Model loading timeout after {seconds} seconds")]
    ModelLoadingTimeout { seconds: u32 },

    #[error("Configuration invalid: {reason}")]
    InvalidConfig { reason: String },

    #[error("Batch size {batch_size} exceeds max {max_batch_size}")]
    BatchSizeExceeded {
        batch_size: usize,
        max_batch_size: usize,
    },

    #[error("Quantization failed: {reason}")]
    QuantizationError { reason: String },

    #[error("Edge deployment not supported for device: {device}")]
    EdgeDeploymentNotSupported { device: String },

    #[error("Backend not available: {backend}")]
    BackendNotAvailable { backend: String },

    #[error("Cost limit exceeded: spent ${spent:.2}, budget ${budget:.2}")]
    CostLimitExceeded { spent: f64, budget: f64 },

    #[error("Metric export failed: {reason}")]
    MetricExportFailed { reason: String },

    #[error("Request rejected: {reason}")]
    RequestRejected { reason: String },

    #[error("Internal error: {message}")]
    InternalError { message: String },
}

impl PyStreamAIError {
    pub fn to_user_message(&self) -> String {
        match self {
            PyStreamAIError::ModelNotFound { model } => {
                format!("Model '{}' not found. Please check the model name or path.", model)
            }
            PyStreamAIError::InvalidModelFormat { format } => {
                format!("Model format '{}' is not supported. Use ONNX, SavedModel, or PyTorch.", format)
            }
            PyStreamAIError::InferenceFailed { reason } => {
                format!("Inference failed: {}. Check model compatibility and input data.", reason)
            }
            PyStreamAIError::InsufficientGPUMemory {
                required_mb,
                available_mb,
            } => {
                format!(
                    "GPU memory insufficient. Required: {}MB, Available: {}MB. Consider using CPU or a larger GPU.",
                    required_mb, available_mb
                )
            }
            PyStreamAIError::ModelLoadingTimeout { seconds } => {
                format!("Model loading timed out after {}s. Try using a smaller model or increasing timeout.", seconds)
            }
            PyStreamAIError::InvalidConfig { reason } => {
                format!("Configuration error: {}. Check your settings.", reason)
            }
            PyStreamAIError::BatchSizeExceeded {
                batch_size,
                max_batch_size,
            } => {
                format!(
                    "Batch size {} exceeds limit of {}. Reduce batch size or increase resources.",
                    batch_size, max_batch_size
                )
            }
            PyStreamAIError::QuantizationError { reason } => {
                format!("Model quantization failed: {}. Try without quantization.", reason)
            }
            PyStreamAIError::EdgeDeploymentNotSupported { device } => {
                format!("Cannot deploy to {}. Supported devices: iOS, Android, Raspberry Pi, Jetson, WASM", device)
            }
            PyStreamAIError::BackendNotAvailable { backend } => {
                format!("Backend {} is not configured. Available backends: local, cloud, kubernetes", backend)
            }
            PyStreamAIError::CostLimitExceeded { spent, budget } => {
                format!(
                    "Cost limit exceeded. Spent: ${:.2}, Budget: ${:.2}. Requests blocked until next billing cycle.",
                    spent, budget
                )
            }
            PyStreamAIError::MetricExportFailed { reason } => {
                format!("Failed to export metrics: {}. Metrics will be collected locally.", reason)
            }
            PyStreamAIError::RequestRejected { reason } => {
                format!("Request rejected: {}", reason)
            }
            PyStreamAIError::InternalError { message } => {
                format!("Internal error: {}. Please report this issue.", message)
            }
        }
    }

    pub fn error_code(&self) -> String {
        match self {
            PyStreamAIError::ModelNotFound { .. } => "ERR_MODEL_NOT_FOUND".to_string(),
            PyStreamAIError::InvalidModelFormat { .. } => "ERR_INVALID_FORMAT".to_string(),
            PyStreamAIError::InferenceFailed { .. } => "ERR_INFERENCE_FAILED".to_string(),
            PyStreamAIError::InsufficientGPUMemory { .. } => "ERR_OUT_OF_MEMORY".to_string(),
            PyStreamAIError::ModelLoadingTimeout { .. } => "ERR_TIMEOUT".to_string(),
            PyStreamAIError::InvalidConfig { .. } => "ERR_INVALID_CONFIG".to_string(),
            PyStreamAIError::BatchSizeExceeded { .. } => "ERR_BATCH_SIZE".to_string(),
            PyStreamAIError::QuantizationError { .. } => "ERR_QUANTIZATION".to_string(),
            PyStreamAIError::EdgeDeploymentNotSupported { .. } => "ERR_EDGE_NOT_SUPPORTED".to_string(),
            PyStreamAIError::BackendNotAvailable { .. } => "ERR_BACKEND_UNAVAILABLE".to_string(),
            PyStreamAIError::CostLimitExceeded { .. } => "ERR_COST_LIMIT".to_string(),
            PyStreamAIError::MetricExportFailed { .. } => "ERR_METRICS".to_string(),
            PyStreamAIError::RequestRejected { .. } => "ERR_REJECTED".to_string(),
            PyStreamAIError::InternalError { .. } => "ERR_INTERNAL".to_string(),
        }
    }

    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            PyStreamAIError::InferenceFailed { .. }
                | PyStreamAIError::ModelLoadingTimeout { .. }
                | PyStreamAIError::MetricExportFailed { .. }
        )
    }
}

pub type Result<T> = std::result::Result<T, PyStreamAIError>;

#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
    pub error_code: String,
    pub message: String,
    pub timestamp: i64,
    pub retryable: bool,
}

impl ErrorResponse {
    pub fn from_error(err: &PyStreamAIError) -> Self {
        ErrorResponse {
            error: format!("{:?}", err),
            error_code: err.error_code(),
            message: err.to_user_message(),
            timestamp: chrono::Utc::now().timestamp(),
            retryable: err.is_retryable(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_messages() {
        let err = PyStreamAIError::ModelNotFound {
            model: "bert".to_string(),
        };
        assert!(err.to_user_message().contains("bert"));
        assert_eq!(err.error_code(), "ERR_MODEL_NOT_FOUND");
    }

    #[test]
    fn test_retryable_errors() {
        let retryable = PyStreamAIError::InferenceFailed {
            reason: "timeout".to_string(),
        };
        assert!(retryable.is_retryable());

        let not_retryable = PyStreamAIError::InvalidConfig {
            reason: "bad config".to_string(),
        };
        assert!(!not_retryable.is_retryable());
    }
}
