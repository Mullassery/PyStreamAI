pub struct Executor {
    workers: Vec<Worker>,
}

#[derive(Debug, Clone)]
pub struct Worker {
    pub id: String,
    pub gpu_available: bool,
    pub memory_mb: usize,
}

impl Executor {
    pub fn new() -> Self {
        Executor {
            workers: vec![],
        }
    }

    pub fn register_worker(&mut self, worker: Worker) {
        self.workers.push(worker);
    }

    pub fn available_workers(&self) -> usize {
        self.workers.len()
    }

    pub fn execute(&self, job_id: &str, model_id: &str) -> Result<String, String> {
        if self.workers.is_empty() {
            return Err("No workers available".to_string());
        }
        Ok(format!("Executing job {} on model {}", job_id, model_id))
    }
}

impl Default for Executor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execute_without_workers_fails() {
        let executor = Executor::new();
        assert!(executor.execute("job-1", "model-1").is_err());
    }

    #[test]
    fn execute_with_a_registered_worker_succeeds() {
        let mut executor = Executor::new();
        executor.register_worker(Worker {
            id: "w1".to_string(),
            gpu_available: true,
            memory_mb: 16384,
        });

        assert_eq!(executor.available_workers(), 1);
        let result = executor.execute("job-1", "model-1").unwrap();
        assert!(result.contains("job-1"));
        assert!(result.contains("model-1"));
    }
}
