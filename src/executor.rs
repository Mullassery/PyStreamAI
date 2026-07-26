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
