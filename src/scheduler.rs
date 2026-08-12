use std::collections::HashMap;
use uuid::Uuid;

pub struct Scheduler {
    jobs: HashMap<String, JobMetadata>,
}

#[derive(Debug, Clone)]
pub struct JobMetadata {
    pub id: String,
    pub model_id: String,
    pub status: JobStatus,
    pub created_at: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobStatus {
    Pending,
    Running,
    Completed,
    Failed,
}

impl Scheduler {
    pub fn new() -> Self {
        Scheduler {
            jobs: HashMap::new(),
        }
    }

    pub fn submit_job(&mut self, model_id: String) -> String {
        let job_id = Uuid::new_v4().to_string();
        let metadata = JobMetadata {
            id: job_id.clone(),
            model_id,
            status: JobStatus::Pending,
            created_at: chrono::Utc::now().timestamp(),
        };
        self.jobs.insert(job_id.clone(), metadata);
        job_id
    }

    pub fn get_job_status(&self, job_id: &str) -> Option<JobStatus> {
        self.jobs.get(job_id).map(|j| j.status)
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn submit_job_starts_pending() {
        let mut scheduler = Scheduler::new();
        let job_id = scheduler.submit_job("model-1".to_string());
        assert_eq!(scheduler.get_job_status(&job_id), Some(JobStatus::Pending));
    }

    #[test]
    fn unknown_job_id_returns_none() {
        let scheduler = Scheduler::new();
        assert_eq!(scheduler.get_job_status("does-not-exist"), None);
    }

    #[test]
    fn each_submission_gets_a_unique_id() {
        let mut scheduler = Scheduler::new();
        let a = scheduler.submit_job("model-1".to_string());
        let b = scheduler.submit_job("model-1".to_string());
        assert_ne!(a, b);
    }
}
