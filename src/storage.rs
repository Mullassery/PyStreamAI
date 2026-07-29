use std::collections::HashMap;

pub struct Storage {
    models: HashMap<String, ModelArtifact>,
}

#[derive(Debug, Clone)]
pub struct ModelArtifact {
    pub id: String,
    pub path: String,
    pub version: String,
    pub created_at: i64,
}

impl Storage {
    pub fn new() -> Self {
        Storage {
            models: HashMap::new(),
        }
    }

    pub fn store_model(&mut self, model_id: String, path: String, version: String) -> Result<(), String> {
        let artifact = ModelArtifact {
            id: model_id.clone(),
            path,
            version,
            created_at: chrono::Utc::now().timestamp(),
        };
        self.models.insert(model_id, artifact);
        Ok(())
    }

    pub fn retrieve_model(&self, model_id: &str) -> Option<ModelArtifact> {
        self.models.get(model_id).cloned()
    }
}
