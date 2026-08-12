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

impl Default for Storage {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_then_retrieve_round_trips() {
        let mut storage = Storage::new();
        storage
            .store_model("m1".to_string(), "/models/m1".to_string(), "v1".to_string())
            .unwrap();

        let artifact = storage.retrieve_model("m1").expect("model should be stored");
        assert_eq!(artifact.path, "/models/m1");
        assert_eq!(artifact.version, "v1");
    }

    #[test]
    fn retrieve_unknown_model_returns_none() {
        let storage = Storage::new();
        assert!(storage.retrieve_model("missing").is_none());
    }

    #[test]
    fn storing_same_model_id_overwrites_previous_version() {
        let mut storage = Storage::new();
        storage
            .store_model("m1".to_string(), "/v1/path".to_string(), "v1".to_string())
            .unwrap();
        storage
            .store_model("m1".to_string(), "/v2/path".to_string(), "v2".to_string())
            .unwrap();

        assert_eq!(storage.retrieve_model("m1").unwrap().version, "v2");
    }
}
