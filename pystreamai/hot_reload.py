"""Model Hot-Reloading - Update models without downtime"""

import logging
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from datetime import datetime
import time
import threading

logger = logging.getLogger(__name__)


class ModelVersion:
    """Versioned model state"""

    def __init__(self, model_id: str, version: str, model: Any):
        self.model_id = model_id
        self.version = version
        self.model = model
        self.loaded_at = datetime.now()
        self.requests_served = 0
        self.active = True


class HotReloadManager:
    """Manage hot-reloading of models"""

    def __init__(self):
        self.models: Dict[str, ModelVersion] = {}
        self.current_versions: Dict[str, str] = {}
        self.lock = threading.RLock()
        self.watchers: Dict[str, Path] = {}

    def load_model(
        self,
        model_id: str,
        version: str,
        model_loader: Callable[[], Any],
    ) -> None:
        """Load a new model version"""
        with self.lock:
            logger.info(f"Loading model {model_id} version {version}")

            try:
                model = model_loader()
                model_version = ModelVersion(model_id, version, model)

                self.models[f"{model_id}:{version}"] = model_version

                # Set as current if first version
                if model_id not in self.current_versions:
                    self.current_versions[model_id] = version

                logger.info(f"Model loaded: {model_id}:{version}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise

    def activate_version(self, model_id: str, version: str) -> None:
        """Activate a model version"""
        with self.lock:
            key = f"{model_id}:{version}"
            if key not in self.models:
                raise ValueError(f"Model version not found: {key}")

            old_version = self.current_versions.get(model_id)
            self.current_versions[model_id] = version

            # Mark old version as inactive
            if old_version:
                old_key = f"{model_id}:{old_version}"
                if old_key in self.models:
                    self.models[old_key].active = False

            logger.info(f"Activated {model_id}:{version}")

    def get_current_model(self, model_id: str) -> Optional[Any]:
        """Get current model version"""
        with self.lock:
            version = self.current_versions.get(model_id)
            if not version:
                return None

            key = f"{model_id}:{version}"
            model_version = self.models.get(key)
            if model_version:
                model_version.requests_served += 1

            return model_version.model if model_version else None

    def watch_for_updates(
        self,
        model_id: str,
        watch_path: Path,
        reload_callback: Callable[[str, Path], Any],
    ) -> None:
        """Watch for model updates and reload"""
        self.watchers[model_id] = watch_path

        def watcher():
            last_mtime = watch_path.stat().st_mtime

            while True:
                time.sleep(5)  # Check every 5 seconds

                try:
                    current_mtime = watch_path.stat().st_mtime
                    if current_mtime > last_mtime:
                        logger.info(f"Model update detected for {model_id}")
                        last_mtime = current_mtime

                        # Reload model
                        try:
                            new_model = reload_callback(model_id, watch_path)
                            version = datetime.now().strftime("%Y%m%d-%H%M%S")
                            self.load_model(model_id, version, lambda: new_model)
                            self.activate_version(model_id, version)
                        except Exception as e:
                            logger.error(f"Failed to reload model: {e}")
                except Exception as e:
                    logger.error(f"Watcher error: {e}")

        thread = threading.Thread(target=watcher, daemon=True)
        thread.start()

    def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """Get model status"""
        with self.lock:
            current_version = self.current_versions.get(model_id)

            versions = {}
            for key, model_version in self.models.items():
                if key.startswith(f"{model_id}:"):
                    versions[model_version.version] = {
                        "loaded_at": model_version.loaded_at.isoformat(),
                        "requests_served": model_version.requests_served,
                        "active": model_version.active,
                        "current": model_version.version == current_version,
                    }

            return {
                "model_id": model_id,
                "current_version": current_version,
                "versions": versions,
            }

    def graceful_shutdown(self) -> None:
        """Graceful shutdown - wait for in-flight requests"""
        logger.info("Initiating graceful shutdown")

        with self.lock:
            for model_key, model_version in self.models.items():
                logger.info(
                    f"{model_key}: {model_version.requests_served} requests served"
                )

        # Wait for pending requests (simplified)
        time.sleep(2)
        logger.info("Graceful shutdown complete")
