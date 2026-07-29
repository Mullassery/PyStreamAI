"""Automatic Model Versioning and Rollback - Zero-config production safety"""

import logging
import time
import json
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status of a model version"""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    ROLLED_BACK = "rolled_back"


@dataclass
class ModelVersionMetadata:
    """Metadata for a model version"""
    version_id: str
    model_id: str
    deployed_at: datetime
    model_hash: str  # Hash of model weights/config
    parent_version: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    health_status: HealthStatus = HealthStatus.INITIALIZING
    error_count: int = 0
    request_count: int = 0
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    promoted_at: Optional[datetime] = None
    rolled_back_at: Optional[datetime] = None
    rollback_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "deployed_at": self.deployed_at.isoformat(),
            "promoted_at": self.promoted_at.isoformat() if self.promoted_at else None,
            "rolled_back_at": self.rolled_back_at.isoformat() if self.rolled_back_at else None,
            "health_status": self.health_status.value,
        }


@dataclass
class HealthThresholds:
    """Thresholds for automatic rollback"""
    error_rate_percent: float = 5.0  # Rollback if error rate > 5%
    p99_latency_ms: float = 2000.0  # Rollback if p99 latency > 2s
    p95_latency_ms: float = 1000.0  # Rollback if p95 latency > 1s
    min_samples: int = 100  # Need at least 100 requests before evaluating
    evaluation_window_seconds: int = 300  # Evaluate over 5-min windows
    consecutive_bad_windows: int = 2  # Rollback after 2 consecutive bad windows


class VersionRegistry:
    """Stores and manages model version history"""

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path or ".pystreamai_versions")
        self.storage_path.mkdir(exist_ok=True)
        self.versions: Dict[str, ModelVersionMetadata] = {}
        self._load_history()

    def _load_history(self) -> None:
        """Load version history from disk"""
        history_file = self.storage_path / "version_history.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    data = json.load(f)
                    for model_id, versions in data.items():
                        for v in versions:
                            meta = ModelVersionMetadata(
                                version_id=v["version_id"],
                                model_id=v["model_id"],
                                deployed_at=datetime.fromisoformat(v["deployed_at"]),
                                model_hash=v["model_hash"],
                                parent_version=v.get("parent_version"),
                                metrics=v.get("metrics", {}),
                                health_status=HealthStatus(v.get("health_status", "initializing")),
                                error_count=v.get("error_count", 0),
                                request_count=v.get("request_count", 0),
                                avg_latency_ms=v.get("avg_latency_ms", 0.0),
                                p99_latency_ms=v.get("p99_latency_ms", 0.0),
                            )
                            if v.get("promoted_at"):
                                meta.promoted_at = datetime.fromisoformat(v["promoted_at"])
                            if v.get("rolled_back_at"):
                                meta.rolled_back_at = datetime.fromisoformat(v["rolled_back_at"])
                            self.versions[meta.version_id] = meta
            except Exception as e:
                logger.error(f"Failed to load version history: {e}")

    def _save_history(self) -> None:
        """Save version history to disk"""
        history_file = self.storage_path / "version_history.json"
        try:
            # Group by model_id
            by_model = {}
            for vid, meta in self.versions.items():
                if meta.model_id not in by_model:
                    by_model[meta.model_id] = []
                by_model[meta.model_id].append(meta.to_dict())

            with open(history_file, "w") as f:
                json.dump(by_model, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save version history: {e}")

    def register_version(
        self,
        model_id: str,
        model_hash: str,
        parent_version: Optional[str] = None,
    ) -> str:
        """Register a new model version"""
        version_id = f"{model_id}-{int(time.time())}-{model_hash[:8]}"

        metadata = ModelVersionMetadata(
            version_id=version_id,
            model_id=model_id,
            deployed_at=datetime.now(),
            model_hash=model_hash,
            parent_version=parent_version,
        )

        self.versions[version_id] = metadata
        self._save_history()

        logger.info(f"Registered version: {version_id}")
        return version_id

    def get_version(self, version_id: str) -> Optional[ModelVersionMetadata]:
        """Get version metadata"""
        return self.versions.get(version_id)

    def get_latest_healthy_version(self, model_id: str) -> Optional[str]:
        """Get the latest promoted (healthy) version of a model"""
        candidates = [
            v for v in self.versions.values()
            if v.model_id == model_id and v.promoted_at
        ]
        if not candidates:
            return None

        # Return most recently promoted version
        return max(candidates, key=lambda v: v.promoted_at).version_id

    def promote_version(self, version_id: str) -> None:
        """Mark version as promoted to production"""
        if version_id in self.versions:
            self.versions[version_id].promoted_at = datetime.now()
            self.versions[version_id].health_status = HealthStatus.HEALTHY
            self._save_history()
            logger.info(f"Promoted version: {version_id}")

    def rollback_version(self, version_id: str, reason: str) -> None:
        """Mark version as rolled back"""
        if version_id in self.versions:
            self.versions[version_id].rolled_back_at = datetime.now()
            self.versions[version_id].rollback_reason = reason
            self.versions[version_id].health_status = HealthStatus.ROLLED_BACK
            self._save_history()
            logger.warning(f"Rolled back version: {version_id} - {reason}")

    def cleanup_old_versions(
        self,
        model_id: str,
        keep_count: int = 5,
        keep_days: int = 7,
    ) -> None:
        """Remove old versions (keep recent ones and recent history)"""
        versions = [
            v for v in self.versions.values()
            if v.model_id == model_id
        ]

        if len(versions) <= keep_count:
            return

        # Sort by deployed_at, newest first
        versions.sort(key=lambda v: v.deployed_at, reverse=True)

        # Keep the most recent versions
        to_keep = set(v.version_id for v in versions[:keep_count])

        # Also keep recent history (within keep_days)
        cutoff = datetime.now() - timedelta(days=keep_days)
        to_keep.update(
            v.version_id for v in versions
            if v.deployed_at > cutoff
        )

        # Remove old versions
        for version_id in list(self.versions.keys()):
            if version_id not in to_keep:
                v = self.versions[version_id]
                if v.model_id == model_id:
                    del self.versions[version_id]
                    logger.info(f"Cleaned up version: {version_id}")

        self._save_history()


class HealthMonitor:
    """Monitor health metrics and detect anomalies"""

    def __init__(self, thresholds: Optional[HealthThresholds] = None):
        self.thresholds = thresholds or HealthThresholds()
        self.latencies: List[float] = []
        self.errors: int = 0
        self.requests: int = 0
        self.window_start = time.time()
        self.bad_window_count = 0

    def record_request(
        self,
        latency_ms: float,
        error: bool = False,
    ) -> None:
        """Record a single inference request"""
        self.latencies.append(latency_ms)
        self.requests += 1
        if error:
            self.errors += 1

    def get_current_health(self) -> Dict[str, Any]:
        """Get current health metrics"""
        if self.requests < self.thresholds.min_samples:
            return {
                "status": "initializing",
                "requests": self.requests,
                "samples_needed": self.thresholds.min_samples - self.requests,
            }

        error_rate = (self.errors / self.requests) * 100 if self.requests > 0 else 0

        # Calculate latency percentiles
        sorted_latencies = sorted(self.latencies)
        p95_idx = max(0, int(len(sorted_latencies) * 0.95) - 1)
        p99_idx = max(0, int(len(sorted_latencies) * 0.99) - 1)

        avg_latency = sum(self.latencies) / len(self.latencies)
        p95_latency = sorted_latencies[p95_idx] if sorted_latencies else 0
        p99_latency = sorted_latencies[p99_idx] if sorted_latencies else 0

        return {
            "error_rate_percent": error_rate,
            "request_count": self.requests,
            "error_count": self.errors,
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency,
        }

    def check_health_thresholds(self) -> tuple[bool, Optional[str]]:
        """Check if health metrics exceed thresholds. Returns (is_healthy, reason)"""
        health = self.get_current_health()

        if health.get("status") == "initializing":
            return True, None

        if health["error_rate_percent"] > self.thresholds.error_rate_percent:
            return False, f"Error rate {health['error_rate_percent']:.1f}% exceeds {self.thresholds.error_rate_percent}%"

        if health["p99_latency_ms"] > self.thresholds.p99_latency_ms:
            return False, f"P99 latency {health['p99_latency_ms']:.0f}ms exceeds {self.thresholds.p99_latency_ms}ms"

        if health["p95_latency_ms"] > self.thresholds.p95_latency_ms:
            return False, f"P95 latency {health['p95_latency_ms']:.0f}ms exceeds {self.thresholds.p95_latency_ms}ms"

        return True, None

    def evaluate_window(self) -> tuple[bool, Optional[str]]:
        """Evaluate current window and reset for next window"""
        is_healthy, reason = self.check_health_thresholds()

        if not is_healthy:
            self.bad_window_count += 1
        else:
            self.bad_window_count = 0

        # Reset for next window
        self.latencies = []
        self.errors = 0
        self.requests = 0
        self.window_start = time.time()

        # Trigger rollback after consecutive bad windows
        should_rollback = self.bad_window_count >= self.thresholds.consecutive_bad_windows

        return should_rollback, reason


class AutoVersionManager:
    """Automatically manages model versioning and rollback"""

    def __init__(self, storage_path: Optional[str] = None):
        self.registry = VersionRegistry(storage_path)
        self.monitors: Dict[str, HealthMonitor] = {}
        self.active_versions: Dict[str, str] = {}  # model_id -> version_id
        self.rollback_handlers: List[Callable[[str, str], None]] = []

    def on_rollback(self, handler: Callable[[str, str], None]) -> None:
        """Register handler for rollback events. Handler receives (version_id, reason)"""
        self.rollback_handlers.append(handler)

    def _notify_rollback(self, version_id: str, reason: str) -> None:
        """Notify registered rollback handlers"""
        for handler in self.rollback_handlers:
            try:
                handler(version_id, reason)
            except Exception as e:
                logger.error(f"Rollback handler failed: {e}")

    def deploy_model(
        self,
        model_id: str,
        model_hash: str,
        thresholds: Optional[HealthThresholds] = None,
    ) -> str:
        """Deploy a new model version"""
        parent_version = self.active_versions.get(model_id)
        version_id = self.registry.register_version(model_id, model_hash, parent_version)

        # Create health monitor for this version
        self.monitors[version_id] = HealthMonitor(thresholds)

        self.active_versions[model_id] = version_id
        logger.info(f"Deployed {model_id} as {version_id}")

        return version_id

    def record_inference(
        self,
        model_id: str,
        latency_ms: float,
        error: bool = False,
    ) -> None:
        """Record inference metrics and check for rollback"""
        version_id = self.active_versions.get(model_id)
        if not version_id:
            return

        monitor = self.monitors.get(version_id)
        if not monitor:
            return

        monitor.record_request(latency_ms, error)

        # Update version metadata
        version = self.registry.get_version(version_id)
        if version:
            version.request_count += 1
            if error:
                version.error_count += 1

            health = monitor.get_current_health()
            version.avg_latency_ms = health.get("avg_latency_ms", 0)
            version.p99_latency_ms = health.get("p99_latency_ms", 0)

    def promote_version(self, version_id: str) -> None:
        """Promote a version to production"""
        version = self.registry.get_version(version_id)
        if not version:
            return

        self.registry.promote_version(version_id)
        self.active_versions[version.model_id] = version_id
        logger.info(f"Promoted {version_id} to production")

    def evaluate_health_windows(self) -> None:
        """Evaluate health for all monitored versions. Call periodically (e.g., every 5 min)"""
        for version_id, monitor in list(self.monitors.items()):
            version = self.registry.get_version(version_id)
            if not version or version.health_status == HealthStatus.ROLLED_BACK:
                continue

            should_rollback, reason = monitor.evaluate_window()

            if should_rollback:
                logger.warning(f"Auto-rollback triggered for {version_id}: {reason}")
                self._auto_rollback(version_id, reason)

    def manual_rollback(self, version_id: str, reason: str = "Manual rollback") -> bool:
        """Manually rollback to a specific previous version. Returns True if successful."""
        version = self.registry.get_version(version_id)
        if not version:
            logger.error(f"Version {version_id} not found")
            return False

        if version.health_status == HealthStatus.ROLLED_BACK:
            logger.warning(f"Version {version_id} is already rolled back")
            return False

        # Rollback the current active version
        current_version_id = self.active_versions.get(version.model_id)
        if current_version_id and current_version_id != version_id:
            self.registry.rollback_version(current_version_id, f"Manual rollback to {version_id}")

        # Switch to the specified version
        self.active_versions[version.model_id] = version_id
        self.registry.promote_version(version_id)

        logger.info(f"Manually rolled back to {version_id}: {reason}")
        self._notify_rollback(current_version_id, reason)

        return True

    def list_versions(self, model_id: str) -> List[Dict[str, Any]]:
        """List all versions of a model (newest first)"""
        versions = [
            v for v in self.registry.versions.values()
            if v.model_id == model_id
        ]
        # Sort by deployed_at, newest first
        versions.sort(key=lambda v: v.deployed_at, reverse=True)
        return [v.to_dict() for v in versions]

    def get_rollback_candidates(self, model_id: str) -> List[Dict[str, Any]]:
        """Get list of versions eligible for rollback (promoted healthy versions)"""
        versions = [
            v for v in self.registry.versions.values()
            if v.model_id == model_id and v.promoted_at and v.health_status == HealthStatus.HEALTHY
        ]
        # Sort by deployed_at, newest first
        versions.sort(key=lambda v: v.promoted_at, reverse=True)
        return [v.to_dict() for v in versions]

    def _auto_rollback(self, version_id: str, reason: str) -> None:
        """Automatically rollback a version"""
        version = self.registry.get_version(version_id)
        if not version:
            return

        # Mark as rolled back
        self.registry.rollback_version(version_id, reason)

        # Switch back to previous healthy version
        previous_version = self.registry.get_latest_healthy_version(version.model_id)
        if previous_version and previous_version != version_id:
            self.active_versions[version.model_id] = previous_version
            logger.warning(f"Switched to previous version: {previous_version}")

        # Notify handlers
        self._notify_rollback(version_id, reason)

    def get_version_status(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a version"""
        version = self.registry.get_version(version_id)
        if not version:
            return None

        monitor = self.monitors.get(version_id)
        health = monitor.get_current_health() if monitor else {}

        return {
            **version.to_dict(),
            "current_health": health,
        }

    def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """Get status of all versions for a model"""
        versions = [
            v for v in self.registry.versions.values()
            if v.model_id == model_id
        ]

        return {
            "model_id": model_id,
            "active_version": self.active_versions.get(model_id),
            "versions": [v.to_dict() for v in versions],
        }
