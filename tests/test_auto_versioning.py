"""Tests for automatic model versioning and rollback"""

import pytest
import hashlib
from datetime import datetime

from pystreamai.auto_versioning import (
    AutoVersionManager,
    HealthMonitor,
    HealthThresholds,
    HealthStatus,
    VersionRegistry,
)


class TestVersionRegistry:
    """Test version registry functionality"""

    def test_register_version(self):
        """Test registering a new version"""
        registry = VersionRegistry()
        model_hash = hashlib.sha256(b"test_model").hexdigest()

        version_id = registry.register_version("test-model", model_hash)

        assert version_id is not None
        assert "test-model" in version_id
        assert registry.get_version(version_id) is not None

    def test_promote_version(self):
        """Test promoting a version"""
        registry = VersionRegistry()
        model_hash = hashlib.sha256(b"test_model").hexdigest()
        version_id = registry.register_version("test-model", model_hash)

        registry.promote_version(version_id)
        version = registry.get_version(version_id)

        assert version.promoted_at is not None
        assert version.health_status == HealthStatus.HEALTHY

    def test_rollback_version(self):
        """Test rolling back a version"""
        registry = VersionRegistry()
        model_hash = hashlib.sha256(b"test_model").hexdigest()
        version_id = registry.register_version("test-model", model_hash)

        registry.rollback_version(version_id, "degraded performance")
        version = registry.get_version(version_id)

        assert version.rolled_back_at is not None
        assert version.health_status == HealthStatus.ROLLED_BACK
        assert version.rollback_reason == "degraded performance"

    def test_get_latest_healthy_version(self):
        """Test getting latest promoted version"""
        registry = VersionRegistry()
        model_hash = hashlib.sha256(b"test_model").hexdigest()

        v1_id = registry.register_version("test-model", model_hash)
        v2_id = registry.register_version("test-model", model_hash + "2")

        # Promote both
        registry.promote_version(v1_id)
        registry.promote_version(v2_id)

        latest = registry.get_latest_healthy_version("test-model")
        assert latest == v2_id  # Most recent promotion


class TestHealthMonitor:
    """Test health monitoring"""

    def test_record_healthy_requests(self):
        """Test recording healthy request metrics"""
        monitor = HealthMonitor()
        thresholds = HealthThresholds(min_samples=10)
        monitor.thresholds = thresholds

        for _ in range(10):
            monitor.record_request(latency_ms=50.0, error=False)

        health = monitor.get_current_health()
        assert health["error_rate_percent"] == 0.0
        assert health["request_count"] == 10
        assert health["avg_latency_ms"] == 50.0

    def test_error_rate_detection(self):
        """Test error rate threshold detection"""
        monitor = HealthMonitor()
        thresholds = HealthThresholds(error_rate_percent=5.0, min_samples=20)
        monitor.thresholds = thresholds

        # Record 20 requests with 10% error rate
        for i in range(20):
            monitor.record_request(latency_ms=50.0, error=(i < 2))

        is_healthy, reason = monitor.check_health_thresholds()
        assert not is_healthy
        assert "error rate" in reason.lower()

    def test_latency_threshold_detection(self):
        """Test latency threshold detection"""
        monitor = HealthMonitor()
        thresholds = HealthThresholds(p99_latency_ms=100.0, min_samples=20)
        monitor.thresholds = thresholds

        # Record 20 requests with high latency
        for _ in range(20):
            monitor.record_request(latency_ms=150.0, error=False)

        is_healthy, reason = monitor.check_health_thresholds()
        assert not is_healthy
        assert "latency" in reason.lower()

    def test_window_evaluation(self):
        """Test window-based evaluation"""
        monitor = HealthMonitor()
        thresholds = HealthThresholds(
            error_rate_percent=5.0,
            min_samples=10,
            consecutive_bad_windows=2,
        )
        monitor.thresholds = thresholds

        # First bad window
        for i in range(10):
            monitor.record_request(latency_ms=50.0, error=(i < 1))  # 10% error

        should_rollback1, _ = monitor.evaluate_window()
        assert not should_rollback1  # Not enough consecutive bad windows yet

        # Second bad window
        for i in range(10):
            monitor.record_request(latency_ms=50.0, error=(i < 1))

        should_rollback2, _ = monitor.evaluate_window()
        assert should_rollback2  # Should trigger after 2 consecutive bad windows


class TestAutoVersionManager:
    """Test automatic version manager"""

    def test_deploy_model(self):
        """Test deploying a model"""
        manager = AutoVersionManager()
        model_hash = hashlib.sha256(b"test_model").hexdigest()

        version_id = manager.deploy_model("test-model", model_hash)

        assert version_id is not None
        assert manager.active_versions["test-model"] == version_id

    def test_record_inference(self):
        """Test recording inference metrics"""
        manager = AutoVersionManager()
        model_hash = hashlib.sha256(b"test_model").hexdigest()
        version_id = manager.deploy_model("test-model", model_hash)

        manager.record_inference("test-model", latency_ms=50.0, error=False)

        version = manager.registry.get_version(version_id)
        assert version.request_count == 1
        assert version.error_count == 0

    def test_promote_version(self):
        """Test promoting a version"""
        manager = AutoVersionManager()
        model_hash = hashlib.sha256(b"test_model").hexdigest()
        version_id = manager.deploy_model("test-model", model_hash)

        manager.promote_version(version_id)

        version = manager.registry.get_version(version_id)
        assert version.promoted_at is not None

    def test_auto_rollback_on_degradation(self):
        """Test automatic rollback on degraded metrics"""
        manager = AutoVersionManager()
        thresholds = HealthThresholds(
            error_rate_percent=5.0,
            min_samples=20,
            consecutive_bad_windows=1,
        )

        # Deploy v1 (stable)
        model_hash_v1 = hashlib.sha256(b"model_v1").hexdigest()
        v1_id = manager.deploy_model("test-model", model_hash_v1, thresholds)
        manager.promote_version(v1_id)

        # Record healthy traffic for v1
        for _ in range(20):
            manager.record_inference("test-model", latency_ms=50.0, error=False)

        # Deploy v2 (broken)
        model_hash_v2 = hashlib.sha256(b"model_v2").hexdigest()
        v2_id = manager.deploy_model("test-model", model_hash_v2, thresholds)

        # Record degraded traffic for v2
        for i in range(20):
            manager.record_inference("test-model", latency_ms=50.0, error=(i < 2))  # 10% error

        # Trigger rollback check
        rollback_called = False
        def on_rollback(vid, reason):
            nonlocal rollback_called
            rollback_called = True

        manager.on_rollback(on_rollback)
        manager.evaluate_health_windows()

        # Should have triggered rollback
        v2_version = manager.registry.get_version(v2_id)
        assert v2_version.health_status == HealthStatus.ROLLED_BACK

        # Should have switched back to v1
        assert manager.active_versions["test-model"] == v1_id

    def test_rollback_handlers(self):
        """Test rollback event handlers"""
        manager = AutoVersionManager()

        called_args = []
        def handler(version_id, reason):
            called_args.append((version_id, reason))

        manager.on_rollback(handler)
        manager._notify_rollback("test-version", "test reason")

        assert len(called_args) == 1
        assert called_args[0] == ("test-version", "test reason")

    def test_get_version_status(self):
        """Test getting version status"""
        manager = AutoVersionManager()
        model_hash = hashlib.sha256(b"test_model").hexdigest()
        version_id = manager.deploy_model("test-model", model_hash)

        for _ in range(20):
            manager.record_inference("test-model", latency_ms=50.0, error=False)

        status = manager.get_version_status(version_id)

        assert status is not None
        assert status["version_id"] == version_id
        assert status["request_count"] == 20
        assert "current_health" in status

    def test_get_model_status(self):
        """Test getting model status with all versions"""
        manager = AutoVersionManager()

        v1_hash = hashlib.sha256(b"model_v1").hexdigest()
        v1_id = manager.deploy_model("test-model", v1_hash)

        v2_hash = hashlib.sha256(b"model_v2").hexdigest()
        v2_id = manager.deploy_model("test-model", v2_hash)

        status = manager.get_model_status("test-model")

        assert status["model_id"] == "test-model"
        assert len(status["versions"]) == 2
        assert status["active_version"] == v2_id

    def test_manual_rollback(self):
        """Test manual rollback to specific version"""
        manager = AutoVersionManager()

        # Deploy v1 and promote
        v1_hash = hashlib.sha256(b"model_v1").hexdigest()
        v1_id = manager.deploy_model("test-model", v1_hash)
        manager.promote_version(v1_id)

        # Deploy v2 and promote
        v2_hash = hashlib.sha256(b"model_v2").hexdigest()
        v2_id = manager.deploy_model("test-model", v2_hash)
        manager.promote_version(v2_id)

        # Rollback to v1
        success = manager.manual_rollback(v1_id, "Manual rollback test")
        assert success
        assert manager.active_versions["test-model"] == v1_id

        # v2 should be marked as rolled back
        v2_version = manager.registry.get_version(v2_id)
        assert v2_version.health_status == HealthStatus.ROLLED_BACK

    def test_list_versions(self):
        """Test listing all versions"""
        manager = AutoVersionManager()

        v1_hash = hashlib.sha256(b"model_v1").hexdigest()
        v1_id = manager.deploy_model("test-model", v1_hash)

        v2_hash = hashlib.sha256(b"model_v2").hexdigest()
        v2_id = manager.deploy_model("test-model", v2_hash)

        versions = manager.list_versions("test-model")
        assert len(versions) == 2
        # Newest first
        assert versions[0]["version_id"] == v2_id
        assert versions[1]["version_id"] == v1_id

    def test_get_rollback_candidates(self):
        """Test getting only healthy, promotable versions"""
        manager = AutoVersionManager()

        # Deploy v1 and promote
        v1_hash = hashlib.sha256(b"model_v1").hexdigest()
        v1_id = manager.deploy_model("test-model", v1_hash)
        manager.promote_version(v1_id)

        # Deploy v2 (canary)
        v2_hash = hashlib.sha256(b"model_v2").hexdigest()
        v2_id = manager.deploy_model("test-model", v2_hash)
        # Don't promote v2

        # Deploy v3 and promote
        v3_hash = hashlib.sha256(b"model_v3").hexdigest()
        v3_id = manager.deploy_model("test-model", v3_hash)
        manager.promote_version(v3_id)

        candidates = manager.get_rollback_candidates("test-model")

        # Should only include promoted versions (v1 and v3)
        candidate_ids = [c["version_id"] for c in candidates]
        assert v1_id in candidate_ids
        assert v3_id in candidate_ids
        assert v2_id not in candidate_ids

    def test_manual_rollback_nonexistent_version(self):
        """Test manual rollback to non-existent version"""
        manager = AutoVersionManager()

        success = manager.manual_rollback("nonexistent-version", "test")
        assert not success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
