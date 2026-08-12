"""Tests for pystreamai.deployment - canary/A-B deployment routing."""

from pystreamai.deployment import (
    DeploymentManager,
    DeploymentStatus,
    CanaryDeployment,
    ABTestDeployment,
    DeploymentVersion,
)


def make_version(model_id="m1", version="v1"):
    return DeploymentVersion(model_id=model_id, version=version, uri=f"s3://models/{version}")


class TestDeploymentManager:
    def test_deploy_model_sets_active_deployment(self):
        manager = DeploymentManager()
        dep = manager.deploy_model("m1", "v1", "s3://models/v1")

        assert dep.status == DeploymentStatus.RUNNING
        assert manager.active_deployment == "v1"

    def test_first_deploy_becomes_active_second_does_not_override(self):
        manager = DeploymentManager()
        manager.deploy_model("m1", "v1", "s3://models/v1")
        manager.deploy_model("m1", "v2", "s3://models/v2")

        assert manager.active_deployment == "v1"
        assert "v2" in manager.deployments

    def test_route_request_without_canary_returns_active(self):
        manager = DeploymentManager()
        manager.deploy_model("m1", "v1", "s3://models/v1")

        routed = manager.route_request()
        assert routed.version == "v1"

    def test_start_canary_then_promote_switches_active(self):
        # Regression test: DeploymentManager.promote_canary() used to read
        # self.canary.canary_version.version *after* calling
        # self.canary.promote_canary(), which already sets canary_version
        # to None - always raising AttributeError. Fixed to read
        # self.canary.current_version (which promote_canary() moves the
        # promoted version into) instead.
        manager = DeploymentManager()
        manager.deploy_model("m1", "v1", "s3://models/v1")
        manager.deploy_model("m1", "v2", "s3://models/v2")

        manager.start_canary_deployment("v2", traffic_percent=50.0)
        assert manager.canary is not None

        manager.promote_canary()
        assert manager.active_deployment == "v2"
        assert manager.canary is None

    def test_start_canary_without_active_deployment_is_noop(self):
        manager = DeploymentManager()
        manager.start_canary_deployment("v2")
        assert manager.canary is None

    def test_rollback_without_canary_does_not_raise(self):
        manager = DeploymentManager()
        manager.rollback_deployment()  # should just log a warning

    def test_ab_test_routes_only_to_known_variants(self):
        manager = DeploymentManager()
        manager.deploy_model("m1", "a", "s3://a")
        manager.deploy_model("m1", "b", "s3://b")
        manager.start_ab_test("a", "b", split_percent=100.0)

        for _ in range(20):
            routed = manager.route_request()
            assert routed.version in ("a", "b")

    def test_get_deployment_status_includes_canary_and_ab_sections(self):
        manager = DeploymentManager()
        manager.deploy_model("m1", "v1", "s3://v1")
        manager.deploy_model("m1", "v2", "s3://v2")
        manager.start_canary_deployment("v2")

        status = manager.get_deployment_status()
        assert "canary" in status
        assert status["active_deployment"] == "v1"


class TestCanaryDeployment:
    def test_route_request_always_returns_current_at_zero_percent(self):
        canary = CanaryDeployment(make_version(version="v1"), make_version(version="v2"), canary_traffic_percent=0.0)
        for _ in range(10):
            assert canary.route_request().version == "v1"

    def test_route_request_always_returns_canary_at_hundred_percent(self):
        canary = CanaryDeployment(make_version(version="v1"), make_version(version="v2"), canary_traffic_percent=100.0)
        for _ in range(10):
            assert canary.route_request().version == "v2"

    def test_get_canary_stats_before_any_traffic_is_empty(self):
        canary = CanaryDeployment(make_version(), make_version())
        assert canary.get_canary_stats() == {}

    def test_promote_canary_clears_canary_slot(self):
        canary = CanaryDeployment(make_version(version="v1"), make_version(version="v2"))
        canary.promote_canary()
        assert canary.current_version.version == "v2"
        assert canary.canary_version is None


class TestABTestDeployment:
    def test_route_request_consistent_for_same_user(self):
        ab = ABTestDeployment(make_version(version="a"), make_version(version="b"), split_percent=50.0)
        first = ab.route_request(user_id="alice")
        second = ab.route_request(user_id="alice")
        assert first.version == second.version

    def test_record_metric_and_get_stats(self):
        ab = ABTestDeployment(make_version(version="a"), make_version(version="b"))
        ab.record_metric("a", "accuracy", 0.9)
        ab.record_metric("a", "accuracy", 1.0)

        stats = ab.get_stats()
        assert stats["variant_a"]["metrics"]["accuracy"] == 0.95
