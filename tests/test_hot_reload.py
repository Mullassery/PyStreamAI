"""Tests for pystreamai.hot_reload - zero-downtime model version switching."""

import pytest

from pystreamai.hot_reload import HotReloadManager


class TestHotReloadManager:
    def test_load_model_sets_first_version_as_current(self):
        manager = HotReloadManager()
        manager.load_model("m1", "v1", model_loader=lambda: "model-v1")

        assert manager.current_versions["m1"] == "v1"
        assert manager.get_current_model("m1") == "model-v1"

    def test_second_load_does_not_change_current_version(self):
        manager = HotReloadManager()
        manager.load_model("m1", "v1", model_loader=lambda: "model-v1")
        manager.load_model("m1", "v2", model_loader=lambda: "model-v2")

        assert manager.current_versions["m1"] == "v1"

    def test_load_model_propagates_loader_exceptions(self):
        manager = HotReloadManager()

        def broken_loader():
            raise ValueError("bad model file")

        with pytest.raises(ValueError):
            manager.load_model("m1", "v1", model_loader=broken_loader)

        assert "m1" not in manager.current_versions

    def test_activate_version_switches_current_and_deactivates_old(self):
        manager = HotReloadManager()
        manager.load_model("m1", "v1", model_loader=lambda: "model-v1")
        manager.load_model("m1", "v2", model_loader=lambda: "model-v2")

        manager.activate_version("m1", "v2")

        assert manager.current_versions["m1"] == "v2"
        assert manager.get_current_model("m1") == "model-v2"
        assert manager.models["m1:v1"].active is False

    def test_activate_unknown_version_raises(self):
        manager = HotReloadManager()
        with pytest.raises(ValueError):
            manager.activate_version("m1", "does-not-exist")

    def test_get_current_model_for_unknown_model_returns_none(self):
        manager = HotReloadManager()
        assert manager.get_current_model("no-such-model") is None

    def test_get_current_model_increments_requests_served(self):
        manager = HotReloadManager()
        manager.load_model("m1", "v1", model_loader=lambda: "model-v1")

        manager.get_current_model("m1")
        manager.get_current_model("m1")

        assert manager.models["m1:v1"].requests_served == 2

    def test_get_model_status_reports_all_versions(self):
        manager = HotReloadManager()
        manager.load_model("m1", "v1", model_loader=lambda: "model-v1")
        manager.load_model("m1", "v2", model_loader=lambda: "model-v2")
        manager.activate_version("m1", "v2")

        status = manager.get_model_status("m1")
        assert status["current_version"] == "v2"
        assert set(status["versions"].keys()) == {"v1", "v2"}
        assert status["versions"]["v2"]["current"] is True
        assert status["versions"]["v1"]["active"] is False
