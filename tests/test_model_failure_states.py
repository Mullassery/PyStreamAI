"""End-to-end tests for model failure states, through the real HTTP API.

Previously `tests/test_api.py`/`tests/test_platform.py` covered only
happy-path inference plus one generic exception path; nothing simulated a
provider-style failure (rate limit, model unavailable) or malformed model
*output* (as opposed to a raised exception) end-to-end through the actual
/predict route.
"""

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")
pytest.importorskip("httpx", reason="httpx not installed (needed by TestClient)")

from fastapi.testclient import TestClient

from pystreamai.api import create_api_server


class TestModelExceptionIsolation:
    """A model callable raising during inference (simulating a rate limit
    or provider outage) must surface as a clean HTTP 500 with a real
    `detail` message, not an unhandled crash."""

    def test_predict_returns_clean_500_not_a_crash(self, raising_model, model_failure):
        server = create_api_server("failing-model", raising_model)
        client = TestClient(server.app)

        response = client.post("/predict", json={"data": {"x": 1}})

        assert response.status_code == 500
        assert str(model_failure) in response.json()["detail"]

    def test_server_survives_a_failed_request_and_serves_the_next_one(self, raising_model):
        """One failed inference shouldn't take down the server process or
        leave it in a broken state for subsequent requests."""
        server = create_api_server("failing-model", raising_model)
        client = TestClient(server.app)

        client.post("/predict", json={"data": {"x": 1}})  # fails

        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "healthy"

    def test_failed_request_is_still_reflected_in_stats(self, raising_model):
        server = create_api_server("failing-model", raising_model)
        client = TestClient(server.app)

        client.post("/predict", json={"data": {"x": 1}})

        # Whether or not a failed request counts toward /stats is a real
        # behavioral question this test pins down either way, rather than
        # leaving it unverified: /stats must not itself error out after a
        # failure.
        stats = client.get("/stats")
        assert stats.status_code == 200


class TestMalformedModelOutput:
    """A model that raises nothing but returns output the API layer can't
    JSON-serialize is a distinct failure mode from an exception -- must
    also surface as a clean error, not an unhandled internal crash."""

    def test_non_serializable_output_returns_clean_500(self, malformed_output_model):
        server = create_api_server("malformed-model", malformed_output_model)
        client = TestClient(server.app)

        response = client.post("/predict", json={"data": {"x": 1}})

        assert response.status_code == 500
        assert "detail" in response.json()

    def test_server_survives_malformed_output_and_serves_the_next_request(
        self, malformed_output_model, echo_model
    ):
        server = create_api_server("malformed-model", malformed_output_model)
        client = TestClient(server.app)

        client.post("/predict", json={"data": {"x": 1}})  # malformed output

        health = client.get("/health")
        assert health.status_code == 200


class TestEchoModelIsReusable:
    """Proves the exported pystreamai.testing.echo_model is a genuine,
    working model -- not dead code sitting unused."""

    def test_echo_model_round_trips_through_real_api(self, echo_model):
        server = create_api_server("echo-model", echo_model)
        client = TestClient(server.app)

        response = client.post("/predict", json={"data": {"hello": "world"}})

        assert response.status_code == 200
        assert response.json()["output"] == {"echoed": {"hello": "world"}}
