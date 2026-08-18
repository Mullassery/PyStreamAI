"""Tests for pystreamai.api - the FastAPI HTTP surface.

fastapi/uvicorn are optional dependencies (see pyproject.toml
[project.optional-dependencies].serving); skipped if unavailable.

IMPORTANT: this API has no authentication (see README "Security" section).
These tests exercise the endpoints exactly as an anonymous caller would -
they do not test auth because there isn't any yet.
"""

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")
pytest.importorskip("httpx", reason="httpx not installed (needed by TestClient)")

from fastapi.testclient import TestClient

from pystreamai.api import create_api_server


def echo_model(data):
    return {"echoed": data}


@pytest.fixture
def client():
    server = create_api_server(
        model_id="demo-model", model=echo_model, gpu_type="A100", num_gpus=1, port=8000
    )
    return TestClient(server.app)


def test_health_endpoint_reports_healthy(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_endpoint_returns_prediction(client):
    response = client.post("/predict", json={"data": {"text": "hello"}})
    assert response.status_code == 200
    body = response.json()
    assert body["model_id"] == "demo-model"
    assert body["latency_ms"] > 0


def test_stats_endpoint_reflects_prior_requests(client):
    client.post("/predict", json={"data": {"text": "hello"}})
    response = client.get("/stats")
    assert response.status_code == 200
    assert response.json()["requests"] == 1


def test_model_info_for_unknown_model_returns_404(client):
    response = client.get("/models/not-the-configured-model/info")
    assert response.status_code == 404


def test_model_info_for_configured_model_returns_ready(client):
    response = client.get("/models/demo-model/info")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_predict_endpoint_has_no_authentication_required(client):
    # Documents current (v0.x) behavior: anyone who can reach the server can
    # run inference. See README's explicit "no built-in auth" warning.
    response = client.post("/predict", json={"data": {}})
    assert response.status_code != 401
    assert response.status_code != 403
