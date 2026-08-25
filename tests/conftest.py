"""Shared pytest fixtures.

Exposes the reusable mock models from `pystreamai.testing` as pytest
fixtures, so this repo's own tests -- and downstream projects' -- get
"a shared, importable mock server" and failure-state fixtures
out-of-the-box instead of each test file hand-writing its own inline
model (previously: `echo_model` was copy-defined locally in `test_api.py`
with no shared/importable version, and nothing simulated a real model
failure at all).
"""

import pytest

from pystreamai.testing import (
    ModelUnavailableError,
    RateLimitError,
    echo_model as _echo_model,
    failing_model,
    malformed_output_model as _malformed_output_model,
)


@pytest.fixture
def echo_model():
    """Happy-path mock model."""
    return _echo_model


@pytest.fixture(params=[RateLimitError, ModelUnavailableError])
def model_failure(request) -> Exception:
    """Parametrized over the simulated model-provider failure states -- a
    test using this fixture runs once per failure type."""
    return request.param("simulated failure")


@pytest.fixture
def raising_model(model_failure):
    """A model callable that always raises `model_failure`."""
    return failing_model(model_failure)


@pytest.fixture
def malformed_output_model():
    """A model that succeeds but returns non-serializable output."""
    return _malformed_output_model
