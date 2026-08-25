"""Reusable test doubles for pystreamai, exported for downstream use.

Previously each test file hand-wrote its own inline mock model (e.g.
`echo_model` in `tests/test_api.py`) with no shared, importable version, and
nothing simulated realistic model-failure states (a provider rate limit, a
model returning malformed/non-serializable output) at all -- only the
happy path was covered.

`Platform.serve()`/`Endpoint` call whatever `model` you pass in directly
(see `platform.py`): a real `.onnx` file, something with `.predict()`, or a
plain callable. These mocks are all plain callables, so they work as a
`model` argument anywhere in the package -- `Platform.serve()`,
`create_api_server()`, `InferenceServer.load_model()`.
"""

from typing import Any, Dict


class RateLimitError(Exception):
    """Raised by a test double to simulate a model provider rate limit."""


class ModelUnavailableError(Exception):
    """Raised by a test double to simulate a model/backend being down."""


def echo_model(data: Any) -> Dict[str, Any]:
    """Happy-path mock: always succeeds, echoing back its input."""
    return {"echoed": data}


def failing_model(failure: Exception):
    """Return a callable model that always raises `failure` -- for testing
    that the real exception-handling path (InferenceEngine._run_batch_inference
    -> InferenceServer.predict -> APIServer's /predict handler) surfaces a
    clean, structured error instead of an unhandled crash.

    Usage: `create_api_server("m", failing_model(RateLimitError("...")))`.
    """

    def _model(data: Any) -> Any:
        raise failure

    return _model


class NotJSONSerializable:
    """A plain object with no JSON/pydantic-compatible representation --
    for testing that malformed model *output* (not an exception) is also
    caught cleanly, not just exceptions raised during inference."""


def malformed_output_model(data: Any) -> "NotJSONSerializable":
    """A model that "succeeds" (raises nothing) but returns output the API
    layer can't serialize into a response -- the other real "malformed
    model output" failure mode, distinct from the model raising."""
    return NotJSONSerializable()
