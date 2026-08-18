"""Tests for Endpoint/Platform wiring to real ONNX inference.

Platform.serve() / Endpoint.predict() used to always return a fabricated
"prediction from {model_id}" string with a hardcoded latency_ms=42.5,
regardless of what model was actually passed in. This verifies serve()
with a real .onnx file runs genuine inference through onnxruntime instead
of the simulated path, and that non-ONNX models still fall back to the
(now explicitly labeled) simulated response.
"""

import pytest

np = pytest.importorskip("numpy")
onnx = pytest.importorskip("onnx", reason="onnx package not installed")
pytest.importorskip("onnxruntime", reason="onnxruntime not installed")

from pystreamai.platform import Platform, Endpoint  # noqa: E402


@pytest.fixture
def add_one_model_path(tmp_path):
    """A minimal real ONNX graph: output = input + 1.0, shape (N, 4)."""
    from onnx import helper, TensorProto

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 4])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 4])
    one = helper.make_tensor("one", TensorProto.FLOAT, [4], [1.0, 1.0, 1.0, 1.0])

    add_node = helper.make_node("Add", ["input", "one"], ["output"])
    graph = helper.make_graph([add_node], "add_one", [x], [y], initializer=[one])
    model = helper.make_model(graph, producer_name="pystreamai-tests")
    model.opset_import[0].version = 14

    path = tmp_path / "add_one.onnx"
    onnx.save(model, str(path))
    return str(path)


def test_serve_with_real_onnx_model_runs_real_inference(add_one_model_path):
    platform = Platform()
    endpoint = platform.serve(add_one_model_path, replicas=1)

    result = endpoint.predict({"input": np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)})

    assert result["simulated"] is False
    np.testing.assert_allclose(result["output"][0], [[2.0, 3.0, 4.0, 5.0]])
    # Real measured latency, not the old hardcoded 42.5
    assert result["latency_ms"] >= 0
    assert result["latency_ms"] != 42.5


def test_serve_with_non_onnx_model_falls_back_to_labeled_simulated_response():
    platform = Platform()
    endpoint = platform.serve(model=object(), replicas=1)

    result = endpoint.predict({"anything": 1})

    assert result["simulated"] is True
    assert result["latency_ms"] == 42.5


def test_serve_with_no_model_falls_back_to_labeled_simulated_response():
    platform = Platform()
    endpoint = platform.serve(model=None, replicas=1)

    result = endpoint.predict({})

    assert result["simulated"] is True


def test_endpoint_accepts_a_preloaded_onnx_model_loader(add_one_model_path):
    from pystreamai.onnx_runtime import ONNXModelLoader

    loader = ONNXModelLoader(add_one_model_path).load(providers=["CPUExecutionProvider"])
    endpoint = Endpoint("model-1", replicas=1, model=loader)

    result = endpoint.predict({"input": np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)})

    assert result["simulated"] is False
    np.testing.assert_allclose(result["output"][0], [[1.0, 1.0, 1.0, 1.0]])
