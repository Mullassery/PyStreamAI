"""Tests for pystreamai.onnx_runtime - real ONNX Runtime inference.

Builds a tiny real .onnx graph (y = x + 1) with the `onnx` package and runs
it through ONNXModelLoader/ONNXBenchmark for real - no mocking - to verify
the wrapper around onnxruntime.InferenceSession actually works end to end.
onnxruntime/numpy/onnx are optional dependencies (see pyproject.toml
[project.optional-dependencies].onnx); tests are skipped if unavailable.
"""

import pytest

np = pytest.importorskip("numpy")
onnx = pytest.importorskip("onnx", reason="onnx package not installed")
pytest.importorskip("onnxruntime", reason="onnxruntime not installed")

from pystreamai.onnx_runtime import ONNXModelLoader, ONNXBenchmark  # noqa: E402


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


class TestONNXModelLoader:
    def test_load_populates_input_output_names(self, add_one_model_path):
        loader = ONNXModelLoader(add_one_model_path).load(providers=["CPUExecutionProvider"])
        assert loader.input_names == ["input"]
        assert loader.output_names == ["output"]

    def test_infer_before_load_raises(self):
        loader = ONNXModelLoader("does-not-matter.onnx")
        with pytest.raises(RuntimeError):
            loader.infer({"input": np.zeros((1, 4), dtype=np.float32)})

    def test_infer_computes_real_result(self, add_one_model_path):
        loader = ONNXModelLoader(add_one_model_path).load(providers=["CPUExecutionProvider"])
        result = loader.infer({"input": np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)})

        np.testing.assert_allclose(result[0], [[2.0, 3.0, 4.0, 5.0]])

    def test_get_input_info_reports_shape(self, add_one_model_path):
        loader = ONNXModelLoader(add_one_model_path).load(providers=["CPUExecutionProvider"])
        info = loader.get_input_info()
        assert "input" in info


class TestONNXBenchmark:
    def test_benchmark_throughput_returns_positive_latency(self, add_one_model_path):
        loader = ONNXModelLoader(add_one_model_path).load(providers=["CPUExecutionProvider"])
        bench = ONNXBenchmark(loader)

        result = bench.benchmark_throughput(batch_size=1, num_iterations=5)

        assert result["latency_ms"] > 0
        assert result["throughput_req_sec"] > 0
