"""BentoML manifest tests."""

from __future__ import annotations

import json
from pathlib import Path

from pystreamai.bentoml_manifest import build_manifest, service_template, write_manifest


def test_build_and_write_manifest(tmp_path: Path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"onnx")
    manifest = build_manifest("demo", model_path=str(model), dependencies=["onnxruntime"])
    out = tmp_path / "bento.json"
    write_manifest(manifest, out)
    data = json.loads(out.read_text())
    assert data["model"]["size_bytes"] == 4
    assert "onnxruntime" in data["dependencies"]


def test_service_template_uses_api_name(tmp_path: Path):
    model = tmp_path / "m"
    model.write_bytes(b"x")
    manifest = build_manifest("demo", model_path=str(model), api_name="score")
    text = service_template(manifest)
    assert "def score" in text
    assert "bentoml.service" in text
