"""BentoML packaging manifest adapter.

PyStreamAI's deployment layer can describe a model's files, dependencies and
API signature. This module emits a small, portable manifest plus a BentoML
service template, so users can hand off to BentoML without PyStreamAI
reimplementing a full model store.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional


def build_manifest(
    name: str,
    *,
    model_path: str,
    api_name: str = "predict",
    python_version: str = "3.11",
    dependencies: Optional[Iterable[str]] = None,
    labels: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(model_path)
    return {
        "name": name,
        "model": {"path": str(path), "size_bytes": path.stat().st_size},
        "api": {"name": api_name, "input": "json", "output": "json"},
        "python_version": python_version,
        "dependencies": sorted(dependencies or []),
        "labels": labels or {},
    }


def write_manifest(manifest: dict[str, Any], path: str | Path) -> str:
    Path(path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return str(path)


def service_template(manifest: dict[str, Any], *, module: str = "service") -> str:
    api = manifest["api"]["name"]
    return f'''"""Generated BentoML service template for {manifest["name"]}."""
import bentoml

@bentoml.service(name="{manifest["name"]}")
class {module.title().replace("_", "")}:
    @bentoml.api
    def {api}(self, payload: dict) -> dict:
        # Load the model from the manifest path or a registered BentoML model.
        return {{"prediction": None, "input": payload}}
'''


__all__ = ["build_manifest", "service_template", "write_manifest"]
