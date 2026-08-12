# Publishing PyStreamAI

How releases are built and published. As of the 2026-08 restoration, the
full source (Python + Rust) is public in this repository - PyStreamAI is
no longer distributed as wheels-only with hidden source. If you're looking
for an older process document that described stripping `src/` and
`pystreamai/*.py` out of the public repo before pushing: that approach was
abandoned. It made three past releases (recorded on PyPI as 0.1.0-1.0.0)
ship a near-empty stub extension with no real functionality behind a
misleading README. Don't resurrect it.

## Versioning

Bump the version in **both** places before tagging - they must match:
- `pyproject.toml` → `[project] version`
- `Cargo.toml` → `[package] version`

`pystreamai/__init__.py` also has a `__version__` string; keep it in sync
too (nothing enforces this automatically yet - a genuine gap, see
`docs/ROADMAP.md`).

## Local build + publish (manual)

```bash
# From a clean checkout, with a Python 3.10+ venv active
pip install maturin twine
maturin build --release          # writes wheels to target/wheels/, NOT dist/
twine upload target/wheels/*
```

`~/.pypirc` (or `TWINE_USERNAME`/`TWINE_PASSWORD` env vars) must already be
configured with a PyPI API token for the `pystreamai` project.

Before uploading, check you're not colliding with an existing version -
PyPI never allows re-uploading a version number, even a bad one:

```bash
pip index versions pystreamai
# or: curl -s https://pypi.org/pypi/pystreamai/json | python3 -m json.tool
```

## CI build + publish

`.github/workflows/build-wheels.yml` builds and publishes automatically
when a `v*` tag is pushed:

```bash
git tag v1.1.0
git push origin v1.1.0
```

It currently authenticates to PyPI with a long-lived `PYPI_TOKEN` repo
secret. **Recommended follow-up**: switch to [PyPI Trusted Publishing](
https://docs.pypi.org/trusted-publishers/) (OIDC) via
`pypa/gh-action-pypi-publish`, which needs no stored token at all. That
requires registering this repo's GitHub Actions workflow as a trusted
publisher on the PyPI project's own settings page
(`pypi.org/manage/project/pystreamai/settings/publishing/`) - a
one-time step done on pypi.org, not something a commit here can do.

## What actually ships in the wheel

The compiled wheel embeds:
- `pystreamai/*.py` - the pure-Python package (see README for what's real
  vs. simulated in this layer)
- `pystreamai/_core.<platform>.so` - the compiled Rust extension
  (`src/*.rs`, built via PyO3/maturin)

Verify locally before tagging a release:

```bash
maturin build --release
pip install --force-reinstall target/wheels/*.whl
python3 -c "import pystreamai; print(pystreamai.__version__); from pystreamai import _core; print(_core.Platform)"
pytest
```
