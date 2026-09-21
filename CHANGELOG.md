# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

This file did not exist before 2026-09-19 - there is no reconstructed
history here for `v0.1.0` through `v2.1.0`. For that history, see
`git log`, the repository's git tags (`v0.1.0`, `v0.2.0`, `v0.2.1`,
`v0.3.0`, `v2.0.0`, ...), and `docs/PUBLISH.md` /
`docs/ROADMAP.md`. Do not treat the absence of pre-2026-09-19 entries as
"nothing happened" - it means this file wasn't being kept yet.

## [Unreleased]

### Fixed
- `pyproject.toml`'s `[onnx]` extra was missing the `onnx` package itself,
  so `tests/test_onnx_runtime.py` / `tests/test_platform_onnx_predict.py`
  were silently skipped in CI. Added `onnx>=1.15.0,<1.23.0` to the
  extra - pinned below 1.23.0 because onnx 1.23.0 is the first release
  that writes IR version 14 graphs, which the currently-resolved
  `onnxruntime` (1.30.0, satisfying `>=1.15.0`) rejects (max supported
  IR version 13; verified directly, not just from the error message).
  onnx 1.15.0-1.22.0 all write IR versions 9-13, which onnxruntime
  1.30.0 loads correctly. All 8 previously-skipped ONNX tests now run
  and pass for real.
- `pyproject.toml`'s `dev` extra was missing `httpx`, so
  `tests/test_api.py` and `tests/test_model_failure_states.py` (15
  tests total) were also silently skipped in CI via
  `pytest.importorskip("httpx", ...)` (needed by
  `fastapi.testclient.TestClient`) - same shape of bug as the ONNX gap
  above, found while investigating it. Added `httpx>=0.24.0` to `dev`.
- Full suite is now 169 passed / 0 skipped (previously 146 passed / 4
  skipped when `[dev,serving,onnx]` was installed without the two fixes
  above), `ruff check .` clean, `cargo test --release` 16 passed.

### Added
- `ROADMAP_HONEST.md` - honest, bucketed roadmap and technical debt list
  (dead Rust code, fake `llm_optimization`/`edge_deployment` modules, the
  ONNX test suite's actual CI/version-compatibility status, etc).
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, this `CHANGELOG.md`.
- `.github/dependabot.yml` (pip, cargo, github-actions ecosystems).
- `.github/ISSUE_TEMPLATE/bug_report.yml`, `feature_request.yml`, and
  `.github/pull_request_template.md`.
- `.github/workflows/security-audit.yml` - `pip-audit` + `cargo audit`,
  scheduled weekly plus manual dispatch (not run on every PR). Not yet
  confirmed green on real GitHub Actions infrastructure - authored and
  `actionlint`-checked in a sandbox with no network access to PyPI/
  crates.io/GitHub's advisory database.
- `Cargo.lock` is now committed (previously gitignored) for reproducible
  builds of this cdylib/PyO3 extension.

### Changed
- `pyproject.toml`: added `[tool.maturin] include = ["LICENSE"]` so
  `maturin sdist` doesn't omit `LICENSE` from the sdist tarball (a
  recurring issue across this org's maturin-based projects that causes
  PyPI to reject the upload).
- `.github/workflows/tests.yml` / `build-wheels.yml`: replaced the
  unmaintained `actions-rs/toolchain@v1` with `dtolnay/rust-toolchain@stable`,
  and bumped `actions/checkout` to v4 and `actions/setup-python` to v5
  (flagged by `actionlint` as running on a deprecated/EOL runner).
- `README.md`: disclosed that `pystreamai.llm_optimization` and
  `pystreamai.edge_deployment` are almost entirely fabricated (return
  hardcoded values, never touch real models/files) - not previously
  documented anywhere outside a test-file comment; disclosed that the
  Rust extension's `Platform.train()`/`serve()`/`predict()` are literal
  string formatters with no real behavior; disclosed that
  `tests/test_onnx_runtime.py` / `tests/test_platform_onnx_predict.py`
  are silently skipped in CI (missing `onnx` package) and actually fail
  if run manually due to an `onnx`/`onnxruntime` IR-version mismatch.

### Removed
- `.env.example` - described `API_KEY`/`DATABASE_URL`/`REDIS_URL`/
  `ELASTICSEARCH_URL`/`SECRET_KEY` etc., none of which are read anywhere
  in this codebase (`grep -r "os.environ\|os.getenv" pystreamai/ src/`
  returns nothing). Kept unremoved, it implied config surface that
  doesn't exist.

## Earlier versions

Version bumps prior to this file (`0.1.0` → `2.1.0`) happened via
`pyproject.toml`/`Cargo.toml` version fields and git tags without a
changelog. See `git log --oneline` and `docs/PUBLISH.md` for the real
mechanics of how releases were built and what changed in the `v1.1.0`
restoration and the `v2.0.0` breaking-change release.
