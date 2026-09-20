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
