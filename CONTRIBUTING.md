# Contributing to PyStreamAI

Thanks for considering a contribution. This project is early-stage
(v2.1.0, source-available, Apache-2.0) - read the README's "Status" and
"How this works today" sections first so you know which parts are real
and which are documented placeholders before you patch either.

## Before you start

- Check `README.md`'s "Known issues" section and `ROADMAP_HONEST.md` -
  your bug or idea may already be tracked there with more detail than an
  issue would have.
- For anything beyond a small fix, open an issue first to discuss the
  approach. This avoids wasted work on something that conflicts with the
  project's direction.

## Development setup

```bash
git clone https://github.com/Mullassery/PyStreamAI.git
cd PyStreamAI
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,serving,onnx]"
maturin develop --release   # builds the Rust extension (src/*.rs) into the venv
```

Requires Python 3.10+ and a Rust toolchain (stable) if you're touching
`src/*.rs`.

## Running checks locally

```bash
pytest                # Python test suite
ruff check .           # lint (pyflakes + core pycodestyle rules only, see pyproject.toml)
cargo test --release   # Rust unit tests (src/*.rs)
```

All three must pass before opening a PR. If you add a new module or
function, add a real test for it - not a smoke test that only checks it
doesn't raise. If you can't add a real test yet (e.g. the feature is
genuinely a hardcoded placeholder), say so explicitly in the code comment
and the PR description; don't leave it implied.

## The "no fake stubs" rule

This codebase has a documented history of shipping code that looked done
but wasn't (see README's "Performance claims" and "Known issues"
sections, and `ROADMAP_HONEST.md`'s technical debt list for current
examples still being cleaned up). When contributing:

- If a function can't do the real thing yet (no test hardware, no real
  backend, etc.), it must say so in its docstring and return values that
  can't be mistaken for real measurements/results - not a plausible-looking
  hardcoded number.
- Don't add "TODO: implement later" methods that silently return success.
  Either implement it for real or leave it unimplemented and raise
  `NotImplementedError`.
- If you're fixing one of the fake modules in `ROADMAP_HONEST.md` (e.g.
  `pystreamai.llm_optimization`, `pystreamai.edge_deployment`), remove the
  "not real" disclosure for that specific function in the same PR that
  makes it real - don't let docs and code drift apart again.

## Commit / PR conventions

- Keep commits focused; one logical change per commit.
- Update `README.md` and the relevant file under `docs/` in the same PR
  as any behavior change - stale docs are treated as a bug here.
- Add an entry to `CHANGELOG.md` under `[Unreleased]` for user-visible
  changes.
- Use the PR template (`.github/pull_request_template.md`) checklist.

## Reporting security issues

Do not open a public issue for a security vulnerability - see
`SECURITY.md`.

## License

By contributing, you agree your contributions are licensed under this
project's Apache License 2.0 (see `LICENSE`).
