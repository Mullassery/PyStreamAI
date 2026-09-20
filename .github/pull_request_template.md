## What does this change do?

<!-- Describe the change and why it's needed. -->

## How was this tested?

<!-- Real commands you ran, e.g. `pytest`, `cargo test --release`, `ruff check .`,
     `maturin develop --release`. If a code path wasn't exercised, say so - don't
     imply test coverage that doesn't exist. -->

## Checklist

- [ ] `pytest` passes locally
- [ ] `ruff check .` passes locally
- [ ] `cargo test --release` passes locally (if `src/*.rs` changed)
- [ ] If this changes behavior described in `README.md` / `docs/`, those docs are updated in the same PR
- [ ] If this adds a new simulated/placeholder/advisory-only code path, it is labeled as such in code comments and in the README (see "no fake stubs" precedent in this repo - don't add unlabeled placeholder logic)
