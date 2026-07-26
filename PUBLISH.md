# Publishing PyStreamAI to GitHub

Guide to publish wheels and documentation to GitHub (source code excluded).

## Prerequisites

- GitHub account
- `gh` CLI installed: https://cli.github.com/
- Git configured with your GitHub account

## Step 1: Create GitHub Repository

```bash
# Create new repository
gh repo create PyStreamAI \
  --owner Mullassery \
  --public \
  --source=. \
  --remote=origin \
  --push

# Or manually at: https://github.com/new
# Name: PyStreamAI
# Owner: Mullassery
# Description: "The simplest way to deploy AI models to production. 40-50x faster inference. Zero YAML."
# Public
```

## Step 2: Verify .gitignore

The `.gitignore` file is already configured to exclude all source code:

```
src/                  # Rust source
pystreamai/*.py       # Python implementation
benchmarks/           # Test code
*.rs                  # Rust files
```

Verify what will be pushed:
```bash
git status
# Should show only:
# - docs/
# - examples/
# - .github/
# - LICENSE
# - README_PUBLIC.md (rename to README.md when pushing)
```

## Step 3: Prepare Repository

Rename README:
```bash
mv README_PUBLIC.md README.md
git add README.md
git commit -m "Update README for public release"
```

Remove development files:
```bash
# Make sure these are in .gitignore
ls -la | grep -E "(Cargo|src|pystreamai.*\.py|benchmarks)"
# Should show nothing (all ignored)
```

## Step 4: Push to GitHub

```bash
# Add remote
git remote add origin https://github.com/Mullassery/PyStreamAI.git

# Rename branch if needed
git branch -M main

# Push
git push -u origin main
```

Verify:
```bash
git remote -v
# Should show origin pointing to GitHub repo
```

## Step 5: Create Release

```bash
# Tag version
git tag v0.1.0 -m "PyStreamAI v0.1: Production-ready ML inference platform"
git push origin v0.1.0

# Or use GitHub CLI
gh release create v0.1.0 \
  --title "PyStreamAI v0.1" \
  --notes "First public release: 40-50x faster inference, zero YAML, production-ready"
```

## Step 6: Build and Publish Wheels

GitHub Actions will automatically build wheels when you tag a release.

Manually (if needed):
```bash
# Install dependencies
pip install maturin twine

# Build wheel
maturin build --release

# Upload to PyPI
twine upload dist/* -u __token__ -p $PYPI_TOKEN
```

Or set up GitHub Actions secrets:
1. Go to repo Settings → Secrets and variables → Actions
2. Add `PYPI_TOKEN` with your PyPI token
3. Commit the `.github/workflows/build-wheels.yml` file
4. On tag push, wheels auto-build and upload

## Step 7: Verify Public Release

```bash
# Check what's in the repo
git ls-files

# Should show only:
# - docs/
# - examples/
# - .github/
# - LICENSE
# - README.md
# - pyproject.toml
# - Cargo.toml (build config only)
# - setup.py (if included)

# Should NOT show:
# - src/ (source code)
# - pystreamai/*.py (implementation)
# - benchmarks/ (tests)
```

## Step 8: Test Installation

After wheels are published to PyPI:

```bash
# Install in fresh environment
python -m venv test_env
source test_env/bin/activate
pip install pystreamai

# Verify it works
python -c "from pystreamai import Platform; print('✓ PyStreamAI installed successfully')"
```

## What Gets Published

### ✅ Included
- Compiled wheels (.whl files)
- Documentation (docs/ folder)
- Examples (examples/ folder)
- LICENSE and README
- GitHub Actions workflows

### ❌ Excluded (by .gitignore)
- Rust source code (src/)
- Python implementation (pystreamai/*.py)
- Build artifacts (target/, build/)
- Test/benchmark code (benchmarks/)
- Development files (.claude/)

## GitHub Repository Structure (Public)

```
PyStreamAI/
├── .github/
│   └── workflows/
│       └── build-wheels.yml      # Auto-build and publish wheels
├── docs/
│   ├── GETTING_STARTED.md        # User onboarding
│   ├── API_REFERENCE.md          # Complete API docs
│   ├── DEPLOYMENT.md             # Production guide
│   └── OPTIMIZATION.md           # Performance tuning
├── examples/
│   ├── serve_model.py            # Basic example
│   └── start_http_server.py      # HTTP server example
├── LICENSE                        # Proprietary license
├── README.md                      # Public-facing readme
├── pyproject.toml                # Python package config
└── Cargo.toml                    # Rust build config (wheels only)
```

## Verification Checklist

- [ ] GitHub repo created
- [ ] .gitignore correctly excludes source
- [ ] `git ls-files` shows only docs/examples/config
- [ ] README.md present (renamed from README_PUBLIC.md)
- [ ] LICENSE included
- [ ] GitHub Actions workflow configured
- [ ] Release tagged (v0.1.0)
- [ ] Wheels built successfully
- [ ] Wheels published to PyPI
- [ ] `pip install pystreamai` works
- [ ] Documentation accessible on GitHub

## PyPI Setup

To publish wheels to PyPI:

1. Create PyPI account: https://pypi.org/account/register/
2. Get API token: https://pypi.org/manage/account/
3. Set GitHub secret: `PYPI_TOKEN`
4. Commit `.github/workflows/build-wheels.yml`
5. Tag release: `git tag v0.1.0 && git push origin v0.1.0`
6. GitHub Actions automatically builds and publishes wheels

## Maintaining Security

The `.gitignore` file ensures:
- No source code leaks
- Only compiled wheels distributed
- Proprietary implementation stays private
- Users get functionality without source access

If you need to update documentation:
```bash
# Edit docs/
git add docs/
git commit -m "Update documentation"
git push origin main

# No rebuild needed - documentation is plain markdown
```

If you need to release a new version:
```bash
# Update version in Cargo.toml and pyproject.toml
# Commit changes
git tag v0.2.0
git push origin v0.2.0

# GitHub Actions automatically builds and publishes v0.2.0 wheels
```

---

For questions or issues during release, check:
- GitHub Actions logs: repo → Actions → build-wheels
- PyPI package page: pypi.org/project/pystreamai/
- Installation troubleshooting: see docs/GETTING_STARTED.md
