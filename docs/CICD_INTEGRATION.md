# CI/CD Integration Guide

Integrate PyStreamAI's automatic model versioning with your CI/CD platform. Supported platforms include ArgoCD, GitHub Actions, GitLab CI, Jenkins, and more.

## Overview

The CI/CD integration layer provides:
- **Multi-platform support** — Works with ArgoCD, GitHub Actions, GitLab CI, Jenkins, etc.
- **Unified API** — Single interface for all CI/CD platforms
- **Automatic tracking** — Every deployment is logged to your CI/CD system
- **Rollback triggers** — Programmatically trigger rollbacks via CI/CD
- **Deployment history** — Query deployment history across platforms

## Quick Start

### Basic Setup

```python
from pystreamai import AutoVersionManager, CICDDispatcher, ArgoCDBackend

# Initialize version manager
manager = AutoVersionManager()

# Initialize CI/CD dispatcher
cicd = CICDDispatcher()

# Register platforms
cicd.register_backend(
    "argocd",
    ArgoCDBackend(
        server="https://argocd.example.com",
        token="<your-token>",
    ),
)

# Report deployment to all platforms
cicd.report_deployment_all(
    version_id="model-v2-1697456789",
    model_id="sentiment-classifier",
    state=DeploymentState.SUCCESS,
    metadata={
        "environment": "production",
        "request_count": 5000,
        "error_rate_percent": 0.5,
        "p99_latency_ms": 150.0,
    },
)
```

## Platform-Specific Setup

### ArgoCD (Kubernetes GitOps)

```python
from pystreamai import ArgoCDBackend, DeploymentState

# Initialize
argocd = ArgoCDBackend(
    server="https://argocd.example.com",
    token="<argocd-token>",
    namespace="argocd",
)

# Report deployment
argocd.report_deployment(
    version_id="sentiment-classifier-v2",
    model_id="sentiment-classifier",
    state=DeploymentState.IN_PROGRESS,
    metadata={
        "request_count": 0,
        "error_rate_percent": 0,
    },
)

# Trigger rollback
success = argocd.trigger_rollback(
    version_id="sentiment-classifier-v1",
    model_id="sentiment-classifier",
)

# Get history
history = argocd.get_deployment_history("sentiment-classifier")
for deployment in history:
    print(f"{deployment['version_id']} - {deployment['state']}")
```

### GitHub Actions

```python
from pystreamai import GitHubActionsBackend, DeploymentState

# Initialize
github = GitHubActionsBackend(
    repo="Mullassery/pystreamai",
    token="<github-token>",
)

# Report via check run
github.report_deployment(
    version_id="model-v2",
    model_id="sentiment-classifier",
    state=DeploymentState.SUCCESS,
    metadata={
        "commit_sha": "abc123def456",
        "request_count": 10000,
        "error_rate_percent": 0.2,
        "p99_latency_ms": 120.0,
    },
)

# Trigger workflow dispatch for rollback
success = github.trigger_rollback(
    version_id="model-v1",
    model_id="sentiment-classifier",
)
```

Requires GitHub Actions workflow to handle rollback:

```yaml
# .github/workflows/model-rollback.yml
name: Model Rollback

on:
  workflow_dispatch:
    inputs:
      model_id:
        description: Model to rollback
        required: true
      version_id:
        description: Version to rollback to
        required: true

jobs:
  rollback:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Rollback Model
        run: |
          pip install pystreamai
          python -c "
            from pystreamai import AutoVersionManager
            manager = AutoVersionManager()
            manager.manual_rollback('${{ inputs.version_id }}', 'Automated rollback')
          "
```

### GitLab CI

```python
from pystreamai import GitLabCIBackend, DeploymentState

# Initialize
gitlab = GitLabCIBackend(
    project_id="123456",
    server="https://gitlab.example.com",
    token="<gitlab-token>",
)

# Report deployment
gitlab.report_deployment(
    version_id="model-v2",
    model_id="sentiment-classifier",
    state=DeploymentState.SUCCESS,
    metadata={
        "environment": "production",
        "request_count": 5000,
    },
)

# Trigger rollback
success = gitlab.trigger_rollback(
    version_id="model-v1",
    model_id="sentiment-classifier",
)
```

### Jenkins

```python
from pystreamai import JenkinsBackend, DeploymentState

# Initialize
jenkins = JenkinsBackend(
    server="https://jenkins.example.com",
    job_name="deploy-model",
    token="<jenkins-token>",
)

# Report via build parameters
jenkins.report_deployment(
    version_id="model-v2",
    model_id="sentiment-classifier",
    state=DeploymentState.SUCCESS,
    metadata={
        "build_id": "123",
    },
)

# Trigger rollback job
success = jenkins.trigger_rollback(
    version_id="model-v1",
    model_id="sentiment-classifier",
)
```

## Multi-Platform Setup

Report deployments to multiple platforms simultaneously:

```python
from pystreamai import (
    CICDDispatcher,
    ArgoCDBackend,
    GitHubActionsBackend,
    DeploymentState,
)

# Initialize dispatcher
cicd = CICDDispatcher()

# Register multiple backends
cicd.register_backend(
    "argocd",
    ArgoCDBackend(
        server="https://argocd.example.com",
        token="<argocd-token>",
    ),
)

cicd.register_backend(
    "github",
    GitHubActionsBackend(
        repo="Mullassery/pystreamai",
        token="<github-token>",
    ),
)

# Report to all platforms at once
results = cicd.report_deployment_all(
    version_id="model-v2",
    model_id="sentiment-classifier",
    state=DeploymentState.SUCCESS,
    metadata={
        "request_count": 5000,
        "error_rate_percent": 0.2,
        "p99_latency_ms": 120.0,
    },
)

# Check results
for platform, success in results.items():
    print(f"{platform}: {'✓' if success else '✗'}")

# Trigger rollback on all platforms
rollback_results = cicd.trigger_rollback_all(
    version_id="model-v1",
    model_id="sentiment-classifier",
)

# Get combined history from all platforms
history = cicd.get_combined_history("sentiment-classifier")
for platform, deployments in history.items():
    print(f"\n{platform}:")
    for d in deployments:
        print(f"  {d['version_id']} - {d['state']}")
```

## Integration with AutoVersionManager

```python
from pystreamai import AutoVersionManager, GitHubDeploymentTracker

# Initialize version manager
manager = AutoVersionManager()

# Initialize GitHub tracker (auto-handles rollback alerts)
tracker = GitHubDeploymentTracker(
    repo="Mullassery/pystreamai",
    auto_version_manager=manager,
    token="<github-token>",
)

# Track a deployment
tracker.track_deployment(
    version_id="model-v2",
    model_id="sentiment-classifier",
    environment="production",
    commit_sha="abc123def456",
    create_release=True,
)

# Deploy model
model_hash = hashlib.sha256(b"weights").hexdigest()
version_id = manager.deploy_model("sentiment-classifier", model_hash)
manager.promote_version(version_id)

# Report success
tracker.post_deployment_success(
    version_id=version_id,
    model_id="sentiment-classifier",
    commit_sha="abc123def456",
)

# If deployment fails or metrics degrade, automatic rollback triggers
# and GitHub issue is automatically created!
```

## Deployment Workflows

### Canary → Production → GitOps

```python
from pystreamai import (
    AutoVersionManager,
    CICDDispatcher,
    ArgoCDBackend,
    DeploymentState,
)
import hashlib

manager = AutoVersionManager()
cicd = CICDDispatcher()
cicd.register_backend("argocd", ArgoCDBackend(...))

# 1. Deploy canary version
model_hash = hashlib.sha256(b"new_weights").hexdigest()
canary_version = manager.deploy_model("model", model_hash)

# Report canary to CI/CD
cicd.report_deployment_all(
    version_id=canary_version,
    model_id="model",
    state=DeploymentState.IN_PROGRESS,
    metadata={"environment": "canary"},
)

# 2. Monitor canary (send 10% traffic via deployment.py)
# ... simulate canary traffic ...

# 3. Promote to production
manager.promote_version(canary_version)

cicd.report_deployment_all(
    version_id=canary_version,
    model_id="model",
    state=DeploymentState.SUCCESS,
    metadata={
        "environment": "production",
        "request_count": 50000,
        "error_rate_percent": 0.1,
    },
)

# 4. ArgoCD syncs automatically (via annotations)
# 5. Health monitoring continues, auto-rollback available
```

### Full Deployment Pipeline

```yaml
# .github/workflows/deploy-model.yml
name: Deploy Model

on:
  push:
    branches: [main]
    paths:
      - 'models/**'

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build Model
        run: |
          # ... build model ...
          echo "MODEL_HASH=$(sha256sum model.onnx)" >> $GITHUB_ENV

      - name: Create Release
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          pip install pystreamai
          python -c "
          from pystreamai import GitHubVersionSync, DeploymentState
          import hashlib
          
          sync = GitHubVersionSync(
              repo='${{ github.repository }}',
              token='${{ secrets.GITHUB_TOKEN }}',
          )
          
          sync.create_release(
              version_id='${{ github.sha }}',
              model_id='my-model',
              tag_name='model-${{ github.sha }}',
              model_hash='${{ env.MODEL_HASH }}',
              commit_sha='${{ github.sha }}',
          )
          "

      - name: Deploy to Production
        run: |
          pip install pystreamai
          python deploy.py

      - name: Monitor and Auto-Rollback
        run: |
          python -c "
          from pystreamai import AutoVersionManager
          import time
          
          manager = AutoVersionManager()
          
          # Monitor for 1 hour
          for _ in range(12):
              manager.evaluate_health_windows()
              time.sleep(300)
          "
```

## Environment Variables

### ArgoCD
```bash
ARGOCD_SERVER=https://argocd.example.com
ARGOCD_TOKEN=your-token
```

### GitHub
```bash
GITHUB_TOKEN=ghp_xxxxxxxxxxxx
GITHUB_REPOSITORY=owner/repo
```

### GitLab
```bash
GITLAB_URL=https://gitlab.example.com
CI_JOB_TOKEN=ci-job-token
CI_PROJECT_ID=123456
```

### Jenkins
```bash
JENKINS_URL=https://jenkins.example.com
JENKINS_USER=admin
JENKINS_TOKEN=your-token
```

## Monitoring Dashboard

Create a dashboard using deployment history:

```python
from pystreamai import CICDDispatcher

cicd = CICDDispatcher()
# ... register backends ...

def show_deployment_dashboard(model_id: str):
    history = cicd.get_combined_history(model_id)
    
    print(f"\n📊 Deployment Dashboard: {model_id}")
    print("=" * 60)
    
    for platform, deployments in history.items():
        print(f"\n{platform}:")
        for d in deployments[:5]:  # Last 5
            state_emoji = {
                "success": "✅",
                "failure": "❌",
                "running": "🔄",
            }.get(d["state"], "❓")
            
            print(f"  {state_emoji} {d['version_id']}")
            print(f"     Time: {d['timestamp']}")

show_deployment_dashboard("sentiment-classifier")
```

## Troubleshooting

### ArgoCD Connection Failed
```python
# Check credentials
argocd = ArgoCDBackend(
    server="https://argocd.example.com",
    token="your-token",
    verify_ssl=False,  # For self-signed certificates
)
```

### GitHub Token Permissions
Required scopes:
- `repo` (full control of private repositories)
- `workflow` (manage GitHub Actions)
- `check:write` (create check runs)

### API Rate Limiting
Implement retry logic:
```python
import time

def retry_report(cicd, version_id, model_id, max_retries=3):
    for attempt in range(max_retries):
        try:
            cicd.report_deployment_all(version_id, model_id, ...)
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
```

## API Reference

### CICDDispatcher

```python
dispatcher = CICDDispatcher()

# Register backend
dispatcher.register_backend("argocd", backend)

# Report to all
results = dispatcher.report_deployment_all(
    version_id: str,
    model_id: str,
    state: DeploymentState,
    metadata: Dict[str, Any],
) -> Dict[str, bool]

# Rollback on all
results = dispatcher.trigger_rollback_all(
    version_id: str,
    model_id: str,
) -> Dict[str, bool]

# Get combined history
history = dispatcher.get_combined_history(
    model_id: str,
) -> Dict[str, List[Dict]]
```

### Platform-Specific Backends

All backends implement:
```python
report_deployment(version_id, model_id, state, metadata) -> bool
get_deployment_history(model_id) -> List[Dict]
trigger_rollback(version_id, model_id) -> bool
```

---

See also:
- [Auto Versioning Guide](AUTO_VERSIONING.md)
- [Deployment Guide](DEPLOYMENT.md)
- [GitHub Integration Guide](../pystreamai/github_integration.py)
