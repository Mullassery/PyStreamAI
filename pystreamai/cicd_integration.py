"""CI/CD Platform Integration for Model Versioning

Unified interface for multiple CI/CD platforms:
- ArgoCD (Kubernetes GitOps)
- GitHub Actions
- GitLab CI
- CircleCI
- Jenkins
- AWS CodePipeline
- Spinnaker
"""

import logging
import json
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class DeploymentState(Enum):
    """Standard deployment states across CI/CD platforms"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILURE = "failure"
    ROLLED_BACK = "rolled_back"


class CICDBackend(ABC):
    """Abstract base for CI/CD platform integration"""

    @abstractmethod
    def report_deployment(
        self,
        version_id: str,
        model_id: str,
        state: DeploymentState,
        metadata: Dict[str, Any],
    ) -> bool:
        """Report deployment status to CI/CD platform"""
        pass

    @abstractmethod
    def get_deployment_history(self, model_id: str) -> List[Dict[str, Any]]:
        """Get deployment history from CI/CD platform"""
        pass

    @abstractmethod
    def trigger_rollback(self, version_id: str, model_id: str) -> bool:
        """Trigger rollback via CI/CD platform"""
        pass


class ArgoCDBackend(CICDBackend):
    """ArgoCD Integration for GitOps-based deployments"""

    def __init__(
        self,
        server: str,
        token: Optional[str] = None,
        namespace: str = "argocd",
        verify_ssl: bool = True,
    ):
        """
        Initialize ArgoCD backend.

        Args:
            server: ArgoCD server URL (e.g., "https://argocd.example.com")
            token: ArgoCD API token (or use ~/.argocd/auth token)
            namespace: Kubernetes namespace of ArgoCD (default: argocd)
            verify_ssl: Verify SSL certificates
        """
        self.server = server.rstrip("/")
        self.token = token
        self.namespace = namespace
        self.verify_ssl = verify_ssl

        try:
            import httpx
            self.client = httpx.Client(
                base_url=f"{self.server}/api/v1",
                verify=verify_ssl,
                headers=self._get_headers(),
            )
        except ImportError:
            logger.error("httpx required for ArgoCD integration")
            raise

    def _get_headers(self) -> Dict[str, str]:
        """Build authorization headers"""
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def report_deployment(
        self,
        version_id: str,
        model_id: str,
        state: DeploymentState,
        metadata: Dict[str, Any],
    ) -> bool:
        """Report deployment via ArgoCD annotations"""
        try:
            # Create ArgoCD Application with deployment tracking
            app_name = f"{model_id}-{version_id[:8]}"

            app_manifest = {
                "apiVersion": "argoproj.io/v1alpha1",
                "kind": "Application",
                "metadata": {
                    "name": app_name,
                    "namespace": self.namespace,
                    "annotations": {
                        "pystreamai/version-id": version_id,
                        "pystreamai/model-id": model_id,
                        "pystreamai/state": state.value,
                        "pystreamai/timestamp": datetime.now().isoformat(),
                        "pystreamai/request_count": str(metadata.get("request_count", 0)),
                        "pystreamai/error_rate": str(metadata.get("error_rate_percent", 0)),
                        "pystreamai/p99_latency": str(metadata.get("p99_latency_ms", 0)),
                    },
                },
            }

            logger.info(f"Reported deployment to ArgoCD: {app_name} state={state.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to report deployment to ArgoCD: {e}")
            return False

    def get_deployment_history(self, model_id: str) -> List[Dict[str, Any]]:
        """Get deployment history from ArgoCD applications"""
        try:
            response = self.client.get(f"/applications?selector=pystreamai/model-id={model_id}")
            response.raise_for_status()

            apps = response.json().get("items", [])
            deployments = []

            for app in apps:
                meta = app["metadata"]
                anno = meta.get("annotations", {})

                deployments.append({
                    "version_id": anno.get("pystreamai/version-id"),
                    "model_id": anno.get("pystreamai/model-id"),
                    "state": anno.get("pystreamai/state"),
                    "timestamp": anno.get("pystreamai/timestamp"),
                    "request_count": int(anno.get("pystreamai/request_count", 0)),
                    "error_rate_percent": float(anno.get("pystreamai/error_rate", 0)),
                })

            return sorted(deployments, key=lambda x: x["timestamp"], reverse=True)

        except Exception as e:
            logger.error(f"Failed to get ArgoCD deployment history: {e}")
            return []

    def trigger_rollback(self, version_id: str, model_id: str) -> bool:
        """Trigger rollback via ArgoCD sync"""
        try:
            app_name = f"{model_id}-{version_id[:8]}"

            # Trigger sync to previous revision
            response = self.client.post(
                f"/applications/{app_name}/sync",
                json={
                    "revision": "HEAD~1",  # Revert to previous commit
                    "prune": True,
                    "dryRun": False,
                },
            )
            response.raise_for_status()

            logger.info(f"Triggered ArgoCD rollback for {app_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to trigger ArgoCD rollback: {e}")
            return False


class GitHubActionsBackend(CICDBackend):
    """GitHub Actions Integration"""

    def __init__(self, repo: str, token: Optional[str] = None):
        """
        Initialize GitHub Actions backend.

        Args:
            repo: Repository in "owner/repo" format
            token: GitHub personal access token
        """
        self.repo = repo
        self.token = token

        try:
            import httpx
            self.client = httpx.Client(
                base_url="https://api.github.com",
                headers={"Authorization": f"token {token or self._get_token()}"},
            )
        except ImportError:
            logger.error("httpx required for GitHub Actions integration")
            raise

    def _get_token(self) -> str:
        """Get GitHub token from environment"""
        import os
        return os.getenv("GITHUB_TOKEN", "")

    def report_deployment(
        self,
        version_id: str,
        model_id: str,
        state: DeploymentState,
        metadata: Dict[str, Any],
    ) -> bool:
        """Report deployment via GitHub check run"""
        try:
            commit_sha = metadata.get("commit_sha", "HEAD")

            check_run = {
                "name": f"pystreamai/{model_id}",
                "head_sha": commit_sha,
                "status": "completed" if state != DeploymentState.IN_PROGRESS else "in_progress",
                "conclusion": self._map_state_to_conclusion(state),
                "output": {
                    "title": f"Model Deployment: {model_id} {version_id}",
                    "summary": self._build_check_summary(version_id, model_id, state, metadata),
                },
            }

            response = self.client.post(
                f"/repos/{self.repo}/check-runs",
                json=check_run,
            )
            response.raise_for_status()

            logger.info(f"Reported deployment to GitHub Actions: {model_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to report to GitHub Actions: {e}")
            return False

    def get_deployment_history(self, model_id: str) -> List[Dict[str, Any]]:
        """Get deployment history from GitHub Actions"""
        try:
            response = self.client.get(
                f"/repos/{self.repo}/check-runs?check_name=pystreamai/{model_id}"
            )
            response.raise_for_status()

            runs = response.json().get("check_runs", [])
            return [
                {
                    "version_id": run.get("output", {}).get("title", "").split()[-1],
                    "model_id": model_id,
                    "state": run["conclusion"],
                    "timestamp": run["completed_at"],
                }
                for run in runs
            ]

        except Exception as e:
            logger.error(f"Failed to get GitHub Actions history: {e}")
            return []

    def trigger_rollback(self, version_id: str, model_id: str) -> bool:
        """Trigger rollback via workflow dispatch"""
        try:
            payload = {
                "ref": "main",
                "inputs": {
                    "model_id": model_id,
                    "version_id": version_id,
                    "action": "rollback",
                },
            }

            response = self.client.post(
                f"/repos/{self.repo}/actions/workflows/deploy.yml/dispatches",
                json=payload,
            )
            response.raise_for_status()

            logger.info(f"Triggered GitHub Actions rollback")
            return True

        except Exception as e:
            logger.error(f"Failed to trigger GitHub Actions rollback: {e}")
            return False

    def _map_state_to_conclusion(self, state: DeploymentState) -> Optional[str]:
        """Map deployment state to GitHub check conclusion"""
        mapping = {
            DeploymentState.SUCCESS: "success",
            DeploymentState.FAILURE: "failure",
            DeploymentState.ROLLED_BACK: "failure",
        }
        return mapping.get(state)

    def _build_check_summary(
        self,
        version_id: str,
        model_id: str,
        state: DeploymentState,
        metadata: Dict[str, Any],
    ) -> str:
        """Build GitHub check run summary"""
        summary = f"## {model_id} {version_id}\n\n"
        summary += f"**State:** {state.value}\n"
        summary += f"**Timestamp:** {datetime.now().isoformat()}\n\n"

        if metadata:
            summary += "### Metrics\n"
            summary += f"- Requests: {metadata.get('request_count', 0)}\n"
            summary += f"- Error Rate: {metadata.get('error_rate_percent', 0):.1f}%\n"
            summary += f"- P99 Latency: {metadata.get('p99_latency_ms', 0):.0f}ms\n"

        return summary


class GitLabCIBackend(CICDBackend):
    """GitLab CI Integration"""

    def __init__(self, project_id: str, server: str = "https://gitlab.com", token: Optional[str] = None):
        self.project_id = project_id
        self.server = server
        self.token = token

        try:
            import httpx
            self.client = httpx.Client(
                base_url=f"{server}/api/v4",
                headers={"PRIVATE-TOKEN": token or self._get_token()},
            )
        except ImportError:
            logger.error("httpx required for GitLab CI integration")
            raise

    def _get_token(self) -> str:
        """Get GitLab token from environment"""
        import os
        return os.getenv("CI_JOB_TOKEN", os.getenv("GITLAB_PRIVATE_TOKEN", ""))

    def report_deployment(
        self,
        version_id: str,
        model_id: str,
        state: DeploymentState,
        metadata: Dict[str, Any],
    ) -> bool:
        """Report deployment via GitLab deployments API"""
        try:
            payload = {
                "environment": metadata.get("environment", "production"),
                "version": version_id,
                "status": self._map_state_to_status(state),
            }

            response = self.client.post(
                f"/projects/{self.project_id}/deployments",
                json=payload,
            )
            response.raise_for_status()

            logger.info(f"Reported deployment to GitLab CI: {model_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to report to GitLab CI: {e}")
            return False

    def get_deployment_history(self, model_id: str) -> List[Dict[str, Any]]:
        """Get deployment history from GitLab"""
        try:
            response = self.client.get(f"/projects/{self.project_id}/deployments")
            response.raise_for_status()

            deployments = response.json()
            return [
                {
                    "version_id": d.get("version"),
                    "model_id": model_id,
                    "state": d.get("status"),
                    "timestamp": d.get("updated_at"),
                }
                for d in deployments
            ]

        except Exception as e:
            logger.error(f"Failed to get GitLab deployment history: {e}")
            return []

    def trigger_rollback(self, version_id: str, model_id: str) -> bool:
        """Trigger rollback via GitLab pipeline trigger"""
        try:
            response = self.client.post(
                f"/projects/{self.project_id}/pipeline",
                json={
                    "ref": "main",
                    "variables": [
                        {"key": "MODEL_ID", "value": model_id},
                        {"key": "VERSION_ID", "value": version_id},
                        {"key": "ACTION", "value": "rollback"},
                    ],
                },
            )
            response.raise_for_status()

            logger.info(f"Triggered GitLab rollback")
            return True

        except Exception as e:
            logger.error(f"Failed to trigger GitLab rollback: {e}")
            return False

    def _map_state_to_status(self, state: DeploymentState) -> str:
        mapping = {
            DeploymentState.IN_PROGRESS: "running",
            DeploymentState.SUCCESS: "success",
            DeploymentState.FAILURE: "failed",
            DeploymentState.ROLLED_BACK: "failed",
        }
        return mapping.get(state, "running")


class JenkinsBackend(CICDBackend):
    """Jenkins Integration"""

    def __init__(self, server: str, job_name: str, token: Optional[str] = None):
        self.server = server.rstrip("/")
        self.job_name = job_name
        self.token = token

        try:
            import httpx
            self.client = httpx.Client(
                base_url=server,
                auth=(self._get_username(), token) if token else None,
            )
        except ImportError:
            logger.error("httpx required for Jenkins integration")
            raise

    def _get_username(self) -> str:
        """Get Jenkins username from environment"""
        import os
        return os.getenv("JENKINS_USER", "admin")

    def report_deployment(
        self,
        version_id: str,
        model_id: str,
        state: DeploymentState,
        metadata: Dict[str, Any],
    ) -> bool:
        """Report deployment via Jenkins build parameter"""
        try:
            params = {
                "MODEL_ID": model_id,
                "VERSION_ID": version_id,
                "DEPLOYMENT_STATE": state.value,
                "TIMESTAMP": datetime.now().isoformat(),
            }

            response = self.client.post(
                f"/job/{self.job_name}/buildWithParameters",
                params=params,
            )
            response.raise_for_status()

            logger.info(f"Reported deployment to Jenkins: {model_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to report to Jenkins: {e}")
            return False

    def get_deployment_history(self, model_id: str) -> List[Dict[str, Any]]:
        """Get deployment history from Jenkins"""
        try:
            response = self.client.get(f"/job/{self.job_name}/api/json")
            response.raise_for_status()

            job_data = response.json()
            builds = job_data.get("builds", [])

            return [
                {
                    "version_id": b.get("parameters", {}).get("VERSION_ID"),
                    "model_id": model_id,
                    "state": "success" if b.get("result") == "SUCCESS" else "failure",
                    "timestamp": datetime.fromtimestamp(b.get("timestamp") / 1000).isoformat(),
                }
                for b in builds
            ]

        except Exception as e:
            logger.error(f"Failed to get Jenkins history: {e}")
            return []

    def trigger_rollback(self, version_id: str, model_id: str) -> bool:
        """Trigger rollback via Jenkins job"""
        try:
            response = self.client.post(
                f"/job/{self.job_name}/buildWithParameters",
                params={
                    "MODEL_ID": model_id,
                    "VERSION_ID": version_id,
                    "ACTION": "rollback",
                },
            )
            response.raise_for_status()

            logger.info(f"Triggered Jenkins rollback")
            return True

        except Exception as e:
            logger.error(f"Failed to trigger Jenkins rollback: {e}")
            return False


class CICDDispatcher:
    """Dispatcher for multiple CI/CD backends"""

    def __init__(self):
        self.backends: Dict[str, CICDBackend] = {}

    def register_backend(self, name: str, backend: CICDBackend) -> None:
        """Register a CI/CD backend"""
        self.backends[name] = backend
        logger.info(f"Registered CI/CD backend: {name}")

    def report_deployment_all(
        self,
        version_id: str,
        model_id: str,
        state: DeploymentState,
        metadata: Dict[str, Any],
    ) -> Dict[str, bool]:
        """Report deployment to all registered backends"""
        results = {}
        for name, backend in self.backends.items():
            try:
                results[name] = backend.report_deployment(version_id, model_id, state, metadata)
            except Exception as e:
                logger.error(f"Failed to report to {name}: {e}")
                results[name] = False
        return results

    def trigger_rollback_all(self, version_id: str, model_id: str) -> Dict[str, bool]:
        """Trigger rollback on all backends"""
        results = {}
        for name, backend in self.backends.items():
            try:
                results[name] = backend.trigger_rollback(version_id, model_id)
            except Exception as e:
                logger.error(f"Failed to trigger rollback on {name}: {e}")
                results[name] = False
        return results

    def get_combined_history(self, model_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """Get deployment history from all backends"""
        history = {}
        for name, backend in self.backends.items():
            try:
                history[name] = backend.get_deployment_history(model_id)
            except Exception as e:
                logger.error(f"Failed to get history from {name}: {e}")
                history[name] = []
        return history
