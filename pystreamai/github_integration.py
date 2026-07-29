"""GitHub Integration for Model Versioning and Deployment Tracking

Sync version history to GitHub releases, tags, and deployment status.
Enables full deployment tracking and rollback management via GitHub.
"""

import logging
import json
from typing import Dict, Any, Optional, List
from dataclasses import asdict
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class GitHubVersionSync:
    """Sync model versions to GitHub releases and tags"""

    def __init__(self, repo: str, token: Optional[str] = None, dry_run: bool = False):
        """
        Initialize GitHub sync.

        Args:
            repo: "owner/repo" format
            token: GitHub personal access token (uses env var GITHUB_TOKEN if not provided)
            dry_run: Don't actually push to GitHub (for testing)
        """
        self.repo = repo
        self.token = token
        self.dry_run = dry_run

        if not dry_run:
            try:
                import httpx
                self.client = httpx.Client(
                    base_url="https://api.github.com",
                    headers={"Authorization": f"token {token or self._get_token()}"},
                )
            except ImportError:
                logger.error("httpx required for GitHub integration: pip install httpx")
                raise

    def _get_token(self) -> str:
        """Get GitHub token from environment"""
        import os
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            raise ValueError("GITHUB_TOKEN environment variable not set")
        return token

    def create_release(
        self,
        version_id: str,
        model_id: str,
        tag_name: Optional[str] = None,
        model_hash: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        commit_sha: Optional[str] = None,
        is_draft: bool = False,
        is_prerelease: bool = False,
    ) -> Dict[str, Any]:
        """
        Create GitHub release for model version.

        Args:
            version_id: Unique version identifier
            model_id: Model name
            tag_name: Git tag name (defaults to version_id)
            model_hash: Hash of model weights
            metrics: Model performance metrics
            commit_sha: Commit that triggered this deployment
            is_draft: Mark as draft release
            is_prerelease: Mark as pre-release

        Returns:
            Release info from GitHub API
        """
        tag_name = tag_name or version_id
        metrics = metrics or {}

        # Build release notes
        release_notes = self._build_release_notes(
            model_id,
            version_id,
            model_hash,
            metrics,
            commit_sha,
        )

        payload = {
            "tag_name": tag_name,
            "name": f"{model_id} {version_id}",
            "body": release_notes,
            "draft": is_draft,
            "prerelease": is_prerelease,
            "target_commitish": commit_sha or "main",
        }

        if self.dry_run:
            logger.info(f"[DRY RUN] Create release: {payload}")
            return {"status": "dry_run", "payload": payload}

        try:
            response = self.client.post(f"/repos/{self.repo}/releases", json=payload)
            response.raise_for_status()
            logger.info(f"Created GitHub release: {tag_name}")
            return response.json()
        except Exception as e:
            logger.error(f"Failed to create GitHub release: {e}")
            raise

    def create_tag(
        self,
        version_id: str,
        model_id: str,
        commit_sha: Optional[str] = None,
        message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create annotated Git tag for version.

        Args:
            version_id: Version identifier
            model_id: Model name
            commit_sha: Commit to tag
            message: Tag message

        Returns:
            Tag info from GitHub API
        """
        tag_name = f"{model_id}-{version_id}"
        message = message or f"Deployed {model_id} {version_id}"

        payload = {
            "tag": tag_name,
            "sha": commit_sha or "HEAD",
            "type": "commit",
            "tagger": {
                "name": "PyStreamAI Bot",
                "email": "bot@pystreamai.io",
                "date": datetime.now().isoformat(),
            },
            "message": message,
        }

        if self.dry_run:
            logger.info(f"[DRY RUN] Create tag: {payload}")
            return {"status": "dry_run", "payload": payload}

        try:
            response = self.client.post(f"/repos/{self.repo}/git/tags", json=payload)
            response.raise_for_status()
            logger.info(f"Created Git tag: {tag_name}")
            return response.json()
        except Exception as e:
            logger.error(f"Failed to create Git tag: {e}")
            raise

    def post_deployment(
        self,
        environment: str,
        version_id: str,
        model_id: str,
        status: str = "in_progress",
        commit_sha: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Post deployment status to GitHub.

        Args:
            environment: Deployment environment (e.g., "production", "staging")
            version_id: Version being deployed
            model_id: Model name
            status: "in_progress", "success", or "failure"
            commit_sha: Commit being deployed
            description: Status description

        Returns:
            Deployment info from GitHub API
        """
        description = description or f"Deploying {model_id} {version_id}"

        payload = {
            "environment": environment,
            "state": status,
            "description": description,
            "environment_url": "",
            "auto_merge": False,
        }

        if self.dry_run:
            logger.info(f"[DRY RUN] Post deployment: {payload}")
            return {"status": "dry_run", "payload": payload}

        try:
            # Use commit statuses API as fallback (more widely supported)
            if not commit_sha:
                logger.warning("No commit SHA provided for deployment status")
                return {}

            response = self.client.post(
                f"/repos/{self.repo}/statuses/{commit_sha}",
                json={
                    "state": status,
                    "description": description,
                    "context": f"deployment/{environment}/{model_id}",
                },
            )
            response.raise_for_status()
            logger.info(f"Posted deployment status: {environment} {status}")
            return response.json()
        except Exception as e:
            logger.error(f"Failed to post deployment status: {e}")
            return {}

    def post_issue_comment(
        self,
        issue_number: int,
        comment: str,
    ) -> Dict[str, Any]:
        """
        Post comment to GitHub issue (for deployment tracking in PRs).

        Args:
            issue_number: GitHub issue/PR number
            comment: Comment text

        Returns:
            Comment info from GitHub API
        """
        payload = {"body": comment}

        if self.dry_run:
            logger.info(f"[DRY RUN] Post comment to issue {issue_number}: {comment[:100]}")
            return {"status": "dry_run", "payload": payload}

        try:
            response = self.client.post(
                f"/repos/{self.repo}/issues/{issue_number}/comments",
                json=payload,
            )
            response.raise_for_status()
            logger.info(f"Posted comment to issue {issue_number}")
            return response.json()
        except Exception as e:
            logger.error(f"Failed to post issue comment: {e}")
            return {}

    def create_rollback_issue(
        self,
        version_id: str,
        model_id: str,
        reason: str,
        labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Create GitHub issue for rollback event.

        Args:
            version_id: Version that was rolled back
            model_id: Model name
            reason: Reason for rollback
            labels: Issue labels

        Returns:
            Issue info from GitHub API
        """
        labels = labels or ["rollback", "incident"]
        title = f"🚨 Auto-Rollback: {model_id} {version_id}"
        body = f"""## Model Rollback Alert

**Model:** {model_id}
**Version:** {version_id}
**Timestamp:** {datetime.now().isoformat()}
**Reason:** {reason}

### Action Required
- [ ] Investigate cause
- [ ] Review metrics
- [ ] Fix issue
- [ ] Re-deploy fixed version

See deployment history in releases: https://github.com/{self.repo}/releases
"""

        payload = {
            "title": title,
            "body": body,
            "labels": labels,
        }

        if self.dry_run:
            logger.info(f"[DRY RUN] Create rollback issue: {title}")
            return {"status": "dry_run", "payload": payload}

        try:
            response = self.client.post(
                f"/repos/{self.repo}/issues",
                json=payload,
            )
            response.raise_for_status()
            logger.info(f"Created rollback issue for {version_id}")
            return response.json()
        except Exception as e:
            logger.error(f"Failed to create rollback issue: {e}")
            return {}

    def update_deployment_history(
        self,
        versions_data: List[Dict[str, Any]],
        branch: str = "gh-pages",
    ) -> bool:
        """
        Update deployment history file in GitHub (for GitHub Pages).

        Args:
            versions_data: List of version metadata
            branch: Branch to update (default: gh-pages for GitHub Pages)

        Returns:
            True if successful
        """
        filename = "deployment-history.json"
        content = json.dumps(versions_data, indent=2)

        if self.dry_run:
            logger.info(f"[DRY RUN] Update {filename} in {branch}")
            return True

        try:
            # Get current file SHA (needed for update)
            try:
                response = self.client.get(
                    f"/repos/{self.repo}/contents/{filename}?ref={branch}"
                )
                response.raise_for_status()
                current_sha = response.json()["sha"]
            except:
                current_sha = None

            # Encode content
            import base64
            encoded = base64.b64encode(content.encode()).decode()

            payload = {
                "message": f"Update deployment history",
                "content": encoded,
                "branch": branch,
            }
            if current_sha:
                payload["sha"] = current_sha

            response = self.client.put(
                f"/repos/{self.repo}/contents/{filename}",
                json=payload,
            )
            response.raise_for_status()
            logger.info(f"Updated deployment history in {branch}")
            return True
        except Exception as e:
            logger.error(f"Failed to update deployment history: {e}")
            return False

    def _build_release_notes(
        self,
        model_id: str,
        version_id: str,
        model_hash: Optional[str],
        metrics: Dict[str, float],
        commit_sha: Optional[str],
    ) -> str:
        """Build formatted release notes"""
        notes = f"""## {model_id} - {version_id}

### Deployment Info
- **Deployed:** {datetime.now().isoformat()}
- **Model Hash:** `{model_hash[:16] if model_hash else "N/A"}...`
"""

        if commit_sha:
            notes += f"- **Commit:** `{commit_sha[:8]}`\n"

        if metrics:
            notes += "\n### Metrics\n"
            for key, value in metrics.items():
                if isinstance(value, float):
                    notes += f"- **{key}:** {value:.2f}\n"
                else:
                    notes += f"- **{key}:** {value}\n"

        notes += "\n### Rollback\n"
        notes += f"To rollback to this version:\n"
        notes += f"```\npystreamai rollback {model_id} {version_id}\n```\n"

        return notes


class GitHubDeploymentTracker:
    """Track deployments in GitHub with AutoVersionManager"""

    def __init__(
        self,
        repo: str,
        auto_version_manager: "AutoVersionManager",
        token: Optional[str] = None,
        dry_run: bool = False,
    ):
        """
        Initialize tracker.

        Args:
            repo: GitHub repo in "owner/repo" format
            auto_version_manager: AutoVersionManager instance
            token: GitHub token
            dry_run: Don't push to GitHub
        """
        self.repo = repo
        self.manager = auto_version_manager
        self.sync = GitHubVersionSync(repo, token, dry_run)

        # Register rollback handler
        self.manager.on_rollback(self._on_rollback)

    def track_deployment(
        self,
        version_id: str,
        model_id: str,
        environment: str = "production",
        commit_sha: Optional[str] = None,
        tags: Optional[List[str]] = None,
        create_release: bool = True,
    ) -> None:
        """
        Track a deployment to GitHub.

        Args:
            version_id: Version being deployed
            model_id: Model name
            environment: Deployment environment
            commit_sha: Commit being deployed
            tags: Additional tags for categorizing version
            create_release: Create GitHub release for this version
        """
        try:
            # Post deployment status
            self.sync.post_deployment(
                environment=environment,
                version_id=version_id,
                model_id=model_id,
                status="in_progress",
                commit_sha=commit_sha,
                description=f"Deploying {model_id} to {environment}",
            )

            # Create GitHub release (optional)
            if create_release:
                version = self.manager.registry.get_version(version_id)
                if version:
                    self.sync.create_release(
                        version_id=version_id,
                        model_id=model_id,
                        model_hash=version.model_hash,
                        metrics=version.metrics,
                        commit_sha=commit_sha,
                        is_prerelease=(environment != "production"),
                    )

            logger.info(f"Tracked deployment: {version_id} to {environment}")
        except Exception as e:
            logger.error(f"Failed to track deployment: {e}")

    def _on_rollback(self, version_id: str, reason: str) -> None:
        """Handle rollback events - post to GitHub"""
        try:
            version = self.manager.registry.get_version(version_id)
            if not version:
                return

            # Create GitHub issue for the rollback
            self.sync.create_rollback_issue(
                version_id=version_id,
                model_id=version.model_id,
                reason=reason,
            )

            logger.info(f"Posted rollback alert to GitHub for {version_id}")
        except Exception as e:
            logger.error(f"Failed to post rollback alert: {e}")

    def post_deployment_success(
        self,
        version_id: str,
        model_id: str,
        environment: str = "production",
        commit_sha: Optional[str] = None,
    ) -> None:
        """Mark deployment as successful in GitHub"""
        try:
            self.sync.post_deployment(
                environment=environment,
                version_id=version_id,
                model_id=model_id,
                status="success",
                commit_sha=commit_sha,
                description=f"Successfully deployed {model_id}",
            )
        except Exception as e:
            logger.error(f"Failed to post deployment success: {e}")

    def post_deployment_failure(
        self,
        version_id: str,
        model_id: str,
        environment: str = "production",
        commit_sha: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Mark deployment as failed in GitHub"""
        try:
            self.sync.post_deployment(
                environment=environment,
                version_id=version_id,
                model_id=model_id,
                status="failure",
                commit_sha=commit_sha,
                description=f"Failed to deploy {model_id}: {error or 'Unknown error'}",
            )
        except Exception as e:
            logger.error(f"Failed to post deployment failure: {e}")

    def sync_version_history_to_github(self, branch: str = "gh-pages") -> bool:
        """
        Sync all version history to GitHub (for visibility/auditing).

        Args:
            branch: Branch to store history (default: gh-pages)

        Returns:
            True if successful
        """
        try:
            models = {}
            for vid, version in self.manager.registry.versions.items():
                if version.model_id not in models:
                    models[version.model_id] = []
                models[version.model_id].append(version.to_dict())

            return self.sync.update_deployment_history(
                [v for versions in models.values() for v in versions],
                branch=branch,
            )
        except Exception as e:
            logger.error(f"Failed to sync version history: {e}")
            return False
