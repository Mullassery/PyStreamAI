"""Deployment Management - Canary deployments, rollbacks, A/B testing"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import random

logger = logging.getLogger(__name__)


class DeploymentStatus(Enum):
    """Deployment status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class DeploymentVersion:
    """Model version for deployment"""
    model_id: str
    version: str
    uri: str
    timestamp: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, float] = field(default_factory=dict)
    status: DeploymentStatus = DeploymentStatus.PENDING


class CanaryDeployment:
    """Canary deployment manager"""

    def __init__(
        self,
        current_version: DeploymentVersion,
        canary_version: DeploymentVersion,
        canary_traffic_percent: float = 10.0,
    ):
        self.current_version = current_version
        self.canary_version = canary_version
        self.canary_traffic_percent = canary_traffic_percent
        self.requests_served = 0
        self.canary_requests = 0
        self.canary_metrics = {}

    def route_request(self) -> DeploymentVersion:
        """Route request to current or canary version"""
        self.requests_served += 1

        if random.random() * 100 < self.canary_traffic_percent:
            self.canary_requests += 1
            return self.canary_version
        else:
            return self.current_version

    def get_canary_stats(self) -> Dict[str, Any]:
        """Get canary deployment statistics"""
        if self.requests_served == 0:
            return {}

        return {
            "total_requests": self.requests_served,
            "canary_requests": self.canary_requests,
            "canary_traffic_percent": (self.canary_requests / self.requests_served) * 100,
            "current_version": self.current_version.version,
            "canary_version": self.canary_version.version,
        }

    def promote_canary(self) -> None:
        """Promote canary to current version"""
        logger.info(f"Promoting canary {self.canary_version.version} to current")
        self.current_version = self.canary_version
        self.canary_version = None
        self.canary_requests = 0

    def rollback_canary(self) -> None:
        """Rollback canary deployment"""
        logger.warning(f"Rolling back canary {self.canary_version.version}")
        self.canary_version = None
        self.canary_requests = 0


class ABTestDeployment:
    """A/B testing deployment manager"""

    def __init__(
        self,
        variant_a: DeploymentVersion,
        variant_b: DeploymentVersion,
        split_percent: float = 50.0,
    ):
        self.variant_a = variant_a
        self.variant_b = variant_b
        self.split_percent = split_percent
        self.requests_a = 0
        self.requests_b = 0
        self.metrics_a = {}
        self.metrics_b = {}

    def route_request(self, user_id: Optional[str] = None) -> DeploymentVersion:
        """Route request to variant A or B"""
        # Consistent routing based on user_id if provided
        if user_id:
            if hash(user_id) % 100 < self.split_percent:
                self.requests_a += 1
                return self.variant_a
            else:
                self.requests_b += 1
                return self.variant_b
        else:
            if random.random() * 100 < self.split_percent:
                self.requests_a += 1
                return self.variant_a
            else:
                self.requests_b += 1
                return self.variant_b

    def record_metric(self, variant: str, metric_name: str, value: float) -> None:
        """Record metric for variant"""
        if variant == "a":
            if metric_name not in self.metrics_a:
                self.metrics_a[metric_name] = []
            self.metrics_a[metric_name].append(value)
        else:
            if metric_name not in self.metrics_b:
                self.metrics_b[metric_name] = []
            self.metrics_b[metric_name].append(value)

    def get_stats(self) -> Dict[str, Any]:
        """Get A/B test statistics"""
        total_requests = self.requests_a + self.requests_b

        stats = {
            "total_requests": total_requests,
            "variant_a": {
                "requests": self.requests_a,
                "percent": (self.requests_a / total_requests * 100) if total_requests > 0 else 0,
                "metrics": self._avg_metrics(self.metrics_a),
            },
            "variant_b": {
                "requests": self.requests_b,
                "percent": (self.requests_b / total_requests * 100) if total_requests > 0 else 0,
                "metrics": self._avg_metrics(self.metrics_b),
            },
        }
        return stats

    @staticmethod
    def _avg_metrics(metrics: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate average metrics"""
        return {
            name: sum(values) / len(values) if values else 0
            for name, values in metrics.items()
        }


class DeploymentManager:
    """Manage model deployments"""

    def __init__(self):
        self.deployments: Dict[str, DeploymentVersion] = {}
        self.active_deployment: Optional[str] = None
        self.canary: Optional[CanaryDeployment] = None
        self.ab_test: Optional[ABTestDeployment] = None

    def deploy_model(
        self,
        model_id: str,
        version: str,
        uri: str,
    ) -> DeploymentVersion:
        """Deploy a model version"""
        deployment = DeploymentVersion(
            model_id=model_id,
            version=version,
            uri=uri,
            status=DeploymentStatus.PENDING,
        )
        deployment.status = DeploymentStatus.RUNNING
        self.deployments[version] = deployment

        if self.active_deployment is None:
            self.active_deployment = version
            logger.info(f"Deployed model {model_id} version {version}")

        return deployment

    def start_canary_deployment(
        self,
        canary_version: str,
        traffic_percent: float = 10.0,
    ) -> None:
        """Start canary deployment"""
        if self.active_deployment is None:
            logger.error("No active deployment to canary")
            return

        current = self.deployments[self.active_deployment]
        canary = self.deployments.get(canary_version)

        if not canary:
            logger.error(f"Canary version {canary_version} not found")
            return

        self.canary = CanaryDeployment(current, canary, traffic_percent)
        logger.info(f"Started canary deployment: {traffic_percent}% traffic to {canary_version}")

    def promote_canary(self) -> None:
        """Promote canary to production"""
        if not self.canary:
            logger.warning("No active canary deployment")
            return

        self.canary.promote_canary()
        self.active_deployment = self.canary.canary_version.version
        self.canary = None
        logger.info(f"Promoted canary to production")

    def rollback_deployment(self) -> None:
        """Rollback to previous deployment"""
        if self.canary:
            self.canary.rollback_canary()
            logger.warning("Canary rolled back")
        else:
            logger.warning("No active deployment to rollback")

    def start_ab_test(
        self,
        variant_a: str,
        variant_b: str,
        split_percent: float = 50.0,
    ) -> None:
        """Start A/B test between two versions"""
        variant_a_deploy = self.deployments.get(variant_a)
        variant_b_deploy = self.deployments.get(variant_b)

        if not variant_a_deploy or not variant_b_deploy:
            logger.error("One or both variants not found")
            return

        self.ab_test = ABTestDeployment(variant_a_deploy, variant_b_deploy, split_percent)
        logger.info(f"Started A/B test: {split_percent}% to {variant_a}")

    def route_request(self) -> DeploymentVersion:
        """Route request to appropriate deployment"""
        if self.canary:
            return self.canary.route_request()
        elif self.ab_test:
            return self.ab_test.route_request()
        else:
            return self.deployments[self.active_deployment]

    def get_deployment_status(self) -> Dict[str, Any]:
        """Get deployment status"""
        status = {
            "active_deployment": self.active_deployment,
            "deployments": {
                version: {
                    "model": dep.model_id,
                    "status": dep.status.value,
                    "timestamp": dep.timestamp.isoformat(),
                }
                for version, dep in self.deployments.items()
            },
        }

        if self.canary:
            status["canary"] = self.canary.get_canary_stats()

        if self.ab_test:
            status["ab_test"] = self.ab_test.get_stats()

        return status
