"""PyStreamAI - Simple ML Deployment Platform"""

from .platform import Platform
from .decorators import train, serve, pipeline
from .monitoring import (
    InferenceMetric,
    MetricBackend,
    MetricCollector,
    get_metrics,
    set_metric_backend,
    log_metric,
)
from .auto_versioning import (
    AutoVersionManager,
    HealthThresholds,
    HealthStatus,
    HealthMonitor,
    VersionRegistry,
    ModelVersionMetadata,
)
from .github_integration import GitHubVersionSync, GitHubDeploymentTracker
from .cicd_integration import (
    CICDBackend,
    CICDDispatcher,
    ArgoCDBackend,
    GitHubActionsBackend,
    GitLabCIBackend,
    JenkinsBackend,
    DeploymentState,
)
from .rollback_strategies import (
    RollbackStrategy,
    RollbackConfig,
    RollbackExecutor,
    RollbackOrchestrator,
)
from .cost_performance import (
    CostCalculator,
    CostModel,
    VersionCostTracker,
    PerformanceBenchmark,
)
from .framework_integration import (
    ModelServer,
    TensorFlowServingBackend,
    TorchServeBackend,
    KubernetesDeployment,
    HuggingFaceHub,
    FrameworkDetector,
)
from .advanced_analytics import (
    RootCauseAnalyzer,
    DriftDetector,
    AnomalyDetector,
)
from .multi_model_orchestration import (
    ModelPipeline,
    PipelineRegistry,
    CrossModelABTest,
    PipelineStage,
)

__version__ = "0.2.0"  # Updated with Phases 5-10
__all__ = [
    # Core
    "Platform",
    "train",
    "serve",
    "pipeline",
    # Monitoring
    "InferenceMetric",
    "MetricBackend",
    "MetricCollector",
    "get_metrics",
    "set_metric_backend",
    "log_metric",
    # Phase 1-2: Auto Versioning
    "AutoVersionManager",
    "HealthThresholds",
    "HealthStatus",
    "HealthMonitor",
    "VersionRegistry",
    "ModelVersionMetadata",
    # Phase 2: GitHub & CI/CD
    "GitHubVersionSync",
    "GitHubDeploymentTracker",
    "CICDBackend",
    "CICDDispatcher",
    "ArgoCDBackend",
    "GitHubActionsBackend",
    "GitLabCIBackend",
    "JenkinsBackend",
    "DeploymentState",
    # Phase 5: Rollback Strategies
    "RollbackStrategy",
    "RollbackConfig",
    "RollbackExecutor",
    "RollbackOrchestrator",
    # Phase 6: Cost & Performance
    "CostCalculator",
    "CostModel",
    "VersionCostTracker",
    "PerformanceBenchmark",
    # Phase 8: Framework Integration
    "ModelServer",
    "TensorFlowServingBackend",
    "TorchServeBackend",
    "KubernetesDeployment",
    "HuggingFaceHub",
    "FrameworkDetector",
    # Phase 9: Advanced Analytics
    "RootCauseAnalyzer",
    "DriftDetector",
    "AnomalyDetector",
    # Phase 10: Multi-Model Orchestration
    "ModelPipeline",
    "PipelineRegistry",
    "CrossModelABTest",
    "PipelineStage",
]
