"""Phase 9: Advanced Analytics - Root Cause Analysis, Drift Detection, Anomaly Detection"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import statistics

logger = logging.getLogger(__name__)


@dataclass
class MetricSnapshot:
    """Snapshot of metrics at a point in time"""
    timestamp: datetime
    error_rate_percent: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    avg_latency_ms: float
    memory_usage_mb: float


@dataclass
class DriftDetectionAlert:
    """Alert when data or model drift detected"""
    detected_at: datetime
    drift_type: str  # data_drift, prediction_drift, model_drift
    severity: str  # low, medium, high, critical
    description: str
    recommended_action: str


class RootCauseAnalyzer:
    """Analyze root causes of model degradation"""

    def __init__(self):
        self.version_snapshots: Dict[str, List[MetricSnapshot]] = {}

    def record_snapshot(self, version_id: str, snapshot: MetricSnapshot) -> None:
        """Record metric snapshot for version"""
        if version_id not in self.version_snapshots:
            self.version_snapshots[version_id] = []

        self.version_snapshots[version_id].append(snapshot)

    def analyze_degradation(
        self,
        degraded_version: str,
        baseline_version: str,
    ) -> Dict[str, Any]:
        """
        Analyze what changed between versions to cause degradation.

        Returns analysis of:
        - Which metrics changed
        - Magnitude of change
        - Likely causes
        - Recommended fixes
        """
        if degraded_version not in self.version_snapshots:
            return {"error": "No data for degraded version"}

        if baseline_version not in self.version_snapshots:
            return {"error": "No data for baseline version"}

        degraded = self.version_snapshots[degraded_version]
        baseline = self.version_snapshots[baseline_version]

        if not degraded or not baseline:
            return {"error": "Insufficient data"}

        # Get latest snapshots
        degraded_latest = degraded[-1]
        baseline_latest = baseline[-1]

        analysis = {
            "degraded_version": degraded_version,
            "baseline_version": baseline_version,
            "analyzed_at": datetime.now().isoformat(),
            "metric_changes": {},
            "probable_causes": [],
            "recommendations": [],
        }

        # Analyze error rate change
        error_rate_change = degraded_latest.error_rate_percent - baseline_latest.error_rate_percent
        if error_rate_change > 1.0:  # Significant increase
            analysis["metric_changes"]["error_rate_percent"] = {
                "baseline": baseline_latest.error_rate_percent,
                "degraded": degraded_latest.error_rate_percent,
                "change_percent": (error_rate_change / baseline_latest.error_rate_percent * 100)
                if baseline_latest.error_rate_percent > 0 else 0,
            }

            analysis["probable_causes"].extend([
                "Dataset distribution changed (data drift)",
                "Model weights corrupted during deployment",
                "Input validation disabled or changed",
                "Dependency version conflict",
                "Upstream service failure",
            ])

        # Analyze latency change
        latency_change = degraded_latest.p99_latency_ms - baseline_latest.p99_latency_ms
        if latency_change > 200:  # Significant increase
            analysis["metric_changes"]["p99_latency_ms"] = {
                "baseline": baseline_latest.p99_latency_ms,
                "degraded": degraded_latest.p99_latency_ms,
                "change_ms": latency_change,
            }

            analysis["probable_causes"].extend([
                "Batch size increased (more processing)",
                "Model size increased without optimization",
                "Resource contention (CPU/GPU/memory)",
                "Network latency increased",
                "Database query performance degraded",
            ])

        # Analyze throughput change
        throughput_change = degraded_latest.throughput_rps - baseline_latest.throughput_rps
        if throughput_change < -10:  # Significant decrease
            analysis["metric_changes"]["throughput_rps"] = {
                "baseline": baseline_latest.throughput_rps,
                "degraded": degraded_latest.throughput_rps,
                "change_percent": (throughput_change / baseline_latest.throughput_rps * 100)
                if baseline_latest.throughput_rps > 0 else 0,
            }

        # Memory usage spike
        memory_change = degraded_latest.memory_usage_mb - baseline_latest.memory_usage_mb
        if memory_change > 100:
            analysis["metric_changes"]["memory_usage_mb"] = {
                "baseline": baseline_latest.memory_usage_mb,
                "degraded": degraded_latest.memory_usage_mb,
                "change_mb": memory_change,
            }

            analysis["probable_causes"].extend([
                "Memory leak in model inference",
                "Caching layer storing too much data",
                "Batch accumulation not cleared",
                "Model loaded multiple times",
            ])

        # Recommendations
        if error_rate_change > 5:
            analysis["recommendations"].extend([
                "Retrain model on recent data to handle drift",
                "Review recent input samples for anomalies",
                "Check data preprocessing pipeline",
            ])

        if latency_change > 500:
            analysis["recommendations"].extend([
                "Profile model to identify bottlenecks",
                "Check resource utilization on deployment nodes",
                "Consider model quantization for faster inference",
            ])

        if throughput_change < -20:
            analysis["recommendations"].extend([
                "Increase batch size if possible",
                "Add more inference replicas",
                "Check upstream service health",
            ])

        # Severity assessment
        max_error_increase = error_rate_change
        max_latency_increase = latency_change
        max_throughput_decrease = abs(throughput_change)

        if max_error_increase > 10 or max_latency_increase > 1000:
            analysis["severity"] = "critical"
        elif max_error_increase > 5 or max_latency_increase > 500:
            analysis["severity"] = "high"
        elif max_error_increase > 2 or max_latency_increase > 200:
            analysis["severity"] = "medium"
        else:
            analysis["severity"] = "low"

        return analysis

    def get_metric_correlation(
        self,
        version_id: str,
        metric1: str,
        metric2: str,
    ) -> Optional[float]:
        """Calculate correlation between two metrics"""
        if version_id not in self.version_snapshots:
            return None

        snapshots = self.version_snapshots[version_id]
        if len(snapshots) < 2:
            return None

        try:
            values1 = [getattr(s, metric1) for s in snapshots]
            values2 = [getattr(s, metric2) for s in snapshots]

            # Simple correlation
            mean1 = statistics.mean(values1)
            mean2 = statistics.mean(values2)

            numerator = sum((v1 - mean1) * (v2 - mean2) for v1, v2 in zip(values1, values2))
            denominator = (
                (sum((v - mean1) ** 2 for v in values1) ** 0.5) *
                (sum((v - mean2) ** 2 for v in values2) ** 0.5)
            )

            if denominator == 0:
                return 0

            return numerator / denominator

        except Exception as e:
            logger.error(f"Failed to calculate correlation: {e}")
            return None


class DriftDetector:
    """Detect data drift and model drift"""

    def __init__(self, baseline_distribution: Optional[Dict[str, Any]] = None):
        self.baseline_distribution = baseline_distribution or {}
        self.current_distribution: Dict[str, Any] = {}
        self.drift_alerts: List[DriftDetectionAlert] = []

    def update_current_distribution(
        self,
        data: List[Any],
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Update current data distribution"""
        if not data:
            return

        # Calculate distribution statistics
        if isinstance(data[0], dict):
            self.current_distribution = self._analyze_dict_distribution(data)
        elif isinstance(data[0], (list, tuple)):
            self.current_distribution = self._analyze_array_distribution(data, feature_names)
        else:
            self.current_distribution = self._analyze_scalar_distribution(data)

    def detect_data_drift(
        self,
        threshold: float = 0.05,  # 5% divergence threshold
    ) -> Tuple[bool, Optional[DriftDetectionAlert]]:
        """Detect if input data distribution has changed"""
        if not self.baseline_distribution or not self.current_distribution:
            return False, None

        # Compare distributions
        divergence = self._calculate_distribution_divergence(
            self.baseline_distribution,
            self.current_distribution,
        )

        if divergence > threshold:
            alert = DriftDetectionAlert(
                detected_at=datetime.now(),
                drift_type="data_drift",
                severity=self._severity_from_divergence(divergence),
                description=f"Input data distribution shifted by {divergence:.1%}",
                recommended_action="Retrain model on recent data or investigate root cause",
            )
            self.drift_alerts.append(alert)
            return True, alert

        return False, None

    def detect_prediction_drift(
        self,
        baseline_predictions: List[float],
        current_predictions: List[float],
        threshold: float = 0.1,  # 10% divergence
    ) -> Tuple[bool, Optional[DriftDetectionAlert]]:
        """Detect if prediction distribution has changed"""
        if not baseline_predictions or not current_predictions:
            return False, None

        baseline_mean = statistics.mean(baseline_predictions)
        current_mean = statistics.mean(current_predictions)

        # Calculate relative change
        change = abs(current_mean - baseline_mean) / baseline_mean if baseline_mean != 0 else 0

        if change > threshold:
            alert = DriftDetectionAlert(
                detected_at=datetime.now(),
                drift_type="prediction_drift",
                severity=self._severity_from_divergence(change),
                description=f"Prediction distribution shifted by {change:.1%}",
                recommended_action="Monitor model performance, may need retraining",
            )
            self.drift_alerts.append(alert)
            return True, alert

        return False, None

    def detect_model_drift(
        self,
        version_metrics_v1: Dict[str, float],
        version_metrics_v2: Dict[str, float],
        threshold: float = 0.15,  # 15% performance degradation
    ) -> Tuple[bool, Optional[DriftDetectionAlert]]:
        """Detect if model performance has degraded"""
        # Compare accuracy/precision/recall between versions
        for metric in ["accuracy", "precision", "recall", "f1"]:
            if metric in version_metrics_v1 and metric in version_metrics_v2:
                v1_score = version_metrics_v1[metric]
                v2_score = version_metrics_v2[metric]

                degradation = (v1_score - v2_score) / v1_score if v1_score > 0 else 0

                if degradation > threshold:
                    alert = DriftDetectionAlert(
                        detected_at=datetime.now(),
                        drift_type="model_drift",
                        severity="critical" if degradation > 0.3 else "high",
                        description=f"{metric} degraded from {v1_score:.3f} to {v2_score:.3f}",
                        recommended_action="Investigate model weights, revert or retrain",
                    )
                    self.drift_alerts.append(alert)
                    return True, alert

        return False, None

    def get_drift_alerts(self, limit: int = 10) -> List[DriftDetectionAlert]:
        """Get recent drift alerts"""
        return sorted(
            self.drift_alerts,
            key=lambda x: x.detected_at,
            reverse=True,
        )[:limit]

    def _analyze_dict_distribution(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze distribution of dict data"""
        distribution = {}
        if not data:
            return distribution

        # Analyze first dict's keys
        first_key = next(iter(data[0].keys()))
        values = [d.get(first_key) for d in data if first_key in d]

        if values and isinstance(values[0], (int, float)):
            distribution[first_key] = {
                "mean": statistics.mean(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                "min": min(values),
                "max": max(values),
            }

        return distribution

    def _analyze_array_distribution(
        self,
        data: List,
        feature_names: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Analyze distribution of array data"""
        distribution = {}

        for idx, feature_name in enumerate(feature_names or []):
            if idx >= len(data[0]):
                break

            values = [d[idx] for d in data if isinstance(d, (list, tuple)) and idx < len(d)]

            if values and isinstance(values[0], (int, float)):
                distribution[feature_name] = {
                    "mean": statistics.mean(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                }

        return distribution

    def _analyze_scalar_distribution(self, data: List) -> Dict[str, Any]:
        """Analyze distribution of scalar data"""
        if not data or not isinstance(data[0], (int, float)):
            return {}

        return {
            "mean": statistics.mean(data),
            "stdev": statistics.stdev(data) if len(data) > 1 else 0,
            "min": min(data),
            "max": max(data),
        }

    def _calculate_distribution_divergence(
        self,
        dist1: Dict[str, Any],
        dist2: Dict[str, Any],
    ) -> float:
        """Calculate divergence between two distributions (Wasserstein distance)"""
        if not dist1 or not dist2:
            return 0

        # Simple divergence based on mean difference
        divergence = 0
        count = 0

        for key in dist1:
            if key in dist2:
                if "mean" in dist1[key] and "mean" in dist2[key]:
                    mean_diff = abs(dist1[key]["mean"] - dist2[key]["mean"])
                    # Normalize by baseline standard deviation
                    baseline_std = dist1[key].get("stdev", 1)
                    if baseline_std > 0:
                        divergence += mean_diff / baseline_std
                    count += 1

        return divergence / count if count > 0 else 0

    def _severity_from_divergence(self, divergence: float) -> str:
        """Map divergence to severity level"""
        if divergence > 0.3:
            return "critical"
        elif divergence > 0.2:
            return "high"
        elif divergence > 0.1:
            return "medium"
        else:
            return "low"


class AnomalyDetector:
    """Detect anomalous behavior in model predictions"""

    def __init__(self, contamination: float = 0.05):
        """
        Initialize anomaly detector.

        Args:
            contamination: Expected proportion of anomalies (0-1)
        """
        self.contamination = contamination
        self.predictions: List[float] = []
        self.anomaly_scores: List[float] = []

    def fit(self, data: List[float]) -> None:
        """Fit detector on training data"""
        self.predictions = data
        self._calculate_anomaly_scores()

    def predict_anomalies(self, data: List[float]) -> List[bool]:
        """Predict which samples are anomalies"""
        anomalies = []

        for value in data:
            score = self._calculate_point_anomaly_score(value)
            threshold = self._calculate_threshold()
            anomalies.append(score > threshold)

        return anomalies

    def _calculate_anomaly_scores(self) -> None:
        """Calculate anomaly scores for all predictions"""
        if not self.predictions:
            return

        mean = statistics.mean(self.predictions)
        stdev = statistics.stdev(self.predictions) if len(self.predictions) > 1 else 1

        # Z-score based anomaly detection
        self.anomaly_scores = [
            abs((p - mean) / stdev) if stdev > 0 else 0
            for p in self.predictions
        ]

    def _calculate_point_anomaly_score(self, value: float) -> float:
        """Calculate anomaly score for single point"""
        if not self.predictions:
            return 0

        mean = statistics.mean(self.predictions)
        stdev = statistics.stdev(self.predictions) if len(self.predictions) > 1 else 1

        return abs((value - mean) / stdev) if stdev > 0 else 0

    def _calculate_threshold(self) -> float:
        """Calculate anomaly threshold"""
        if not self.anomaly_scores:
            return 3.0  # Default: 3 sigma

        sorted_scores = sorted(self.anomaly_scores)
        threshold_idx = int(len(sorted_scores) * (1 - self.contamination))
        return sorted_scores[threshold_idx] if threshold_idx < len(sorted_scores) else 3.0
