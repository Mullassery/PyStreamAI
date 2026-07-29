"""Phase 10: Multi-Model Orchestration - Deploy and manage multiple models as coordinated systems"""

import logging
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class PipelineStageType(Enum):
    """Types of pipeline stages"""
    SEQUENTIAL = "sequential"  # Run one after another
    PARALLEL = "parallel"  # Run simultaneously
    CONDITIONAL = "conditional"  # Run based on condition
    ENSEMBLE = "ensemble"  # Combine outputs


@dataclass
class PipelineStage:
    """Single stage in model pipeline"""
    name: str
    model_id: str
    stage_type: PipelineStageType = PipelineStageType.SEQUENTIAL
    input_from: Optional[str] = None  # Name of previous stage (for sequential)
    condition: Optional[Callable[[Dict], bool]] = None  # For conditional stages
    ensemble_weights: Optional[Dict[str, float]] = None  # For ensemble stages


@dataclass
class PipelineMetrics:
    """Metrics for entire pipeline"""
    pipeline_id: str
    total_latency_ms: float
    stage_latencies: Dict[str, float] = field(default_factory=dict)
    error_rate_percent: float = 0.0
    throughput_rps: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class ModelPipeline:
    """Orchestrate multiple models as a coordinated pipeline"""

    def __init__(self, pipeline_id: str):
        self.pipeline_id = pipeline_id
        self.stages: List[PipelineStage] = []
        self.version_map: Dict[str, str] = {}  # model_id -> version_id
        self.metrics_history: List[PipelineMetrics] = []

    def add_stage(self, stage: PipelineStage) -> None:
        """Add stage to pipeline"""
        self.stages.append(stage)
        logger.info(f"Added stage {stage.name} to pipeline {self.pipeline_id}")

    def add_sequential_stage(self, name: str, model_id: str) -> None:
        """Add sequential stage"""
        stage = PipelineStage(
            name=name,
            model_id=model_id,
            stage_type=PipelineStageType.SEQUENTIAL,
            input_from=self.stages[-1].name if self.stages else None,
        )
        self.add_stage(stage)

    def add_parallel_stage(self, name: str, model_ids: List[str]) -> None:
        """Add parallel stages (one for each model)"""
        for idx, model_id in enumerate(model_ids):
            stage = PipelineStage(
                name=f"{name}_{idx}",
                model_id=model_id,
                stage_type=PipelineStageType.PARALLEL,
            )
            self.add_stage(stage)

    def add_conditional_stage(
        self,
        name: str,
        model_id: str,
        condition: Callable[[Dict], bool],
    ) -> None:
        """Add conditional stage (runs if condition is true)"""
        stage = PipelineStage(
            name=name,
            model_id=model_id,
            stage_type=PipelineStageType.CONDITIONAL,
            condition=condition,
        )
        self.add_stage(stage)

    def add_ensemble_stage(
        self,
        name: str,
        model_ids: List[str],
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Add ensemble stage (combine multiple models)"""
        # Default equal weights
        if not weights:
            weight = 1.0 / len(model_ids)
            weights = {mid: weight for mid in model_ids}

        for model_id in model_ids:
            stage = PipelineStage(
                name=f"{name}_{model_id}",
                model_id=model_id,
                stage_type=PipelineStageType.ENSEMBLE,
                ensemble_weights=weights,
            )
            self.add_stage(stage)

    async def execute(
        self,
        input_data: Dict[str, Any],
        inference_fn: Callable[[str, str, Dict], Tuple[Any, float]],
    ) -> Tuple[Any, PipelineMetrics]:
        """
        Execute entire pipeline.

        Args:
            input_data: Input to first stage
            inference_fn: Function to run inference (model_id, version_id, data) -> (output, latency_ms)

        Returns:
            (final_output, pipeline_metrics)
        """
        import asyncio
        import time

        start_time = time.time()
        stage_outputs: Dict[str, Any] = {}
        stage_latencies: Dict[str, float] = {}
        error_rate = 0.0

        # Group stages by type
        sequential_stages = [s for s in self.stages if s.stage_type == PipelineStageType.SEQUENTIAL]
        parallel_stages = [s for s in self.stages if s.stage_type == PipelineStageType.PARALLEL]
        conditional_stages = [s for s in self.stages if s.stage_type == PipelineStageType.CONDITIONAL]
        ensemble_stages = [s for s in self.stages if s.stage_type == PipelineStageType.ENSEMBLE]

        try:
            # Execute sequential stages
            current_input = input_data
            for stage in sequential_stages:
                stage_input = stage_outputs.get(stage.input_from, current_input) if stage.input_from else current_input

                output, latency = await self._execute_stage(
                    stage,
                    stage_input,
                    inference_fn,
                )

                stage_outputs[stage.name] = output
                stage_latencies[stage.name] = latency
                current_input = output

            # Execute parallel stages
            if parallel_stages:
                parallel_tasks = [
                    self._execute_stage(stage, current_input, inference_fn)
                    for stage in parallel_stages
                ]
                parallel_results = await asyncio.gather(*parallel_tasks, return_exceptions=True)

                for stage, result in zip(parallel_stages, parallel_results):
                    if isinstance(result, Exception):
                        logger.error(f"Parallel stage {stage.name} failed: {result}")
                        error_rate += 1
                    else:
                        output, latency = result
                        stage_outputs[stage.name] = output
                        stage_latencies[stage.name] = latency

            # Execute conditional stages
            for stage in conditional_stages:
                if stage.condition and stage.condition(stage_outputs):
                    output, latency = await self._execute_stage(
                        stage,
                        current_input,
                        inference_fn,
                    )
                    stage_outputs[stage.name] = output
                    stage_latencies[stage.name] = latency

            # Execute ensemble stages and combine
            if ensemble_stages:
                ensemble_outputs = []
                for stage in ensemble_stages:
                    output, latency = await self._execute_stage(
                        stage,
                        current_input,
                        inference_fn,
                    )
                    ensemble_outputs.append(output)
                    stage_latencies[stage.name] = latency

                # Combine ensemble outputs (weighted average for numeric, voting for categorical)
                final_output = self._combine_ensemble_outputs(ensemble_outputs, ensemble_stages[0].ensemble_weights)
                stage_outputs["ensemble"] = final_output

            final_output = stage_outputs.get(self.stages[-1].name if self.stages else "output", current_input)

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            error_rate = 1.0
            final_output = None

        # Calculate metrics
        total_latency = (time.time() - start_time) * 1000

        metrics = PipelineMetrics(
            pipeline_id=self.pipeline_id,
            total_latency_ms=total_latency,
            stage_latencies=stage_latencies,
            error_rate_percent=error_rate * 100,
            throughput_rps=1000 / total_latency if total_latency > 0 else 0,
        )

        self.metrics_history.append(metrics)

        return final_output, metrics

    async def _execute_stage(
        self,
        stage: PipelineStage,
        input_data: Dict[str, Any],
        inference_fn: Callable,
    ) -> Tuple[Any, float]:
        """Execute single stage"""
        import asyncio

        version_id = self.version_map.get(stage.model_id)
        if not version_id:
            raise ValueError(f"No version set for model {stage.model_id}")

        # Run inference (wrap sync function in async)
        loop = asyncio.get_event_loop()
        output, latency = await loop.run_in_executor(
            None,
            lambda: inference_fn(stage.model_id, version_id, input_data),
        )

        return output, latency

    def _combine_ensemble_outputs(
        self,
        outputs: List[Any],
        weights: Optional[Dict[str, float]],
    ) -> Any:
        """Combine ensemble outputs"""
        if not outputs:
            return None

        # Try numeric averaging
        try:
            return sum(outputs) / len(outputs)
        except:
            pass

        # Try list/vector averaging
        try:
            if isinstance(outputs[0], (list, tuple)):
                return [sum(x) / len(outputs) for x in zip(*outputs)]
        except:
            pass

        # Fallback: return first output
        return outputs[0]

    def set_version(self, model_id: str, version_id: str) -> None:
        """Set version for model in pipeline"""
        self.version_map[model_id] = version_id

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline metrics"""
        if not self.metrics_history:
            return {}

        latencies = [m.total_latency_ms for m in self.metrics_history]
        errors = [m.error_rate_percent for m in self.metrics_history]

        import statistics
        return {
            "total_executions": len(self.metrics_history),
            "avg_latency_ms": statistics.mean(latencies),
            "p95_latency_ms": self._percentile(latencies, 0.95),
            "p99_latency_ms": self._percentile(latencies, 0.99),
            "avg_error_rate_percent": statistics.mean(errors),
            "max_error_rate_percent": max(errors),
        }

    @staticmethod
    def _percentile(data: List[float], percentile: float) -> float:
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        return sorted_data[min(index, len(sorted_data) - 1)]


class PipelineRegistry:
    """Registry of model pipelines"""

    def __init__(self):
        self.pipelines: Dict[str, ModelPipeline] = {}
        self.pipeline_versions: Dict[str, List[str]] = {}  # pipeline_id -> [version_ids]

    def register_pipeline(self, pipeline: ModelPipeline) -> None:
        """Register pipeline"""
        self.pipelines[pipeline.pipeline_id] = pipeline
        self.pipeline_versions[pipeline.pipeline_id] = []
        logger.info(f"Registered pipeline: {pipeline.pipeline_id}")

    def get_pipeline(self, pipeline_id: str) -> Optional[ModelPipeline]:
        """Get pipeline"""
        return self.pipelines.get(pipeline_id)

    def version_pipeline(self, pipeline_id: str, version_id: str) -> None:
        """Create version of pipeline"""
        if pipeline_id not in self.pipeline_versions:
            self.pipeline_versions[pipeline_id] = []

        self.pipeline_versions[pipeline_id].append(version_id)
        logger.info(f"Versioned pipeline {pipeline_id} as {version_id}")

    def promote_pipeline_version(self, pipeline_id: str, version_id: str) -> None:
        """Promote pipeline version to production"""
        if pipeline_id in self.pipelines:
            logger.info(f"Promoted {pipeline_id} {version_id} to production")

    def rollback_pipeline(self, pipeline_id: str, version_id: str) -> bool:
        """Rollback pipeline to previous version"""
        pipeline = self.pipelines.get(pipeline_id)
        if not pipeline:
            return False

        logger.warning(f"Rolled back {pipeline_id} to {version_id}")
        return True


class CrossModelABTest:
    """Run A/B tests across multiple models in pipeline"""

    def __init__(self, test_id: str):
        self.test_id = test_id
        self.variant_a: Dict[str, str] = {}  # model_id -> version_id
        self.variant_b: Dict[str, str] = {}  # model_id -> version_id
        self.metrics_a: Dict[str, Any] = {}
        self.metrics_b: Dict[str, Any] = {}
        self.split_percent: float = 50.0

    def set_variant_a(self, model_versions: Dict[str, str]) -> None:
        """Set variant A (model_id -> version_id)"""
        self.variant_a = model_versions

    def set_variant_b(self, model_versions: Dict[str, str]) -> None:
        """Set variant B"""
        self.variant_b = model_versions

    def set_split(self, percent: float) -> None:
        """Set A/B split percentage"""
        self.split_percent = percent

    def record_metric(self, variant: str, metric: str, value: float) -> None:
        """Record metric for variant"""
        if variant == "a":
            if metric not in self.metrics_a:
                self.metrics_a[metric] = []
            self.metrics_a[metric].append(value)
        else:
            if metric not in self.metrics_b:
                self.metrics_b[metric] = []
            self.metrics_b[metric].append(value)

    def get_stats(self) -> Dict[str, Any]:
        """Get A/B test statistics"""
        import statistics

        stats = {
            "test_id": self.test_id,
            "variant_a": self._calculate_variant_stats(self.metrics_a),
            "variant_b": self._calculate_variant_stats(self.metrics_b),
            "winners": {},
        }

        # Determine winners for each metric
        for metric in self.metrics_a:
            if metric in self.metrics_b:
                avg_a = statistics.mean(self.metrics_a[metric])
                avg_b = statistics.mean(self.metrics_b[metric])

                if metric in ["latency_ms", "error_rate_percent"]:
                    # Lower is better
                    stats["winners"][metric] = "a" if avg_a < avg_b else "b"
                else:
                    # Higher is better (throughput, accuracy)
                    stats["winners"][metric] = "a" if avg_a > avg_b else "b"

        return stats

    def _calculate_variant_stats(self, metrics: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate statistics for variant"""
        import statistics

        stats = {}
        for metric, values in metrics.items():
            if values:
                stats[metric] = {
                    "mean": statistics.mean(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }

        return stats

    def get_recommendation(self) -> Tuple[str, str]:
        """Get recommendation for best variant"""
        stats = self.get_stats()

        a_wins = sum(1 for w in stats["winners"].values() if w == "a")
        b_wins = sum(1 for w in stats["winners"].values() if w == "b")

        if a_wins > b_wins:
            return "a", f"Variant A wins {a_wins} of {a_wins + b_wins} metrics"
        else:
            return "b", f"Variant B wins {b_wins} of {a_wins + b_wins} metrics"
