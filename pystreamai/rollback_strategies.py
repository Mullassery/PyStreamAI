"""Phase 5: Advanced Rollback Strategies - Canary, Blue-Green, Shadow Rollback"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RollbackStrategy(Enum):
    """Rollback strategy types"""
    INSTANT = "instant"  # Immediate 100% rollback
    CANARY = "canary"  # Gradual rollback with health checks
    BLUE_GREEN = "blue_green"  # Instant switch, keep old for quick revert
    SHADOW = "shadow"  # Route % to old version in parallel
    GRADUAL = "gradual"  # Smooth percentage increase over time


@dataclass
class RollbackConfig:
    """Configuration for rollback execution"""
    strategy: RollbackStrategy = RollbackStrategy.CANARY
    traffic_steps: List[int] = field(default_factory=lambda: [10, 25, 50, 100])
    step_duration_seconds: int = 300  # 5 minutes per step
    health_check_interval_seconds: int = 30
    rollback_on_error: bool = True
    error_rate_threshold: float = 5.0  # % errors to abort this step
    latency_threshold_ms: float = 2000  # ms for p99 to abort


class RollbackExecutor(ABC):
    """Abstract base for rollback execution strategies"""

    @abstractmethod
    async def execute(
        self,
        current_version: str,
        target_version: str,
        config: RollbackConfig,
        health_check: Callable[[], Dict[str, Any]],
    ) -> bool:
        """Execute rollback. Returns True if successful."""
        pass

    @abstractmethod
    def get_traffic_split(self) -> Dict[str, float]:
        """Get current traffic split between versions"""
        pass

    @abstractmethod
    async def abort(self) -> None:
        """Abort ongoing rollback"""
        pass


class InstantRollback(RollbackExecutor):
    """Instant 100% rollback - no gradual transition"""

    async def execute(
        self,
        current_version: str,
        target_version: str,
        config: RollbackConfig,
        health_check: Callable[[], Dict[str, Any]],
    ) -> bool:
        """Rollback immediately to target version"""
        try:
            logger.warning(f"Instant rollback: {current_version} → {target_version}")

            # Switch immediately
            await self._switch_traffic(target_version, 100)

            # Brief health check
            await asyncio.sleep(5)
            health = health_check()

            if health.get("error_rate_percent", 0) > config.error_rate_threshold:
                logger.error("Health check failed after rollback")
                # Attempt to switch back (but likely to fail)
                await self._switch_traffic(current_version, 100)
                return False

            logger.info(f"Instant rollback successful")
            return True

        except Exception as e:
            logger.error(f"Instant rollback failed: {e}")
            return False

    async def _switch_traffic(self, version: str, percent: float) -> None:
        """Switch traffic to version"""
        # Implement based on deployment backend
        await asyncio.sleep(0.1)

    def get_traffic_split(self) -> Dict[str, float]:
        return {"target": 100}

    async def abort(self) -> None:
        pass


class CanaryRollback(RollbackExecutor):
    """Gradual rollback - shift traffic incrementally with health checks"""

    def __init__(self):
        self.current_step = 0
        self.traffic_splits: Dict[str, float] = {}
        self.is_aborting = False

    async def execute(
        self,
        current_version: str,
        target_version: str,
        config: RollbackConfig,
        health_check: Callable[[], Dict[str, Any]],
    ) -> bool:
        """Gradually rollback with health checks at each step"""
        try:
            logger.info(
                f"Canary rollback: {current_version} → {target_version} "
                f"({config.traffic_steps})"
            )

            for step_idx, target_percent in enumerate(config.traffic_steps):
                if self.is_aborting:
                    logger.warning("Rollback aborted by user")
                    return False

                logger.info(f"Rollback step {step_idx + 1}: {target_percent}% to {target_version}")

                # Shift traffic
                current_percent = 100 - target_percent
                self.traffic_splits = {
                    "current": current_percent,
                    "target": target_percent,
                }
                await self._shift_traffic(current_version, target_version, target_percent)

                # Monitor health
                step_start = time.time()
                step_healthy = False

                while time.time() - step_start < config.step_duration_seconds:
                    if self.is_aborting:
                        # Revert to current version on abort
                        await self._shift_traffic(current_version, target_version, 0)
                        return False

                    health = health_check()
                    error_rate = health.get("error_rate_percent", 0)
                    latency = health.get("p99_latency_ms", 0)

                    if (
                        error_rate > config.error_rate_threshold
                        or latency > config.latency_threshold_ms
                    ):
                        logger.error(
                            f"Health check failed at {target_percent}%: "
                            f"error_rate={error_rate}%, latency={latency}ms"
                        )

                        if config.rollback_on_error:
                            # Revert to current version
                            await self._shift_traffic(current_version, target_version, 0)
                            logger.warning("Rolled back to previous step")
                            return False

                    await asyncio.sleep(config.health_check_interval_seconds)
                    step_healthy = True

                if target_percent == 100:
                    logger.info("Canary rollback successful - 100% on target version")
                    return True

            return True

        except Exception as e:
            logger.error(f"Canary rollback failed: {e}")
            return False

    async def _shift_traffic(self, current: str, target: str, target_percent: float) -> None:
        """Shift traffic to target version"""
        await asyncio.sleep(0.1)  # Simulated shift

    def get_traffic_split(self) -> Dict[str, float]:
        return self.traffic_splits

    async def abort(self) -> None:
        """Abort ongoing rollback"""
        logger.warning("Aborting canary rollback")
        self.is_aborting = True


class BlueGreenRollback(RollbackExecutor):
    """Blue-green rollback - instant switch, keep both versions running"""

    def __init__(self):
        self.active_version = None
        self.standby_version = None

    async def execute(
        self,
        current_version: str,
        target_version: str,
        config: RollbackConfig,
        health_check: Callable[[], Dict[str, Any]],
    ) -> bool:
        """Instant switch to target, keep current running for quick revert"""
        try:
            logger.info(f"Blue-green rollback: {current_version} (blue) ← {target_version} (green)")

            # Set blue as standby (for quick revert)
            self.standby_version = current_version
            self.active_version = target_version

            # Switch traffic to green
            await self._switch_traffic(target_version)

            # Brief validation
            await asyncio.sleep(5)
            health = health_check()

            if health.get("error_rate_percent", 0) > config.error_rate_threshold:
                logger.warning("Health check failed, reverting to blue")
                await self._switch_traffic(current_version)
                return False

            logger.info("Blue-green rollback successful")
            return True

        except Exception as e:
            logger.error(f"Blue-green rollback failed: {e}")
            return False

    async def quick_revert(self) -> bool:
        """Quick revert to previous version if something goes wrong"""
        if self.standby_version:
            logger.warning(f"Quick revert to {self.standby_version}")
            await self._switch_traffic(self.standby_version)
            return True
        return False

    async def _switch_traffic(self, version: str) -> None:
        """Switch all traffic to version"""
        await asyncio.sleep(0.1)

    def get_traffic_split(self) -> Dict[str, float]:
        if self.active_version:
            return {self.active_version: 100}
        return {}

    async def abort(self) -> None:
        """Revert to blue immediately"""
        if self.standby_version:
            await self._switch_traffic(self.standby_version)


class ShadowRollback(RollbackExecutor):
    """Shadow rollback - route % to target in parallel (no impact on users)"""

    def __init__(self):
        self.shadow_percent = 0

    async def execute(
        self,
        current_version: str,
        target_version: str,
        config: RollbackConfig,
        health_check: Callable[[], Dict[str, Any]],
    ) -> bool:
        """Shadow route traffic to validate target before cutover"""
        try:
            logger.info(f"Shadow rollback: Testing {target_version} with shadow traffic")

            # Gradually increase shadow traffic
            for shadow_percent in [5, 10, 25, 50]:
                self.shadow_percent = shadow_percent
                await self._enable_shadow(target_version, shadow_percent)

                # Monitor shadow metrics separately
                shadow_start = time.time()
                while time.time() - shadow_start < 300:
                    health = health_check()

                    if health.get("error_rate_percent", 0) > config.error_rate_threshold:
                        logger.error(
                            f"Shadow traffic unhealthy at {shadow_percent}%, "
                            f"aborting shadow"
                        )
                        await self._disable_shadow()
                        return False

                    await asyncio.sleep(30)

            # Shadow traffic validated, do full cutover
            logger.info("Shadow traffic validated, switching 100% to target")
            await self._full_cutover(current_version, target_version)

            return True

        except Exception as e:
            logger.error(f"Shadow rollback failed: {e}")
            await self._disable_shadow()
            return False

    async def _enable_shadow(self, version: str, percent: float) -> None:
        """Enable shadow traffic to version"""
        await asyncio.sleep(0.1)

    async def _disable_shadow(self) -> None:
        """Disable shadow traffic"""
        self.shadow_percent = 0
        await asyncio.sleep(0.1)

    async def _full_cutover(self, current: str, target: str) -> None:
        """Switch 100% to target"""
        await asyncio.sleep(0.1)

    def get_traffic_split(self) -> Dict[str, float]:
        return {"shadow_percent": self.shadow_percent}

    async def abort(self) -> None:
        """Abort shadow traffic"""
        await self._disable_shadow()


class RollbackOrchestrator:
    """Orchestrate rollbacks with different strategies"""

    def __init__(self, auto_version_manager: "AutoVersionManager"):
        self.manager = auto_version_manager
        self.current_rollback: Optional[RollbackExecutor] = None
        self.rollback_history: List[Dict[str, Any]] = []

    async def rollback(
        self,
        version_id: str,
        reason: str,
        strategy: RollbackStrategy = RollbackStrategy.CANARY,
        config: Optional[RollbackConfig] = None,
    ) -> bool:
        """Execute rollback with specified strategy"""
        config = config or RollbackConfig(strategy=strategy)

        version = self.manager.registry.get_version(version_id)
        if not version or not version.promoted_at:
            logger.error(f"Cannot rollback: version {version_id} not found or not promoted")
            return False

        # Get current version
        current_version_id = self.manager.active_versions.get(version.model_id)
        if not current_version_id:
            logger.error("No active version to rollback from")
            return False

        try:
            # Select executor based on strategy
            executor = self._get_executor(strategy)
            self.current_rollback = executor

            # Create health checker
            def health_check_fn() -> Dict[str, Any]:
                monitor = self.manager.monitors.get(current_version_id)
                return monitor.get_current_health() if monitor else {}

            # Execute rollback
            rollback_start = datetime.now()
            success = await executor.execute(current_version_id, version_id, config, health_check_fn)

            if success:
                # Update version status
                self.manager.active_versions[version.model_id] = version_id
                self.manager.registry.promote_version(version_id)

                # Record in history
                self.rollback_history.append({
                    "from_version": current_version_id,
                    "to_version": version_id,
                    "strategy": strategy.value,
                    "reason": reason,
                    "timestamp": rollback_start.isoformat(),
                    "duration_seconds": (datetime.now() - rollback_start).total_seconds(),
                    "success": True,
                })

                logger.info(f"Rollback successful: {version_id} via {strategy.value}")

            else:
                self.rollback_history.append({
                    "from_version": current_version_id,
                    "to_version": version_id,
                    "strategy": strategy.value,
                    "reason": reason,
                    "timestamp": rollback_start.isoformat(),
                    "duration_seconds": (datetime.now() - rollback_start).total_seconds(),
                    "success": False,
                })

                logger.error(f"Rollback failed: {version_id} via {strategy.value}")

            return success

        except Exception as e:
            logger.error(f"Rollback error: {e}")
            return False

    async def abort_rollback(self) -> None:
        """Abort currently running rollback"""
        if self.current_rollback:
            await self.current_rollback.abort()
            self.current_rollback = None

    def _get_executor(self, strategy: RollbackStrategy) -> RollbackExecutor:
        """Get executor for strategy"""
        executors = {
            RollbackStrategy.INSTANT: InstantRollback(),
            RollbackStrategy.CANARY: CanaryRollback(),
            RollbackStrategy.BLUE_GREEN: BlueGreenRollback(),
            RollbackStrategy.SHADOW: ShadowRollback(),
            RollbackStrategy.GRADUAL: CanaryRollback(),  # Same as canary
        }
        return executors.get(strategy, CanaryRollback())

    def get_rollback_history(self, model_id: str) -> List[Dict[str, Any]]:
        """Get rollback history for a model"""
        return [r for r in self.rollback_history]

    def get_current_rollback_status(self) -> Optional[Dict[str, Any]]:
        """Get status of current rollback"""
        if not self.current_rollback:
            return None

        return {
            "traffic_split": self.current_rollback.get_traffic_split(),
            "can_abort": True,
        }
