"""Routing policy engine: complexity tier -> backend, with circuit-breaker failover."""
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from src.client.base import ChatTurn, LLMClient, LLMResult, UpstreamError
from src.config import Settings
from src.routing.circuit_breaker import CircuitBreaker
from src.routing.classifier import ComplexityResult, RouteTier, score_prompt_complexity

logger = logging.getLogger(__name__)


class RouteTarget(StrEnum):
    """Low-cardinality route labels (safe for headers and future Prometheus labels)."""

    CACHE = "REDIS_VECTOR_CACHE"
    LOCAL = "LOCAL"
    CLOUD = "CLOUD"
    LOCAL_FAILOVER = "LOCAL_AUTO_FAILOVER_FROM_CLOUD"
    LOCAL_CIRCUIT_OPEN = "LOCAL_CIRCUIT_OPEN_FALLBACK"


@dataclass(frozen=True, slots=True)
class DispatchResult:
    result: LLMResult
    route: RouteTarget
    cost_usd: float

    @property
    def failover_triggered(self) -> bool:
        return self.route in (RouteTarget.LOCAL_FAILOVER, RouteTarget.LOCAL_CIRCUIT_OPEN)


class ModelRouter:
    """Decides the tier for a prompt and executes it with graceful degradation."""

    def __init__(
        self, local: LLMClient, cloud: LLMClient, breaker: CircuitBreaker, settings: Settings
    ) -> None:
        self._local = local
        self._cloud = cloud
        self._breaker = breaker
        self._settings = settings

    @property
    def breaker(self) -> CircuitBreaker:
        return self._breaker

    def classify(
        self, prompt: str, vector: bytes | None = None, model: str = "auto-router"
    ) -> ComplexityResult:
        result = score_prompt_complexity(
            prompt, threshold=self._settings.complexity_threshold, vector=vector
        )
        target = model.strip().lower()
        if target in {"local", self._settings.local_model.lower()}:
            return result.model_copy(update={"tier": RouteTier.LOCAL_SMALL})
        if target in {"cloud", self._settings.cloud_model.lower()}:
            return result.model_copy(update={"tier": RouteTier.CLOUD_FRONTIER})
        return result

    async def close(self) -> None:
        """Release pooled HTTP connections held by the model clients."""
        await self._local.aclose()
        await self._cloud.aclose()

    async def dispatch(
        self,
        messages: Sequence[ChatTurn],
        tier: RouteTier,
        simulate_cloud_outage: bool = False,
        *,
        max_tokens: int | None = 512,
        temperature: float | None = None,
    ) -> DispatchResult:
        """Run the request. Raises UpstreamError only if every eligible backend failed."""
        if tier is RouteTier.LOCAL_SMALL and not simulate_cloud_outage:
            return await self._run_local(
                messages, RouteTarget.LOCAL, max_tokens=max_tokens, temperature=temperature
            )
        return await self._run_cloud_with_failover(
            messages, simulate_cloud_outage, max_tokens=max_tokens, temperature=temperature
        )

    async def _run_local(
        self,
        messages: Sequence[ChatTurn],
        route: RouteTarget,
        *,
        max_tokens: int | None = 512,
        temperature: float | None = None,
    ) -> DispatchResult:
        result = await self._local.complete(messages, max_tokens=max_tokens, temperature=temperature)
        return DispatchResult(result=result, route=route, cost_usd=self._settings.local_cost_per_request_usd)

    async def _run_cloud_with_failover(
        self,
        messages: Sequence[ChatTurn],
        simulate_cloud_outage: bool,
        *,
        max_tokens: int | None = 512,
        temperature: float | None = None,
    ) -> DispatchResult:
        if not self._breaker.allow_request():
            logger.info("circuit_open_bypass", extra={"state": self._breaker.state})
            return await self._run_local(
                messages, RouteTarget.LOCAL_CIRCUIT_OPEN, max_tokens=max_tokens, temperature=temperature
            )

        try:
            if simulate_cloud_outage:
                raise UpstreamError(self._cloud.name, "Simulated Cloud API Outage (503 Service Unavailable)")
            result = await self._cloud.complete(messages, max_tokens=max_tokens, temperature=temperature)
        except UpstreamError as exc:
            self._breaker.record_failure()
            logger.warning(
                "cloud_failover",
                extra={
                    "error": str(exc),
                    "failures": self._breaker.failure_count,
                    "threshold": self._settings.breaker_failure_threshold,
                    "state": self._breaker.state,
                },
            )
            return await self._run_local(
                messages, RouteTarget.LOCAL_FAILOVER, max_tokens=max_tokens, temperature=temperature
            )
        finally:
            self._breaker.release_probe()  # no-op unless a probe was cancelled mid-flight

        self._breaker.record_success()
        return DispatchResult(
            result=result, route=RouteTarget.CLOUD, cost_usd=self._settings.cloud_cost_per_request_usd
        )
