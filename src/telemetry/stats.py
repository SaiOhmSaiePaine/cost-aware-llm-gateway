"""Unified in-process request ledger and Prometheus metric recorder."""
from pydantic import BaseModel

from src.telemetry.metrics import (
    COST_SAVED_USD_TOTAL,
    GUARDRAIL_BLOCKS_TOTAL,
    ONNX_EMBED_LATENCY_SECONDS,
    REQUEST_LATENCY_SECONDS,
    REQUESTS_TOTAL,
    UPSTREAM_ERRORS_TOTAL,
    set_breaker_state,
)


# shortcut: single-worker in-memory fields alongside Prometheus collectors; aggregate via Prometheus when --workers > 1
class GatewayStats(BaseModel):
    total_requests: int = 0
    cache_hits: int = 0
    guardrail_blocks: int = 0
    local_routed_calls: int = 0
    cloud_routed_calls: int = 0
    failover_saves: int = 0
    upstream_errors: int = 0
    money_saved_usd: float = 0.0

    def add_savings(self, amount_usd: float) -> None:
        saved = max(0.0, amount_usd)
        self.money_saved_usd = round(self.money_saved_usd + saved, 6)
        if saved > 0.0:
            COST_SAVED_USD_TOTAL.inc(saved)

    def record_embed(self, embed_ms: float | None) -> None:
        if embed_ms is not None:
            ONNX_EMBED_LATENCY_SECONDS.observe(embed_ms / 1000.0)

    def record_guardrail_block(self) -> None:
        self.guardrail_blocks += 1
        GUARDRAIL_BLOCKS_TOTAL.inc()

    def record_cache_hit(self, route: str, latency_ms: float, saved_usd: float, breaker_state: str) -> None:
        self.total_requests += 1
        self.cache_hits += 1
        self.add_savings(saved_usd)
        REQUESTS_TOTAL.labels(route_target=route, cache_hit="true", failover="false").inc()
        REQUEST_LATENCY_SECONDS.labels(route_target=route).observe(latency_ms / 1000.0)
        set_breaker_state(breaker_state)

    def record_dispatch(
        self,
        route: str,
        latency_ms: float,
        saved_usd: float,
        breaker_state: str,
        *,
        failover: bool = False,
    ) -> None:
        self.total_requests += 1
        if route == "CLOUD":
            self.cloud_routed_calls += 1
        else:
            self.local_routed_calls += 1
            self.add_savings(saved_usd)
            if failover:
                self.failover_saves += 1
        REQUESTS_TOTAL.labels(
            route_target=route, cache_hit="false", failover=str(failover).lower()
        ).inc()
        REQUEST_LATENCY_SECONDS.labels(route_target=route).observe(latency_ms / 1000.0)
        set_breaker_state(breaker_state)

    def record_upstream_error(self, breaker_state: str = "OPEN") -> None:
        self.total_requests += 1
        self.upstream_errors += 1
        UPSTREAM_ERRORS_TOTAL.inc()
        REQUESTS_TOTAL.labels(route_target="UPSTREAM_ERROR", cache_hit="false", failover="true").inc()
        set_breaker_state(breaker_state)
