"""Health probe and admin stats endpoints."""
from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from redis.exceptions import RedisError

from src.api.dependencies import GatewayServices, get_services
from src.telemetry.stats import GatewayStats

router = APIRouter(tags=["ops"])
ServicesDep = Annotated[GatewayServices, Depends(get_services)]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    redis: Literal["up", "down"]
    circuit_breaker_state: str


class AdminStatsResponse(GatewayStats):
    circuit_breaker_state: str
    consecutive_cloud_failures: int


@router.get("/health", response_model=HealthResponse)
async def health(services: ServicesDep) -> HealthResponse:
    """Liveness is implied by any response; 'degraded' means the cache is being bypassed."""
    try:
        redis_ok = bool(await services.redis.ping())
    except (RedisError, OSError):
        redis_ok = False
    return HealthResponse(
        status="ok" if redis_ok else "degraded",
        redis="up" if redis_ok else "down",
        circuit_breaker_state=services.router.breaker.state,
    )


@router.get("/v1/admin/stats", response_model=AdminStatsResponse)
async def admin_stats(services: ServicesDep) -> AdminStatsResponse:
    breaker = services.router.breaker
    return AdminStatsResponse(
        **services.stats.model_dump(),
        circuit_breaker_state=breaker.state,
        consecutive_cloud_failures=breaker.failure_count,
    )
