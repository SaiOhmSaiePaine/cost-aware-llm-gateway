"""FastAPI application factory with lifespan-managed connection pools."""
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api import health, router
from src.api.dependencies import GatewayServices
from src.cache.embedder import OnnxEmbedder
from src.cache.redis_client import create_redis_client, ensure_vector_index
from src.cache.semantic_cache import SemanticCache
from src.client.base import UpstreamError
from src.client.cloud_client import GeminiClient
from src.client.vllm_client import OpenAICompatibleClient
from src.config import Settings, get_settings
from src.routing.circuit_breaker import CircuitBreaker
from src.routing.router import ModelRouter
from src.telemetry.logger import configure_logging
from src.telemetry.stats import GatewayStats

logger = logging.getLogger(__name__)


def _build_services(settings: Settings) -> GatewayServices:
    redis = create_redis_client(settings)
    embedder = OnnxEmbedder(settings.embedding_model, settings.embedding_dim)
    api_key = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else None
    if api_key is None:
        logger.error("gemini_api_key_missing", extra={"effect": "cloud tier will always fail over"})
    model_router = ModelRouter(
        local=OpenAICompatibleClient(settings.local_base_url, settings.local_model, settings.local_timeout_seconds),
        cloud=GeminiClient(api_key, settings.cloud_model, settings.cloud_timeout_seconds),
        breaker=CircuitBreaker(settings.breaker_failure_threshold, settings.breaker_recovery_timeout_seconds),
        settings=settings,
    )
    return GatewayServices(
        settings=settings,
        redis=redis,
        embedder=embedder,
        cache=SemanticCache(redis, settings),
        router=model_router,
        stats=GatewayStats(),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    services = _build_services(settings)
    services.embedder.warmup()
    await ensure_vector_index(services.redis, settings)  # never raises; logs if Redis is down
    app.state.services = services
    logger.info("gateway_started", extra={"local_model": settings.local_model, "cloud_model": settings.cloud_model})
    try:
        yield
    finally:
        await services.router.close()
        await services.redis.aclose()
        logger.info("gateway_stopped")


def create_app() -> FastAPI:
    app = FastAPI(title=get_settings().app_name, version="0.7.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
                       expose_headers=["X-Gateway-Route", "X-Gateway-Latency-Ms", "X-Gateway-Cache-Hit"])

    @app.exception_handler(UpstreamError)
    async def _upstream_error_handler(request: Request, exc: UpstreamError) -> JSONResponse:
        request.app.state.services.stats.upstream_errors += 1
        logger.error("all_upstreams_failed", extra={"provider": exc.provider, "error": str(exc)})
        return JSONResponse(
            status_code=503,
            content={"error": {"message": "All model backends are unavailable.", "type": "upstream_unavailable"}},
        )

    app.include_router(router.router)
    app.include_router(health.router)
    return app


app = create_app()
