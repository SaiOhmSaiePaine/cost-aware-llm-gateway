"""Dependency-injection container built once in the FastAPI lifespan."""
from dataclasses import dataclass

from fastapi import Request
from redis.asyncio import Redis

from src.cache.embedder import OnnxEmbedder
from src.cache.semantic_cache import SemanticCache
from src.config import Settings
from src.routing.router import ModelRouter
from src.telemetry.stats import GatewayStats


@dataclass(slots=True)
class GatewayServices:
    settings: Settings
    redis: Redis
    embedder: OnnxEmbedder
    cache: SemanticCache
    router: ModelRouter
    stats: GatewayStats


def get_services(request: Request) -> GatewayServices:
    """FastAPI dependency returning the shared, pooled service container."""
    services: GatewayServices = request.app.state.services
    return services
