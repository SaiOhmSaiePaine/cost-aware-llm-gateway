"""Async Redis connection pool and HNSW vector index bootstrap."""
import logging

from redis.asyncio import Redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.exceptions import RedisError, ResponseError

from src.config import Settings

logger = logging.getLogger(__name__)


def create_redis_client(settings: Settings) -> Redis:
    """Create a pooled async client with short timeouts so a dead Redis fails fast."""
    return Redis.from_url(
        settings.redis_url,
        decode_responses=False,  # vectors are raw bytes
        socket_timeout=settings.redis_socket_timeout_seconds,
        socket_connect_timeout=settings.redis_socket_timeout_seconds,
    )


async def ensure_vector_index(client: Redis, settings: Settings) -> bool:
    """Create the HNSW index if missing. Returns False (never raises) if Redis is unavailable."""
    search = client.ft(settings.cache_index_name)
    try:
        await search.info()
        logger.info("redis_index_exists", extra={"index": settings.cache_index_name})
        return True
    except ResponseError:
        pass  # Index does not exist yet -> create it below.
    except (RedisError, OSError) as exc:
        logger.warning("redis_unavailable_at_startup", extra={"error": str(exc)})
        return False

    schema = (
        TextField("prompt"),
        TextField("system_prompt"),
        TextField("answer"),
        VectorField(
            "vector",
            "HNSW",
            {"TYPE": "FLOAT32", "DIM": settings.embedding_dim, "DISTANCE_METRIC": "COSINE"},
        ),
    )
    definition = IndexDefinition(prefix=[settings.cache_key_prefix], index_type=IndexType.HASH)
    try:
        await search.create_index(fields=schema, definition=definition)
        logger.info("redis_index_created", extra={"index": settings.cache_index_name})
        return True
    except (RedisError, OSError) as exc:
        logger.warning("redis_index_create_failed", extra={"error": str(exc)})
        return False
