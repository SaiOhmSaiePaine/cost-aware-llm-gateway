"""Async semantic cache: KNN lookup with guardrails, and fire-and-forget TTL writes."""
import logging
import uuid
from dataclasses import dataclass

from redis.asyncio import Redis
from redis.commands.search.query import Query
from redis.exceptions import RedisError

from src.cache.guardrails import passes_guardrails
from src.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CacheLookupResult:
    """Outcome of a semantic cache lookup."""

    hit: bool
    answer: str | None = None
    matched_prompt: str | None = None
    similarity: float | None = None
    guardrail_blocked: bool = False


_MISS = CacheLookupResult(hit=False)


def _to_str(value: str | bytes) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else value


class SemanticCache:
    """Redis Stack HNSW-backed semantic cache. Never raises on Redis failure."""

    def __init__(self, client: Redis, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self._query = (
            Query("*=>[KNN 1 @vector $vec AS vector_distance]")
            .return_fields("prompt", "answer", "vector_distance")
            .dialect(2)
        )

    async def lookup(self, prompt: str, vector: bytes) -> CacheLookupResult:
        """Return the nearest cached answer if similarity and guardrails both pass."""
        try:
            results = await self._client.ft(self._settings.cache_index_name).search(
                self._query, query_params={"vec": vector}
            )
        except (RedisError, OSError) as exc:
            logger.warning("cache_lookup_bypassed", extra={"error": str(exc)})
            return _MISS

        if not results.docs:
            return _MISS

        top = results.docs[0]
        similarity = 1.0 - float(top.vector_distance)
        if similarity < self._settings.similarity_threshold:
            return _MISS

        cached_prompt = _to_str(top.prompt)
        if not passes_guardrails(prompt, cached_prompt):
            logger.info("cache_guardrail_block", extra={"similarity": round(similarity, 3)})
            return CacheLookupResult(
                hit=False, matched_prompt=cached_prompt, similarity=similarity, guardrail_blocked=True
            )

        return CacheLookupResult(
            hit=True, answer=_to_str(top.answer), matched_prompt=cached_prompt, similarity=similarity
        )

    async def store(self, prompt: str, answer: str, vector: bytes) -> None:
        """Write an entry with TTL in one round-trip. Intended to run as a background task."""
        key = f"{self._settings.cache_key_prefix}{uuid.uuid4()}"
        try:
            async with self._client.pipeline(transaction=False) as pipe:
                pipe.hset(key, mapping={"prompt": prompt, "answer": answer, "vector": vector})
                pipe.expire(key, self._settings.cache_ttl_seconds)
                await pipe.execute()
        except (RedisError, OSError) as exc:
            logger.warning("cache_store_failed", extra={"error": str(exc)})
