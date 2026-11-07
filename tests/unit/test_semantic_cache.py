"""Unit tests for Phase 2.5: KNN 3 semantic cache, system isolation, LRU embed cache, model override, and bounded generation."""
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.api.schemas import ChatCompletionRequest
from src.cache.embedder import OnnxEmbedder
from src.cache.semantic_cache import SemanticCache
from src.client.base import ChatTurn, LLMClient, LLMResult
from src.config import Settings
from src.routing.circuit_breaker import CircuitBreaker
from src.routing.classifier import RouteTier
from src.routing.router import ModelRouter


class _DummyLLM(LLMClient):
    def __init__(self, name: str) -> None:
        self.name = name
        self.last_kwargs: dict[str, int | float | None] = {}

    async def complete(
        self,
        messages: list[ChatTurn],
        *,
        max_tokens: int | None = 512,
        temperature: float | None = None,
    ) -> LLMResult:
        self.last_kwargs = {"max_tokens": max_tokens, "temperature": temperature}
        return LLMResult(text="ok", model=self.name)


def _make_fake_embedder() -> OnnxEmbedder:
    call_count = 0

    def _fake_embed_list(texts: list[str]):
        nonlocal call_count
        call_count += 1
        time.sleep(0.01)  # simulate 10ms cold ONNX inference
        yield np.ones(384, dtype=np.float32)

    embedder = OnnxEmbedder("fake-onnx", 384, model=SimpleNamespace(embed=_fake_embed_list))
    embedder.call_count = lambda: call_count  # type: ignore[attr-defined]
    return embedder


@pytest.mark.anyio
async def test_lru_embedding_cache_sub_1ms_on_repeat() -> None:
    import asyncio

    embedder = _make_fake_embedder()
    results = await asyncio.gather(*(embedder.aembed("What is the capital of Japan?") for _ in range(5)))
    t0 = time.perf_counter()
    vec2 = await embedder.aembed("  what is the CAPITAL of japan?  ")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert all(v == vec2 for v in results)
    assert elapsed_ms <= 1.0
    assert embedder.call_count() == 1  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_aembed_updates_lru_recency_and_evicts_least_recently_used(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.cache.embedder as embedder_mod

    monkeypatch.setattr(embedder_mod, "_LRU_MAXSIZE", 3)
    embedder = _make_fake_embedder()

    await embedder.aembed("k1")
    await embedder.aembed("k2")
    await embedder.aembed("k3")
    assert embedder.call_count() == 3  # type: ignore[attr-defined]

    # Touch "k1" via aembed hit so "k2" becomes the least-recently-used entry
    await embedder.aembed("k1")
    await embedder.aembed("k4")  # evicts "k2", keeps "k1", "k3", "k4"
    assert embedder.call_count() == 4  # type: ignore[attr-defined]

    await embedder.aembed("k1")  # must still be cached in memory
    assert embedder.call_count() == 4  # type: ignore[attr-defined]
    await embedder.aembed("k2")  # was evicted, so this re-runs cold embed
    assert embedder.call_count() == 5  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_knn3_returns_second_neighbor_when_first_blocked_by_guardrail() -> None:
    settings = Settings()
    docs = [
        SimpleNamespace(
            prompt="Calculate 15% tip on $60",
            system_prompt="",
            answer="$9.00",
            vector_distance="0.04",  # similarity 0.96, blocked by number mismatch ($60 vs $90)
        ),
        SimpleNamespace(
            prompt="What is a 15% tip on $90?",
            system_prompt="",
            answer="$13.50",
            vector_distance="0.08",  # similarity 0.92, passes guardrails!
        ),
    ]
    ft_mock = MagicMock()
    ft_mock.search = AsyncMock(return_value=SimpleNamespace(docs=docs))
    redis_mock = MagicMock()
    redis_mock.ft.return_value = ft_mock

    cache = SemanticCache(redis_mock, settings)
    res = await cache.lookup("Calculate 15% tip on $90", b"\x00" * 1536)

    assert res.hit is True
    assert res.answer == "$13.50"
    assert res.matched_prompt == "What is a 15% tip on $90?"
    assert res.guardrail_blocked is False


@pytest.mark.anyio
async def test_system_prompt_isolation_prevents_cache_collision() -> None:
    settings = Settings()
    docs = [
        SimpleNamespace(
            prompt="Explain recursion",
            system_prompt="you are a pirate",
            answer="Arrr, recursion calls itself!",
            vector_distance="0.02",
        ),
        SimpleNamespace(
            prompt="Explain recursion",
            system_prompt="you are a formal professor",
            answer="Recursion is a method of self-referential definition.",
            vector_distance="0.03",
        ),
    ]
    ft_mock = MagicMock()
    ft_mock.search = AsyncMock(return_value=SimpleNamespace(docs=docs))
    redis_mock = MagicMock()
    redis_mock.ft.return_value = ft_mock

    cache = SemanticCache(redis_mock, settings)
    res_prof = await cache.lookup(
        "Explain recursion", b"\x00" * 1536, system_prompt="You are a formal professor"
    )
    assert res_prof.hit is True
    assert res_prof.answer == "Recursion is a method of self-referential definition."

    res_chef = await cache.lookup(
        "Explain recursion", b"\x00" * 1536, system_prompt="You are a chef"
    )
    assert res_chef.hit is False


@pytest.mark.anyio
async def test_explicit_model_override_and_bounded_generation_forwarding() -> None:
    settings = Settings()
    local = _DummyLLM("llama3.2:1b")
    cloud = _DummyLLM("gemini-3.5-flash-lite")
    router = ModelRouter(local, cloud, CircuitBreaker(3, 60.0), settings)

    # Complex prompt forced to local via model="local"
    c_local = router.classify("Write a Python binary search function", model="local")
    assert c_local.tier is RouteTier.LOCAL_SMALL

    # Simple prompt forced to cloud via model="cloud"
    c_cloud = router.classify("Hello there", model="cloud")
    assert c_cloud.tier is RouteTier.CLOUD_FRONTIER

    # Default max_tokens=None on ChatCompletionRequest so Cloud isn't truncated, while explicit values forward
    req_default = ChatCompletionRequest.model_validate({"prompt": "Hello", "temperature": 0.2})
    assert req_default.max_tokens is None
    assert req_default.temperature == 0.2

    await router.dispatch(
        req_default.to_turns(),
        c_cloud.tier,
        max_tokens=req_default.max_tokens,
        temperature=req_default.temperature,
    )
    assert cloud.last_kwargs == {"max_tokens": None, "temperature": 0.2}

    req_bounded = ChatCompletionRequest.model_validate(
        {"prompt": "Hello", "max_tokens": 256, "temperature": 0.1}
    )
    await router.dispatch(
        req_bounded.to_turns(),
        c_local.tier,
        max_tokens=req_bounded.max_tokens,
        temperature=req_bounded.temperature,
    )
    assert local.last_kwargs == {"max_tokens": 256, "temperature": 0.1}


@pytest.mark.anyio
async def test_missing_redis_index_self_heals_on_lookup() -> None:
    from redis.exceptions import ResponseError

    settings = Settings()
    ft_mock = MagicMock()
    ft_mock.search = AsyncMock(side_effect=ResponseError("idx:llm_cache: no such index"))
    ft_mock.info = AsyncMock(side_effect=ResponseError("Unknown Index name"))
    ft_mock.create_index = AsyncMock(return_value=True)
    redis_mock = MagicMock()
    redis_mock.ft.return_value = ft_mock

    cache = SemanticCache(redis_mock, settings)
    res = await cache.lookup("Hello", b"\x00" * 1536)
    assert res.hit is False
    ft_mock.create_index.assert_awaited_once()


@pytest.mark.anyio
async def test_route_isolation_prevents_local_cache_from_serving_cloud_override() -> None:
    settings = Settings()
    docs = [
        SimpleNamespace(
            prompt="Hello",
            system_prompt="",
            route="LOCAL",
            answer="1B local reply",
            vector_distance="0.01",
        )
    ]
    ft_mock = MagicMock()
    ft_mock.search = AsyncMock(return_value=SimpleNamespace(docs=docs))
    redis_mock = MagicMock()
    redis_mock.ft.return_value = ft_mock

    cache = SemanticCache(redis_mock, settings)
    assert (await cache.lookup("Hello", b"\x00" * 1536, route="LOCAL")).hit is True
    assert (await cache.lookup("Hello", b"\x00" * 1536, route="CLOUD")).hit is False

    # Legacy/untagged cache entry (route="") must not match when a specific route is requested
    docs[0].route = ""
    assert (await cache.lookup("Hello", b"\x00" * 1536, route="CLOUD")).hit is False


@pytest.mark.anyio
async def test_knn_finds_fourth_variant_past_three_guardrail_blocked_neighbors() -> None:
    settings = Settings()
    docs = [
        SimpleNamespace(prompt=f"Calculate taxes for {yr}", system_prompt="", route="LOCAL", answer=f"Tax {yr}", vector_distance="0.02")
        for yr in (2021, 2022, 2023, 2024)
    ]
    ft_mock = MagicMock()
    ft_mock.search = AsyncMock(return_value=SimpleNamespace(docs=docs))
    redis_mock = MagicMock()
    redis_mock.ft.return_value = ft_mock

    cache = SemanticCache(redis_mock, settings)
    assert "KNN 10" in cache._query.query_string()
    res = await cache.lookup("Calculate taxes for 2024", b"\x00" * 1536, route="LOCAL")
    assert res.hit is True
    assert res.answer == "Tax 2024"



